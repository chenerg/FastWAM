from __future__ import annotations

import inspect
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from omegaconf import DictConfig, OmegaConf

from fastwam.utils.logging_config import get_logger

from .helpers.loader import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_action_dit import WanVideoActionDiT

logger = get_logger(__name__)


def _to_plain_dict(cfg, name: str) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(cfg)}")
    return dict(cfg)


def _filter_kwargs_for(cls, cfg: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    return {k: v for k, v in cfg.items() if k in allowed}


class FastWAMOne(nn.Module):
    """FastWAM-one: one shared DiT stack for video, depth, and action."""

    def __init__(
        self,
        dit: WanVideoActionDiT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_depth: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_per_frame: Optional[int] = None,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = None
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_depth = float(loss_lambda_depth)
        self.loss_lambda_action = float(loss_lambda_action)
        self.action_per_frame = None if action_per_frame is None else int(action_per_frame)
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_action_dit_config: dict[str, Any] | None = None,
        skip_dit_load_from_pretrain: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_depth: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_per_frame: Optional[int] = None,
    ) -> "FastWAMOne":
        if video_action_dit_config is None:
            raise ValueError("`video_action_dit_config` is required for FastWAMOne.")
        dit_cfg = _to_plain_dict(video_action_dit_config, "video_action_dit_config")
        if "text_dim" not in dit_cfg:
            raise ValueError("`video_action_dit_config['text_dim']` is required.")

        from .wan_video_dit import WanVideoDiT

        base_video_cfg = _filter_kwargs_for(WanVideoDiT, dit_cfg)
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=base_video_cfg,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        dit = WanVideoActionDiT(**_filter_kwargs_for(WanVideoActionDiT, dit_cfg)).to(device=device, dtype=torch_dtype)
        missing, unexpected = dit.load_state_dict(components.dit.state_dict(), strict=False)
        logger.info(
            "Initialized FastWAMOne DiT from WanVideoDiT weights: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

        model = cls(
            dit=dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(dit_cfg["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_depth=loss_lambda_depth,
            loss_lambda_action=loss_lambda_action,
            action_per_frame=action_per_frame,
        )
        model.model_paths = {
            "video_action_dit_init": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(device=self.device), torch.ones_like(mask, dtype=torch.bool)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be [B,D], got {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1))
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor: torch.Tensor, tiled: bool = False):
        return self.vae.encode(video_tensor, device=self.device, tiled=tiled)

    @staticmethod
    def _check_resize_height_width(height: int, width: int, num_frames: int):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled: bool = False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        image = input_image.to(device=self.device, dtype=self.torch_dtype)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _prepare_video_like_tensor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"`sample['{name}']` must be [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1, -1)
        if x.shape[1] != 3:
            raise ValueError(f"`sample['{name}']` channel dim must be 1 or 3, got {x.shape[1]}")
        return x.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

    def build_inputs(self, sample: dict[str, Any], tiled: bool = False):
        if "video" not in sample:
            raise ValueError("`sample['video']` is required.")
        if "action" not in sample:
            raise ValueError("`sample['action']` is required.")
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("FastWAMOne training requires cached `context/context_mask`.")

        video = self._prepare_video_like_tensor(sample["video"], "video")
        input_latents = self._encode_video_latents(video, tiled=tiled)
        first_frame_latents = input_latents[:, :, 0:1].clone() if self.dit.fuse_vae_embedding_in_latents else None

        depth_latents = None
        first_depth_latents = None
        if sample.get("depth") is not None:
            depth = self._prepare_video_like_tensor(sample["depth"], "depth")
            depth_latents = self._encode_video_latents(depth, tiled=tiled)
            first_depth_latents = depth_latents[:, :, 0:1].clone() if self.dit.fuse_vae_embedding_in_latents else None

        context = sample["context"]
        context_mask = sample["context_mask"]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}")
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio = sample.get("proprio", None)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim == 3:
                proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")
        if action.shape[2] != self.dit.action_dim:
            raise ValueError(f"Action dim mismatch: sample={action.shape[2]} model={self.dit.action_dim}")
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "video_latents": input_latents,
            "depth_latents": depth_latents,
            "first_frame_latents": first_frame_latents,
            "first_depth_latents": first_depth_latents,
            "action": action,
            "action_is_pad": sample.get("action_is_pad", None),
            "image_is_pad": sample.get("image_is_pad", None),
            "depth_is_pad": sample.get("depth_is_pad", sample.get("image_is_pad", None)),
        }

    @staticmethod
    def _loss_video_like_per_sample(pred: torch.Tensor, target: torch.Tensor, is_pad: Optional[torch.Tensor]) -> torch.Tensor:
        loss_token = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 3, 4))
        if is_pad is None:
            return loss_token.mean(dim=1)
        if is_pad.ndim != 2:
            raise ValueError(f"`*_is_pad` must be [B,T], got {tuple(is_pad.shape)}")
        # Latent time usually maps first frame plus groups of original frames.
        if is_pad.shape[1] != loss_token.shape[1]:
            if is_pad.shape[1] > loss_token.shape[1]:
                is_pad = is_pad[:, : loss_token.shape[1]]
            else:
                pad = torch.zeros(
                    (is_pad.shape[0], loss_token.shape[1] - is_pad.shape[1]),
                    dtype=is_pad.dtype,
                    device=is_pad.device,
                )
                is_pad = torch.cat([is_pad, pad], dim=1)
        valid = (~is_pad.to(device=loss_token.device, dtype=torch.bool)).to(dtype=loss_token.dtype)
        return (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _build_joint_pre_states(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        latents_depth: Optional[torch.Tensor] = None,
        timestep_depth: Optional[torch.Tensor] = None,
    ) -> dict[str, dict[str, Any]]:
        pre_states = {
            "video": self.dit.pre_dit_video_like(
                latents_video,
                timestep_video,
                context,
                context_mask,
                modality="video",
                grid_t=0,
            ),
            "action": self.dit.pre_dit_action(
                latents_action,
                timestep_action,
                context,
                context_mask,
                action_per_frame=self.action_per_frame,
                grid_t=1,
            ),
        }
        if latents_depth is not None:
            if timestep_depth is None:
                timestep_depth = timestep_video
            pre_states["depth"] = self.dit.pre_dit_video_like(
                latents_depth,
                timestep_depth,
                context,
                context_mask,
                modality="depth",
                grid_t=0,
            )
        return pre_states

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        video = inputs["video_latents"]
        depth = inputs["depth_latents"]
        action = inputs["action"]
        batch_size = video.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        noise_video = torch.randn_like(video)
        timestep_video = self.train_video_scheduler.sample_training_t(batch_size, self.device, video.dtype)
        noisy_video = self.train_video_scheduler.add_noise(video, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(video, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            noisy_video[:, :, 0:1] = inputs["first_frame_latents"]

        noisy_depth = None
        target_depth = None
        timestep_depth = None
        if depth is not None:
            noise_depth = torch.randn_like(depth)
            timestep_depth = self.train_video_scheduler.sample_training_t(batch_size, self.device, depth.dtype)
            noisy_depth = self.train_video_scheduler.add_noise(depth, noise_depth, timestep_depth)
            target_depth = self.train_video_scheduler.training_target(depth, noise_depth, timestep_depth)
            if inputs["first_depth_latents"] is not None:
                noisy_depth[:, :, 0:1] = inputs["first_depth_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(batch_size, self.device, action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        pre_states = self._build_joint_pre_states(
            latents_video=noisy_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            latents_action=noisy_action,
            timestep_action=timestep_action,
            latents_depth=noisy_depth,
            timestep_depth=timestep_depth,
        )
        pred = self.dit.forward_joint(pre_states)

        pred_video = pred["video"]
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._loss_video_like_per_sample(
            pred_video,
            target_video,
            None if inputs["image_is_pad"] is None else inputs["image_is_pad"].to(self.device, dtype=torch.bool),
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(loss_video_per_sample)
        loss_video = (loss_video_per_sample * video_weight).mean()

        loss_depth = torch.zeros_like(loss_video)
        if depth is not None and target_depth is not None:
            pred_depth = pred["depth"]
            if inputs["first_depth_latents"] is not None:
                pred_depth = pred_depth[:, :, 1:]
                target_depth = target_depth[:, :, 1:]
            loss_depth_per_sample = self._loss_video_like_per_sample(
                pred_depth,
                target_depth,
                None if inputs["depth_is_pad"] is None else inputs["depth_is_pad"].to(self.device, dtype=torch.bool),
            )
            depth_weight = self.train_video_scheduler.training_weight(timestep_depth).to(loss_depth_per_sample)
            loss_depth = (loss_depth_per_sample * depth_weight).mean()

        action_loss_token = F.mse_loss(pred["action"].float(), target_action.float(), reduction="none").mean(dim=2)
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            valid = (~action_is_pad.to(device=action_loss_token.device, dtype=torch.bool)).to(action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(action_loss_per_sample)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_depth * loss_depth
            + self.loss_lambda_action * loss_action
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_depth": self.loss_lambda_depth * float(loss_depth.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "dit" not in payload:
            raise ValueError(f"FastWAMOne checkpoint missing `dit` key: {path}")
        self.dit.load_state_dict(payload["dit"], strict=False)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, action_cfg_scale
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, int(num_frames))
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f"`input_image` must be pre-resized to multiples of 16, got HxW=({height},{width})")
        if checked_t != int(num_frames):
            raise ValueError(f"`num_frames` must satisfy T % 4 == 1, got {num_frames}")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        latent_t = (int(num_frames) - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, int(action_horizon), self.dit.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        first_frame_latents = self._encode_input_image_latents_tensor(input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            pre_states = self._build_joint_pre_states(
                latents_video=latents_video,
                timestep_video=step_t_video.unsqueeze(0).to(dtype=latents_video.dtype),
                context=context,
                context_mask=context_mask,
                latents_action=latents_action,
                timestep_action=step_t_action.unsqueeze(0).to(dtype=latents_action.dtype),
            )
            pred = self.dit.forward_joint(pre_states)
            latents_video = self.infer_video_scheduler.step(pred["video"], step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred["action"], step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
