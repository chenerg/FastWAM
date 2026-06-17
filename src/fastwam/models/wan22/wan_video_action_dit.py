from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import DiTBlock, Head, sinusoidal_embedding_1d


class WanGridRotaryPosEmbed(nn.Module):
    """LingBot-VA style grid-id RoPE for video-like and action tokens."""

    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = int(attention_head_dim)
        self.max_seq_len = int(max_seq_len)
        self.theta = float(theta)

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.register_buffer("f_freqs_base", f_freqs_base, persistent=False)
        self.register_buffer("h_freqs_base", h_freqs_base, persistent=False)
        self.register_buffer("w_freqs_base", w_freqs_base, persistent=False)

    def _precompute_freqs_base(self):
        f_freqs_base = 1.0 / (
            self.theta ** (torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double() / self.f_dim)
        )
        h_freqs_base = 1.0 / (
            self.theta ** (torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double() / self.h_dim)
        )
        w_freqs_base = 1.0 / (
            self.theta ** (torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double() / self.w_dim)
        )
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        if grid_ids.ndim == 2:
            grid_ids = grid_ids.unsqueeze(0)
        if grid_ids.ndim != 3 or grid_ids.shape[1] < 3:
            raise ValueError(f"`grid_ids` must be [B,3+,L] or [3+,L], got {tuple(grid_ids.shape)}")
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            return torch.polar(torch.ones_like(freqs), freqs)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


def get_mesh_id(
    f: int,
    h: int,
    w: int,
    t: int | float,
    f_w: int | float = 1,
    f_shift: int | float = 0,
    *,
    action: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build LingBot-VA compatible grid ids.

    The fourth row carries diffusion/streaming metadata for compatibility with
    LingBot-VA's layout, but RoPE consumes only the first three rows.
    """

    f_idx = torch.arange(f_shift, f + f_shift, device=device) * f_w
    h_idx = torch.arange(h, device=device)
    w_idx = torch.arange(w, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h], device=device).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], float(t))], dim=0)
    return grid_id


class WanVideoActionDiT(nn.Module):
    """Single-block-stack DiT for joint video, depth-as-video, and action generation."""

    VIDEO_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
        "in_dim",
        "out_dim",
        "patch_size",
    )

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool = False,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = True,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_dim: int = 7,
        rope_max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        del has_image_input, has_image_pos_emb, has_ref_conv, add_control_adapter
        del in_dim_control_adapter, require_vae_embedding, require_clip_embedding

        self.hidden_dim = int(hidden_dim)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.freq_dim = int(freq_dim)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.seperated_timestep = bool(seperated_timestep)
        self.fuse_vae_embedding_in_latents = bool(fuse_vae_embedding_in_latents)
        self.action_dim = int(action_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {self.attn_head_dim}")
        if not self.seperated_timestep:
            raise NotImplementedError("WanVideoActionDiT currently expects `seperated_timestep=true`.")

        self.patch_embedding = nn.Conv3d(self.in_dim, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.depth_patch_embedding = deepcopy(self.patch_embedding)
        self.action_embedding = nn.Linear(self.action_dim, self.hidden_dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(self.hidden_dim, self.out_dim, self.patch_size, eps)
        self.depth_head = deepcopy(self.head)
        self.action_head = ActionHead(self.hidden_dim, self.action_dim, eps)
        self.rope = WanGridRotaryPosEmbed(
            attention_head_dim=self.attn_head_dim,
            max_seq_len=rope_max_seq_len,
            theta=rope_theta,
        )

    @classmethod
    def from_pretrained_payload(
        cls,
        video_action_dit_config: dict[str, Any],
        video_action_dit_pretrained_path: str | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "WanVideoActionDiT":
        if video_action_dit_config is None:
            raise ValueError("`video_action_dit_config` is required.")
        model = cls(**video_action_dit_config).to(device=device, dtype=torch_dtype)
        if not video_action_dit_pretrained_path:
            return model

        p = Path(video_action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        video_action_dit_pretrained_path = str(p)
        if not os.path.isfile(video_action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`video_action_dit_pretrained_path` does not exist: {video_action_dit_pretrained_path}"
            )

        payload = torch.load(video_action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid WanVideoActionDiT payload type from {video_action_dit_pretrained_path}: {type(payload)}"
            )

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {video_action_dit_pretrained_path}")
        expected_meta = {
            "hidden_dim": int(video_action_dit_config["hidden_dim"]),
            "ffn_dim": int(video_action_dit_config["ffn_dim"]),
            "num_layers": int(video_action_dit_config["num_layers"]),
            "num_heads": int(video_action_dit_config["num_heads"]),
            "attn_head_dim": int(video_action_dit_config["attn_head_dim"]),
            "text_dim": int(video_action_dit_config["text_dim"]),
            "freq_dim": int(video_action_dit_config["freq_dim"]),
            "eps": float(video_action_dit_config["eps"]),
            "in_dim": int(video_action_dit_config["in_dim"]),
            "out_dim": int(video_action_dit_config["out_dim"]),
            "patch_size": tuple(int(v) for v in video_action_dit_config["patch_size"]),
        }
        for key in cls.VIDEO_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {video_action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {video_action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif key == "patch_size":
                if tuple(int(v) for v in got_value) != tuple(expected_value):
                    raise ValueError(
                        f"`meta.{key}` mismatch in {video_action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {video_action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        initialized_state_dict = payload.get("initialized_state_dict")
        if not isinstance(initialized_state_dict, dict):
            raise ValueError(
                f"`initialized_state_dict` must be a dict in {video_action_dit_pretrained_path}, "
                f"got {type(initialized_state_dict)}"
            )

        current_state = model.state_dict()
        unexpected_keys = sorted(set(initialized_state_dict) - set(current_state))
        if unexpected_keys:
            raise ValueError(
                "WanVideoActionDiT init payload has unexpected keys: "
                f"{unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )
        merged_state = dict(current_state)
        for key, value in initialized_state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`initialized_state_dict[{key}]` must be torch.Tensor in "
                    f"{video_action_dit_pretrained_path}, got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {video_action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)
        model.load_state_dict(merged_state, strict=True)
        return model

    def patchify(self, x: torch.Tensor, *, modality: str) -> torch.Tensor:
        if modality == "video":
            return self.patch_embedding(x)
        if modality == "depth":
            return self.depth_patch_embedding(x)
        raise ValueError(f"Patchify only supports video/depth, got {modality!r}")

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        f, h, w = grid_size
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f,
            h=h,
            w=w,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def _build_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        if context_mask.ndim != 2:
            raise ValueError(f"`context_mask` must be [B,L], got {tuple(context_mask.shape)}")
        context_emb = self.text_embedding(context)
        return context_emb, context_mask.unsqueeze(1).expand(-1, seq_len, -1)

    def _token_timestep_mod(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        seq_shape: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        *,
        zero_first_frame: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must be 1 or batch size {batch_size}, got {timestep.shape[0]}")

        num_groups, tokens_per_group = seq_shape
        token_timesteps = torch.ones((batch_size, num_groups, tokens_per_group), dtype=timestep.dtype, device=device)
        token_timesteps = token_timesteps * timestep.view(batch_size, 1, 1)
        if zero_first_frame and num_groups > 0:
            token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim).to(dtype=dtype)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        return t, t_mod

    def pre_dit_video_like(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        modality: str,
        grid_t: int | float = 0,
        frame_shift: int | float = 0,
    ) -> Dict[str, Any]:
        if x.ndim != 5:
            raise ValueError(f"`{modality}` latents must be [B,C,T,H,W], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        x = self.patchify(x, modality=modality)
        f, h, w = x.shape[2:]
        tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        tokens_per_frame = h * w

        t, t_mod = self._token_timestep_mod(
            timestep=timestep,
            batch_size=batch_size,
            seq_shape=(f, tokens_per_frame),
            dtype=tokens.dtype,
            device=tokens.device,
            zero_first_frame=bool(self.fuse_vae_embedding_in_latents),
        )
        context_emb, context_attn_mask = self._build_context(context, context_mask, tokens.shape[1])
        grid_id = get_mesh_id(f, h, w, grid_t, f_shift=frame_shift, device=tokens.device)
        freqs = self.rope(grid_id)[0].view(tokens.shape[1], 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "modality": modality,
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def pre_dit_action(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        action_per_frame: Optional[int] = None,
        grid_t: int | float = 1,
        frame_shift: int | float = 0,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}")
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}")
        batch_size, seq_len, _ = action_tokens.shape
        if action_per_frame is None:
            action_per_frame = seq_len
        if seq_len % int(action_per_frame) != 0:
            raise ValueError(f"Action length {seq_len} must be divisible by action_per_frame={action_per_frame}")
        f = seq_len // int(action_per_frame)
        h = int(action_per_frame)
        w = 1
        tokens = self.action_embedding(action_tokens)
        t, t_mod = self._token_timestep_mod(
            timestep=timestep,
            batch_size=batch_size,
            seq_shape=(f, h * w),
            dtype=tokens.dtype,
            device=tokens.device,
            zero_first_frame=False,
        )
        context_emb, context_attn_mask = self._build_context(context, context_mask, tokens.shape[1])
        grid_id = get_mesh_id(f, h, w, grid_t, f_shift=frame_shift, action=True, device=tokens.device)
        freqs = self.rope(grid_id)[0].view(tokens.shape[1], 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "modality": "action",
                "grid_size": (f, h, w),
                "action_per_frame": int(action_per_frame),
                "batch_size": batch_size,
            },
        }

    def _run_blocks(
        self,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask, self_attn_mask=self_attn_mask)
        return x

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        modality = pre_state["meta"]["modality"]
        if modality == "action":
            return self.action_head(tokens, pre_state["t"])
        if modality == "video":
            x = self.head(tokens, pre_state["t"])
            return self.unpatchify(x, pre_state["meta"]["grid_size"])
        if modality == "depth":
            x = self.depth_head(tokens, pre_state["t"])
            return self.unpatchify(x, pre_state["meta"]["grid_size"])
        raise ValueError(f"Unsupported modality: {modality}")

    def forward_modality(self, pre_state: Dict[str, Any], self_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self._run_blocks(
            tokens=pre_state["tokens"],
            freqs=pre_state["freqs"],
            t_mod=pre_state["t_mod"],
            context=pre_state["context"],
            context_mask=pre_state["context_mask"],
            self_attn_mask=self_attn_mask,
        )
        return self.post_dit(tokens, pre_state)

    def forward_joint(
        self,
        pre_states: Dict[str, Dict[str, Any]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        order = [name for name in ("video", "depth", "action") if name in pre_states and pre_states[name] is not None]
        if not order:
            raise ValueError("`pre_states` must contain at least one modality.")

        tokens = torch.cat([pre_states[name]["tokens"] for name in order], dim=1)
        freqs = torch.cat([pre_states[name]["freqs"] for name in order], dim=0)
        t_mod = torch.cat([pre_states[name]["t_mod"] for name in order], dim=1)
        context = pre_states[order[0]]["context"]
        context_mask = torch.cat([pre_states[name]["context_mask"] for name in order], dim=1)

        if attention_mask is not None and attention_mask.shape != (tokens.shape[1], tokens.shape[1]):
            raise ValueError(
                f"`attention_mask` must be [{tokens.shape[1]},{tokens.shape[1]}], got {tuple(attention_mask.shape)}"
            )
        mixed_tokens = self._run_blocks(tokens, freqs, t_mod, context, context_mask, self_attn_mask=attention_mask)

        out: Dict[str, torch.Tensor] = {}
        start = 0
        for name in order:
            seq_len = pre_states[name]["tokens"].shape[1]
            out[name] = self.post_dit(mixed_tokens[:, start : start + seq_len], pre_states[name])
            start += seq_len
        return out
