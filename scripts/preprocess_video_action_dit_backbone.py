import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.models.wan22.wan_video_action_dit import WanVideoActionDiT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _filter_kwargs_for(cls, cfg: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    return {k: v for k, v in cfg.items() if k in allowed}


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    cfg = OmegaConf.load(str(path))
    if "video_action_dit_config" not in cfg:
        raise ValueError(f"`{path}` must contain `video_action_dit_config` at top level.")

    dit_cfg = OmegaConf.to_container(cfg.video_action_dit_config, resolve=False)
    if not isinstance(dit_cfg, dict):
        raise ValueError("`video_action_dit_config` must resolve to a dict.")

    if _is_unresolved_interpolation(dit_cfg.get("action_dim")):
        print("[WARN] `video_action_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.")
        dit_cfg["action_dim"] = 7

    video_cfg = _filter_kwargs_for(WanVideoDiT, dit_cfg)
    return video_cfg, dit_cfg, cfg


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def _copy_or_resize(
    src_key: str,
    dst_key: str,
    video_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    initialized_state_dict: dict[str, torch.Tensor],
) -> str:
    if src_key not in video_state:
        raise ValueError(f"Key `{src_key}` not found in video expert state dict.")
    if dst_key not in target_state:
        raise ValueError(f"Key `{dst_key}` not found in WanVideoActionDiT state dict.")
    src = video_state[src_key]
    target = target_state[dst_key]
    if tuple(src.shape) == tuple(target.shape):
        value = src
        mode = "copied"
    else:
        value = _resize_tensor_to_shape(src, tuple(target.shape))
        mode = "interpolated"
    initialized_state_dict[dst_key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess WanVideoActionDiT partial initialization weights from WanVideoDiT "
            "and save as .pt payload."
        )
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml, e.g. configs/model/fastwam_one.yaml")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed WanVideoActionDiT init.")
    parser.add_argument("--device", default="cpu", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_cfg, dit_cfg, cfg = _load_model_config(model_config_path)
    torch_dtype = _parse_dtype(args.dtype)
    redirect_common_files = bool(cfg.get("redirect_common_files", False))

    int_fields = [
        "hidden_dim",
        "in_dim",
        "ffn_dim",
        "out_dim",
        "text_dim",
        "freq_dim",
        "num_heads",
        "attn_head_dim",
        "num_layers",
        "action_dim",
    ]
    for key in int_fields:
        dit_cfg[key] = _require_int_config(dit_cfg, key)
    dit_cfg["eps"] = _require_float_config(dit_cfg, "eps")
    dit_cfg["patch_size"] = [int(v) for v in dit_cfg["patch_size"]]

    print(
        f"[INFO] Loaded model config from {model_config_path}. "
        f"Preprocessing WanVideoActionDiT partial init with dtype={torch_dtype} on device={args.device}."
    )
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        redirect_common_files=redirect_common_files,
        dit_config=video_cfg,
        load_text_encoder=False,
    )
    video_expert = components.dit
    video_action_expert = WanVideoActionDiT(**dit_cfg).to(device=args.device, dtype=torch_dtype)

    target_state = video_action_expert.state_dict()
    video_state = video_expert.state_dict()
    initialized_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0

    for dst_key in sorted(target_state.keys()):
        src_key = None
        if dst_key in video_state:
            src_key = dst_key
        elif dst_key.startswith("depth_patch_embedding."):
            src_key = "patch_embedding." + dst_key.removeprefix("depth_patch_embedding.")
        elif dst_key.startswith("depth_head."):
            src_key = "head." + dst_key.removeprefix("depth_head.")

        if src_key is None:
            continue
        mode = _copy_or_resize(
            src_key=src_key,
            dst_key=dst_key,
            video_state=video_state,
            target_state=target_state,
            initialized_state_dict=initialized_state_dict,
        )
        if mode == "copied":
            copied += 1
        else:
            interpolated += 1

    payload = {
        "policy": {
            "source": "WanVideoDiT",
            "direct_keys": "same-name WanVideoActionDiT keys are initialized from WanVideoDiT",
            "mapped_keys": {
                "depth_patch_embedding.*": "patch_embedding.*",
                "depth_head.*": "head.*",
            },
            "random_kept_prefixes": ["action_embedding.", "action_head."],
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "initialized_state_dict": initialized_state_dict,
        "meta": {
            "hidden_dim": int(dit_cfg["hidden_dim"]),
            "ffn_dim": int(dit_cfg["ffn_dim"]),
            "num_layers": int(dit_cfg["num_layers"]),
            "num_heads": int(dit_cfg["num_heads"]),
            "attn_head_dim": int(dit_cfg["attn_head_dim"]),
            "text_dim": int(dit_cfg["text_dim"]),
            "freq_dim": int(dit_cfg["freq_dim"]),
            "eps": float(dit_cfg["eps"]),
            "in_dim": int(dit_cfg["in_dim"]),
            "out_dim": int(dit_cfg["out_dim"]),
            "patch_size": tuple(int(v) for v in dit_cfg["patch_size"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(target_state) - len(initialized_state_dict)
    print(
        "[INFO] Saved WanVideoActionDiT init payload to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
