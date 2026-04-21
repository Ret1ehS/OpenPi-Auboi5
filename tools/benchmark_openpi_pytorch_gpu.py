#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.machinery
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import safetensors.torch
import sentencepiece as spm
import torch
import torch.nn.functional as F  # noqa: N812


def _install_pytest_stub() -> None:
    if "pytest" in sys.modules:
        return

    try:
        importlib.import_module("pytest")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("pytest")

    class Cache:  # pragma: no cover - import stub only
        pass

    module.Cache = Cache
    module.__spec__ = importlib.machinery.ModuleSpec("pytest", loader=None)
    sys.modules["pytest"] = module


def _install_gemma_stub() -> None:
    if "openpi.models.gemma" in sys.modules:
        return

    parent = importlib.import_module("openpi.models")
    module = types.ModuleType("openpi.models.gemma")

    @dataclasses.dataclass
    class Config:
        width: int
        depth: int
        mlp_dim: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        lora_configs: dict[str, object] = dataclasses.field(default_factory=dict)

    def get_config(variant: str) -> Config:
        if variant == "dummy":
            return Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
        if variant == "gemma_300m":
            return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
        if variant == "gemma_2b":
            return Config(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
        if variant == "gemma_2b_lora":
            return Config(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
        if variant == "gemma_300m_lora":
            return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
        raise ValueError(f"Unsupported gemma variant: {variant}")

    module.Config = Config
    module.Variant = str
    module.get_config = get_config
    sys.modules["openpi.models.gemma"] = module
    setattr(parent, "gemma", module)


def _resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + rem_w
    value = 0 if images.dtype == torch.uint8 else -1.0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=value)

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)
        if batch_size == 1:
            padded = padded.squeeze(0)
    return padded


def _install_image_tools_stub() -> None:
    if "openpi.shared.image_tools" in sys.modules:
        return

    parent = importlib.import_module("openpi.shared")
    module = types.ModuleType("openpi.shared.image_tools")
    module.resize_with_pad_torch = _resize_with_pad_torch
    sys.modules["openpi.shared.image_tools"] = module
    setattr(parent, "image_tools", module)


def _prepare_imports(repo_root: Path) -> None:
    paths = [
        repo_root / "src",
        repo_root / "packages" / "openpi-client" / "src",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    _install_pytest_stub()
    _install_gemma_stub()
    _install_image_tools_stub()


@dataclasses.dataclass(frozen=True)
class Pi0TorchConfig:
    action_dim: int
    action_horizon: int
    paligemma_variant: str
    action_expert_variant: str
    precision: str = "bfloat16"
    pi05: bool = True
    dtype: str = "bfloat16"
    max_token_len: int = 200
    pytorch_compile_mode: str | None = None


@dataclasses.dataclass
class SimpleObservation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None


class PaligemmaTokenizerLite:
    def __init__(self, tokenizer_model: Path, max_len: int) -> None:
        self._max_len = max_len
        self._tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_model))

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized.tolist()))
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")

        if len(tokens) < self._max_len:
            pad_len = self._max_len - len(tokens)
            mask = [True] * len(tokens) + [False] * pad_len
            tokens = tokens + [False] * pad_len
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)


def _load_checkpoint_config(checkpoint_dir: Path, *, compile_mode: str | None) -> Pi0TorchConfig:
    config_path = checkpoint_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    precision = str(config_data.get("precision", "bfloat16"))
    return Pi0TorchConfig(
        action_dim=int(config_data["action_dim"]),
        action_horizon=int(config_data["action_horizon"]),
        paligemma_variant=str(config_data["paligemma_variant"]),
        action_expert_variant=str(config_data["action_expert_variant"]),
        precision=precision,
        dtype=precision,
        pytorch_compile_mode=compile_mode,
    )


def _find_norm_stats_path(checkpoint_dir: Path) -> Path:
    matches = sorted(checkpoint_dir.glob("assets/**/norm_stats.json"))
    if not matches:
        raise FileNotFoundError(f"No norm_stats.json found under {checkpoint_dir / 'assets'}")
    return matches[0]


def _load_norm_stats(norm_stats_path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(norm_stats_path.read_text(encoding="utf-8"))["norm_stats"]
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, stats in raw.items():
        result[key] = {
            stat_name: None if stat_value is None else np.asarray(stat_value, dtype=np.float32)
            for stat_name, stat_value in stats.items()
        }
    return result


def _normalize_quantile(x: np.ndarray, stats: dict[str, np.ndarray | None]) -> np.ndarray:
    q01 = stats["q01"]
    q99 = stats["q99"]
    if q01 is None or q99 is None:
        raise ValueError("Quantile stats are required for PI05 normalization")
    q01 = q01[..., : x.shape[-1]]
    q99 = q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _unnormalize_quantile(x: np.ndarray, stats: dict[str, np.ndarray | None]) -> np.ndarray:
    q01 = stats["q01"]
    q99 = stats["q99"]
    if q01 is None or q99 is None:
        raise ValueError("Quantile stats are required for PI05 unnormalization")
    if q01.shape[-1] < x.shape[-1]:
        dim = q01.shape[-1]
        head = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _pad_to_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[-1] >= target_dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - x.shape[-1])
    return np.pad(x, pad_width, constant_values=0.0)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
    return image


def _image_to_model_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = _parse_image(image)
    tensor = torch.from_numpy(image[None, ...]).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor / 255.0 * 2.0 - 1.0


def _make_observation(
    *,
    device: torch.device,
    config: Pi0TorchConfig,
    tokenizer: PaligemmaTokenizerLite,
    norm_stats: dict[str, dict[str, np.ndarray]],
    prompt: str,
    state: np.ndarray,
    base_image: np.ndarray,
    wrist_image: np.ndarray,
) -> SimpleObservation:
    state = np.asarray(state, dtype=np.float32)
    state = _normalize_quantile(state, norm_stats["state"])
    tokenized_prompt, tokenized_prompt_mask = tokenizer.tokenize(prompt, state)
    padded_state = _pad_to_dim(state[None, :], config.action_dim)
    base_image = _parse_image(base_image)
    wrist_image = _parse_image(wrist_image)
    right_wrist_image = np.zeros_like(base_image)

    return SimpleObservation(
        images={
            "base_0_rgb": _image_to_model_tensor(base_image, device),
            "left_wrist_0_rgb": _image_to_model_tensor(wrist_image, device),
            "right_wrist_0_rgb": _image_to_model_tensor(right_wrist_image, device),
        },
        image_masks={
            "base_0_rgb": torch.tensor([True], dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.tensor([True], dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.tensor([False], dtype=torch.bool, device=device),
        },
        state=torch.from_numpy(padded_state).to(device=device),
        tokenized_prompt=torch.from_numpy(tokenized_prompt[None, ...]).to(device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.from_numpy(tokenized_prompt_mask[None, ...]).to(device=device, dtype=torch.bool),
    )


def _synthetic_inputs(state_dim: int, image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.zeros((state_dim,), dtype=np.float32)
    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    wrist_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    return state, base_image, wrist_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenPI PyTorch checkpoint on GPU without JAX runtime.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, default=Path("/home/niic/.cache/openpi/big_vision/paligemma_tokenizer.model"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="pick up the blue cube")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--compile-mode", type=str, default="none")
    args = parser.parse_args()

    compile_mode = None if args.compile_mode.strip().lower() in {"", "none", "null"} else args.compile_mode.strip()
    _prepare_imports(args.repo_root.resolve())

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    checkpoint_dir = args.checkpoint_dir.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    config = _load_checkpoint_config(checkpoint_dir, compile_mode=compile_mode)
    tokenizer = PaligemmaTokenizerLite(args.tokenizer_model.resolve(), config.max_token_len)
    norm_stats = _load_norm_stats(_find_norm_stats_path(checkpoint_dir))
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing PyTorch weight file: {weight_path}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    load_t0 = time.perf_counter()
    model = PI0Pytorch(config=config)
    if compile_mode is None:
        try:
            torch_dynamo = importlib.import_module("torch._dynamo")
            model.sample_actions = torch_dynamo.disable(model.sample_actions)
        except Exception:
            pass
    safetensors.torch.load_model(model, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params(config.dtype)
    model = model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_s = time.perf_counter() - load_t0

    state, base_image, wrist_image = _synthetic_inputs(args.state_dim, args.image_size)
    observation = _make_observation(
        device=device,
        config=config,
        tokenizer=tokenizer,
        norm_stats=norm_stats,
        prompt=args.prompt,
        state=state,
        base_image=base_image,
        wrist_image=wrist_image,
    )

    timings_ms: list[float] = []
    action_preview: list[float] | None = None
    for index in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            actions = model.sample_actions(str(device), observation, num_steps=args.num_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        timings_ms.append(infer_ms)

        actions_np = actions[0].detach().cpu().numpy()
        actions_np = _unnormalize_quantile(actions_np, norm_stats["actions"])
        actions_np = actions_np[:, :7]
        if action_preview is None:
            action_preview = actions_np[0].astype(float).tolist()

        print(
            json.dumps(
                {
                    "iteration": index,
                    "infer_ms": round(infer_ms, 3),
                    "action_shape": list(actions_np.shape),
                    "first_action": [round(value, 6) for value in actions_np[0].astype(float).tolist()],
                },
                ensure_ascii=False,
            )
        )

    summary = {
        "device": str(device),
        "load_s": round(load_s, 3),
        "iterations": len(timings_ms),
        "num_steps": args.num_steps,
        "compile_mode": compile_mode,
        "infer_ms_first": round(timings_ms[0], 3),
        "infer_ms_best": round(min(timings_ms), 3),
        "infer_ms_last": round(timings_ms[-1], 3),
        "infer_ms_mean_excl_first": round(float(np.mean(timings_ms[1:] or timings_ms)), 3),
        "action_preview": [round(value, 6) for value in (action_preview or [])],
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
