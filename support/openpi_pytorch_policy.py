#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np
import safetensors.torch
import sentencepiece as spm
import torch
import torch.nn.functional as F  # noqa: N812


DEFAULT_TOKENIZER_MODEL = Path("~/.cache/openpi/big_vision/paligemma_tokenizer.model").expanduser()
IMAGE_SIZE = 224
_SYSTEM_TENSORRT_PYTHON_PATHS = (
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.10/dist-packages",
)


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


def prepare_openpi_pytorch_imports(repo_root: Path) -> None:
    for path in (repo_root / "src", repo_root / "packages" / "openpi-client" / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    _install_pytest_stub()
    _install_gemma_stub()
    _install_image_tools_stub()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _add_system_tensorrt_python_paths() -> None:
    for raw_path in _SYSTEM_TENSORRT_PYTHON_PATHS:
        path = Path(raw_path)
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)


class _TrtImageEmbedWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._model.paligemma_with_expert.embed_image(image)


def _maybe_build_trt_image_embedder(model: torch.nn.Module, example_input: torch.Tensor) -> torch.nn.Module | None:
    if not _env_truthy("OPENPI_PYTORCH_TRT_VISION"):
        return None

    _add_system_tensorrt_python_paths()
    import torch_tensorrt

    cache_dir = Path(
        os.environ.get(
            "OPENPI_PYTORCH_TRT_CACHE_DIR",
            "~/.cache/openpi/torch_tensorrt/vision_embed",
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    wrapper = _TrtImageEmbedWrapper(model).eval().to(device=example_input.device)
    return torch_tensorrt.compile(
        wrapper,
        ir="dynamo",
        inputs=[example_input],
        enabled_precisions={torch.float32, torch.bfloat16},
        cache_built_engines=True,
        reuse_cached_engines=True,
        engine_cache_dir=str(cache_dir),
    )


def _install_mask_aware_prefix_fastpath(model: torch.nn.Module) -> None:
    original = getattr(model, "embed_prefix")

    def embed_prefix_mask_aware(self, images, img_masks, lang_tokens, lang_masks):
        embs = []
        pad_masks = []
        att_masks = []
        cached_img_shape: tuple[int, int] | None = None
        cached_img_dtype: torch.dtype | None = None

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(image_tensor: torch.Tensor) -> torch.Tensor:
                return self.paligemma_with_expert.embed_image(image_tensor)

            batch_size = img.shape[0]
            all_masked = bool((~img_mask).all().item())
            if all_masked and cached_img_shape is not None and cached_img_dtype is not None:
                num_img_embs, emb_dim = cached_img_shape
                img_emb = torch.zeros((batch_size, num_img_embs, emb_dim), dtype=cached_img_dtype, device=img.device)
            else:
                trt_image_embedder = getattr(self, "_openpi_trt_image_embedder", None)
                if trt_image_embedder is not None:
                    img_emb = trt_image_embedder(img)
                else:
                    img_emb = self._apply_checkpoint(image_embed_func, img)
                cached_img_shape = (img_emb.shape[1], img_emb.shape[2])
                cached_img_dtype = img_emb.dtype

            batch_size, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_token_tensor: torch.Tensor) -> torch.Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_token_tensor)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_tensor = att_masks_tensor[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks_tensor

    setattr(model, "_openpi_original_embed_prefix", original)
    model.embed_prefix = types.MethodType(embed_prefix_mask_aware, model)


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
            tokens = tokens + [0] * pad_len
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)


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


def _parse_image(image: Any) -> np.ndarray:
    if isinstance(image, str):
        from PIL import Image

        return np.asarray(Image.open(image))
    if isinstance(image, (bytes, bytearray)):
        import io
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(image)))
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            import io
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(image["bytes"])))
        if image.get("path"):
            from PIL import Image

            return np.asarray(Image.open(image["path"]))

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {array.shape}")
    return array


def _image_to_model_tensor(image: Any, device: torch.device) -> torch.Tensor:
    array = _parse_image(image)
    tensor = torch.from_numpy(array[None, ...]).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        tensor = _resize_with_pad_torch(tensor, IMAGE_SIZE, IMAGE_SIZE)
    return tensor / 255.0 * 2.0 - 1.0


def _find_norm_stats_path(checkpoint_dir: Path) -> Path:
    matches = sorted(checkpoint_dir.glob("assets/**/norm_stats.json"))
    if not matches:
        raise FileNotFoundError(f"No norm_stats.json found under {checkpoint_dir / 'assets'}")
    return matches[0]


def _load_norm_stats(norm_stats_path: Path) -> dict[str, dict[str, np.ndarray | None]]:
    raw = json.loads(norm_stats_path.read_text(encoding="utf-8"))["norm_stats"]
    result: dict[str, dict[str, np.ndarray | None]] = {}
    for key, stats in raw.items():
        result[key] = {
            stat_name: None if stat_value is None else np.asarray(stat_value, dtype=np.float32)
            for stat_name, stat_value in stats.items()
        }
    return result


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


def _resolve_prompt(prompt_value: Any, default_prompt: str | None) -> str:
    if prompt_value is None:
        if default_prompt is None:
            raise ValueError("Prompt is required but neither observation nor default_prompt provided.")
        return default_prompt
    if isinstance(prompt_value, np.ndarray):
        if prompt_value.shape == ():
            prompt_value = prompt_value.item()
        elif prompt_value.size == 1:
            prompt_value = prompt_value.reshape(()).item()
    if isinstance(prompt_value, bytes):
        prompt_value = prompt_value.decode("utf-8")
    return str(prompt_value)


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device


def _resolve_compile_mode(explicit: str | None = None) -> str | None:
    raw = explicit if explicit is not None else os.environ.get("OPENPI_PYTORCH_COMPILE_MODE", "")
    value = str(raw).strip().lower()
    if value in {"", "none", "null", "false", "off"}:
        return None
    return str(raw).strip()


@dataclasses.dataclass
class SyntheticObservationSpec:
    prompt: str = "pick up the blue cube"
    state_dim: int = 8
    image_size: int = IMAGE_SIZE


class OpenPIPyTorchPolicy:
    def __init__(
        self,
        *,
        repo_root: Path,
        checkpoint_dir: Path,
        pytorch_device: str | None = None,
        default_prompt: str | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        tokenizer_model: Path | None = None,
        compile_mode: str | None = None,
    ) -> None:
        repo_root = Path(repo_root).resolve()
        checkpoint_dir = Path(checkpoint_dir).resolve()
        prepare_openpi_pytorch_imports(repo_root)

        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        self._device = _resolve_device(pytorch_device)
        self._default_prompt = default_prompt
        self._num_steps = int((sample_kwargs or {}).get("num_steps", 10))
        self._compile_mode = _resolve_compile_mode(compile_mode)
        self._checkpoint_dir = checkpoint_dir

        tokenizer_path = Path(
            os.environ.get("OPENPI_PYTORCH_TOKENIZER_MODEL", str(tokenizer_model or DEFAULT_TOKENIZER_MODEL))
        ).expanduser()
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"PyTorch tokenizer model not found: {tokenizer_path}. "
                "Set OPENPI_PYTORCH_TOKENIZER_MODEL to a valid paligemma_tokenizer.model path."
            )

        config = _load_checkpoint_config(checkpoint_dir, compile_mode=self._compile_mode)
        norm_stats = _load_norm_stats(_find_norm_stats_path(checkpoint_dir))
        weight_path = checkpoint_dir / "model.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing PyTorch weight file: {weight_path}")

        torch.backends.cudnn.benchmark = True

        load_t0 = time.perf_counter()
        model = PI0Pytorch(config=config)
        if self._compile_mode is None:
            try:
                torch_dynamo = importlib.import_module("torch._dynamo")
                model.sample_actions = torch_dynamo.disable(model.sample_actions)
            except Exception:
                pass
        safetensors.torch.load_model(model, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params(config.dtype)
        model = model.to(self._device)
        _install_mask_aware_prefix_fastpath(model)
        model.eval()
        self._model = model
        self._config = config
        self._tokenizer = PaligemmaTokenizerLite(tokenizer_path.resolve(), config.max_token_len)
        self._norm_stats = norm_stats
        self._masked_right_wrist_image = torch.full(
            (1, 3, IMAGE_SIZE, IMAGE_SIZE),
            -1.0,
            device=self._device,
            dtype=torch.float32,
        )
        trt_image_embedder = None
        trt_vision_enabled = False
        trt_vision_error: str | None = None
        if self._device.type == "cuda":
            try:
                trt_image_embedder = _maybe_build_trt_image_embedder(model, self._masked_right_wrist_image)
                if trt_image_embedder is not None:
                    setattr(model, "_openpi_trt_image_embedder", trt_image_embedder)
                    trt_vision_enabled = True
            except Exception as exc:
                trt_vision_error = f"{type(exc).__name__}: {exc}"
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        load_s = time.perf_counter() - load_t0
        self._metadata = {
            "policy_backend": "pytorch",
            "checkpoint_dir": str(checkpoint_dir),
            "pytorch_device": str(self._device),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "compile_mode": self._compile_mode or "none",
            "sample_num_steps": self._num_steps,
            "tokenizer_model": str(tokenizer_path.resolve()),
            "runtime_python": sys.executable,
            "load_s": float(load_s),
            "mask_aware_prefix_fastpath": True,
            "trt_vision_enabled": trt_vision_enabled,
        }
        if trt_vision_error is not None:
            self._metadata["trt_vision_error"] = trt_vision_error

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = _resolve_prompt(obs.get("prompt"), self._default_prompt)
        state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        state = _normalize_quantile(state, self._norm_stats["state"])
        tokenized_prompt, tokenized_prompt_mask = self._tokenizer.tokenize(prompt, state)
        padded_state = _pad_to_dim(state[None, :], self._config.action_dim)

        base_image = _image_to_model_tensor(obs["observation/image"], self._device)
        wrist_image = _image_to_model_tensor(obs["observation/wrist_image"], self._device)
        observation = SimpleObservation(
            images={
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": self._masked_right_wrist_image,
            },
            image_masks={
                "base_0_rgb": torch.tensor([True], dtype=torch.bool, device=self._device),
                "left_wrist_0_rgb": torch.tensor([True], dtype=torch.bool, device=self._device),
                "right_wrist_0_rgb": torch.tensor([False], dtype=torch.bool, device=self._device),
            },
            state=torch.from_numpy(padded_state).to(device=self._device),
            tokenized_prompt=torch.from_numpy(tokenized_prompt[None, ...]).to(device=self._device, dtype=torch.long),
            tokenized_prompt_mask=torch.from_numpy(tokenized_prompt_mask[None, ...]).to(
                device=self._device, dtype=torch.bool
            ),
        )

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            actions = self._model.sample_actions(str(self._device), observation, num_steps=self._num_steps)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        actions_np = actions[0].detach().cpu().numpy()
        actions_np = _unnormalize_quantile(actions_np, self._norm_stats["actions"]).astype(np.float32)
        return {
            "actions": actions_np[:, :7].copy(),
            "policy_timing": {"infer_ms": float(infer_ms)},
        }


def build_synthetic_observation(spec: SyntheticObservationSpec | None = None) -> dict[str, Any]:
    spec = spec or SyntheticObservationSpec()
    return {
        "observation/state": np.zeros((spec.state_dim,), dtype=np.float32),
        "observation/image": np.zeros((spec.image_size, spec.image_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((spec.image_size, spec.image_size, 3), dtype=np.uint8),
        "prompt": spec.prompt,
    }


__all__ = [
    "DEFAULT_TOKENIZER_MODEL",
    "SyntheticObservationSpec",
    "OpenPIPyTorchPolicy",
    "build_synthetic_observation",
    "prepare_openpi_pytorch_imports",
]
