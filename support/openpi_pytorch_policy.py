#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
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


_NDARRAY_MARKER = "__openpi_worker_ndarray__"
_NUMPY_SCALAR_MARKER = "__openpi_worker_numpy_scalar__"


def encode_worker_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_MARKER: True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return {
            _NUMPY_SCALAR_MARKER: True,
            "dtype": value.dtype.str,
            "value": value.item(),
        }
    if isinstance(value, dict):
        return {key: encode_worker_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_worker_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(encode_worker_value(item) for item in value)
    return value


def decode_worker_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_NDARRAY_MARKER):
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(value["shape"]).copy()
        if value.get(_NUMPY_SCALAR_MARKER):
            return np.asarray(value["value"], dtype=np.dtype(value["dtype"]))[()]
        return {key: decode_worker_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_worker_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_worker_value(item) for item in value)
    return value


def _resolve_attention_backend(explicit: str | None = None) -> str:
    raw = explicit if explicit is not None else os.environ.get("OPENPI_PYTORCH_ATTN_BACKEND", "")
    value = str(raw).strip().lower()
    if value in {"", "eager"}:
        return "eager"
    if value == "sdpa":
        return "sdpa"
    raise ValueError(f"Unsupported OPENPI_PYTORCH_ATTN_BACKEND={raw!r}; expected eager|sdpa")


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


class _TrtDenoiseWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.cache_utils import DynamicCache

        legacy = tuple(
            (k, v)
            for k, v in zip(
                torch.unbind(key_cache, dim=0),
                torch.unbind(value_cache, dim=0),
                strict=True,
            )
        )
        past_key_values = DynamicCache.from_legacy_cache(legacy)
        return self._model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)


def _maybe_build_trt_denoise_engine(policy: "OpenPIPyTorchPolicy", model: torch.nn.Module) -> torch.nn.Module | None:
    if not _env_truthy("OPENPI_PYTORCH_TRT_DENOISE"):
        return None

    _add_system_tensorrt_python_paths()
    import torch_tensorrt
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from transformers.cache_utils import DynamicCache

    cache_dir = Path(
        os.environ.get(
            "OPENPI_PYTORCH_TRT_DENOISE_CACHE_DIR",
            "~/.cache/openpi/torch_tensorrt/denoise_step",
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_obs = build_synthetic_observation()
    sample_prompt = _resolve_prompt(sample_obs.get("prompt"), policy._default_prompt)
    sample_state_np = np.asarray(sample_obs["observation/state"], dtype=np.float32).reshape(-1)
    sample_state_np = _normalize_quantile(sample_state_np, policy._norm_stats["state"])
    sample_padded_state = _pad_to_dim(sample_state_np[None, :], policy._config.action_dim)
    tokenized_prompt, tokenized_prompt_mask = policy._tokenizer.tokenize(sample_prompt, sample_state_np)
    sample_observation = SimpleObservation(
        images={
            "base_0_rgb": _image_to_model_tensor(sample_obs["observation/image"], policy._device),
            "left_wrist_0_rgb": _image_to_model_tensor(sample_obs["observation/wrist_image"], policy._device),
            "right_wrist_0_rgb": policy._masked_right_wrist_image,
        },
        image_masks={
            "base_0_rgb": torch.tensor([True], dtype=torch.bool, device=policy._device),
            "left_wrist_0_rgb": torch.tensor([True], dtype=torch.bool, device=policy._device),
            "right_wrist_0_rgb": torch.tensor([False], dtype=torch.bool, device=policy._device),
        },
        state=torch.from_numpy(sample_padded_state).to(device=policy._device),
        tokenized_prompt=torch.from_numpy(tokenized_prompt[None, ...]).to(device=policy._device, dtype=torch.long),
        tokenized_prompt_mask=torch.from_numpy(tokenized_prompt_mask[None, ...]).to(
            device=policy._device, dtype=torch.bool
        ),
    )

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(sample_observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, sample_past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    legacy_cache = sample_past_key_values.to_legacy_cache()
    key_cache = torch.stack([key for key, _ in legacy_cache], dim=0)
    value_cache = torch.stack([value for _, value in legacy_cache], dim=0)
    sample_x_t = torch.randn(
        (1, model.config.action_horizon, model.config.action_dim),
        device=policy._device,
        dtype=torch.float32,
    )
    sample_timestep = torch.tensor([1.0], device=policy._device, dtype=torch.float32)

    wrapper = _TrtDenoiseWrapper(model).eval().to(device=policy._device)
    return torch_tensorrt.compile(
        wrapper,
        ir="dynamo",
        inputs=[state, prefix_pad_masks, sample_x_t, sample_timestep, key_cache, value_cache],
        enabled_precisions={torch.float32, torch.bfloat16},
        min_block_size=1,
        truncate_double=True,
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


def _install_sdpa_attention_fastpath(model: torch.nn.Module) -> None:
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    @torch.no_grad()
    def sample_actions_sdpa(self, device, observation, noise=None, num_steps=10):
        batch_size = observation.state.shape[0]
        if noise is None:
            actions_shape = (batch_size, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_q_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=prefix_q_dtype)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=prefix_q_dtype)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time_value = torch.tensor(1.0, dtype=torch.float32, device=device)
        while bool(time_value >= (-dt / 2)):
            expanded_time = time_value.expand(batch_size)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time_value += dt
        return x_t

    def denoise_step_sdpa(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_q_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=suffix_q_dtype)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks).to(dtype=suffix_q_dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "sdpa"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    setattr(model, "_openpi_original_sample_actions", model.sample_actions)
    setattr(model, "_openpi_original_denoise_step", model.denoise_step)
    model.sample_actions = types.MethodType(sample_actions_sdpa, model)
    model.denoise_step = types.MethodType(denoise_step_sdpa, model)


def _install_trt_denoise_fastpath(model: torch.nn.Module, trt_denoise: torch.nn.Module) -> None:
    original = getattr(model, "denoise_step")

    def denoise_step_trt(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        legacy = past_key_values.to_legacy_cache()
        key_cache = torch.stack([key for key, _ in legacy], dim=0)
        value_cache = torch.stack([value for _, value in legacy], dim=0)
        return trt_denoise(state, prefix_pad_masks, x_t, timestep, key_cache, value_cache)

    setattr(model, "_openpi_original_denoise_step", original)
    setattr(model, "_openpi_trt_denoise", trt_denoise)
    model.denoise_step = types.MethodType(denoise_step_trt, model)


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
        self._attention_backend = _resolve_attention_backend()
        self._trt_denoise_requested = _env_truthy("OPENPI_PYTORCH_TRT_DENOISE")

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
        if self._attention_backend == "sdpa" and not self._trt_denoise_requested:
            _install_sdpa_attention_fastpath(model)
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
        trt_denoise = None
        trt_denoise_enabled = False
        trt_denoise_error: str | None = None
        if self._device.type == "cuda":
            try:
                trt_image_embedder = _maybe_build_trt_image_embedder(model, self._masked_right_wrist_image)
                if trt_image_embedder is not None:
                    setattr(model, "_openpi_trt_image_embedder", trt_image_embedder)
                    trt_vision_enabled = True
            except Exception as exc:
                trt_vision_error = f"{type(exc).__name__}: {exc}"
            try:
                trt_denoise = _maybe_build_trt_denoise_engine(self, model)
                if trt_denoise is not None:
                    _install_trt_denoise_fastpath(model, trt_denoise)
                    trt_denoise_enabled = True
            except Exception as exc:
                trt_denoise_error = f"{type(exc).__name__}: {exc}"
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        load_s = time.perf_counter() - load_t0
        effective_attention_backend = "eager" if trt_denoise_enabled else self._attention_backend
        self._metadata = {
            "policy_backend": "pytorch",
            "checkpoint_dir": str(checkpoint_dir),
            "pytorch_device": str(self._device),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "compile_mode": self._compile_mode or "none",
            "attention_backend": effective_attention_backend,
            "requested_attention_backend": self._attention_backend,
            "sample_num_steps": self._num_steps,
            "tokenizer_model": str(tokenizer_path.resolve()),
            "runtime_python": sys.executable,
            "load_s": float(load_s),
            "mask_aware_prefix_fastpath": True,
            "trt_vision_enabled": trt_vision_enabled,
            "trt_denoise_enabled": trt_denoise_enabled,
        }
        if trt_vision_error is not None:
            self._metadata["trt_vision_error"] = trt_vision_error
        if trt_denoise_error is not None:
            self._metadata["trt_denoise_error"] = trt_denoise_error

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def infer(self, obs: dict[str, Any], noise: np.ndarray | None = None) -> dict[str, Any]:
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
        noise_tensor = None
        if noise is not None:
            noise_array = np.asarray(noise, dtype=np.float32)
            if noise_array.ndim == 2:
                noise_array = noise_array[None, ...]
            noise_tensor = torch.from_numpy(noise_array).to(device=self._device, dtype=torch.float32)
        with torch.inference_mode():
            actions = self._model.sample_actions(
                str(self._device),
                observation,
                noise=noise_tensor,
                num_steps=self._num_steps,
            )
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


def _read_worker_message() -> dict[str, Any] | None:
    header = sys.stdin.buffer.read(8)
    if not header:
        return None
    size = int.from_bytes(header, "little", signed=False)
    payload = sys.stdin.buffer.read(size)
    while len(payload) < size:
        chunk = sys.stdin.buffer.read(size - len(payload))
        if not chunk:
            raise EOFError("Unexpected EOF while reading worker payload.")
        payload += chunk
    return decode_worker_value(pickle.loads(payload))


def _write_worker_message(message: dict[str, Any]) -> None:
    payload = pickle.dumps(encode_worker_value(message), protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(len(payload).to_bytes(8, "little", signed=False))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def run_worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenPI PyTorch local policy worker.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--default-prompt", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--compile-mode", type=str, default="")
    args = parser.parse_args(argv)

    try:
        policy = OpenPIPyTorchPolicy(
            repo_root=args.repo_root,
            checkpoint_dir=args.checkpoint_dir,
            pytorch_device=args.device,
            default_prompt=args.default_prompt or None,
            sample_kwargs={"num_steps": int(args.num_steps)},
            compile_mode=args.compile_mode or None,
        )
        _write_worker_message({"ok": True, "metadata": policy.metadata})
    except Exception as exc:
        _write_worker_message(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    while True:
        try:
            message = _read_worker_message()
            if message is None:
                return 0
            op = message.get("op")
            if op == "infer":
                _write_worker_message(
                    {
                        "ok": True,
                        "result": policy.infer(message["obs"], noise=message.get("noise")),
                    }
                )
            elif op == "reset":
                policy.reset()
                _write_worker_message({"ok": True})
            elif op == "close":
                policy.close()
                _write_worker_message({"ok": True})
                return 0
            else:
                _write_worker_message({"ok": False, "error": f"Unsupported worker op: {op!r}"})
        except Exception as exc:
            _write_worker_message(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )


__all__ = [
    "DEFAULT_TOKENIZER_MODEL",
    "SyntheticObservationSpec",
    "OpenPIPyTorchPolicy",
    "build_synthetic_observation",
    "decode_worker_value",
    "encode_worker_value",
    "prepare_openpi_pytorch_imports",
    "run_worker_main",
]


if __name__ == "__main__":
    raise SystemExit(run_worker_main())
