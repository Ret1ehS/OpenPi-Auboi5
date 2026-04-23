#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    for path in (repo_root, repo_root / "src", repo_root / "packages" / "openpi-client" / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _load_saved_observation(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
        state = np.asarray(data["observation_state"], dtype=np.float32)
        image = np.asarray(data["observation_image"], dtype=np.uint8)
        wrist = np.asarray(data["observation_wrist_image"], dtype=np.uint8)
        prompt_raw = data["prompt"]
        noise = np.asarray(data["noise"], dtype=np.float32)

    prompt = str(prompt_raw.item() if isinstance(prompt_raw, np.ndarray) and prompt_raw.shape == () else prompt_raw)
    obs = {
        "image": image,
        "wrist_image": wrist,
        "state": state,
        "prompt": prompt,
    }
    return obs, noise


def _build_policy_input_transform(train_cfg: Any, checkpoint_dir: Path) -> Any:
    import openpi.policies.policy_config as policy_config
    import openpi.training.checkpoints as checkpoints
    import openpi.transforms as transforms

    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", data_cfg.asset_id)
    return transforms.compose(
        [
            *data_cfg.repack_transforms.inputs,
            transforms.InjectDefaultPrompt(None),
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ]
    )


def _summarize_array(x: np.ndarray) -> dict[str, Any]:
    flat = x.reshape(-1).astype(np.float64)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


def _as_numeric_array(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x)
    try:
        return array.astype(np.float64)
    except (TypeError, ValueError):
        return np.asarray(array.tolist(), dtype=np.float64)


def _summarize_diff(reference: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    ref = _as_numeric_array(reference)
    oth = _as_numeric_array(other)
    delta = oth - ref
    ref_flat = ref.reshape(-1)
    oth_flat = oth.reshape(-1)
    flat = delta.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(oth_flat))
    return {
        "shape": list(np.asarray(reference).shape),
        "max_abs": float(np.max(np.abs(flat))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "l2": float(np.linalg.norm(flat)),
        "cosine": None if denom == 0.0 else float(np.dot(ref_flat, oth_flat) / denom),
    }


def _select_block_indices(depth: int) -> list[int]:
    if depth <= 0:
        return []
    return sorted({0, depth // 2, depth - 1})


def _vision_key(image_name: str, suffix: str) -> str:
    return f"vision__{image_name}__{suffix}"


def _flatten_spatial_tokens(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
    return x


def _extract_jax_block_output(block_out: Any) -> Any:
    if hasattr(block_out, "keys"):
        for key in ("+mlp", "pre_ln", "+sa", "mlp", "sa"):
            if key in block_out:
                return block_out[key]
        first_key = next(iter(block_out.keys()))
        return block_out[first_key]
    return block_out


def _batched_numpy_inputs(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = {sub_key: np.asarray(sub_value)[None, ...] for sub_key, sub_value in value.items()}
        else:
            result[key] = np.asarray(value)[None, ...]
    return result


def _batched_torch_inputs(data: dict[str, Any], *, device: str) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = {
                sub_key: torch.from_numpy(np.asarray(sub_value)).to(device)[None, ...] for sub_key, sub_value in value.items()
            }
        else:
            result[key] = torch.from_numpy(np.asarray(value)).to(device)[None, ...]
    return result


def _collect_jax_intermediates(model: Any, observation: Any, noise: np.ndarray) -> dict[str, np.ndarray]:
    import einops
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask

    batch_size = observation.state.shape[0]
    noise_jax = jnp.asarray(noise[None, ...], dtype=jnp.float32)
    time_jax = jnp.ones((batch_size,), dtype=jnp.float32)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_pad_masks, prefix_att_masks)
    prefix_positions = jnp.cumsum(prefix_pad_masks, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_embs, None], mask=prefix_attn_mask, positions=prefix_positions)

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(observation, noise_jax, time_jax)
    suffix_attn_mask = make_attn_mask(suffix_pad_masks, suffix_att_masks)
    prefix_cross_mask = einops.repeat(prefix_pad_masks, "b p -> b s p", s=suffix_embs.shape[1])
    full_attn_mask = jnp.concatenate([prefix_cross_mask, suffix_attn_mask], axis=-1)
    suffix_positions = jnp.sum(prefix_pad_masks, axis=-1)[:, None] + jnp.cumsum(suffix_pad_masks, axis=-1) - 1

    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_embs],
        mask=full_attn_mask,
        positions=suffix_positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    suffix_tail = suffix_out[:, -model.action_horizon :]
    v_t = model.action_out_proj(suffix_tail)

    result = {
        "prefix_embs": np.asarray(prefix_embs, dtype=np.float32),
        "prefix_pad_masks": np.asarray(prefix_pad_masks, dtype=bool),
        "prefix_att_masks": np.asarray(prefix_att_masks, dtype=bool),
        "suffix_embs_t1": np.asarray(suffix_embs, dtype=np.float32),
        "suffix_pad_masks_t1": np.asarray(suffix_pad_masks, dtype=bool),
        "suffix_att_masks_t1": np.asarray(suffix_att_masks, dtype=bool),
        "adarms_cond_t1": np.asarray(adarms_cond, dtype=np.float32),
        "suffix_out_t1": np.asarray(suffix_tail, dtype=np.float32),
        "v_t_t1": np.asarray(v_t, dtype=np.float32),
    }
    for image_name, image_value in observation.images.items():
        _, vision_out = model.PaliGemma.img(image_value, train=False)
        stem = np.asarray(vision_out["stem"], dtype=np.float32)
        result[_vision_key(image_name, "stem_tokens")] = _flatten_spatial_tokens(stem)
        result[_vision_key(image_name, "with_posemb")] = np.asarray(vision_out["with_posemb"], dtype=np.float32)

        encoder_out = vision_out["encoder"]
        block_keys = sorted(key for key in encoder_out.keys() if str(key).startswith("block"))
        for block_index in _select_block_indices(len(block_keys)):
            block_key = block_keys[block_index]
            block_tensor = _extract_jax_block_output(encoder_out[block_key])
            result[_vision_key(image_name, f"{block_key}_out")] = np.asarray(block_tensor, dtype=np.float32)

        result[_vision_key(image_name, "encoded")] = np.asarray(vision_out["encoded"], dtype=np.float32)
        image_features = np.asarray(vision_out["logits_2d"], dtype=np.float32)
        result[_vision_key(image_name, "image_features")] = _flatten_spatial_tokens(image_features)
    return result


def _collect_pytorch_visual_intermediates(model: Any, image_names: list[str], images: list[Any]) -> dict[str, np.ndarray]:
    import torch

    result: dict[str, np.ndarray] = {}
    paligemma = model.paligemma_with_expert.paligemma
    vision_tower = paligemma.vision_tower
    vision_model = vision_tower.vision_model
    embeddings_module = vision_model.embeddings
    encoder_layers = vision_model.encoder.layers
    target_dtype = embeddings_module.patch_embedding.weight.dtype

    def to_numpy(t: torch.Tensor) -> np.ndarray:
        tensor = t.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu().numpy()

    for image_name, image_value in zip(image_names, images, strict=True):
        patch_embeds = embeddings_module.patch_embedding(image_value.to(dtype=target_dtype))
        stem_tokens = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings_module(image_value)
        vision_outputs = vision_tower(pixel_values=image_value, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states

        result[_vision_key(image_name, "stem_tokens")] = to_numpy(stem_tokens)
        result[_vision_key(image_name, "with_posemb")] = to_numpy(embeddings)
        for block_index in _select_block_indices(len(encoder_layers)):
            result[_vision_key(image_name, f"block{block_index:02d}_out")] = to_numpy(hidden_states[block_index + 1])
        result[_vision_key(image_name, "encoded")] = to_numpy(vision_outputs.last_hidden_state)
        result[_vision_key(image_name, "image_features")] = to_numpy(
            paligemma.multi_modal_projector(vision_outputs.last_hidden_state)
        )
    return result


def _collect_pytorch_intermediates(model: Any, observation: Any, noise: np.ndarray) -> dict[str, np.ndarray]:
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    device = torch.device("cpu")
    noise_torch = torch.from_numpy(noise[None, ...]).to(device=device, dtype=torch.float32)
    image_names = list(observation.images.keys())
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001

    with torch.inference_mode():
        vision_data = _collect_pytorch_visual_intermediates(model, image_names, images)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        timestep = torch.ones((observation.state.shape[0],), dtype=torch.float32, device=device)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, noise_torch, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        suffix_positions = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)  # noqa: SLF001
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=suffix_positions,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -model.config.action_horizon :]
        v_t = model.action_out_proj(suffix_out.to(dtype=torch.float32))

    def to_numpy(t: torch.Tensor | None) -> np.ndarray:
        if t is None:
            return np.asarray([])
        tensor = t.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu().numpy()

    result = {
        "prefix_embs": to_numpy(prefix_embs),
        "prefix_pad_masks": to_numpy(prefix_pad_masks),
        "prefix_att_masks": to_numpy(prefix_att_masks),
        "suffix_embs_t1": to_numpy(suffix_embs),
        "suffix_pad_masks_t1": to_numpy(suffix_pad_masks),
        "suffix_att_masks_t1": to_numpy(suffix_att_masks),
        "adarms_cond_t1": to_numpy(adarms_cond),
        "suffix_out_t1": to_numpy(suffix_out),
        "v_t_t1": to_numpy(v_t),
    }
    result.update(vision_data)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare official JAX and official PyTorch intermediates on one observation.")
    parser.add_argument("--backend", choices=("jax", "pytorch", "report"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--observation-npz", type=Path, required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--jax-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "intermediate_compare")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    _ensure_repo_imports(repo_root)

    from openpi.training.config import get_config

    raw_obs, noise = _load_saved_observation(args.observation_npz)
    train_cfg = get_config(args.config_name)
    raw_action_dim = int(np.asarray(raw_obs["state"]).shape[-1])
    raw_obs.setdefault(
        "actions",
        np.zeros(
            (int(train_cfg.model.action_horizon), raw_action_dim),
            dtype=np.float32,
        ),
    )
    input_transform = _build_policy_input_transform(train_cfg, args.jax_checkpoint_dir.expanduser().resolve())
    transformed = input_transform(dict(raw_obs))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        run_dir / "transformed_inputs.npz",
        state=np.asarray(transformed["state"], dtype=np.float32),
        tokenized_prompt=np.asarray(transformed["tokenized_prompt"], dtype=np.int32),
        tokenized_prompt_mask=np.asarray(transformed["tokenized_prompt_mask"], dtype=bool),
        base_image=np.asarray(transformed["image"]["base_0_rgb"]),
        left_wrist_image=np.asarray(transformed["image"]["left_wrist_0_rgb"]),
        right_wrist_image=np.asarray(transformed["image"]["right_wrist_0_rgb"]),
        base_image_mask=np.asarray(transformed["image_mask"]["base_0_rgb"]),
        left_wrist_image_mask=np.asarray(transformed["image_mask"]["left_wrist_0_rgb"]),
        right_wrist_image_mask=np.asarray(transformed["image_mask"]["right_wrist_0_rgb"]),
        noise=np.asarray(noise, dtype=np.float32),
    )

    import openpi.models.model as model_api

    if args.backend == "jax":
        import jax.numpy as jnp

        jax_model = train_cfg.model.load(
            model_api.restore_params(args.jax_checkpoint_dir.expanduser().resolve() / "params", dtype=jnp.bfloat16)
        )
        jax_inputs = _batched_numpy_inputs(transformed)
        jax_inputs = {
            key: {sub_key: jnp.asarray(sub_value) for sub_key, sub_value in value.items()}
            if isinstance(value, dict)
            else jnp.asarray(value)
            for key, value in jax_inputs.items()
        }
        jax_obs = model_api.Observation.from_dict(jax_inputs)
        jax_data = _collect_jax_intermediates(jax_model, jax_obs, noise)
        np.savez_compressed(run_dir / "intermediates_jax.npz", **jax_data)
        summary = {
            "backend": "jax",
            "prompt": raw_obs["prompt"],
            "run_dir": str(run_dir),
            "jax": {key: _summarize_array(value) for key, value in jax_data.items()},
        }
        (run_dir / "jax_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    if args.backend == "pytorch":
        pytorch_model = train_cfg.model.load_pytorch(
            train_cfg,
            str(args.pytorch_checkpoint_dir.expanduser().resolve() / "model.safetensors"),
        )
        pytorch_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        pytorch_model = pytorch_model.to("cpu").eval()
        pytorch_obs = model_api.Observation.from_dict(_batched_torch_inputs(transformed, device="cpu"))
        pytorch_data = _collect_pytorch_intermediates(pytorch_model, pytorch_obs, noise)
        np.savez_compressed(run_dir / "intermediates_pytorch.npz", **pytorch_data)
        summary = {
            "backend": "pytorch",
            "prompt": raw_obs["prompt"],
            "run_dir": str(run_dir),
            "pytorch": {key: _summarize_array(value) for key, value in pytorch_data.items()},
        }
        (run_dir / "pytorch_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    with np.load(run_dir / "intermediates_jax.npz", allow_pickle=False) as data:
        jax_data = {key: np.asarray(data[key]) for key in data.files}
    with np.load(run_dir / "intermediates_pytorch.npz", allow_pickle=False) as data:
        pytorch_data = {key: np.asarray(data[key]) for key in data.files}
    with np.load(run_dir / "transformed_inputs.npz", allow_pickle=False) as data:
        transformed_inputs = {key: np.asarray(data[key]) for key in data.files}

    diff_keys = sorted(
        key
        for key in jax_data
        if key in pytorch_data
        and (
            key in ("prefix_embs", "suffix_embs_t1", "adarms_cond_t1", "suffix_out_t1", "v_t_t1")
            or key.startswith("vision__")
        )
    )
    diff_report = {key: _summarize_diff(jax_data[key], pytorch_data[key]) for key in diff_keys}
    summary = {
        "backend": "report",
        "prompt": raw_obs["prompt"],
        "state_summary": _summarize_array(np.asarray(transformed_inputs["state"], dtype=np.float32)),
        "tokenized_prompt_nonzero": int(np.count_nonzero(np.asarray(transformed_inputs["tokenized_prompt_mask"], dtype=bool))),
        "noise_summary": _summarize_array(np.asarray(transformed_inputs["noise"], dtype=np.float32)),
        "image_masks": {
            "base_0_rgb": bool(np.asarray(transformed_inputs["base_image_mask"]).item()),
            "left_wrist_0_rgb": bool(np.asarray(transformed_inputs["left_wrist_image_mask"]).item()),
            "right_wrist_0_rgb": bool(np.asarray(transformed_inputs["right_wrist_image_mask"]).item()),
        },
        "jax": {key: _summarize_array(value) for key, value in jax_data.items()},
        "pytorch": {key: _summarize_array(value) for key, value in pytorch_data.items()},
        "diffs": diff_report,
        "artifacts": {
            "run_dir": str(run_dir),
            "jax_npz": str(run_dir / "intermediates_jax.npz"),
            "pytorch_npz": str(run_dir / "intermediates_pytorch.npz"),
            "inputs_npz": str(run_dir / "transformed_inputs.npz"),
        },
    }
    (run_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
