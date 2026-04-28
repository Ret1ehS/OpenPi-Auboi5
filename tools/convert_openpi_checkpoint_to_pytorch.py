#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import safetensors.torch
import torch

try:
    import tyro
except ModuleNotFoundError:
    tyro = None


def _ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    for path in (repo_root, repo_root / "src", repo_root / "packages" / "openpi-client" / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _lora_scale(config: Any | None) -> float:
    if config is None:
        return 1.0
    rank = float(config.rank)
    alpha = float(config.alpha)
    return alpha / math.sqrt(rank) if bool(getattr(config, "rslora", False)) else alpha / rank


def _merge_linear_lora(base: np.ndarray, lora_a: np.ndarray | None, lora_b: np.ndarray | None, scale: float) -> np.ndarray:
    if lora_a is None or lora_b is None:
        return base
    return np.asarray(base, dtype=np.float32) + np.matmul(lora_a.astype(np.float32), lora_b.astype(np.float32)) * scale


def _merge_tensor_lora(
    state_dict: dict[str, np.ndarray],
    *,
    base_key: str,
    lora_a_key: str,
    lora_b_key: str,
    einsum_expr: str,
    scale: float,
) -> None:
    if lora_a_key not in state_dict or lora_b_key not in state_dict:
        return
    base = np.asarray(state_dict[base_key], dtype=np.float32)
    lora_a = np.asarray(state_dict.pop(lora_a_key), dtype=np.float32)
    lora_b = np.asarray(state_dict.pop(lora_b_key), dtype=np.float32)
    delta = np.einsum(einsum_expr, lora_a, lora_b, optimize=True)
    state_dict[base_key] = base + delta.astype(np.float32) * scale


def _merge_attn_vec_lora(
    state_dict: dict[str, np.ndarray],
    *,
    base_key: str,
    lora_a_key: str,
    lora_b_key: str,
    scale: float,
) -> None:
    if lora_a_key not in state_dict or lora_b_key not in state_dict:
        return
    base = np.asarray(state_dict[base_key], dtype=np.float32)
    lora_a = np.asarray(state_dict.pop(lora_a_key), dtype=np.float32)
    lora_b = np.asarray(state_dict.pop(lora_b_key), dtype=np.float32)
    # For attn_vec_einsum, the JAX runtime LoRA path is:
    #   einsum("BTNH,NHL->BTL"), then einsum("BTL,NLD->BTD")
    # The second einsum sums over N instead of preserving the same head index,
    # so the equivalent merged weight uses sum_N(lora_b) rather than a per-head product.
    delta = np.einsum("lnhr,lrd->lnhd", lora_a, np.sum(lora_b, axis=1), optimize=True)
    state_dict[base_key] = base + delta.astype(np.float32) * scale


def _merge_lora_weights(flat_state_dict: dict[str, np.ndarray], model_config: Any) -> dict[str, np.ndarray]:
    import openpi.models.gemma

    merged = dict(flat_state_dict)
    paligemma_cfg = openpi.models.gemma.get_config(model_config.paligemma_variant)
    action_expert_cfg = openpi.models.gemma.get_config(model_config.action_expert_variant)

    def merge_attention(prefix: str, config: Any) -> None:
        attn_cfg = config.lora_configs.get("attn") if getattr(config, "lora_configs", None) else None
        if attn_cfg is None:
            return
        scale = _lora_scale(attn_cfg)
        qkv_key = f"{prefix}/qkv_einsum/w"
        if qkv_key in merged:
            _merge_tensor_lora(
                merged,
                base_key=qkv_key,
                lora_a_key=f"{prefix}/qkv_einsum/lora_a",
                lora_b_key=f"{prefix}/qkv_einsum/lora_b",
                einsum_expr="lqndr,lqnrh->lqndh",
                scale=scale,
            )
        else:
            _merge_tensor_lora(
                merged,
                base_key=f"{prefix}/q_einsum/w",
                lora_a_key=f"{prefix}/q_einsum/lora_a",
                lora_b_key=f"{prefix}/q_einsum/lora_b",
                einsum_expr="lndr,lnrh->lndh",
                scale=scale,
            )
            _merge_tensor_lora(
                merged,
                base_key=f"{prefix}/kv_einsum/w",
                lora_a_key=f"{prefix}/kv_einsum/lora_a",
                lora_b_key=f"{prefix}/kv_einsum/lora_b",
                einsum_expr="labdr,labrh->labdh",
                scale=scale,
            )
        _merge_attn_vec_lora(
            merged,
            base_key=f"{prefix}/attn_vec_einsum/w",
            lora_a_key=f"{prefix}/attn_vec_einsum/lora_a",
            lora_b_key=f"{prefix}/attn_vec_einsum/lora_b",
            scale=scale,
        )

    def merge_mlp(prefix: str, config: Any) -> None:
        ffn_cfg = config.lora_configs.get("ffn") if getattr(config, "lora_configs", None) else None
        if ffn_cfg is None:
            return
        # openpi.models.lora.FeedForward._dot() adds LoRA deltas without applying scaling_value,
        # so the exported merged MLP weights must preserve that runtime behavior.
        scale = 1.0
        _merge_tensor_lora(
            merged,
            base_key=f"{prefix}/gating_einsum",
            lora_a_key=f"{prefix}/gating_einsum_lora_a",
            lora_b_key=f"{prefix}/gating_einsum_lora_b",
            einsum_expr="lafr,larh->lafh",
            scale=scale,
        )
        linear_key = f"{prefix}/linear"
        lora_a_key = f"{prefix}/linear_lora_a"
        lora_b_key = f"{prefix}/linear_lora_b"
        if lora_a_key in merged and lora_b_key in merged:
            merged[linear_key] = _merge_linear_lora(
                np.asarray(merged[linear_key], dtype=np.float32),
                np.asarray(merged.pop(lora_a_key), dtype=np.float32),
                np.asarray(merged.pop(lora_b_key), dtype=np.float32),
                scale,
            )

    merge_attention("llm/layers/attn", paligemma_cfg)
    merge_mlp("llm/layers/mlp", paligemma_cfg)
    merge_mlp("llm/layers/mlp_1", action_expert_cfg)

    # The flattened checkpoint names expert attention blocks with a suffix on the leaf name instead of the path.
    if any(key.startswith("llm/layers/attn/q_einsum_1") for key in merged):
        attn_cfg = action_expert_cfg.lora_configs.get("attn") if getattr(action_expert_cfg, "lora_configs", None) else None
        if attn_cfg is not None:
            scale = _lora_scale(attn_cfg)
            _merge_tensor_lora(
                merged,
                base_key="llm/layers/attn/q_einsum_1/w",
                lora_a_key="llm/layers/attn/q_einsum_1/lora_a",
                lora_b_key="llm/layers/attn/q_einsum_1/lora_b",
                einsum_expr="lndr,lnrh->lndh",
                scale=scale,
            )
            _merge_tensor_lora(
                merged,
                base_key="llm/layers/attn/kv_einsum_1/w",
                lora_a_key="llm/layers/attn/kv_einsum_1/lora_a",
                lora_b_key="llm/layers/attn/kv_einsum_1/lora_b",
                einsum_expr="labdr,labrh->labdh",
                scale=scale,
            )
            _merge_attn_vec_lora(
                merged,
                base_key="llm/layers/attn/attn_vec_einsum_1/w",
                lora_a_key="llm/layers/attn/attn_vec_einsum_1/lora_a",
                lora_b_key="llm/layers/attn/attn_vec_einsum_1/lora_b",
                scale=scale,
            )

    return merged


def _build_paligemma_bridge_config(model_config: Any) -> Any:
    import openpi.models.gemma

    text_cfg = openpi.models.gemma.get_config(model_config.paligemma_variant)

    class PaliGemmaBridgeConfig:
        def __init__(self) -> None:
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": text_cfg.width,
                    "num_hidden_layers": text_cfg.depth,
                    "num_attention_heads": text_cfg.num_heads,
                    "head_dim": text_cfg.head_dim,
                    "intermediate_size": text_cfg.mlp_dim,
                },
            )()

    return PaliGemmaBridgeConfig()


def _assets_source(checkpoint_dir: Path) -> Path | None:
    direct = checkpoint_dir / "assets"
    if direct.exists():
        return direct
    parent = checkpoint_dir.parent / "assets"
    if parent.exists():
        return parent
    return None


def convert_checkpoint(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    config_name: str,
    output_path: Path,
    precision: str = "float32",
) -> None:
    import examples.convert_jax_model_to_pytorch as upstream
    import openpi.models.gemma
    import openpi.models.pi0_config
    import openpi.models_pytorch.pi0_pytorch
    import openpi.training.config as training_config

    model_config = training_config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    initial_params = upstream.slice_initial_orbax_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        restore_precision="float32",
    )
    merged_flat_params = _merge_lora_weights(initial_params["paligemma_params"], model_config)

    if model_config.pi05:
        projection_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        projection_keys = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]

    projection_params: dict[str, torch.Tensor] = {}
    for key in projection_keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params
        projection_params[f"{key}.weight"] = torch.from_numpy(np.asarray(weight, dtype=np.float32)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.asarray(bias, dtype=np.float32))

    paligemma_bridge_config = _build_paligemma_bridge_config(model_config)
    action_expert_config = openpi.models.gemma.get_config(model_config.action_expert_variant)

    paligemma_params, expert_params = upstream.slice_paligemma_state_dict(dict(merged_flat_params), paligemma_bridge_config)
    gemma_params = upstream.slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        checkpoint_dir=str(checkpoint_dir),
        pi05=model_config.pi05,
    )

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    all_params = {**paligemma_params, **gemma_params, **projection_params}
    load_result = model.load_state_dict(all_params, strict=False)

    # `lm_head.weight` is tied / unused for action inference; anything else is suspicious.
    unexpected = list(load_result.unexpected_keys)
    missing = [key for key in load_result.missing_keys if not key.endswith("lm_head.weight")]
    if unexpected or missing:
        raise RuntimeError(
            "Converted state_dict still has unresolved keys: "
            + json.dumps({"missing": missing, "unexpected": unexpected}, ensure_ascii=False)
        )

    if precision == "float32":
        model = model.to(torch.float32)
    elif precision == "bfloat16":
        model = model.to(torch.bfloat16)
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    output_path.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, str(output_path / "model.safetensors"))

    assets_source = _assets_source(checkpoint_dir)
    if assets_source is not None:
        assets_dest = output_path / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "discrete_state_input": bool(getattr(model_config, "discrete_state_input", True)),
        "precision": precision,
    }
    (output_path / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    summary = {
        "output_path": str(output_path),
        "precision": precision,
        "missing_lm_heads": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(
    repo_root: Path,
    checkpoint_dir: Path,
    config_name: str,
    output_path: Path,
    precision: str = "float32",
) -> None:
    repo_root = repo_root.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    _ensure_repo_imports(repo_root)
    convert_checkpoint(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        config_name=config_name,
        output_path=output_path,
        precision=precision,
    )


if __name__ == "__main__":
    if tyro is not None:
        tyro.cli(main)
    else:
        parser = argparse.ArgumentParser(description="Convert an OpenPI JAX checkpoint to PyTorch format.")
        parser.add_argument("--repo-root", type=Path, required=True)
        parser.add_argument("--checkpoint_dir", type=Path, required=True)
        parser.add_argument("--config_name", type=str, required=True)
        parser.add_argument("--output_path", type=Path, required=True)
        parser.add_argument("--precision", type=str, default="float32")
        args = parser.parse_args()
        main(
            repo_root=args.repo_root,
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config_name,
            output_path=args.output_path,
            precision=args.precision,
        )
