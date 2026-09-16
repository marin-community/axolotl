"""Convert independent Qwen3.5 Q/K/V LoRA weights into an exact fused adapter."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_PROJECTIONS = ("q", "k", "v")
_PROJECTION_COUNT = len(_PROJECTIONS)
_FUSED_PROJECTION = "in_proj_qkv"
_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"


@dataclass(frozen=True)
class FusedFactors:
    factor_a: torch.Tensor
    factor_b: torch.Tensor
    split_rank: int


def _split_prefixes(weights: dict[str, torch.Tensor]) -> set[str]:
    suffix = ".in_proj_q.lora_A.weight"
    return {key[: -len(suffix)] for key in weights if key.endswith(suffix)}


def _projection_key(prefix: str, projection: str, factor: str) -> str:
    return f"{prefix}.in_proj_{projection}.lora_{factor}.weight"


def _fused_key(prefix: str, factor: str) -> str:
    fused_prefix = (
        prefix
        if prefix.endswith(f".{_FUSED_PROJECTION}")
        else f"{prefix}.{_FUSED_PROJECTION}"
    )
    return f"{fused_prefix}.lora_{factor}.weight"


def _fused_factors(
    weights: dict[str, torch.Tensor], prefix: str
) -> FusedFactors:
    factors = {
        projection: (
            weights[_projection_key(prefix, projection, "A")],
            weights[_projection_key(prefix, projection, "B")],
        )
        for projection in _PROJECTIONS
    }
    ranks = {factor_a.shape[0] for factor_a, _ in factors.values()}
    input_dims = {factor_a.shape[1] for factor_a, _ in factors.values()}
    if len(ranks) != 1 or len(input_dims) != 1:
        raise ValueError(f"Split Q/K/V LoRA factors have incompatible shapes under {prefix}")

    rank = ranks.pop()
    for projection, (_, factor_b) in factors.items():
        if factor_b.shape[1] != rank:
            raise ValueError(
                f"in_proj_{projection} LoRA A/B ranks disagree under {prefix}"
            )

    fused_a = torch.cat([factors[projection][0] for projection in _PROJECTIONS], dim=0)
    output_dims = [factors[projection][1].shape[0] for projection in _PROJECTIONS]
    fused_b = factors["q"][1].new_zeros((sum(output_dims), rank * _PROJECTION_COUNT))
    output_start = 0
    for rank_block, projection in enumerate(_PROJECTIONS):
        factor_b = factors[projection][1]
        output_stop = output_start + factor_b.shape[0]
        rank_start = rank_block * rank
        fused_b[output_start:output_stop, rank_start : rank_start + rank] = factor_b
        output_start = output_stop
    return FusedFactors(factor_a=fused_a, factor_b=fused_b, split_rank=rank)


def _fused_config(config: dict, fused_rank: int) -> dict:
    if config.get("use_rslora") or config.get("use_dora"):
        raise ValueError("Split-QKV adapter fusion supports standard LoRA scaling only")

    base_rank = config["r"]
    if fused_rank != base_rank * _PROJECTION_COUNT:
        raise ValueError(
            f"Expected fused QKV rank {base_rank * _PROJECTION_COUNT}, found {fused_rank}"
        )

    target_modules = set(config["target_modules"])
    split_targets = {f"in_proj_{projection}" for projection in _PROJECTIONS}
    if not split_targets.issubset(target_modules):
        raise ValueError("Adapter config does not target all split Q/K/V projections")
    target_modules.difference_update(split_targets)
    target_modules.add(_FUSED_PROJECTION)

    fused = dict(config)
    fused["target_modules"] = sorted(target_modules)
    rank_pattern = dict(fused.get("rank_pattern") or {})
    alpha_pattern = dict(fused.get("alpha_pattern") or {})
    rank_pattern[_FUSED_PROJECTION] = fused_rank
    alpha_pattern[_FUSED_PROJECTION] = config["lora_alpha"] * _PROJECTION_COUNT
    fused["rank_pattern"] = rank_pattern
    fused["alpha_pattern"] = alpha_pattern
    return fused


def fuse_split_qkv_adapter(input_path: Path, output_path: Path) -> None:
    """Write a stock-model-compatible adapter with exact fused QKV updates."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Fused adapter output must differ from the split adapter input")

    config_path = input_path / _ADAPTER_CONFIG
    weights_path = input_path / _ADAPTER_WEIGHTS
    config = json.loads(config_path.read_text())
    weights = load_file(weights_path)
    prefixes = _split_prefixes(weights)
    if not prefixes:
        raise ValueError("Adapter contains no split Qwen3.5 QKV LoRA weights")

    fused_weights = dict(weights)
    fused_rank: int | None = None
    for prefix in sorted(prefixes):
        factors = _fused_factors(weights, prefix)
        candidate_rank = factors.split_rank * _PROJECTION_COUNT
        if fused_rank is not None and candidate_rank != fused_rank:
            raise ValueError("Split QKV LoRA ranks differ across layers")
        fused_rank = candidate_rank
        for projection in _PROJECTIONS:
            for factor in ("A", "B"):
                del fused_weights[_projection_key(prefix, projection, factor)]
        fused_weights[_fused_key(prefix, "A")] = factors.factor_a
        fused_weights[_fused_key(prefix, "B")] = factors.factor_b

    assert fused_rank is not None
    output_path.mkdir(parents=True, exist_ok=False)
    for source in input_path.iterdir():
        if source.name in {_ADAPTER_CONFIG, _ADAPTER_WEIGHTS}:
            continue
        if source.is_dir():
            shutil.copytree(source, output_path / source.name)
        else:
            shutil.copy2(source, output_path / source.name)

    with safe_open(weights_path, framework="pt") as handle:
        metadata = handle.metadata()
    save_file(fused_weights, output_path / _ADAPTER_WEIGHTS, metadata=metadata)
    (output_path / _ADAPTER_CONFIG).write_text(
        json.dumps(_fused_config(config, fused_rank), indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse independent Qwen3.5 Q/K/V LoRA factors for stock serving"
    )
    parser.add_argument("input", type=Path, help="Split-QKV PEFT adapter directory")
    parser.add_argument("output", type=Path, help="New fused PEFT adapter directory")
    args = parser.parse_args()
    fuse_split_qkv_adapter(args.input, args.output)


if __name__ == "__main__":
    main()
