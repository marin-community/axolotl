"""Expose Qwen3.5's fused linear-attention QKV projection as three LoRA targets."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _linear_slice(linear: nn.Linear, start: int, stop: int) -> nn.Linear:
    sliced = nn.Linear(
        linear.in_features,
        stop - start,
        bias=linear.bias is not None,
        device="meta",
        dtype=linear.weight.dtype,
    )
    sliced.weight = nn.Parameter(
        linear.weight[start:stop].detach().clone(),
        requires_grad=linear.weight.requires_grad,
    )
    if linear.bias is not None:
        sliced.bias = nn.Parameter(
            linear.bias[start:stop].detach().clone(),
            requires_grad=linear.bias.requires_grad,
        )
    return sliced


class SplitQKVProjection(nn.Module):
    """A fused-compatible projection backed by independent Q, K, and V linears."""

    def __init__(self, fused: nn.Linear, key_dim: int, value_dim: int):
        super().__init__()
        expected_rows = key_dim * 2 + value_dim
        if fused.out_features != expected_rows:
            raise ValueError(
                "Qwen3.5 in_proj_qkv output dimension does not match "
                f"Q/K/V dimensions: {fused.out_features} != {expected_rows}"
            )

        self.in_proj_q = _linear_slice(fused, 0, key_dim)
        self.in_proj_k = _linear_slice(fused, key_dim, key_dim * 2)
        self.in_proj_v = _linear_slice(fused, key_dim * 2, expected_rows)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.in_proj_q(hidden_states),
                self.in_proj_k(hidden_states),
                self.in_proj_v(hidden_states),
            ),
            dim=-1,
        )


def split_qwen35_qkv_projections(model: nn.Module) -> int:
    """Replace dense Qwen3.5 Gated DeltaNet QKV projections and return the count."""
    replaced = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        if isinstance(module.in_proj_qkv, SplitQKVProjection):
            continue
        module.in_proj_qkv = SplitQKVProjection(
            module.in_proj_qkv,
            key_dim=module.key_dim,
            value_dim=module.value_dim,
        )
        replaced += 1
    return replaced


class Qwen35SplitQKVPlugin(BasePlugin):
    """Split Qwen3.5 linear-attention QKV before PEFT installs LoRA layers."""

    def post_model_build(self, cfg: DictDefault, model: PreTrainedModel):
        target_modules = set(cfg.lora_target_modules or [])
        required = {"in_proj_q", "in_proj_k", "in_proj_v"}
        missing = required - target_modules
        if missing:
            raise ValueError(
                "Qwen35SplitQKVPlugin requires separate linear-attention LoRA targets: "
                + ", ".join(sorted(missing))
            )
        if "linear_attn.in_proj_qkv" in target_modules or "in_proj_qkv" in target_modules:
            raise ValueError(
                "Qwen35SplitQKVPlugin cannot target linear_attn.in_proj_qkv together "
                "with the split Q/K/V projections"
            )

        replaced = split_qwen35_qkv_projections(model)
        if replaced == 0:
            raise ValueError(
                "Qwen35SplitQKVPlugin found no Qwen3.5 Gated DeltaNet projections"
            )
        LOG.info("Split %d Qwen3.5 linear-attention QKV projections", replaced)
