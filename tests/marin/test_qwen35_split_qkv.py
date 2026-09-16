"""Behavioral coverage for independent Qwen3.5 linear-attention LoRA factors."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from axolotl.integrations.qwen35_split_qkv.adapter import fuse_split_qkv_adapter
from axolotl.integrations.qwen35_split_qkv.plugin import (
    Qwen35SplitQKVPlugin,
    SplitQKVProjection,
)
from axolotl.utils.dict import DictDefault


def _linear_attention() -> Qwen3_5GatedDeltaNet:
    config = Qwen3_5TextConfig(
        hidden_size=8,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
    )
    return Qwen3_5GatedDeltaNet(config, layer_idx=0)


def _add_split_lora(
    attention: Qwen3_5GatedDeltaNet, target_modules: list[str]
) -> PeftModel:
    return get_peft_model(
        attention,
        LoraConfig(
            r=2, lora_alpha=1, lora_dropout=0, target_modules=target_modules
        ),
    )


def _fill_split_lora(model: PeftModel) -> None:
    split = model.base_model.model.in_proj_qkv
    for index, projection in enumerate(("q", "k", "v"), start=1):
        layer = getattr(split, f"in_proj_{projection}")
        layer.lora_A["default"].weight.data.fill_(index / 10)
        layer.lora_B["default"].weight.data.fill_(index / 20)


def test_split_qkv_lora_matches_three_independent_updates() -> None:
    torch.manual_seed(17)
    attention = _linear_attention()
    reference_inputs = torch.randn(2, 4, attention.hidden_size, requires_grad=True)
    expected_base = attention.in_proj_qkv(reference_inputs)
    expected_gradient = torch.autograd.grad(expected_base.square().sum(), reference_inputs)[0]

    Qwen35SplitQKVPlugin().post_model_build(
        DictDefault(
            {"lora_target_modules": ["in_proj_q", "in_proj_k", "in_proj_v"]}
        ),
        attention,
    )
    assert isinstance(attention.in_proj_qkv, SplitQKVProjection)
    inputs = reference_inputs.detach().clone().requires_grad_(True)
    actual_base = attention.in_proj_qkv(inputs)
    actual_gradient = torch.autograd.grad(actual_base.square().sum(), inputs)[0]
    torch.testing.assert_close(actual_base, expected_base)
    torch.testing.assert_close(actual_gradient, expected_gradient)

    model = _add_split_lora(attention, ["in_proj_q", "in_proj_k", "in_proj_v"])
    _fill_split_lora(model)
    split = model.base_model.model.in_proj_qkv
    expected_updates = []
    for projection in ("q", "k", "v"):
        layer = getattr(split, f"in_proj_{projection}")
        expected_updates.append(
            layer.lora_B["default"](layer.lora_A["default"](inputs))
            * layer.scaling["default"]
        )

    expected = expected_base + torch.cat(expected_updates, dim=-1)
    torch.testing.assert_close(split(inputs), expected)


def _write_split_adapter(path: Path) -> dict[str, torch.Tensor]:
    attention = _linear_attention()
    Qwen35SplitQKVPlugin().post_model_build(
        DictDefault(
            {"lora_target_modules": ["in_proj_q", "in_proj_k", "in_proj_v"]}
        ),
        attention,
    )
    model = _add_split_lora(
        attention, ["in_proj_q", "in_proj_k", "in_proj_v", "out_proj"]
    )
    _fill_split_lora(model)
    model.base_model.model.out_proj.lora_A["default"].weight.data.normal_()
    model.base_model.model.out_proj.lora_B["default"].weight.data.normal_()
    model.save_pretrained(path, safe_serialization=True)
    return load_file(path / "adapter_model.safetensors")


def _load_fused_projection(path: Path, inputs: torch.Tensor) -> torch.Tensor:
    model = _linear_attention()
    model.in_proj_qkv.weight.data.zero_()
    loaded = PeftModel.from_pretrained(model, path)
    return loaded.base_model.model.in_proj_qkv(inputs)


def test_fused_adapter_preserves_split_qkv_delta(tmp_path: Path) -> None:
    split_path = tmp_path / "split"
    fused_path = tmp_path / "fused"
    split_path.mkdir()
    split_weights = _write_split_adapter(split_path)

    fuse_split_qkv_adapter(split_path, fused_path)

    fused_weights = load_file(fused_path / "adapter_model.safetensors")
    config = json.loads((fused_path / "adapter_config.json").read_text())
    prefix = next(
        key.removesuffix(".in_proj_q.lora_A.weight")
        for key in split_weights
        if key.endswith(".in_proj_q.lora_A.weight")
    )
    split_delta = torch.cat(
        [
            split_weights[f"{prefix}.in_proj_{projection}.lora_B.weight"]
            @ split_weights[f"{prefix}.in_proj_{projection}.lora_A.weight"]
            for projection in ("q", "k", "v")
        ]
    ) * (1 / 2)
    fused_delta = (
        fused_weights[f"{prefix}.lora_B.weight"]
        @ fused_weights[f"{prefix}.lora_A.weight"]
    ) * (config["alpha_pattern"]["in_proj_qkv"] / config["rank_pattern"]["in_proj_qkv"])

    torch.testing.assert_close(fused_delta, split_delta)
    inputs = torch.randn(2, 8)
    torch.testing.assert_close(_load_fused_projection(fused_path, inputs), inputs @ split_delta.T)
    assert config["rank_pattern"]["in_proj_qkv"] == 6
    assert config["alpha_pattern"]["in_proj_qkv"] == 3
    assert config["target_modules"] == ["in_proj_qkv", "out_proj"]
    assert not any(".in_proj_q." in key for key in fused_weights)
    out_proj_key = next(key for key in split_weights if key.endswith("out_proj.lora_A.weight"))
    torch.testing.assert_close(
        fused_weights[out_proj_key],
        split_weights[out_proj_key],
    )
