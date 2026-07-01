# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MFU computation, ported from LLaMA-Factory extras/misc.py:245-435.

Kept as a small self-contained module (only torch for dtype/device probing, which
is present in any training runtime) so it can be unit-tested against a mock trainer
state. Peak-TFLOPs LUT covers A100/H100/H200/B200 bf16/fp16 with a
``PEAK_TFLOPS_PER_GPU`` env override.
"""

import os
from typing import Any, Optional

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def get_world_size() -> int:
    """LF misc.py:236 — dist world size, defaulting to 1."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def peak_tflops_lookup(device_name: str, dtype: "torch.dtype") -> Optional[float]:
    """Approximate per-GPU peak TFLOPs for known accelerators + dtype (LF misc.py:245).

    Ballpark values — override via PEAK_TFLOPS_PER_GPU for accuracy.
    """
    name = device_name.upper()
    is_bf16 = dtype == torch.bfloat16
    peaks = {
        "A100": 312.0 if is_bf16 else 312.0,  # FP16/BF16 similar on A100
        "H100": 989.0 if is_bf16 else 1979.0,
        "H200": 1415.0 if is_bf16 else 2829.0,
        "B200": 2000.0 if is_bf16 else 4000.0,
    }
    for key, val in peaks.items():
        if key in name:
            return val
    return None


def parse_peak_env() -> Optional[float]:
    """PEAK_TFLOPS_PER_GPU env override (LF misc.py:266). Comma-sep allowed; first used."""
    val = os.getenv("PEAK_TFLOPS_PER_GPU")
    if not val:
        return None
    try:
        first = str(val).split(",")[0].strip()
        return float(first)
    except Exception:
        LOG.warning(f"Invalid PEAK_TFLOPS_PER_GPU={val!r}; ignoring override.")
        return None


def compute_mfu_from_trainer(
    trainer: Any, train_runtime: float
) -> Optional[dict[str, float]]:
    """Achieved per-GPU TFLOPs + MFU from HF Trainer's cumulative total_flos (LF misc.py:279).

    Returns {achieved_tflops_per_gpu[, mfu_percent]} or None if total_flos unavailable.
    When the device peak is unknown (and no env override), returns achieved TFLOPs only.
    """
    total_flos = getattr(getattr(trainer, "state", None), "total_flos", None)
    if not isinstance(total_flos, (int, float)) or total_flos <= 0 or train_runtime <= 0:
        return None

    world = get_world_size()
    achieved_tflops_per_gpu = (total_flos / train_runtime) / (1e12 * max(world, 1))

    dtype = getattr(getattr(trainer, "model", None), "dtype", None) or torch.bfloat16
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    peak = parse_peak_env() or peak_tflops_lookup(device_name, dtype)
    if peak is None or peak <= 0:
        LOG.warning(
            "MFU: Unknown peak TFLOPs for device '%s' and dtype '%s'. "
            "Set PEAK_TFLOPS_PER_GPU to report MFU.",
            device_name,
            str(dtype),
        )
        return {"achieved_tflops_per_gpu": achieved_tflops_per_gpu}

    mfu = 100.0 * achieved_tflops_per_gpu / peak
    return {"achieved_tflops_per_gpu": achieved_tflops_per_gpu, "mfu_percent": mfu}


def _compute_model_flops_from_cfg(
    cfg: Any,
    total_batch_size: int,
    seq_length: int,
    include_backward: bool = True,
    include_recompute: bool = False,
    include_flashattn: bool = False,
) -> int:
    """Analytic FLOPs/step from model config (LF misc.py:310)."""
    hidden_size = getattr(cfg, "hidden_size", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    intermediate_size = getattr(cfg, "intermediate_size", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    num_key_value_heads = getattr(cfg, "num_key_value_heads", None)
    num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
    tie_word_embeddings = getattr(cfg, "tie_word_embeddings", False)

    BASE = 2  # gemm (add + mul)

    mlp_flops_per_token = 3 * BASE * hidden_size * intermediate_size
    mlp_flops = total_batch_size * seq_length * num_hidden_layers * mlp_flops_per_token

    q_flops_per_token = BASE * hidden_size * hidden_size
    o_flops_per_token = BASE * hidden_size * hidden_size
    k_flops_per_token = (
        BASE * hidden_size * hidden_size * num_key_value_heads // num_attention_heads
    )
    v_flops_per_token = (
        BASE * hidden_size * hidden_size * num_key_value_heads // num_attention_heads
    )
    attn_proj_flops_per_token = (
        q_flops_per_token + o_flops_per_token + k_flops_per_token + v_flops_per_token
    )
    attn_proj_flops = (
        total_batch_size * seq_length * num_hidden_layers * attn_proj_flops_per_token
    )

    sdpa_flops_per_layer = 2 * BASE * hidden_size * seq_length * seq_length
    sdpa_flops = total_batch_size * num_hidden_layers * sdpa_flops_per_layer

    embedding_flops_per_token = hidden_size * vocab_size
    embedding_flops = total_batch_size * seq_length * embedding_flops_per_token
    if tie_word_embeddings is False:
        embedding_flops *= 2

    non_embedding_flops = mlp_flops + attn_proj_flops + sdpa_flops
    non_embedding_coeff, embedding_coeff = 1, 1
    if include_backward:
        non_embedding_coeff += 2
        embedding_coeff += 2
    if include_recompute:
        non_embedding_coeff += 1

    total_flops = (
        non_embedding_coeff * non_embedding_flops + embedding_coeff * embedding_flops
    )
    if include_flashattn:
        total_flops += sdpa_flops
    return int(total_flops)


def compute_mfu_theoretical_from_trainer(
    trainer: Any,
    model_name_or_path: str,
    total_batch_size: int,
    seq_length: int,
    steps_per_second: float,
) -> Optional[dict[str, float]]:
    """Theoretical MFU from analytic FLOPs/step + measured steps/s (LF misc.py:391)."""
    try:
        if steps_per_second is None or steps_per_second <= 0:
            return None

        world = get_world_size()
        cfg = getattr(getattr(trainer, "model", None), "config", None)
        if cfg is not None:
            flops_per_step = _compute_model_flops_from_cfg(
                cfg, total_batch_size, seq_length
            )
        else:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_name_or_path)
            flops_per_step = _compute_model_flops_from_cfg(
                cfg, total_batch_size, seq_length
            )
        cluster_flops_per_s = steps_per_second * flops_per_step
        achieved_tflops_per_gpu = cluster_flops_per_s / (1e12 * max(world, 1))

        dtype = getattr(getattr(trainer, "model", None), "dtype", None) or torch.bfloat16
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        peak = parse_peak_env() or peak_tflops_lookup(device_name, dtype)
        if peak is None or peak <= 0:
            LOG.warning(
                "MFU (theoretical): Unknown peak TFLOPs for device '%s' dtype '%s'. "
                "Set PEAK_TFLOPS_PER_GPU.",
                device_name,
                str(dtype),
            )
            return {"achieved_tflops_per_gpu_theoretical": achieved_tflops_per_gpu}

        mfu = 100.0 * achieved_tflops_per_gpu / peak
        return {
            "achieved_tflops_per_gpu_theoretical": achieved_tflops_per_gpu,
            "mfu_percent_theoretical": mfu,
        }
    except Exception:
        return None
