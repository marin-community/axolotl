"""Stage 3 — MFU compute unit gate (CPU-feasible).

Unit-drives the ported compute_mfu_from_trainer against a MOCK trainer state and
asserts finite, in-range mfu_percent. Also asserts:
  - flag-off (no total_flos) → None (no MFU keys)
  - unknown-peak fallback returns achieved TFLOPs only (matching LF)
  - PEAK_TFLOPS_PER_GPU env override is honored

compute.py imports `axolotl.utils.logging` (which chains into the un-installable axolotl
stack). We stub that module and import compute.py by path — the compute logic itself is
torch-only (torch 2.9.0 is present).
"""
import importlib.util
import sys
import types
from pathlib import Path

# ---- stub axolotl.utils.logging so compute.py imports without the full stack ----
_pkg = types.ModuleType("axolotl")
_pkg.__path__ = []
_utils = types.ModuleType("axolotl.utils")
_utils.__path__ = []
_logging = types.ModuleType("axolotl.utils.logging")


class _L:
    def warning(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


_logging.get_logger = lambda *a, **k: _L()
sys.modules["axolotl"] = _pkg
sys.modules["axolotl.utils"] = _utils
sys.modules["axolotl.utils.logging"] = _logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ok  # noqa: E402

_COMPUTE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "axolotl"
    / "integrations"
    / "mfu"
    / "compute.py"
)
_spec = importlib.util.spec_from_file_location("mfu_compute", _COMPUTE)
mfu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mfu)


class MockState:
    def __init__(self, total_flos):
        self.total_flos = total_flos


class MockModel:
    def __init__(self, dtype):
        self.dtype = dtype


class MockTrainer:
    def __init__(self, total_flos, dtype):
        self.state = MockState(total_flos)
        self.model = MockModel(dtype)


def main():
    import os

    import torch

    # 1. flag-off / no data → None (no MFU keys)
    assert mfu.compute_mfu_from_trainer(MockTrainer(None, torch.bfloat16), 10.0) is None
    assert mfu.compute_mfu_from_trainer(MockTrainer(0, torch.bfloat16), 10.0) is None
    ok("flag-off / missing total_flos → None (no MFU keys emitted)")

    # 2. known peak via env override → finite in-range mfu_percent
    os.environ["PEAK_TFLOPS_PER_GPU"] = "989"  # H100 bf16
    # ~1e15 FLOPs over 10s single GPU → 100 TFLOP/s → ~10% of 989
    m = mfu.compute_mfu_from_trainer(MockTrainer(1e15, torch.bfloat16), 10.0)
    assert m is not None and "mfu_percent" in m, m
    assert 0.0 < m["mfu_percent"] <= 100.0, f"mfu_percent out of range: {m}"
    import math

    assert math.isfinite(m["mfu_percent"]) and math.isfinite(m["achieved_tflops_per_gpu"])
    # 1e15/10 = 1e14 FLOP/s = 100 TFLOP/s; /989 *100 ≈ 10.11%
    assert abs(m["achieved_tflops_per_gpu"] - 100.0) < 1e-6, m
    assert abs(m["mfu_percent"] - (100.0 * 100.0 / 989.0)) < 1e-6, m
    ok(f"env-override peak → finite in-range mfu_percent={m['mfu_percent']:.3f}% (achieved {m['achieved_tflops_per_gpu']:.1f} TFLOP/s)")

    # 3. unknown peak (no env, CPU device) → achieved TFLOPs only, no mfu_percent (LF fallback)
    del os.environ["PEAK_TFLOPS_PER_GPU"]
    m2 = mfu.compute_mfu_from_trainer(MockTrainer(1e15, torch.bfloat16), 10.0)
    assert m2 is not None and "achieved_tflops_per_gpu" in m2
    assert "mfu_percent" not in m2, f"expected no mfu_percent on unknown peak, got {m2}"
    ok("unknown-peak fallback → achieved_tflops_per_gpu only (matches LF)")

    # 4. peak LUT sanity
    assert mfu.peak_tflops_lookup("NVIDIA A100-SXM4-80GB", torch.bfloat16) == 312.0
    assert mfu.peak_tflops_lookup("NVIDIA H100 80GB HBM3", torch.bfloat16) == 989.0
    assert mfu.peak_tflops_lookup("CPU", torch.bfloat16) is None
    ok("peak-TFLOPs LUT resolves A100/H100 and returns None for unknown")

    print("\nSTAGE 3 MFU GATE: PASS (CPU-feasible)")
    print("DEFERRED-CLUSTER: a 20-step 1-GPU run producing a real state.total_flos + trainer.log emission.")


if __name__ == "__main__":
    main()
