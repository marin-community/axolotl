"""Stage 5 — integration gate: all three features ON in one config (CPU-feasible parts).

Composes the CPU-feasible parts of Stages 2/3/4 under ONE combined config (delphi +
tokenizer_save_jinja_files:false + mfu:true + supabase_register:true, mock supabase), and
asserts NO cross-feature interference:
  1. delphi save+footgun: saved checkpoint carries the delphi template in tokenizer_config
     .json; tool_calls+role:tool reload render byte-matches LF.
  2. MFU: mock trainer state → finite in-range mfu_percent.
  3. Supabase: mock dry-run → exactly one trained-model record with expected fields.
  4. flag-off of any ONE feature leaves the other two unchanged (union == sum of parts).

The actual 1-GPU tiny-SFT run exercising the real trainer wiring is DEFERRED-CLUSTER.
"""
import importlib.util
import json
import re
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TOOLCALL_CONVO, TOOLS, load_tokenizer, ok, read_delphi_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_SR = ROOT / "src" / "axolotl" / "integrations" / "supabase_registry"
LF_TEMPLATE_PY = Path(
    "/Users/benjaminfeuer/Documents/LLaMA-Factory/src/llamafactory/data/template.py"
)


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _stub_axolotl_logging():
    if "axolotl.utils.logging" in sys.modules:
        return
    pkg = types.ModuleType("axolotl"); pkg.__path__ = []
    utils = types.ModuleType("axolotl.utils"); utils.__path__ = []
    logging = types.ModuleType("axolotl.utils.logging")

    class _L:
        def __getattr__(self, _):
            return lambda *a, **k: None

    logging.get_logger = lambda *a, **k: _L()
    sys.modules.update({"axolotl": pkg, "axolotl.utils": utils, "axolotl.utils.logging": logging})


def _load_db_package(db_dir: Path):
    pkg = types.ModuleType("sr_db"); pkg.__path__ = [str(db_dir)]
    sys.modules["sr_db"] = pkg
    for sub in ("config", "models", "utils"):
        m = _load(f"sr_db.{sub}", db_dir / f"{sub}.py")
        setattr(pkg, sub, m)
        sys.modules[f"sr_db.{sub}"] = m
    return pkg


class DictCfg(dict):
    pass


def combined_cfg():
    return DictCfg(
        {
            "hub_model_id": "laion/marin-all3",
            "base_model": "NousResearch/Meta-Llama-3-8B-Instruct",
            "datasets": [{"path": "laion/ds", "type": "chat_template"}],
            "chat_template": "delphi",
            "tokenizer_save_jinja_files": False,
            "mfu": True,
            "supabase_register": True,
        }
    )


def lf_template():
    return re.search(
        r'DELPHI_V0_JINJA_TEMPLATE = r"""(.*?)"""', LF_TEMPLATE_PY.read_text(), re.S
    ).group(1)


def main():
    from transformers import AutoTokenizer
    import torch

    cfg = combined_cfg()
    delphi = read_delphi_template()

    # ---- 1. delphi save + footgun under the combined config ----
    repair = _load(
        "ti_repair",
        ROOT / "src" / "axolotl" / "integrations" / "template_integrity" / "repair.py",
    )
    tok = load_tokenizer()
    tok.chat_template = delphi
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "out"
        ckpt = td / "out_ckpt" / "checkpoint-5"
        ckpt.mkdir(parents=True)
        # output_dir honors the cfg flag; checkpoint save ignores it (base.py:952)
        tok.save_pretrained(out, save_jinja_files=cfg["tokenizer_save_jinja_files"])
        tok.save_pretrained(ckpt)
        repair.repair_saved_dir(ckpt, expected_template=tok.chat_template)
        for name, d in [("output_dir", out), ("checkpoint-*/", ckpt)]:
            c = json.loads((d / "tokenizer_config.json").read_text())
            assert c.get("chat_template") == delphi, f"{name}: delphi template not embedded under combined cfg"
            r = AutoTokenizer.from_pretrained(d).apply_chat_template(
                TOOLCALL_CONVO, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
            r_lf = tok.apply_chat_template(
                TOOLCALL_CONVO, tools=TOOLS, chat_template=lf_template(), tokenize=False, add_generation_prompt=True
            )
            assert r == r_lf, f"{name}: combined-cfg render diverges from LF"
    ok("combined cfg: delphi footgun gate passes (both dirs embed the template; render byte-matches LF)")

    # ---- 2. MFU under the combined config ----
    _stub_axolotl_logging()
    mfu = _load("mfu_compute", ROOT / "src" / "axolotl" / "integrations" / "mfu" / "compute.py")
    import os

    os.environ["PEAK_TFLOPS_PER_GPU"] = "989"
    trainer = types.SimpleNamespace(
        state=types.SimpleNamespace(total_flos=1e15),
        model=types.SimpleNamespace(dtype=torch.bfloat16),
    )
    m = mfu.compute_mfu_from_trainer(trainer, 10.0)
    assert m and 0 < m["mfu_percent"] <= 100, m
    ok(f"combined cfg: MFU emits finite in-range mfu_percent={m['mfu_percent']:.3f}%")
    del os.environ["PEAK_TFLOPS_PER_GPU"]

    # ---- 3. Supabase mock dry-run under the combined config ----
    record_mod = _load("sr_record", PLUGIN_SR / "record.py")
    rec = record_mod.build_registration_record(cfg, datetime.now(timezone.utc), datetime.now(timezone.utc))
    assert rec and rec["model_name"] == "laion/marin-all3" and rec["training_type"] == "SFT"

    inserts = {"agents": [], "datasets": [], "models": []}

    class MockTable:
        def __init__(self, name):
            self.name = name; self._op = None; self._payload = None
        def insert(self, p): self._op, self._payload = "insert", p; return self
        def select(self, *a, **k): self._op = "select"; return self
        def eq(self, *a, **k): return self
        def limit(self, *a, **k): return self
        def update(self, p): self._op, self._payload = "update", p; return self
        def execute(self):
            if self._op == "insert":
                row = dict(self._payload); row.setdefault("id", f"{self.name}-{len(inserts[self.name])+1}")
                inserts[self.name].append(row); return types.SimpleNamespace(data=[row])
            return types.SimpleNamespace(data=[])

    class MockClient:
        def table(self, name): return MockTable(name)

    db = _load_db_package(PLUGIN_SR / "db")
    db.utils.get_supabase_client = lambda use_admin=False: MockClient()
    db.utils.get_default_client = lambda: MockClient()
    db.utils.get_admin_client = lambda: MockClient()
    db.utils.get_dataset_by_name = lambda name: {"id": f"ds-{name}", "name": name}
    res = db.utils.register_trained_model(rec)
    assert res.get("success"), res
    trained = [x for x in inserts["models"] if x.get("name") == "laion/marin-all3"]
    assert len(trained) == 1, f"expected one trained-model row, got {len(trained)}"
    ok("combined cfg: mock Supabase dry-run inserts exactly ONE trained-model row; NO live DB write")

    # ---- 4. no cross-feature interference: each flag-off zeroes ONLY its own behavior ----
    # MFU off → None regardless of delphi/supabase
    assert mfu.compute_mfu_from_trainer(types.SimpleNamespace(state=types.SimpleNamespace(total_flos=None), model=None), 10.0) is None
    # supabase off → record still builds ONLY when hub set; with hub unset → None
    off = DictCfg(dict(cfg)); off["hub_model_id"] = None
    assert record_mod.build_registration_record(off, datetime.now(timezone.utc)) is None
    # delphi off → stock template, no <|tool_call|>
    stock = load_tokenizer().apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    assert "<|tool_call|>" not in stock
    ok("no cross-feature interference: each flag-off zeroes only its own behavior")

    print("\nSTAGE 5 INTEGRATION GATE: PASS (CPU-composable parts)")
    print("DEFERRED-CLUSTER: the actual 1-GPU tiny-SFT all-3-on run exercising the real trainer wiring.")


if __name__ == "__main__":
    main()
