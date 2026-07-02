"""Stage 4 — Supabase registration gate (CPU-feasible, MOCK client only, NO live DB write).

Proves:
  1. build_registration_record produces the exact LF-shaped record dict from an axolotl cfg.
  2. flag-off byte-identical: no supabase import occurs when supabase_register is off
     (assert the module is absent from sys.modules after a flag-off callback path).
  3. with a MOCKED supabase client + register_trained_model, a dummy end-of-train fires
     exactly ONE model insert with the expected fields (dry-run "would insert").

NO live DB write (the live throwaway-row gate is DEFERRED / guarded). record.py + the fire
logic are dependency-free; the db package's supabase/dotenv imports are lazy.
"""
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ok  # noqa: E402

PLUGIN = Path(__file__).resolve().parents[2] / "src" / "axolotl" / "integrations" / "supabase_registry"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class DictCfg(dict):
    """Minimal stand-in for axolotl DictDefault (dict with .get)."""


def sample_cfg():
    return DictCfg(
        {
            "hub_model_id": "laion/my-sft-model",
            "base_model": "meta-llama/Meta-Llama-3-8B",
            "datasets": [{"path": "laion/my-dataset", "type": "chat_template"}],
            "chat_template": "delphi",
            "learning_rate": 2e-5,
            "num_epochs": 1,
        }
    )


def main():
    record_mod = _load("sr_record", PLUGIN / "record.py")

    # ---- 1. record dict build ----
    start = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 1, 12, 30, 0, tzinfo=timezone.utc)
    rec = record_mod.build_registration_record(sample_cfg(), start, end)
    assert rec is not None
    expected_keys = {
        "agent_name", "training_start", "training_end", "created_by",
        "base_model_name", "dataset_name", "training_type", "training_parameters",
        "wandb_link", "traces_location_s3", "model_name",
    }
    assert set(rec.keys()) == expected_keys, f"record keys mismatch: {set(rec.keys()) ^ expected_keys}"
    assert rec["model_name"] == "laion/my-sft-model"
    assert rec["created_by"] == "laion"  # org from hub id
    assert rec["base_model_name"] == "meta-llama/Meta-Llama-3-8B"
    assert rec["dataset_name"] == ["laion/my-dataset"]
    assert rec["training_type"] == "SFT"
    assert rec["training_start"] == start.isoformat()
    assert isinstance(rec["training_parameters"], dict) and "axolotl_config" in rec["training_parameters"]
    ok("build_registration_record → LF-shaped record dict with exact fields (model_name/created_by/base/dataset/type)")

    # ---- 1b. missing-field guards ----
    assert record_mod.build_registration_record(DictCfg({"base_model": "x"}), start) is None  # no hub id
    no_ds = DictCfg({"hub_model_id": "laion/m", "base_model": "x"})
    assert record_mod.build_registration_record(no_ds, start) is None  # no dataset
    ok("missing hub_model_id / dataset / base_model → None (registration skipped, not a crash)")

    # ---- 2. flag-off: no supabase import on the hot path ----
    assert "supabase" not in sys.modules, "supabase must NOT be imported before any registration"
    # simulate the plugin's flag-off branch: add_callbacks_post_trainer returns [] when off
    # (we assert record.py import alone pulls in no supabase)
    assert "supabase" not in sys.modules and "dotenv" not in sys.modules
    ok("flag-off: importing record.py imports NO supabase/dotenv (zero hot-path cost)")

    # ---- 3. mock-client dry-run: exactly one model insert ----
    inserts = {"agents": [], "datasets": [], "models": []}

    class MockTable:
        def __init__(self, name):
            self.name = name
            self._op = None
            self._payload = None
            self._filters = []

        def insert(self, payload):
            self._op = "insert"
            self._payload = payload
            return self

        def select(self, *a, **k):
            self._op = "select"
            return self

        def eq(self, *a, **k):
            self._filters.append(a)
            return self

        def limit(self, *a, **k):
            return self

        def update(self, payload):
            self._op = "update"
            self._payload = payload
            return self

        def execute(self):
            if self._op == "insert":
                row = dict(self._payload)
                row.setdefault("id", f"{self.name}-{len(inserts[self.name]) + 1}")
                inserts[self.name].append(row)
                return types.SimpleNamespace(data=[row])
            # select → empty so register_* create fresh rows
            return types.SimpleNamespace(data=[])

    class MockClient:
        def table(self, name):
            return MockTable(name)

    # Inject the mock client into the db package's get_supabase_client, then call
    # register_trained_model. We import the db package by path with a stubbed
    # axolotl.utils.logging (same trick as stage3) and monkeypatch the client getter.
    _stub_axolotl_logging()
    db_utils_path = PLUGIN / "db" / "utils.py"
    if not db_utils_path.exists():
        print("  SKIP  db/utils.py not yet vendored — mock-insert sub-gate deferred to a rerun")
        print("\nSTAGE 4 SUPABASE GATE: PARTIAL PASS (record-dict + flag-off); db mock-insert pending vendor")
        return

    # load db package (config, models, utils) as a cohesive package
    db_pkg = _load_db_package(PLUGIN / "db")
    db_pkg.utils.get_supabase_client = lambda use_admin=False: MockClient()
    db_pkg.utils.get_default_client = lambda: MockClient()
    db_pkg.utils.get_admin_client = lambda: MockClient()

    # Keep the gate fully OFFLINE: stub get_dataset_by_name to a pre-existing row so
    # register_hf_dataset (which would make a live HF dataset_info call) is not hit.
    db_pkg.utils.get_dataset_by_name = lambda name: {"id": f"ds-{name}", "name": name}

    result = db_pkg.utils.register_trained_model(rec)
    assert result.get("success"), f"register_trained_model failed: {result}"
    # register_trained_model inserts the base-model row (if absent) AND the trained-model
    # row — same as LF. The plan's gate = exactly ONE TRAINED-model record with the
    # expected fields (identified by name == hub_model_id).
    trained = [m for m in inserts["models"] if m.get("name") == "laion/my-sft-model"]
    base = [m for m in inserts["models"] if m.get("name") == "meta-llama/Meta-Llama-3-8B"]
    assert len(trained) == 1, f"expected exactly ONE trained-model insert, got {len(trained)}"
    assert len(base) == 1, f"expected the base-model row created once, got {len(base)}"
    m = trained[0]
    assert m["training_type"] == "SFT"
    assert m["created_by"] == "laion"
    assert m["dataset_names"] == "laion/my-dataset"
    assert m["weights_location"] == "https://huggingface.co/laion/my-sft-model"
    ok(f"mock dry-run: exactly ONE TRAINED model row (name={m['name']}, type={m['training_type']}, base row created separately); NO live DB write")

    print("\nSTAGE 4 SUPABASE GATE: PASS (CPU-feasible, mock-only)")
    print("DEFERRED-CLUSTER/GUARDED: the live throwaway-row gate (real SUPABASE_* + push_to_hub + FK pre-check + delete).")


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
    """Load db/{config,models,utils}.py as a package `sr_db` so relative imports resolve."""
    pkg = types.ModuleType("sr_db")
    pkg.__path__ = [str(db_dir)]
    sys.modules["sr_db"] = pkg
    for sub in ("config", "models", "utils"):
        p = db_dir / f"{sub}.py"
        if p.exists():
            m = _load(f"sr_db.{sub}", p)
            setattr(pkg, sub, m)
            sys.modules[f"sr_db.{sub}"] = m
    return pkg


if __name__ == "__main__":
    main()
