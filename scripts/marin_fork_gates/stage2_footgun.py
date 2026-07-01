"""Stage 2 — save-time template-integrity FOOTGUN GATE (HARD BLOCKER, CPU-feasible).

Reproduces axolotl's TWO save paths with the delphi template on a Llama-3 tokenizer:
  - top-level output_dir save: axolotl calls save_pretrained(save_jinja_files=cfg
    .tokenizer_save_jinja_files). Default True → footgun split. We set False → embed.
  - per-checkpoint save (core/trainers/base.py:952): save_pretrained(NO flag) → transformers
    default True → footgun split EVEN WITH tokenizer_save_jinja_files:false. The plugin's
    TemplateIntegrityCallback.on_save repairs this.

A/B discipline (debug-reproduce-on-real-config): we FIRST prove the CONTROL (no fix) leaves
the checkpoint dir split (chat_template absent from tokenizer_config.json) — otherwise a green
with-fix result is inconclusive. THEN prove the fix embeds it in BOTH dirs and the reload
render byte-matches LF.
"""
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import TOOLCALL_CONVO, TOOLS, load_tokenizer, ok, read_delphi_template  # noqa: E402

# repair.py is deliberately dependency-free (no torch/axolotl), so import it directly by
# path — the full axolotl package won't import without its CUDA stack (not installable on
# arm64 Mac). This is exactly the module the trainer callback calls.
_REPAIR_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "axolotl"
    / "integrations"
    / "template_integrity"
    / "repair.py"
)
_spec = importlib.util.spec_from_file_location("ti_repair", _REPAIR_PATH)
_repair_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repair_mod)
repair_saved_dir = _repair_mod.repair_saved_dir

LF_TEMPLATE_PY = Path(
    "/Users/benjaminfeuer/Documents/LLaMA-Factory/src/llamafactory/data/template.py"
)


def lf_template() -> str:
    return re.search(
        r'DELPHI_V0_JINJA_TEMPLATE = r"""(.*?)"""', LF_TEMPLATE_PY.read_text(), re.S
    ).group(1)


def config_has_template(d: Path) -> bool:
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    return bool(cfg.get("chat_template"))


def config_template(d: Path) -> str:
    return json.loads((d / "tokenizer_config.json").read_text()).get("chat_template")


def render(tok) -> str:
    return tok.apply_chat_template(
        TOOLCALL_CONVO, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )


def main():
    from transformers import AutoTokenizer

    delphi = read_delphi_template()
    tok = load_tokenizer()
    tok.chat_template = delphi  # simulate axolotl loaders/tokenizer.py:326 setting resolved template

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---------- CONTROL: no fix ----------
        # output_dir with the shipped DEFAULT (save_jinja_files=True) -> split
        ctrl_out = td / "control_output_dir"
        tok.save_pretrained(ctrl_out, save_jinja_files=True)
        # simulated per-checkpoint save (base.py:952 -> NO flag -> default True) -> split
        ctrl_ckpt = td / "control_output_dir_ckpt" / "checkpoint-10"
        ctrl_ckpt.mkdir(parents=True)
        tok.save_pretrained(ctrl_ckpt)  # NO flag = the trainer per-ckpt path

        assert not config_has_template(ctrl_out), "CONTROL broken: default output_dir save should split"
        assert not config_has_template(ctrl_ckpt), "CONTROL broken: per-ckpt save should split"
        ok("CONTROL reproduced: WITHOUT fix, both output_dir AND checkpoint-*/ LACK chat_template in tokenizer_config.json (the footgun)")

        # ---------- FIX ----------
        # (a) output_dir: axolotl sets save_jinja_files=cfg.tokenizer_save_jinja_files=False -> embed
        fix_out = td / "fixed_output_dir"
        tok.save_pretrained(fix_out, save_jinja_files=False)

        # (b) checkpoint dir: per-ckpt save IGNORES the flag -> still split, then plugin repairs it
        fix_ckpt = td / "fixed_output_dir_ckpt" / "checkpoint-10"
        fix_ckpt.mkdir(parents=True)
        tok.save_pretrained(fix_ckpt)  # NO flag (base.py:952) -> split...
        assert not config_has_template(fix_ckpt), "expected pre-repair split at the checkpoint"
        report = repair_saved_dir(fix_ckpt, expected_template=tok.chat_template)  # plugin on_save
        assert report["repaired"], f"plugin repair did not fire: {report}"
        ok("flag-ignoring per-checkpoint save (base.py:952) repaired by the plugin callback")

        # ---------- ASSERT: both dirs now carry the delphi template ----------
        for name, d in [("output_dir", fix_out), ("checkpoint-*/", fix_ckpt)]:
            assert config_has_template(d), f"{name}: tokenizer_config.json missing chat_template AFTER fix"
            embedded = config_template(d)
            assert embedded == delphi, f"{name}: embedded chat_template != delphi string"
            # if a chat_template.jinja exists it must be byte-equal
            jf = d / "chat_template.jinja"
            if jf.exists():
                assert jf.read_text() == delphi, f"{name}: chat_template.jinja not byte-equal to embedded"
            ok(f"{name}: tokenizer_config.json carries the delphi chat_template (embedded, not jinja-only)")

        # ---------- reload + render byte-match to LF ----------
        lf = lf_template()
        for name, d in [("output_dir", fix_out), ("checkpoint-*/", fix_ckpt)]:
            reloaded = AutoTokenizer.from_pretrained(d)
            r = render(reloaded)
            # reference render straight from the LF template string
            r_lf = tok.apply_chat_template(
                TOOLCALL_CONVO, tools=TOOLS, chat_template=lf, tokenize=False, add_generation_prompt=True
            )
            assert r == r_lf, f"{name}: reloaded render diverges from LF render"
            for marker in ("<|tool_call|>", "<|tool_call_end|>", "<|tool_result|>", "<|tool_result_end|>"):
                assert marker in r, f"{name}: reloaded render missing {marker}"
            ok(f"{name}: reload -> apply_chat_template(tool_calls+role:tool) byte-identical to LF render")

    print("\nSTAGE 2 FOOTGUN GATE: PASS (HARD BLOCKER cleared, CPU-feasible)")
    print("DEFERRED-CLUSTER: a real 1-GPU tiny-SFT save that exercises the axolotl trainer's actual on_save path.")


if __name__ == "__main__":
    main()
