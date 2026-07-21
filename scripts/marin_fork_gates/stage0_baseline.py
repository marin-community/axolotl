"""Stage 0 — baseline manifest + footgun control (CPU-feasible).

The full 1-GPU baseline SFT smoke is DEFERRED-CLUSTER (axolotl's CUDA stack won't install
on arm64 Mac). What we CAN capture CPU-side, per the plan's Stage-0 GO gate, is the
saved-tokenizer manifest under axolotl's two save modes, which is the CONTROL for the
Stage-2 footgun A/B:

  (A) default save (save_jinja_files=True, upstream default): the footgun state —
      chat_template popped out of tokenizer_config.json, lands only in chat_template.jinja.
  (B) save_jinja_files=False: legacy embed — chat_template stays in tokenizer_config.json.

We do this with a plain Llama-3 tokenizer + a placeholder chat_template so Stage 1/2 can
diff against it. Also records the flag-off (no chat_template override) manifest.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_tokenizer, ok, saved_tokenizer_manifest  # noqa: E402


def main():
    tok = load_tokenizer()
    baseline_template = tok.chat_template  # the stock Llama-3 template string
    assert isinstance(baseline_template, str) and baseline_template, "expected a stock chat_template"

    out = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # (A) upstream default: save_jinja_files=True -> footgun split
        a = td / "default_true"
        tok.save_pretrained(a, save_jinja_files=True)
        out["A_default_true"] = saved_tokenizer_manifest(a)

        # (B) save_jinja_files=False -> legacy embed (the Stage-2 mitigation shape)
        b = td / "flag_false"
        tok.save_pretrained(b, save_jinja_files=False)
        out["B_flag_false"] = saved_tokenizer_manifest(b)

    # Assertions documenting the pre-port footgun CONTROL:
    assert out["A_default_true"]["config_has_chat_template"] is False, (
        "CONTROL BROKEN: default save should POP chat_template out of tokenizer_config.json"
    )
    assert out["A_default_true"]["jinja_file_exists"] is True, (
        "CONTROL BROKEN: default save should write chat_template.jinja"
    )
    ok("footgun control (A): default save_jinja_files=True splits chat_template OUT of tokenizer_config.json")

    assert out["B_flag_false"]["config_has_chat_template"] is True, (
        "save_jinja_files=False should re-embed chat_template into tokenizer_config.json"
    )
    ok("mitigation shape (B): save_jinja_files=False embeds chat_template in tokenizer_config.json")

    manifest_path = Path(__file__).resolve().parent / "stage0_baseline_manifest.json"
    # trim the (long) template bodies for the recorded manifest
    trimmed = json.loads(json.dumps(out))
    for k in trimmed:
        for tk in ("config_chat_template", "jinja_file_contents"):
            v = trimmed[k].get(tk)
            if isinstance(v, str):
                trimmed[k][tk] = f"<{len(v)} chars>"
    manifest_path.write_text(json.dumps(trimmed, indent=2))
    ok(f"baseline manifest written -> {manifest_path.name}")
    print("\nSTAGE 0 BASELINE GATE: PASS (CPU-feasible portion)")
    print("DEFERRED-CLUSTER: the 1-GPU tiny-SFT baseline smoke (axolotl CUDA stack; arm64 Mac cannot install).")


if __name__ == "__main__":
    main()
