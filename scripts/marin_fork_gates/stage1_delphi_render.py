"""Stage 1 — delphi named-template resolution + render byte-match to LF (CPU-feasible).

Axolotl auto-registers any `templates/*.jinja` as a named template via the glob at
utils/chat_templates/base.py:19-23 (`_CHAT_TEMPLATES[filename[:-6]] = f.read()`). We can't
import axolotl (CUDA stack won't install on arm64), so we replicate that exact glob logic
here and assert `delphi` resolves to the delphi jinja bytes.

Render gate: rendering a tool_calls + role:tool conversation with the delphi template must
be byte-identical to the LF DELPHI_V0_JINJA_TEMPLATE render of the SAME conversation. Since
the axolotl asset is a byte-identical copy of OT-Agent's delphi_v0.jinja2 which is itself
byte-identical to LF's embedded DELPHI_V0_JINJA_TEMPLATE (verified sha256 04a181f5b75f), we
render with BOTH the axolotl asset string and the LF-embedded string and assert equal output
(same-template ⇒ same render, and we also byte-compare the two template strings).
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    TOOLCALL_CONVO,
    TOOLS,
    load_tokenizer,
    ok,
    read_delphi_template,
)

TEMPLATE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "axolotl"
    / "utils"
    / "chat_templates"
    / "templates"
)
LF_TEMPLATE_PY = Path(
    "/Users/benjaminfeuer/Documents/LLaMA-Factory/src/llamafactory/data/template.py"
)


def replicate_axolotl_glob() -> dict:
    """Exact copy of axolotl utils/chat_templates/base.py:19-23 glob logic."""
    templates = {}
    for filename in [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".jinja")]:
        with open(TEMPLATE_DIR / filename, "r", encoding="utf-8") as f:
            templates[filename[:-6]] = f.read()
    return templates


def lf_embedded_template() -> str:
    src = LF_TEMPLATE_PY.read_text()
    return re.search(r'DELPHI_V0_JINJA_TEMPLATE = r"""(.*?)"""', src, re.S).group(1)


def render(tok, template_str: str) -> str:
    return tok.apply_chat_template(
        TOOLCALL_CONVO,
        tools=TOOLS,
        chat_template=template_str,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    # 1. glob registration
    templates = replicate_axolotl_glob()
    assert "delphi" in templates, "delphi.jinja did not register as 'delphi' via the glob"
    ok("axolotl glob registers delphi.jinja as chat_template name 'delphi'")

    asset = read_delphi_template()
    assert templates["delphi"] == asset
    ok("resolved 'delphi' == the delphi.jinja asset bytes")

    # 2. template byte-identity vs LF
    lf = lf_embedded_template()
    assert asset == lf, "delphi asset != LF DELPHI_V0_JINJA_TEMPLATE"
    ok(f"delphi asset byte-identical to LF DELPHI_V0_JINJA_TEMPLATE ({len(asset)} chars)")

    # 3. render gate — tool_calls + role:tool convo, byte-compare axolotl-asset vs LF render
    tok = load_tokenizer()
    r_axolotl = render(tok, asset)
    r_lf = render(tok, lf)
    assert r_axolotl == r_lf, "render diverges between axolotl asset and LF template"
    # protocol tokens present
    for marker in ("<|tool_call|>", "<|tool_call_end|>", "<|tool_result|>", "<|tool_result_end|>", "<|start_think|>"):
        assert marker in r_axolotl, f"delphi render missing protocol marker {marker}"
    ok("tool_calls+role:tool render byte-identical to LF + carries the full <|tool_call|>/<|tool_result|> protocol")

    # 4. flag-off: no chat_template override -> stock tokenizer template unaffected (sanity)
    stock = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    assert "<|tool_call|>" not in stock  # stock llama3 has no delphi protocol
    ok("flag-off: no chat_template override leaves the stock (non-delphi) template in effect")

    print("\nSTAGE 1 DELPHI RENDER GATE: PASS")


if __name__ == "__main__":
    main()
