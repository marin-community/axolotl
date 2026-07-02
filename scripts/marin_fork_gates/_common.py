"""Shared helpers for the marin-fork CPU gates (Stages 0-5).

These gates run transformers-only (no axolotl CUDA stack, which does not install on
arm64 Mac). They reproduce axolotl's *save path* and *chat-template resolution* so the
flag-off byte-identical + footgun invariants can be proven CPU-side. The actual 1-GPU
training smokes are DEFERRED-CLUSTER.

Env: /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python (transformers 4.57.3;
the fork pins 5.12.1 — the save_jinja_files pop/re-embed mechanism is identical in both).
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DELPHI_JINJA_PATH = REPO / "src" / "axolotl" / "utils" / "chat_templates" / "templates" / "delphi.jinja"

# A Llama-3 tokenizer (delphi is a Llama-3 template). Cached locally.
BASE_TOKENIZER = "NousResearch/Meta-Llama-3-8B-Instruct"


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_TOKENIZER)


def read_delphi_template() -> str:
    """The delphi jinja string the axolotl asset ships (== LF DELPHI_V0_JINJA_TEMPLATE)."""
    return DELPHI_JINJA_PATH.read_text()


def saved_tokenizer_manifest(save_dir: Path) -> dict:
    """Capture the footgun-relevant state of a saved tokenizer dir:
    the file list, whether tokenizer_config.json carries a chat_template, and
    whether a separate chat_template.jinja file exists (+ its bytes)."""
    save_dir = Path(save_dir)
    files = sorted(p.name for p in save_dir.iterdir() if p.is_file())
    tok_cfg = json.loads((save_dir / "tokenizer_config.json").read_text())
    jinja_file = save_dir / "chat_template.jinja"
    return {
        "files": files,
        "config_has_chat_template": "chat_template" in tok_cfg,
        "config_chat_template": tok_cfg.get("chat_template"),
        "jinja_file_exists": jinja_file.exists(),
        "jinja_file_contents": jinja_file.read_text() if jinja_file.exists() else None,
    }


# ---- a canonical tool_calls + role:tool conversation for the render gate ----
TOOLCALL_CONVO = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "The user wants weather. I should call the tool.",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "content": '{"temp_c": 18, "cond": "cloudy"}'},
    {"role": "assistant", "content": "It's 18C and cloudy in Paris."},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def ok(msg):
    print(f"  PASS  {msg}")


def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)
