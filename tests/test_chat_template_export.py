"""Regression tests for tool-capable chat-template preservation at export.

Guards the "0% SWE-bench" footgun: a config that sets e.g. ``chat_template: chatml``
on top of a base model whose own template is tool-capable used to overwrite the
saved ``tokenizer.chat_template`` with a bare, tool-blind ChatML template. The
exported model could then no longer render tool calls at serve time (vLLM silently
drops ``tools``) even though training was healthy.

These tests assert that a tool-capable base template SURVIVES export.
"""

import json
import os

import pytest

from axolotl.loaders import load_tokenizer
from axolotl.utils.chat_templates import (
    _CHAT_TEMPLATES,
    resolve_export_chat_template,
    template_handles_tools,
)
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

# A minimal, self-contained tool-capable template: it references the `tools` kwarg
# (so tool *definitions* are rendered) and emits `<tool_call>` markers for
# assistant tool calls — the two things a bare ChatML template lacks.
TOOLS_AWARE_TEMPLATE = (
    "{% if tools %}{{ '<tools>' + (tools | tojson) + '</tools>' }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message.get('tool_calls') %}"
    "{{ '<tool_call>' + (message['tool_calls'] | tojson) + '</tool_call>' }}"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{{ '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

CHATML_TEMPLATE = _CHAT_TEMPLATES["chatml"]


def _read_exported_template(save_dir: str) -> str:
    """Return the chat template a serving stack would load from ``save_dir``.

    transformers may split the template into a standalone ``chat_template.jinja``
    (popping it from ``tokenizer_config.json``); a serving stack resolves whichever
    is present. We union both so the assertion is robust to that split.
    """
    parts = []
    jinja_path = os.path.join(save_dir, "chat_template.jinja")
    if os.path.exists(jinja_path):
        with open(jinja_path, "r", encoding="utf-8") as handle:
            parts.append(handle.read())
    cfg_path = os.path.join(save_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            tmpl = json.load(handle).get("chat_template")
            if tmpl:
                parts.append(tmpl)
    return "\n".join(parts)


class TestTemplateHandlesTools:
    """Unit tests for the tool-capability detector."""

    def test_chatml_is_tool_blind(self):
        assert template_handles_tools(CHATML_TEMPLATE) is False

    def test_none_and_empty_are_tool_blind(self):
        assert template_handles_tools(None) is False
        assert template_handles_tools("") is False

    def test_tools_aware_template_detected(self):
        assert template_handles_tools(TOOLS_AWARE_TEMPLATE) is True

    def test_whitespace_control_variant_detected(self):
        # `{%- if tools %}` must be recognized the same as `{% if tools %}`.
        assert (
            template_handles_tools("{%- if tools -%}{{ tools }}{%- endif -%}") is True
        )

    def test_shipped_tool_templates_detected(self):
        # The qwen3 template shipped with axolotl is tool-capable.
        assert template_handles_tools(_CHAT_TEMPLATES["qwen3"]) is True


class TestResolveExportChatTemplate:
    """Unit tests for the export-template resolver."""

    def test_downgrade_is_averted(self):
        resolved, averted = resolve_export_chat_template(
            base_template=TOOLS_AWARE_TEMPLATE,
            configured_template=CHATML_TEMPLATE,
        )
        assert averted is True
        assert resolved == TOOLS_AWARE_TEMPLATE

    def test_explicit_tool_choice_is_respected(self):
        # Configured template is itself tool-capable -> keep it (no override).
        resolved, averted = resolve_export_chat_template(
            base_template=TOOLS_AWARE_TEMPLATE,
            configured_template=TOOLS_AWARE_TEMPLATE,
        )
        assert averted is False
        assert resolved == TOOLS_AWARE_TEMPLATE

    def test_no_base_template_keeps_configured(self):
        resolved, averted = resolve_export_chat_template(
            base_template=None,
            configured_template=CHATML_TEMPLATE,
        )
        assert averted is False
        assert resolved == CHATML_TEMPLATE

    def test_tool_blind_base_keeps_configured(self):
        resolved, averted = resolve_export_chat_template(
            base_template=CHATML_TEMPLATE,
            configured_template=CHATML_TEMPLATE,
        )
        assert averted is False
        assert resolved == CHATML_TEMPLATE

    def test_preserve_disabled_keeps_configured(self):
        resolved, averted = resolve_export_chat_template(
            base_template=TOOLS_AWARE_TEMPLATE,
            configured_template=CHATML_TEMPLATE,
            preserve_tool_capable=False,
        )
        assert averted is False
        assert resolved == CHATML_TEMPLATE


class TestLoadTokenizerToolPreservation:
    """End-to-end tests through ``load_tokenizer`` + a save round-trip."""

    @enable_hf_offline
    def _make_tool_capable_base(self, base_dir: str) -> None:
        """Materialize a small base model whose tokenizer template is tool-capable."""
        base_cfg = DictDefault({"tokenizer_config": "HuggingFaceTB/SmolLM2-135M"})
        base_tok = load_tokenizer(base_cfg)
        base_tok.chat_template = TOOLS_AWARE_TEMPLATE
        base_tok.save_pretrained(base_dir)
        with open(
            os.path.join(base_dir, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"model_type": "llama"}, handle)

    @enable_hf_offline
    def test_chatml_override_preserves_base_tools(self, temp_dir):
        base_dir = os.path.join(temp_dir, "base")
        out_dir = os.path.join(temp_dir, "out")
        self._make_tool_capable_base(base_dir)

        # chat_template: chatml on top of a tool-capable base must NOT downgrade
        # the SAVED template to bare ChatML.
        cfg = DictDefault(
            {
                "tokenizer_config": base_dir,
                "chat_template": "chatml",
                "output_dir": out_dir,
            }
        )
        tokenizer = load_tokenizer(cfg)

        assert template_handles_tools(tokenizer.chat_template)
        assert "<tool_call>" in tokenizer.chat_template

        # ... and it survives an actual export (even with the transformers
        # save_jinja_files split that pops chat_template out of tokenizer_config).
        tokenizer.save_pretrained(out_dir, save_jinja_files=True)
        exported = _read_exported_template(out_dir)
        assert template_handles_tools(exported)
        assert "<tool_call>" in exported

    @enable_hf_offline
    def test_flag_off_exports_configured_template(self, temp_dir):
        base_dir = os.path.join(temp_dir, "base")
        self._make_tool_capable_base(base_dir)

        # Opt-out: the configured (tool-blind) template is exported as-is, proving
        # the preservation is what saves tool-calling in the default path.
        cfg = DictDefault(
            {
                "tokenizer_config": base_dir,
                "chat_template": "chatml",
                "preserve_tool_capable_chat_template": False,
            }
        )
        tokenizer = load_tokenizer(cfg)

        assert not template_handles_tools(tokenizer.chat_template)
        assert tokenizer.chat_template == CHATML_TEMPLATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
