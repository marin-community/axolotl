"""
utility functions for chat templates
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

LOG = get_logger("axolotl.utils.chat_templates")

_JINJA_TEMPLATE_CHOICE = "jinja"
_DEFAULT_TEMPLATE_CHOICE = "tokenizer_default"
_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX = "tokenizer_default_fallback_"

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CHAT_TEMPLATES: dict[str, str] = {}
for filename in [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".jinja")]:
    with open(os.path.join(TEMPLATE_DIR, filename), "r", encoding="utf-8") as f:
        _CHAT_TEMPLATES[filename[:-6]] = f.read()


def get_chat_template(
    user_choice: str,
    jinja_template: str | None = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> str:
    """
    Finds the correct chat_template based on the user's choice, jinja_template, and tokenizer.

    Args:
        user_choice (str): The user's choice of template.
        jinja_template (str, optional): The jinja template string or Path to a valid jinja template file. Defaults to None.
        tokenizer (PreTrainedTokenizerBase, optional): The tokenizer. Defaults to None.

    Returns:
        str: The chosen template string.

    Raises:
        ValueError: If the user_choice is not found in the templates.
    """
    if user_choice == _JINJA_TEMPLATE_CHOICE:
        if not jinja_template:
            raise ValueError(
                f"`jinja_template` cannot be None when `chat_template` choice is {_JINJA_TEMPLATE_CHOICE}"
            )
        if os.path.exists(jinja_template) and os.path.isfile(jinja_template):
            with open(jinja_template, "r", encoding="utf-8") as file:
                jinja_template = file.read()
        return jinja_template

    if user_choice == _DEFAULT_TEMPLATE_CHOICE:
        if not tokenizer:
            raise ValueError(
                f"`tokenizer` cannot be None when chat_template choice is {_DEFAULT_TEMPLATE_CHOICE}"
            )
        if not tokenizer.chat_template:
            raise ValueError(
                f"`chat_template choice is {_DEFAULT_TEMPLATE_CHOICE} but tokenizer's chat_template is null. "
                f"Please add a chat_template in tokenizer config"
            )
        return tokenizer.chat_template  # type: ignore

    if user_choice.startswith(_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX):
        if not tokenizer:
            raise ValueError(
                f"`tokenizer` cannot be None when chat_template choice starts with {_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX}"
            )
        if tokenizer.chat_template:
            return tokenizer.chat_template  # type: ignore

        user_choice = user_choice[
            len(_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX) :
        ]
        LOG.warning(
            f"No chat template found on tokenizer, falling back to {user_choice}. It is recommended to set --train_on_inputs to True for the model to learn this chat template."
        )

    if user_choice in _CHAT_TEMPLATES:
        return _CHAT_TEMPLATES[user_choice]

    raise ValueError(f"Template '{user_choice}' not found.")


def extract_chat_template_args(cfg, ds_cfg: Dict[str, Any] | None = None):
    if ds_cfg and ds_cfg.get("chat_template"):
        chat_template_choice = ds_cfg.get("chat_template") or _DEFAULT_TEMPLATE_CHOICE
        chat_template_jinja = ds_cfg.get("chat_template_jinja")
    else:
        chat_template_choice = cfg.get("chat_template") or _DEFAULT_TEMPLATE_CHOICE
        chat_template_jinja = cfg.get("chat_template_jinja")
    return chat_template_choice, chat_template_jinja


def get_chat_template_from_config(
    cfg,
    ds_cfg: Dict[str, Any] | None = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> str:
    chat_template_choice, chat_template_jinja = extract_chat_template_args(
        cfg=cfg, ds_cfg=ds_cfg
    )
    return get_chat_template(
        user_choice=chat_template_choice,
        jinja_template=chat_template_jinja,
        tokenizer=tokenizer,
    )


def register_chat_template(template_name: str, chat_template: str):
    """
    Registers chat templates.

    Args:
        template_name (str): The name of the template.
        chat_template (str): The template string.
    """

    if template_name in _CHAT_TEMPLATES:
        raise ValueError(f"Template '{template_name}' already exists.")

    _CHAT_TEMPLATES[template_name] = chat_template


# Substrings that a chat template must contain to be able to render tool
# definitions (the ``tools`` kwarg) and/or tool-call / tool-result turns. A
# template lacking all of these cannot express tool-calling, so a serving stack
# (e.g. vLLM) that is handed ``tools`` will silently drop them.
_TOOL_TEMPLATE_MARKERS = (
    "tool_calls",
    "tool_call",
    "tool_response",
    "if tools",  # `{% if tools %}` / `{%- if tools %}` after whitespace-strip normalization
    "'tool'",
    '"tool"',
)


def template_handles_tools(template: str | None) -> bool:
    """Best-effort check for whether a chat template can render tool calls.

    Detects the two things a tool-capable template needs: a reference to the
    ``tools`` kwarg (so tool *definitions* injected at inference time are rendered)
    and/or handling of ``tool_calls`` / ``role == "tool"`` turns. A bare ChatML
    template (``<|im_start|>{role}\\n{content}<|im_end|>``) matches none of these.

    This is intentionally conservative (substring based, no jinja parse): it only
    reports ``True`` when explicit tool markers are present, which is enough to
    flag a tool-capable -> tool-blind downgrade at save time.
    """
    if not template:
        return False
    # Normalize jinja whitespace-control so `{%- if tools %}` matches `if tools`.
    normalized = template.replace("{%-", "{%").replace("-%}", "%}")
    return any(marker in normalized for marker in _TOOL_TEMPLATE_MARKERS)


def resolve_export_chat_template(
    base_template: str | None,
    configured_template: str,
    preserve_tool_capable: bool = True,
) -> tuple[str, bool]:
    """Pick the chat template to persist on the tokenizer for inference/export.

    Axolotl overwrites ``tokenizer.chat_template`` with the config-chosen template
    purely so it is saved for inference (training itself renders with the resolved
    template passed explicitly to ``apply_chat_template``). When a config sets e.g.
    ``chat_template: chatml`` on top of a base model whose own template was
    tool-capable (e.g. Qwen3's ``{% if tools %}`` / ``<tool_call>`` template), that
    overwrite silently downgrades the *saved* template to a tool-blind one -> the
    exported model can no longer tool-call at serve time (the "0% SWE-bench" footgun).

    Returns ``(template_to_save, downgrade_averted)``:
      * If ``preserve_tool_capable`` and the base template handles tools but the
        configured template does not, returns ``(base_template, True)`` so the
        tool-capable template survives export.
      * Otherwise returns ``(configured_template, False)`` — the explicit config
        choice is respected (never silently overridden when no tool capability is lost).
    """
    if (
        preserve_tool_capable
        and base_template
        and template_handles_tools(base_template)
        and not template_handles_tools(configured_template)
    ):
        return base_template, True
    return configured_template, False
