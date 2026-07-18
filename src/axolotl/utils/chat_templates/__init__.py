"""
This module provides functionality for selecting chat templates based on user choices.
These templates are used for formatting messages in a conversation.
"""

from .base import (
    _CHAT_TEMPLATES,
    extract_chat_template_args,
    get_chat_template,
    get_chat_template_from_config,
    register_chat_template,
    resolve_export_chat_template,
    template_handles_tools,
)

__all__ = [
    "get_chat_template",
    "extract_chat_template_args",
    "get_chat_template_from_config",
    "register_chat_template",
    "resolve_export_chat_template",
    "template_handles_tools",
    "_CHAT_TEMPLATES",
]
