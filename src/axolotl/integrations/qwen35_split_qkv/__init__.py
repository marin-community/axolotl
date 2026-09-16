"""Independent Q/K/V LoRA support for Qwen3.5 linear attention."""

from axolotl.integrations.qwen35_split_qkv.adapter import fuse_split_qkv_adapter
from axolotl.integrations.qwen35_split_qkv.plugin import Qwen35SplitQKVPlugin

__all__ = ["Qwen35SplitQKVPlugin", "fuse_split_qkv_adapter"]
