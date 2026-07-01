"""Vendored subset of the LLaMA-Factory unified Supabase DB package.

Exposes the model-registration entrypoints used by the supabase_registry plugin.
"""

from .utils import load_supabase_keys, register_trained_model

__all__ = ["load_supabase_keys", "register_trained_model"]
