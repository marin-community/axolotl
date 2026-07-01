# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-of-train Supabase model-registration callback (rank-0, opt-in)."""

import os
from datetime import datetime, timezone

from transformers import TrainerCallback

from axolotl.utils.logging import get_logger

from .record import build_registration_record

LOG = get_logger(__name__)

_REQUIRED_SUPABASE_KEYS = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY")


def _supabase_ready() -> bool:
    """True if the 3 SUPABASE_* env vars resolve (directly or via a KEYS file).

    The KEYS-file loader is imported lazily so nothing supabase-related is imported
    on the flag-off hot path.
    """
    if all(os.environ.get(k) for k in _REQUIRED_SUPABASE_KEYS):
        return True
    try:
        from .db import load_supabase_keys

        return bool(load_supabase_keys()) and all(
            os.environ.get(k) for k in _REQUIRED_SUPABASE_KEYS
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Supabase key load failed: %s", exc)
        return False


class SupabaseRegistryCallback(TrainerCallback):
    """Rank-0 on_train_end callback that registers one model row.

    Fires ONLY if: rank-0 ∧ supabase-ready ∧ a hub push is configured
    (``cfg.hub_model_id``). Never raises into training.
    """

    def __init__(self, cfg=None, client_factory=None):
        self.cfg = cfg
        # client_factory lets the gate inject a MOCK client (no live DB).
        self._client_factory = client_factory
        self._training_start = datetime.now(timezone.utc)

    def on_train_begin(self, args, state, control, **kwargs):
        self._training_start = datetime.now(timezone.utc)

    def on_train_end(self, args, state, control, **kwargs):
        if state is not None and not state.is_world_process_zero:
            return
        cfg = self.cfg
        if not cfg or not cfg.get("hub_model_id"):
            return
        if not _supabase_ready():
            LOG.warning("Supabase registration skipped: SUPABASE_* creds unavailable.")
            return

        record = build_registration_record(
            cfg, self._training_start, datetime.now(timezone.utc)
        )
        if record is None:
            LOG.warning(
                "Supabase registration skipped: missing base_model / dataset / hub_model_id."
            )
            return

        try:
            from .db import register_trained_model

            result = register_trained_model(record)
            if result.get("success"):
                LOG.info("Registered trained model metadata with Supabase.")
            else:
                LOG.warning("Supabase registration failed: %s", result.get("error"))
        except Exception as exc:  # never raise into training
            LOG.warning("Supabase registration raised (ignored): %s", exc)
