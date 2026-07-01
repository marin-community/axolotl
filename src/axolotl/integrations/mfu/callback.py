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

"""End-of-train MFU logging callback."""

import time
from typing import Optional

from transformers import TrainerCallback

from axolotl.utils.logging import get_logger

from .compute import compute_mfu_from_trainer

LOG = get_logger(__name__)


class MFUCallback(TrainerCallback):
    """Rank-0 callback that logs achieved MFU at end-of-train.

    Ports LF's end-of-run path: after training, compute achieved per-GPU TFLOPs +
    MFU% from the trainer's cumulative ``state.total_flos`` and the measured runtime.
    Logs the metrics and stashes them on ``state.mfu_metrics`` for downstream readers.
    """

    def __init__(self, trainer=None):
        self._trainer = trainer
        self._t_start: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t_start = time.time()

    def _train_runtime(self, state) -> float:
        # Prefer HF's own train_runtime metric if present in the log history.
        for entry in reversed(getattr(state, "log_history", []) or []):
            if "train_runtime" in entry:
                return float(entry["train_runtime"])
        if self._t_start is not None:
            return max(time.time() - self._t_start, 1e-9)
        return 0.0

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        trainer = self._trainer
        metrics = compute_mfu_from_trainer(trainer, self._train_runtime(state))
        if not metrics:
            LOG.warning(
                "MFU: total_flos unavailable or runtime non-positive; skipping MFU report."
            )
            return
        # stash for downstream readers + log
        try:
            state.mfu_metrics = metrics
        except Exception:
            pass
        if trainer is not None and hasattr(trainer, "log"):
            try:
                trainer.log(dict(metrics))
            except Exception:
                pass
        LOG.info("MFU: %s", metrics)
