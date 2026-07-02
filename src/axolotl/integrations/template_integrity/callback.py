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

"""TrainerCallback that repairs the saved chat-template on save + train-end."""

import os

from transformers import TrainerCallback

from axolotl.utils.logging import get_logger

from .repair import repair_saved_dir

LOG = get_logger(__name__)


class TemplateIntegrityCallback(TrainerCallback):
    """Rank-0 callback ensuring saved dirs carry ``chat_template`` in the config.

    Fires on:
      - ``on_save``: after each checkpoint save (covers the flag-ignoring per-checkpoint
        save at ``core/trainers/base.py`` that would otherwise leave ``checkpoint-*/``
        split-out).
      - ``on_train_end``: covers the final ``output_dir`` save.

    The expected template is the trainer tokenizer's ``chat_template`` (the resolved,
    in-effect template), passed to the pure-stdlib :func:`repair_saved_dir`.
    """

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    def _expected_template(self):
        tok = self._tokenizer
        tmpl = getattr(tok, "chat_template", None) if tok is not None else None
        return tmpl if isinstance(tmpl, str) and tmpl else None

    def _repair(self, save_dir, state):
        if state is not None and not state.is_world_process_zero:
            return
        if not save_dir or not os.path.isdir(save_dir):
            return
        report = repair_saved_dir(save_dir, expected_template=self._expected_template())
        if report.get("repaired"):
            LOG.info(
                "template-integrity: embedded chat_template into %s/tokenizer_config.json",
                save_dir,
            )

    def on_save(self, args, state, control, **kwargs):
        # transformers names the per-checkpoint dir checkpoint-<global_step>.
        ckpt = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        self._repair(ckpt, state)

    def on_train_end(self, args, state, control, **kwargs):
        self._repair(args.output_dir, state)
