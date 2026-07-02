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

"""MFU (Model FLOPs Utilization) logging plugin.

Generic / upstream-shaped: an end-of-train rank-0 callback that reports achieved
per-GPU TFLOPs + MFU% from the HF Trainer's cumulative ``state.total_flos``, with a
peak-TFLOPs LUT (A100/H100/H200/B200) and a ``PEAK_TFLOPS_PER_GPU`` env override.
Ported from LLaMA-Factory extras/misc.py.

Enable via ``plugins: [axolotl.integrations.mfu.MFUPlugin]`` + ``mfu: true``.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import MFUArgs as MFUArgs

LOG = get_logger(__name__)


class MFUPlugin(BasePlugin):
    """Plugin adding an end-of-train MFU logging callback."""

    def get_input_args(self):
        return "axolotl.integrations.mfu.MFUArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        if not cfg.get("mfu", False):
            return []
        from .callback import MFUCallback

        return [MFUCallback(trainer=trainer)]
