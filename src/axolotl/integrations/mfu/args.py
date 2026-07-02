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

"""Input args for the MFU logging plugin."""

from typing import Optional

from pydantic import BaseModel


class MFUArgs(BaseModel):
    """Input args for the MFUPlugin (mirrors LF finetuning_args.py:524-539).

    Peak per-GPU TFLOPs may be overridden via the ``PEAK_TFLOPS_PER_GPU`` env var
    (comma-separated allowed; first entry used) for accurate MFU on unknown devices.
    """

    # default False → plugin is a no-op unless explicitly enabled (default-off invariant)
    mfu: Optional[bool] = False
    mfu_profile_every: Optional[int] = 50
    mfu_warmup_steps: Optional[int] = 5
