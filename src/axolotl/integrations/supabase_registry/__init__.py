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

"""Supabase model-registration plugin (lab-private).

On rank-0 at train-end, when ``supabase_register: true`` (or the 3 SUPABASE_* env vars
resolve) AND a hub push is configured, registers one model-registration row in the lab's
Supabase registry — mirroring LLaMA-Factory's post-train registration block, relocated
from a core patch to a plugin.

Default OFF (opt-in). Creds by env-var name only. No supabase import on the flag-off hot
path (the callback + db package import supabase lazily, only when actually registering).

Enable via ``plugins: [axolotl.integrations.supabase_registry.SupabaseRegistryPlugin]`` +
``supabase_register: true`` + SUPABASE_* env (e.g. via a ``KEYS`` secrets file).
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import SupabaseRegistryArgs as SupabaseRegistryArgs

LOG = get_logger(__name__)


class SupabaseRegistryPlugin(BasePlugin):
    """Plugin adding a rank-0 end-of-train Supabase registration callback."""

    def get_input_args(self):
        return "axolotl.integrations.supabase_registry.SupabaseRegistryArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        if not cfg.get("supabase_register", False):
            return []
        from .callback import SupabaseRegistryCallback

        return [SupabaseRegistryCallback(cfg=cfg)]
