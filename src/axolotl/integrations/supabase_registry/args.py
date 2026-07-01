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

"""Input args for the Supabase model-registration plugin."""

from typing import Optional

from pydantic import BaseModel


class SupabaseRegistryArgs(BaseModel):
    """Input args for SupabaseRegistryPlugin.

    Default OFF (opt-in), mirroring LLaMA-Factory's ``enable_db_registration: false``
    posture. Registration fires only on rank-0 at train-end when ALL of: this flag is
    true (or the 3 SUPABASE_* env vars resolve), AND a hub push is configured
    (``hub_model_id`` set). Creds are supplied by env-var name only
    (``SUPABASE_URL`` / ``SUPABASE_ANON_KEY`` / ``SUPABASE_SERVICE_ROLE_KEY``, or a
    ``KEYS``-pointed secrets file) — never in a YAML or the fork.
    """

    supabase_register: Optional[bool] = False
