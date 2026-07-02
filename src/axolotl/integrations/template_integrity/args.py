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

"""Input args for the save-time chat-template-integrity guard."""

from typing import Optional

from pydantic import BaseModel


class TemplateIntegrityArgs(BaseModel):
    """Input args for the TemplateIntegrityPlugin.

    Guards against the transformers save-time footgun where
    ``save_jinja_files=True`` (the transformers default) writes the chat template
    ONLY to ``chat_template.jinja`` and pops ``chat_template`` out of
    ``tokenizer_config.json`` — so a serving stack that reads only
    ``tokenizer_config.json`` sees no template and silently drops
    ``tool_calls`` / ``role:tool`` turns (the 0%-SWE-bench OOD failure).

    Setting ``tokenizer_save_jinja_files: false`` fixes the top-level ``output_dir``
    save, but the per-checkpoint trainer save at ``core/trainers/base.py`` calls
    ``save_pretrained`` with NO flag, so intermediate ``checkpoint-*/`` dirs stay
    split. This plugin's callback re-embeds ``chat_template`` into
    ``tokenizer_config.json`` on every ``on_save`` (covering checkpoints) and on
    ``on_train_end`` (covering ``output_dir``), rank-0 only, idempotently.
    """

    # Default False → the plugin is a no-op unless explicitly enabled (default-off
    # invariant). Enabling requires adding the plugin to the `plugins:` list.
    template_integrity_repair: Optional[bool] = True
