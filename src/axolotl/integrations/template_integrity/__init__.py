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

"""Save-time chat-template-integrity guard plugin.

Ensures a saved checkpoint presents its chat template where the serving stack reads
it — ``tokenizer_config.json['chat_template']`` — rather than only in a separate
``chat_template.jinja`` (the transformers save-time footgun). Generic /
upstream-shaped: no lab- or delphi-specific coupling; it repairs whatever template
the tokenizer carries.

Enable via ``plugins: [axolotl.integrations.template_integrity.TemplateIntegrityPlugin]``.
Pair with ``tokenizer_save_jinja_files: false`` for the top-level ``output_dir`` save;
this plugin additionally covers the flag-ignoring per-checkpoint save.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import TemplateIntegrityArgs as TemplateIntegrityArgs

LOG = get_logger(__name__)


class TemplateIntegrityPlugin(BasePlugin):
    """Plugin adding a rank-0 save-time chat-template-integrity callback."""

    def get_input_args(self):
        return "axolotl.integrations.template_integrity.TemplateIntegrityArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        if not cfg.get("template_integrity_repair", True):
            return []
        from .callback import TemplateIntegrityCallback

        tokenizer = getattr(trainer, "processing_class", None) or getattr(
            trainer, "tokenizer", None
        )
        return [TemplateIntegrityCallback(tokenizer=tokenizer)]
