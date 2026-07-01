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

"""Pure-stdlib save-time chat-template-integrity repair.

Kept dependency-free (no torch / transformers / axolotl imports) so it can be
unit-tested CPU-side and called from the trainer callback alike. The repair is
idempotent and additive: it only ever ENSURES ``tokenizer_config.json`` carries a
non-empty ``chat_template`` matching the on-disk ``chat_template.jinja`` (when the
latter exists). It never deletes the jinja file and never touches anything else.
"""

import json
import os
from pathlib import Path

CHAT_TEMPLATE_JINJA = "chat_template.jinja"
TOKENIZER_CONFIG = "tokenizer_config.json"


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def repair_saved_dir(save_dir, expected_template: str | None = None) -> dict:
    """Ensure ``tokenizer_config.json`` in ``save_dir`` carries ``chat_template``.

    Resolution order for the template string:
      1. ``expected_template`` if provided,
      2. else the existing ``tokenizer_config.json['chat_template']`` (already fine),
      3. else the bytes of ``chat_template.jinja`` (the split-out footgun case).

    Returns a small report dict describing what was found / changed. A no-op (already
    embedded, no jinja file) leaves the file byte-untouched.
    """
    save_dir = Path(save_dir)
    cfg_path = save_dir / TOKENIZER_CONFIG
    jinja_path = save_dir / CHAT_TEMPLATE_JINJA

    report = {
        "dir": str(save_dir),
        "had_config_chat_template": False,
        "had_jinja_file": jinja_path.exists(),
        "repaired": False,
        "reason": None,
    }

    if not cfg_path.exists():
        # Nothing to repair (e.g. a sharded-weights-only merge dir with no tokenizer).
        report["reason"] = "no tokenizer_config.json"
        return report

    cfg = _read_json(cfg_path)
    existing = cfg.get("chat_template")
    report["had_config_chat_template"] = bool(existing)

    jinja_bytes = jinja_path.read_text(encoding="utf-8") if jinja_path.exists() else None

    template = expected_template
    if template is None:
        template = existing if existing else jinja_bytes

    if not template:
        report["reason"] = "no chat_template available anywhere (nothing to embed)"
        return report

    # If a jinja file exists, keep it byte-consistent with what we embed.
    if jinja_bytes is not None and jinja_bytes != template:
        # Trust the explicit/embedded template as canonical; rewrite the jinja to match.
        jinja_path.write_text(template, encoding="utf-8")
        report["jinja_rewritten"] = True

    if existing == template:
        report["reason"] = "already embedded + consistent"
        return report

    cfg["chat_template"] = template
    _write_json(cfg_path, cfg)
    report["repaired"] = True
    report["reason"] = "embedded chat_template into tokenizer_config.json"
    return report


def iter_repair_targets(output_dir) -> list:
    """The output_dir itself plus any ``checkpoint-*`` subdirs under it."""
    output_dir = Path(output_dir)
    targets = []
    if (output_dir / TOKENIZER_CONFIG).exists():
        targets.append(output_dir)
    if output_dir.is_dir():
        for name in sorted(os.listdir(output_dir)):
            sub = output_dir / name
            if name.startswith("checkpoint-") and (sub / TOKENIZER_CONFIG).exists():
                targets.append(sub)
    return targets
