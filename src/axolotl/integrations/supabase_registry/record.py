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

"""Build the model-registration record dict (mirrors LF tuner.py:162-246).

Dependency-free (no supabase/torch/axolotl imports) so it's unit-testable CPU-side and
carries zero import cost on the training hot path. The record shape matches LF exactly so
the shared OT-Agent registry sees identical rows regardless of trainer.
"""

import dataclasses
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional


def _first_str(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        return _first_str(value[0] if value else None)
    if isinstance(value, set):
        return _first_str(next(iter(value)) if value else None)
    return str(value) if value is not None else None


def _to_jsonable(obj: Any) -> Any:
    try:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
    except Exception:
        pass
    try:
        if hasattr(obj, "to_json_string"):
            return json.loads(obj.to_json_string())
    except Exception:
        pass
    if dataclasses.is_dataclass(obj):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _wandb_link() -> Optional[str]:
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            return wandb.run.url
    except Exception:
        pass
    return None


def build_registration_record(
    cfg: Any,
    training_start: datetime,
    training_end: Optional[datetime] = None,
) -> Optional[dict]:
    """Build the LF-shaped registration record from an axolotl cfg.

    Returns None (with the caller logging why) if a required field is missing.
    ``cfg`` is the axolotl DictDefault config; we read hub_model_id, base_model,
    datasets, and stash a jsonable subset of the config as training_parameters.
    """
    hub_repo_id = cfg.get("hub_model_id")
    if not hub_repo_id:
        return None

    # dataset name(s): axolotl datasets is a list of dataset dicts with a `path`.
    dataset_name = None
    datasets = cfg.get("datasets")
    if datasets:
        names = []
        for d in datasets:
            path = d.get("path") if isinstance(d, dict) else getattr(d, "path", None)
            if path:
                names.append(str(path))
        dataset_name = names if names else None
    if not dataset_name:
        dataset_name = _first_str(cfg.get("dataset")) or _first_str(cfg.get("dataset_dir"))
    if not dataset_name:
        return None

    base_model_name = cfg.get("base_model") or cfg.get("base_model_config")
    if not base_model_name:
        return None

    created_by = ""
    if "/" in str(hub_repo_id):
        created_by = str(hub_repo_id).split("/", 1)[0]
    created_by = (
        created_by
        or os.environ.get("HF_USERNAME")
        or os.environ.get("JOB_CREATOR", "")
    )

    agent_name = (
        os.environ.get("TRAINING_AGENT_NAME")
        or os.environ.get("DC_AGENT_NAME")
        or cfg.get("rl")  # axolotl RL mode marker, else None
        or "axolotl"
    )
    # axolotl is an SFT trainer here; RL uses a different path.
    training_type = "RL" if cfg.get("rl") else "SFT"

    training_parameters = {"axolotl_config": _to_jsonable(dict(cfg))}

    return {
        "agent_name": agent_name,
        "training_start": training_start.isoformat(),
        "training_end": (training_end or datetime.now(timezone.utc)).isoformat(),
        "created_by": created_by,
        "base_model_name": base_model_name,
        "dataset_name": dataset_name,
        "training_type": training_type,
        "training_parameters": training_parameters,
        "wandb_link": _wandb_link(),
        "traces_location_s3": os.environ.get("TRACE_S3_PATH"),
        "model_name": hub_repo_id,
    }
