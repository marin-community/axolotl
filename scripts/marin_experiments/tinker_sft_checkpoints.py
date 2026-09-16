"""Publish remote completion markers for Axolotl SFT checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from fsspec import AbstractFileSystem
from s3fs import S3FileSystem

CHECKPOINT_PATTERN = re.compile(r"checkpoint-([1-9][0-9]*)")
REQUIRED_FILES = frozenset(
    {
        "adapter_config.json",
        "chat_template.jinja",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokens_state.json",
    }
)
COMMIT_FILENAME = "checkpoint-commit.json"
LOCAL_CHECKPOINTS_TO_RETAIN = 2


def completed_checkpoint_inventory(checkpoint: Path, step: int, world_size: int) -> dict[str, int] | None:
    """Return the files in a finished trainer checkpoint, or None while it is being saved."""
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != step:
        raise ValueError(f"{state_path} does not describe step {step}")
    files = {path.name: path.stat().st_size for path in checkpoint.iterdir() if path.is_file()}
    required = REQUIRED_FILES | {f"rng_state_{rank}.pth" for rank in range(world_size)}
    if not required.issubset(files) or not any(name.startswith("adapter_model") and name.endswith(".safetensors") for name in files):
        return None
    if any(size == 0 for size in files.values()):
        return None
    return files


def publish_completed_checkpoints(
    output_root: Path,
    output_uri: str,
    *,
    cadence: int,
    world_size: int,
    filesystem: AbstractFileSystem | None = None,
) -> list[int]:
    """Mark each complete remote checkpoint after all objects match its local inventory."""
    checkpoint_root = output_root / "peft"
    if not checkpoint_root.is_dir():
        return []
    candidates = []
    for checkpoint in checkpoint_root.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if checkpoint.is_dir() and match and int(match.group(1)) % cadence == 0:
            candidates.append((int(match.group(1)), checkpoint))
    published = []
    remote_root = output_uri.removeprefix("s3://").rstrip("/")
    for step, checkpoint in sorted(candidates):
        inventory = completed_checkpoint_inventory(checkpoint, step, world_size)
        if inventory is None:
            continue
        filesystem = filesystem or S3FileSystem()
        remote_checkpoint = f"{remote_root}/peft/{checkpoint.name}"
        commit_path = f"{remote_checkpoint}/{COMMIT_FILENAME}"
        if filesystem.exists(commit_path):
            with filesystem.open(commit_path, "r") as source:
                committed = json.load(source)
            if committed.get("step") != step or committed.get("files") != inventory:
                raise ValueError(f"Remote checkpoint marker conflicts with local step {step}: {commit_path}")
            published.append(step)
            continue
        for name, size in inventory.items():
            path = f"{remote_checkpoint}/{name}"
            if not filesystem.exists(path) or filesystem.info(path)["size"] != size:
                break
        else:
            with filesystem.open(commit_path, "w") as destination:
                json.dump({"schema_version": 1, "step": step, "files": inventory}, destination, sort_keys=True)
            published.append(step)
    return published


def prune_published_checkpoints(output_root: Path, published: list[int]) -> None:
    """Keep the newest local checkpoints; older copies remain at their committed remote paths."""
    checkpoint_root = (output_root / "peft").resolve()
    for step in sorted(published)[:-LOCAL_CHECKPOINTS_TO_RETAIN]:
        checkpoint = checkpoint_root / f"checkpoint-{step}"
        if checkpoint.resolve().parent != checkpoint_root:
            raise ValueError(f"Checkpoint path escapes {checkpoint_root}: {checkpoint}")
        shutil.rmtree(checkpoint)


def missing_remote_checkpoints(
    output_uri: str, expected_steps: range, filesystem: AbstractFileSystem | None = None
) -> list[int]:
    """Find expected checkpoints without a remote completion marker."""
    filesystem = filesystem or S3FileSystem()
    remote_root = output_uri.removeprefix("s3://").rstrip("/")
    return [
        step
        for step in expected_steps
        if not filesystem.exists(f"{remote_root}/peft/checkpoint-{step}/{COMMIT_FILENAME}")
    ]
