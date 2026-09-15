"""Run the pinned OpenThoughts3 SFT control reported by the Tinker cookbook."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

MODEL_REPOSITORY = "Qwen/Qwen3.5-9B-Base"
MODEL_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
DATASET_REPOSITORY = "open-thoughts/OpenThoughts3-1.2M"
DATASET_REVISION = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
WORLD_SIZE = 8
FULL_STEPS = 3_000
FULL_BATCH_SIZE = 128
FULL_SEQUENCE_LENGTH = 16_384
FULL_SHUFFLE_BUFFER = FULL_STEPS * FULL_BATCH_SIZE
SYNC_INTERVAL = 300
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
KNOWN_DEVIATIONS = (
    "Axolotl and Tinker use different distributed data loaders and training kernels.",
    "PEFT wraps Qwen3.5's fused Gated DeltaNet QKV projection with one adapter; Tinker exposes separate Q/K/V adapters.",
    "Tinker's service-side LoRA initialization and scaling are not published.",
)


class Stage(StrEnum):
    PLUMBING = "plumbing"
    FIDELITY_STEP = "fidelity_step"
    FULL = "full"


@dataclass(frozen=True)
class StageShape:
    steps: int
    sequence_length: int
    global_batch_size: int
    shuffle_buffer: int
    materialized_rows: int


STAGES = {
    Stage.PLUMBING: StageShape(1, 2_048, 8, 128, 8),
    Stage.FIDELITY_STEP: StageShape(1, FULL_SEQUENCE_LENGTH, FULL_BATCH_SIZE, FULL_SHUFFLE_BUFFER, FULL_SHUFFLE_BUFFER),
    Stage.FULL: StageShape(FULL_STEPS, FULL_SEQUENCE_LENGTH, FULL_BATCH_SIZE, FULL_SHUFFLE_BUFFER, FULL_SHUFFLE_BUFFER),
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_revisions() -> dict[str, str]:
    api = HfApi()
    model = api.model_info(MODEL_REPOSITORY, revision=MODEL_REVISION)
    dataset = api.dataset_info(DATASET_REPOSITORY, revision=DATASET_REVISION)
    if model.sha != MODEL_REVISION or dataset.sha != DATASET_REVISION:
        raise ValueError("Hugging Face did not resolve the pinned model and dataset revisions")
    return {"model": model.sha, "dataset": dataset.sha}


def runtime_inventory(source_commit: str) -> dict[str, object]:
    baked_commit = os.environ.get("AXOLOTL_SOURCE_COMMIT")
    if not COMMIT_PATTERN.fullmatch(source_commit) or baked_commit != source_commit:
        raise ValueError("--source-commit must match the commit baked into the task image")
    import torch

    if torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError(f"The SFT control requires exactly {WORLD_SIZE} visible GPUs")
    return {
        "source_commit": source_commit,
        "python": sys.version,
        "axolotl": importlib.metadata.version("axolotl"),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "nvidia_smi": subprocess.run(["nvidia-smi", "-q"], check=True, capture_output=True, text=True).stdout,
    }


def materialize_dataset(destination: Path, shape: StageShape) -> int:
    dataset = load_dataset(DATASET_REPOSITORY, split="train", streaming=True, revision=DATASET_REVISION)
    rows = dataset.shuffle(seed=0, buffer_size=shape.shuffle_buffer).take(shape.materialized_rows)
    with destination.open("x", encoding="utf-8") as output:
        count = 0
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    if count != shape.materialized_rows:
        raise RuntimeError(f"OpenThoughts3 yielded {count} rows; expected {shape.materialized_rows}")
    return count


def resolved_config(config_path: Path, work_root: Path, shape: StageShape) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["datasets"][0]["path"] = str(work_root / "openthoughts3-tinker-order.jsonl")
    config["dataset_prepared_path"] = str(work_root / "prepared")
    config["output_dir"] = str(work_root / "output" / "peft")
    config["sequence_len"] = shape.sequence_length
    config["gradient_accumulation_steps"] = shape.global_batch_size // WORLD_SIZE
    config["max_steps"] = shape.steps
    return config


def adapter_inventory(output_dir: Path) -> list[dict[str, str | int]]:
    config_path = output_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("r") != 128 or config.get("lora_alpha") != 1:
        raise RuntimeError("The final adapter does not retain rank=128 and alpha=1")
    paths = [config_path, *sorted(output_dir.glob("adapter_model*.safetensors"))]
    if len(paths) == 1:
        raise RuntimeError("Training completed without adapter safetensors")
    return [{"path": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)} for path in paths]


def sync_output(output_root: Path, output_uri: str) -> None:
    if not output_uri.startswith("s3://") or not output_uri.removeprefix("s3://").strip("/"):
        raise ValueError("--output-uri must be a non-root s3:// prefix")
    subprocess.run(["aws", "s3", "sync", str(output_root), output_uri.rstrip("/"), "--only-show-errors"], check=True)


@contextmanager
def periodic_output_sync(output_root: Path, output_uri: str, interval: int = SYNC_INTERVAL):
    stop = threading.Event()

    def upload_until_stopped() -> None:
        while not stop.wait(interval):
            sync_output(output_root, output_uri)

    sync_output(output_root, output_uri)
    uploader = threading.Thread(target=upload_until_stopped, daemon=True, name="tinker-sft-output-sync")
    uploader.start()
    try:
        yield
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_output(output_root, output_uri)


def run(stage: Stage, config_path: Path, work_root: Path, source_commit: str, output_uri: str) -> int:
    if work_root.exists():
        raise ValueError(f"Work root already exists: {work_root}")
    shape = STAGES[stage]
    work_root.mkdir(parents=True)
    output_root = work_root / "output"
    output_root.mkdir()
    runtime = runtime_inventory(source_commit)
    revisions = validate_revisions()
    dataset_path = work_root / "openthoughts3-tinker-order.jsonl"
    rows = materialize_dataset(dataset_path, shape)
    config = resolved_config(config_path, work_root, shape)
    resolved_path = output_root / "resolved-axolotl-config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = (
        "axolotl",
        "train",
        str(resolved_path),
        "--launcher",
        "torchrun",
        "--",
        f"--nproc_per_node={WORLD_SIZE}",
        "--nnodes=1",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "started",
        "stage": stage,
        "shape": asdict(shape),
        "source_commit": source_commit,
        "runtime": runtime,
        "revisions": revisions,
        "dataset": {"rows": rows, "seed": 0, "sha256": file_sha256(dataset_path)},
        "config_sha256": file_sha256(resolved_path),
        "command": command,
        "known_deviations": KNOWN_DEVIATIONS,
    }
    manifest_path = output_root / "tinker-sft-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with periodic_output_sync(output_root, output_uri):
        try:
            result = subprocess.run(command, check=False)
            manifest["returncode"] = result.returncode
            manifest["status"] = "complete" if result.returncode == 0 else "failed"
            if result.returncode == 0:
                manifest["adapter_files"] = adapter_inventory(Path(config["output_dir"]))
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failure"] = str(error)
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(Stage), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args(argv)
    return run(Stage(args.stage), args.config, args.work_root, args.source_commit, args.output_uri)


if __name__ == "__main__":
    sys.exit(main())
