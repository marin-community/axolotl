"""Run the pinned OpenThoughts3 SFT control reported by the Tinker cookbook."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_REPOSITORY = "open-thoughts/OpenThoughts3-1.2M"
DATASET_REVISION = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
DATASET_FILENAME = "openthoughts3-tinker-order.jsonl"
WORLD_SIZE = 8
FULL_STEPS = 3_000
FULL_BATCH_SIZE = 128
FULL_SEQUENCE_LENGTH = 16_384
FULL_SHUFFLE_BUFFER = FULL_STEPS * FULL_BATCH_SIZE
SYNC_INTERVAL = 300
AWS_MAX_ATTEMPTS = "20"
AWS_RETRY_MODE = "adaptive"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
LOGGER = logging.getLogger(__name__)
KNOWN_DEVIATIONS = (
    "Axolotl and Tinker use different distributed data loaders and training kernels.",
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


def validate_revisions(config: dict[str, Any]) -> dict[str, str]:
    model_repository = config["base_model"]
    model_revision = config["revision_of_model"]
    api = HfApi()
    model = api.model_info(model_repository, revision=model_revision)
    dataset = api.dataset_info(DATASET_REPOSITORY, revision=DATASET_REVISION)
    if model.sha != model_revision or dataset.sha != DATASET_REVISION:
        raise ValueError("Hugging Face did not resolve the pinned model and dataset revisions")
    return {"model": model.sha, "dataset": dataset.sha}


def validate_runtime(source_commit: str) -> dict[str, object]:
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


def validate_source_commit(source_commit: str) -> None:
    baked_commit = os.environ.get("AXOLOTL_SOURCE_COMMIT")
    if not COMMIT_PATTERN.fullmatch(source_commit) or baked_commit != source_commit:
        raise ValueError("--source-commit must match the commit baked into the task image")


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


def dataset_inventory(path: Path, rows: int) -> dict[str, str | int]:
    return {"rows": rows, "seed": 0, "sha256": file_sha256(path)}


def staged_dataset_inventory(path: Path, expected_rows: int) -> dict[str, str | int]:
    with path.open(encoding="utf-8") as source:
        rows = sum(1 for _ in source)
    if rows != expected_rows:
        raise RuntimeError(f"Prepared dataset contains {rows} rows; expected {expected_rows}")
    return dataset_inventory(path, rows)


def resolved_config(config_path: Path, work_root: Path, shape: StageShape) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["datasets"][0]["path"] = str(work_root / DATASET_FILENAME)
    config["dataset_prepared_path"] = str(work_root / "prepared")
    config["output_dir"] = str(work_root / "output" / "peft")
    config["sequence_len"] = shape.sequence_length
    config["gradient_accumulation_steps"] = shape.global_batch_size // WORLD_SIZE
    config["max_steps"] = shape.steps
    return config


def adapter_inventory(output_dir: Path, expected_rank: int, expected_alpha: int | float) -> list[dict[str, str | int]]:
    config_path = output_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("r") != expected_rank or config.get("lora_alpha") != expected_alpha:
        raise RuntimeError(f"The final adapter does not retain rank={expected_rank} and alpha={expected_alpha}")
    paths = [config_path, *sorted(output_dir.glob("adapter_model*.safetensors"))]
    if len(paths) == 1:
        raise RuntimeError("Training completed without adapter safetensors")
    return [{"path": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)} for path in paths]


def configure_aws_environment(work_root: Path) -> dict[str, str]:
    aws_config = work_root / "aws-config"
    inherited_config = Path(os.environ.get("AWS_CONFIG_FILE", Path.home() / ".aws" / "config"))
    if inherited_config.is_file() and inherited_config != aws_config:
        shutil.copyfile(inherited_config, aws_config)
    environment = os.environ | {
        "AWS_CONFIG_FILE": str(aws_config),
        "AWS_MAX_ATTEMPTS": AWS_MAX_ATTEMPTS,
        "AWS_RETRY_MODE": AWS_RETRY_MODE,
    }
    subprocess.run(
        ["aws", "configure", "set", "default.s3.addressing_style", "virtual"],
        check=True,
        env=environment,
    )
    return environment


def validate_s3_location(location: str, option: str) -> None:
    bucket, separator, key = location.removeprefix("s3://").partition("/")
    if not location.startswith("s3://") or not bucket or not separator or not key.strip("/"):
        raise ValueError(f"{option} must identify a non-root s3:// location")


def sync_output(output_root: Path, output_uri: str) -> None:
    environment = configure_aws_environment(output_root.parent)
    subprocess.run(
        ["aws", "s3", "sync", str(output_root), output_uri.rstrip("/"), "--only-show-errors"],
        check=True,
        env=environment,
    )


def stage_dataset(dataset_uri: str, destination: Path) -> None:
    subprocess.run(
        ["aws", "s3", "cp", dataset_uri, str(destination), "--only-show-errors"],
        check=True,
        env=configure_aws_environment(destination.parent),
    )


def create_output_root(work_root: Path) -> Path:
    if work_root.exists():
        raise ValueError(f"Work root already exists: {work_root}")
    work_root.mkdir(parents=True)
    output_root = work_root / "output"
    output_root.mkdir()
    return output_root


def prepare_dataset(stage: Stage, work_root: Path, source_commit: str, output_uri: str) -> None:
    validate_source_commit(source_commit)
    validate_s3_location(output_uri, "--output-uri")
    output_root = create_output_root(work_root)
    dataset_path = output_root / DATASET_FILENAME
    shape = STAGES[stage]
    rows = materialize_dataset(dataset_path, shape)
    inventory = dataset_inventory(dataset_path, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "stage": stage,
        "source_commit": source_commit,
        "dataset": inventory,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
    }
    (output_root / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sync_output(output_root, output_uri)


def _upload_until_stopped(output_root: Path, output_uri: str, stop: threading.Event, interval: int) -> None:
    while not stop.wait(interval):
        try:
            sync_output(output_root, output_uri)
        except subprocess.CalledProcessError:
            LOGGER.exception("Periodic S3 output sync failed; retrying in %s seconds", interval)


@contextmanager
def periodic_output_sync(output_root: Path, output_uri: str, interval: int = SYNC_INTERVAL):
    stop = threading.Event()
    sync_output(output_root, output_uri)
    uploader = threading.Thread(
        target=_upload_until_stopped,
        args=(output_root, output_uri, stop, interval),
        daemon=True,
        name="tinker-sft-output-sync",
    )
    uploader.start()
    try:
        yield
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_output(output_root, output_uri)


def run(
    stage: Stage,
    config_path: Path,
    work_root: Path,
    source_commit: str,
    output_uri: str,
    dataset_uri: str | None = None,
) -> int:
    validate_source_commit(source_commit)
    validate_s3_location(output_uri, "--output-uri")
    if dataset_uri is not None:
        validate_s3_location(dataset_uri, "--dataset-uri")
    shape = STAGES[stage]
    output_root = create_output_root(work_root)
    runtime = validate_runtime(source_commit)
    dataset_path = work_root / DATASET_FILENAME
    if dataset_uri is None:
        rows = materialize_dataset(dataset_path, shape)
        dataset = dataset_inventory(dataset_path, rows)
    else:
        stage_dataset(dataset_uri, dataset_path)
        dataset = staged_dataset_inventory(dataset_path, shape.materialized_rows)
    config = resolved_config(config_path, work_root, shape)
    revisions = validate_revisions(config)
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
        "dataset": dataset | {"uri": dataset_uri},
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
                manifest["adapter_files"] = adapter_inventory(
                    Path(config["output_dir"]), config["lora_r"], config["lora_alpha"]
                )
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failure"] = str(error)
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Materialize and publish the pinned dataset")
    train_parser = subparsers.add_parser("train", help="Run the SFT control")
    for command_parser in (prepare_parser, train_parser):
        command_parser.add_argument("--stage", choices=tuple(Stage), required=True)
        command_parser.add_argument("--work-root", type=Path, required=True)
        command_parser.add_argument("--source-commit", required=True)
        command_parser.add_argument("--output-uri", required=True)
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--dataset-uri")
    args = parser.parse_args(argv)
    stage = Stage(args.stage)
    if args.command == "prepare":
        prepare_dataset(stage, args.work_root, args.source_commit, args.output_uri)
        return 0
    return run(stage, args.config, args.work_root, args.source_commit, args.output_uri, args.dataset_uri)


if __name__ == "__main__":
    sys.exit(main())
