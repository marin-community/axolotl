"""Convert a committed Tinker SFT checkpoint into a stock-Qwen3.5 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from fsspec import AbstractFileSystem
from s3fs import S3FileSystem

from axolotl.integrations.qwen35_split_qkv.adapter import fuse_split_qkv_adapter
from scripts.marin_experiments.tinker_sft import (
    WORLD_SIZE,
    validate_s3_location,
    validate_source_commit,
)
from scripts.marin_experiments.tinker_sft_checkpoints import (
    COMMIT_FILENAME,
    REQUIRED_FILES,
)

CHECKPOINT_PATTERN = re.compile(r"checkpoint-([1-9][0-9]*)")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CONVERSION_MANIFEST = "conversion-manifest.json"


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def convert_committed_checkpoint(
    checkpoint_uri: str,
    output_uri: str,
    source_commit: str,
    filesystem: AbstractFileSystem | None = None,
) -> dict:
    """Verify a remote SFT checkpoint, fuse its adapter, and commit the converted output."""
    validate_s3_location(checkpoint_uri, "--checkpoint-uri")
    validate_s3_location(output_uri, "--output-uri")
    source = checkpoint_uri.removeprefix("s3://").rstrip("/")
    target = output_uri.removeprefix("s3://").rstrip("/")
    match = CHECKPOINT_PATTERN.fullmatch(source.rsplit("/", 1)[-1])
    if not match or target == source or target.startswith(f"{source}/") or source.startswith(f"{target}/"):
        raise ValueError("Source must be a checkpoint-N directory and output must be distinct")
    step = int(match.group(1))
    filesystem = filesystem or S3FileSystem()
    commit_bytes = filesystem.cat_file(f"{source}/{COMMIT_FILENAME}")
    commit = json.loads(commit_bytes)
    inventory = commit.get("files")
    if commit.get("schema_version") != 1 or commit.get("step") != step or not isinstance(inventory, dict):
        raise ValueError(f"Invalid SFT checkpoint commit record: {checkpoint_uri}")
    required = REQUIRED_FILES | set(ADAPTER_FILES) | {f"rng_state_{rank}.pth" for rank in range(WORLD_SIZE)}
    for name in required:
        size = inventory.get(name)
        if not isinstance(size, int) or size <= 0 or filesystem.info(f"{source}/{name}")["size"] != size:
            raise ValueError(f"Incomplete committed SFT checkpoint file: {checkpoint_uri}/{name}")

    manifest_path = f"{target}/{CONVERSION_MANIFEST}"
    source_commit_sha256 = hashlib.sha256(commit_bytes).hexdigest()
    if filesystem.exists(manifest_path):
        manifest = json.loads(filesystem.cat_file(manifest_path))
        if (
            manifest.get("source_commit_sha256") != source_commit_sha256
            or manifest.get("converter_revision") != source_commit
        ):
            raise ValueError(f"Converted adapter already exists for a different source: {output_uri}")
        for name, artifact in manifest["outputs"].items():
            if filesystem.info(f"{target}/{name}")["size"] != artifact["size"]:
                raise ValueError(f"Committed converted adapter has changed: {output_uri}/{name}")
        return manifest

    with tempfile.TemporaryDirectory(prefix="tinker-sft-adapter-") as temporary:
        root = Path(temporary)
        raw = root / "raw"
        raw.mkdir()
        fused = root / "fused"
        source_hashes = {}
        for name in ADAPTER_FILES:
            local = raw / name
            filesystem.get_file(f"{source}/{name}", str(local))
            if local.stat().st_size != inventory[name]:
                raise ValueError(f"Downloaded SFT adapter file has changed: {checkpoint_uri}/{name}")
            source_hashes[name] = file_sha256(local)
        fuse_split_qkv_adapter(raw, fused)
        outputs = {}
        for name in ADAPTER_FILES:
            local = fused / name
            outputs[name] = {"sha256": file_sha256(local), "size": local.stat().st_size}
            filesystem.put_file(str(local), f"{target}/{name}")
            if filesystem.info(f"{target}/{name}")["size"] != local.stat().st_size:
                raise IOError(f"Converted adapter upload size mismatch: {output_uri}/{name}")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "step": step,
            "checkpoint_uri": checkpoint_uri,
            "source_commit_sha256": source_commit_sha256,
            "source_sha256": source_hashes,
            "converter_revision": source_commit,
            "output_uri": output_uri,
            "outputs": outputs,
        }
        filesystem.pipe_file(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    validate_source_commit(args.source_commit)
    manifest = convert_committed_checkpoint(args.checkpoint_uri, args.output_uri, args.source_commit)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
