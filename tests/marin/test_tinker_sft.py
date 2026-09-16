import json
import subprocess
from pathlib import Path

import pytest

from scripts.marin_experiments import tinker_sft

RECIPE = Path(__file__).parents[2] / "examples" / "marin" / "tinker-openthoughts3-sft.yaml"


def test_full_stage_resolves_the_published_training_contract(tmp_path: Path) -> None:
    config = tinker_sft.resolved_config(RECIPE, tmp_path, tinker_sft.STAGES[tinker_sft.Stage.FULL])

    assert config["base_model"] == "Qwen/Qwen3.5-9B-Base"
    assert config["revision_of_model"] == "68c46c4b3498877f3ef123c856ecfde50c39f404"
    assert config["sequence_len"] == 16_384
    assert config["max_steps"] == 3_000
    assert config["micro_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 16
    assert config["excess_length_strategy"] == "truncate"
    assert config["save_steps"] == 2
    assert (config["lora_r"], config["lora_alpha"]) == (128, 1)
    assert config["learning_rate"] == 1e-3
    assert (config["adam_beta1"], config["adam_beta2"], config["adam_epsilon"]) == (0.9, 0.95, 1e-8)
    assert config["datasets"][0]["roles_to_train"] == ["assistant"]
    assert config["datasets"][0]["train_on_eos"] == "turn"


def test_materialization_preserves_the_external_stream_order(tmp_path: Path, monkeypatch) -> None:
    class FakeStream:
        def shuffle(self, *, seed: int, buffer_size: int):
            assert (seed, buffer_size) == (0, 4)
            return self

        def take(self, rows: int):
            assert rows == 3
            return ({"conversations": [{"from": "human", "value": str(index)}]} for index in (2, 0, 1))

    monkeypatch.setattr(tinker_sft, "load_dataset", lambda *args, **kwargs: FakeStream())
    destination = tmp_path / "data.jsonl"

    count = tinker_sft.materialize_dataset(destination, tinker_sft.StageShape(1, 32, 1, 4, 3))

    assert count == 3
    assert [json.loads(line)["conversations"][0]["value"] for line in destination.read_text().splitlines()] == [
        "2",
        "0",
        "1",
    ]


def test_dataset_inventory_rejects_an_incomplete_prepared_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "data.jsonl"
    destination.write_text('{"row":1}\n{"row":2}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="2 rows; expected 3"):
        tinker_sft.staged_dataset_inventory(destination, expected_rows=3)


def test_adapter_inventory_rejects_wrong_lora_shape(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 1}), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="rank=128"):
        tinker_sft.adapter_inventory(tmp_path, expected_rank=128, expected_alpha=1)


def test_periodic_output_sync_survives_transient_upload_failure(tmp_path: Path, monkeypatch) -> None:
    class StopAfterTwoAttempts:
        waits = 0

        def wait(self, _interval: int) -> bool:
            self.waits += 1
            return self.waits > 2

    attempts = 0

    def sync_output(_output_root: Path, _output_uri: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.CalledProcessError(1, ["aws", "s3", "sync"])

    monkeypatch.setattr(tinker_sft, "sync_output", sync_output)

    tinker_sft._upload_until_stopped(tmp_path, "s3://bucket/prefix", StopAfterTwoAttempts(), interval=1)

    assert attempts == 2


def test_periodic_output_sync_preserves_final_artifacts_when_publication_fails(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "output"
    adapter_dir = output_root / "peft"
    adapter_dir.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "adapter_files": [
            {"path": "adapter_config.json", "size": 2, "sha256": "unused"},
            {"path": "adapter_model.safetensors", "size": 7, "sha256": "unused"},
        ],
    }
    manifest_text = json.dumps(manifest, sort_keys=True) + "\n"
    (output_root / "tinker-sft-manifest.json").write_text(manifest_text, encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    checkpoint = adapter_dir / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"large transient state")
    iris_output_dir = tmp_path / "iris-output"
    iris_output_dir.mkdir()
    attempts = 0

    def sync_output(_output_root: Path, _output_uri: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise subprocess.CalledProcessError(1, ["aws", "s3", "sync"])

    monkeypatch.setattr(tinker_sft, "sync_output", sync_output)

    with pytest.raises(tinker_sft.ArtifactPublicationError):
        with tinker_sft.periodic_output_sync(
            output_root,
            "s3://bucket/prefix",
            iris_output_dir=iris_output_dir,
        ):
            pass

    fallback = iris_output_dir / "tinker-sft-output"
    assert (fallback / "tinker-sft-manifest.json").read_text(encoding="utf-8") == manifest_text
    assert (fallback / "peft" / "adapter_config.json").read_text(encoding="utf-8") == "{}"
    assert (fallback / "peft" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert not (fallback / "peft" / "checkpoint-2").exists()
