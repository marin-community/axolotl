import json
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
