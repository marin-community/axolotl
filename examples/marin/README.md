# Marin experiment controls

`tinker-openthoughts3-sft.yaml` reproduces the public input and optimizer contract for the rank-128 SFT checkpoint used
by Tinker's reasoning-distillation example. The reported calibration result is approximately 65% on AIME 2024 after
3,000 steps.

The runner pins `Qwen/Qwen3.5-9B-Base` and `open-thoughts/OpenThoughts3-1.2M` to immutable Hugging Face revisions. It
uses the cookbook's seed-0 streaming shuffle with a 384,000-row buffer and consumes 384,000 rows: 3,000 global batches
of 128, truncated to 16,384 tokens. The rank-128 LoRA trains assistant turns and the turn-ending token with AdamW at
`1e-3`, linear decay, no warmup, betas `0.9/0.95`, epsilon `1e-8`, and no weight decay. A per-device microbatch of one
makes Axolotl's token-mean loss a per-example mean before gradient accumulation, matching the Tinker cookbook's datum
normalization.

Campaign runs save every two optimizer steps. The external MarinSkyRL
evaluator consumes each durable checkpoint on AIME 2024; evaluation is not part
of the Axolotl training process.

Qwen3.5 stores each linear-attention Q/K/V projection in one fused base tensor, but Tinker trains three independent
rank-128 adapters. The configured plugin exposes those projections independently during training. Before stock
Transformers or vLLM serving, convert a checkpoint to an exactly equivalent fused rank-384 adapter:

```bash
python -m axolotl.integrations.qwen35_split_qkv.adapter \
  /path/to/checkpoint \
  /path/to/fused-checkpoint
```

The converter uses block-diagonal output factors and preserves the original LoRA scaling. The external evaluator
performs this conversion for each durable checkpoint without modifying the training artifact.

Run the plumbing stage on one eight-H100 node:

```bash
python -m scripts.marin_experiments.tinker_sft train \
  --stage plumbing \
  --config examples/marin/tinker-openthoughts3-sft.yaml \
  --work-root /tmp/tinker-sft-plumbing \
  --source-commit "$AXOLOTL_SOURCE_COMMIT" \
  --output-uri s3://bucket/path/tinker-sft/plumbing
```

`plumbing` runs one 2,048-token global batch of eight. Before the full-shape stages, materialize the published shuffle
once on a CPU task and publish it to the training cluster's S3 region:

```bash
python -m scripts.marin_experiments.tinker_sft prepare \
  --stage full \
  --work-root /tmp/tinker-sft-dataset \
  --source-commit "$AXOLOTL_SOURCE_COMMIT" \
  --output-uri s3://bucket/path/tinker-sft/dataset
```

Then pass that exact object to both the one-step and full runs:

```bash
python -m scripts.marin_experiments.tinker_sft train \
  --stage fidelity_step \
  --config examples/marin/tinker-openthoughts3-sft.yaml \
  --work-root /tmp/tinker-sft-fidelity \
  --source-commit "$AXOLOTL_SOURCE_COMMIT" \
  --dataset-uri s3://bucket/path/tinker-sft/dataset/openthoughts3-tinker-order.jsonl \
  --output-uri s3://bucket/path/tinker-sft/fidelity
```

`fidelity_step` consumes one full-shape batch, while `full` consumes all 3,000 steps. Each stage requires a new local
and S3 prefix. Training verifies the prepared artifact's 384,000-row count and records its checksum. The runner uploads
its resolved config, runtime and input provenance, checkpoints, and final adapter hashes every five minutes and at
exit.

This is a cross-runtime reproduction, not a claim of identical optimizer trajectories. Axolotl and Tinker use
different distributed loaders and kernels, and Tinker's LoRA initialization and scaling are not public.
