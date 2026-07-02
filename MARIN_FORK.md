# marin axolotl fork

Lab fork of [`axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl).

- **Fork point:** upstream `0bda5a13e4d52ceec58104f44fabb7bd314f9c02` (transformers pin `==5.12.1`).
- **Feature branch:** `feuer/marin-fork-3feature-port`.

## Ported features (each default-off; no behavior change to upstream when its key is unset)

1. **delphi chat template** (`chat_template: delphi`) — a named jinja asset
   (`src/axolotl/utils/chat_templates/templates/delphi.jinja`) implementing the delphi
   think/tool token protocol (`<|start_think|>`/`<|tool_call|>`/`<|tool_result|>` …).
2. **Save-time template-integrity guard** — a plugin (`integrations/template_integrity/`)
   + the per-YAML `tokenizer_save_jinja_files: false` that guarantee the saved checkpoint
   presents its chat_template in `tokenizer_config.json` (not `chat_template.jinja`-only),
   which the serving stack reads — fixing the 0%-SWE-bench "tokenizer restoration" footgun.
3. **MFU logging** (`mfu: true`) — a plugin (`integrations/mfu/`) computing achieved +
   theoretical MFU from `trainer.state.total_flos` at end-of-train.
4. **Supabase model-registration** (`supabase_register: true`) — a plugin
   (`integrations/supabase_registry/`) that, on rank-0 at train-end with a hub push
   configured, registers one model row in the lab's Supabase registry. Creds by env-var
   name only (`SUPABASE_URL` / `SUPABASE_ANON_KEY` / `SUPABASE_SERVICE_ROLE_KEY`); default OFF.

## CPU gates

`scripts/marin_fork_gates/` — transformers-only reproductions of axolotl's save path +
chat-template resolution, so the flag-off byte-identical + footgun invariants are provable
without the CUDA stack (which does not install on arm64 Mac). The 1-GPU training smokes are
deferred to a cluster.
