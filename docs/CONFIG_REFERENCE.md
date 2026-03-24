# Config Reference

All main scripts use JSON configs.

## `run_experiment.py` config

Required keys:

- `run_name`: unique run identifier.
- `variant`: one of `baseline`, `stm`, `ltm_only`, `full_uniform`, `full_prioritized`, `nrem_off`.
- `task`: one of `passkey`, `needle`, `char_lm`.

Optional keys:

- `eval_task`: defaults to `task`.
- `seed`: default `42`.
- `device`: `auto`, `cuda`, `mps`, or `cpu` (default `auto`).
- `output_dir`: root output directory (default `artifacts/runs`).
- `resume_checkpoint`: path to checkpoint for resume.
- `corpus_path`: required when `task=char_lm`.
- `eval_corpus_path`: optional eval corpus for `char_lm`.
- `model_overrides`: keys from `DreamFormerConfig`.
- `train_overrides`: keys from `TrainingConfig`.

## `run_ablation_sweep.py` config

Uses the same base config as `run_experiment.py` and overrides `variant` per sweep run.

## `run_continual_eval.py` config

Base keys:

- `run_name`
- `variant`
- `seed`
- `device`
- `output_dir`
- `model_overrides`
- `train_overrides`
- `phases`: ordered list of phase objects.

Each phase object:

- `task`: `passkey`, `needle`, or `char_lm`.
- `steps`: number of steps to train in this phase.
- `corpus_path`: optional (required for `char_lm`).
- `eval_corpus_path`: optional.

## `DreamFormerConfig` override fields

- Transformer:
  - `vocab_size`
  - `max_seq_len`
  - `d_model`
  - `n_heads`
  - `n_layers`
  - `ffn_dim`
  - `dropout`
- Memory:
  - `memory_layer_index`
  - `num_stm_slots`
  - `memory_key_dim`
  - `memory_value_dim`
  - `stm_top_k`
  - `stm_update_threshold`
  - `stm_usage_decay`
- Replay and consolidation:
  - `replay_capacity`
  - `replay_alpha`
  - `replay_epsilon`
  - `ltm_update_rate`
  - `nrem_threshold`
- Ablation controls:
  - `enable_stm`
  - `enable_ltm`
  - `enable_replay`
  - `enable_nrem`
  - `replay_strategy`
  - `fixed_memory_gate`
  - `memory_gate_init`
  - `memory_gate_target`
  - `memory_gate_band`
  - `memory_gate_regularization_weight`
  - `normalize_memory_reads`
  - `memory_read_norm_eps`
  - `stm_fusion_scale_init`
  - `ltm_fusion_scale_init`

## `TrainingConfig` override fields

- `steps`
- `batch_size`
- `seq_len`
- `learning_rate`
- `weight_decay`
- `grad_clip_norm`
- `log_every`
- `eval_every`
- `checkpoint_every`
- `nrem_every`
- `replay_beta_start`
- `replay_beta_end`
- `eval_batches`
- `amp`
- `compile_model`
