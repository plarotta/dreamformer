# DreamFormer Runbook

This runbook covers how to run the project from local smoke checks to real-compute experiments.

## 1. Environment setup

```bash
uv sync
uv run pytest
```

If tests fail, do not launch long runs.

## 2. Config-driven runs

All production runs are JSON-config driven.

Core scripts:

- `scripts/run_experiment.py`: single experiment run.
- `scripts/run_ablation_sweep.py`: multiple variants from one base config.
- `scripts/run_continual_eval.py`: sequential continual-learning phases.
- `scripts/report_results.py`: aggregate one or more `*.jsonl` metrics files.

Reference configs:

- `configs/phase_a_smoke.json`
- `configs/phase_a_real_compute.json`
- `configs/phase_a_needle_real_compute.json`
- `configs/char_lm_real_compute.json`
- `configs/continual_real_compute.json`

## 3. Local smoke run

```bash
uv run python scripts/run_experiment.py --config configs/phase_a_smoke.json
```

Expected outputs:

- `artifacts/runs/<run_name>/resolved_config.json`
- `artifacts/runs/<run_name>/<run_name>_metrics.jsonl`
- `artifacts/runs/<run_name>/<run_name>_summary.json`
- checkpoints (`*_checkpoint_last.pt` and periodic checkpoints)

Expected console logs during training:

- `experiment_resolved ...`
- `run_start ...`
- periodic `train step=...` lines with loss, replay size, STM occupancy, gate value, NREM selections, steps/sec, and ETA
- periodic `eval step=...` lines
- `checkpoint_saved ...`
- `best_checkpoint_updated ...`
- `run_complete ...`

## 4. Real compute single run

Example:

```bash
uv run python scripts/run_experiment.py --config configs/phase_a_real_compute.json
```

This config is now the safer first-stage real-compute curriculum (`passkey`, shorter context, lower LR, softer consolidation).

Harder retrieval follow-up:

```bash
uv run python scripts/run_experiment.py --config configs/phase_a_needle_real_compute.json
```

For text corpus LM:

1. Put training corpus at `data/train.txt`
2. Put validation corpus at `data/valid.txt`
3. Run:

```bash
uv run python scripts/run_experiment.py --config configs/char_lm_real_compute.json
```

## 5. Resume from checkpoint

Set `resume_checkpoint` in the config JSON to the path of `*_checkpoint_last.pt` (or best checkpoint), then run `run_experiment.py` again with that updated config.

The trainer restores:

- model weights
- optimizer state
- global step
- random states (Python, NumPy, Torch, and CUDA when available)

## 6. Ablation sweep

```bash
uv run python scripts/run_ablation_sweep.py \
  --config configs/phase_a_smoke.json \
  --variants baseline stm full_uniform full_prioritized nrem_off
```

Sweep summary is written to `artifacts/ablation_summary.json` by default.

## 7. Continual learning run

```bash
uv run python scripts/run_continual_eval.py --config configs/continual_real_compute.json
```

This writes per-phase evaluation snapshots and a continual summary JSON with forgetting statistics.

## 8. Reporting

Aggregate metrics across runs:

```bash
uv run python scripts/report_results.py \
  --inputs artifacts/runs/*/*_metrics.jsonl \
  --out artifacts/report_summary.json
```

## 9. Operational checks before long jobs

- Verify device selection in config (`cuda`, `mps`, `cpu`, or `auto`).
- Confirm `vocab_size` is high enough for selected task. `char_lm` uses character IDs offset by `+3`.
- Confirm `seq_len` does not exceed `model_overrides.max_seq_len`.
- Confirm checkpoint cadence (`checkpoint_every`) is frequent enough for your cluster preemption policy.
- Start with a short run (for example 100 to 500 steps) and inspect metrics for:
  - loss moving down
  - `memory_gate_mean` not pinned at exactly 0 or 1 too early
  - non-zero NREM selection when replay/NREM are enabled

## 10. Known failure patterns and mitigations

- Device mismatch:
  - Symptom: immediate crash when moving model to device.
  - Fix: use `device=auto` or a valid available backend.
- Corpus vocab mismatch:
  - Symptom: `vocab_size ... too small for corpus token max`.
  - Fix: increase `model_overrides.vocab_size`.
- Divergence/instability:
  - Symptom: exploding loss or NaNs.
  - Fix: lower `learning_rate`, tighten `grad_clip_norm`, increase logging frequency, and inspect checkpoints.
