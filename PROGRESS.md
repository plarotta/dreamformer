# Progress Report

Last updated: 2026-03-24

## 1. Current status

The project is now in a runnable, config-driven state for single-node experiments on real compute.

Completed:

- Core DreamFormer implementation:
  - `STM`, `LTM`, replay buffer, memory-gated transformer, wake + NREM path.
- Training infrastructure:
  - reusable trainer with logging, evaluation cadence, checkpointing, resume support, grad clipping, replay-beta annealing, optional AMP.
- Experiment workflows:
  - single-run script (`run_experiment.py`)
  - ablation sweep script (`run_ablation_sweep.py`)
  - continual-learning script (`run_continual_eval.py`)
  - metrics aggregation script (`report_results.py`)
- Task generators:
  - `passkey`, `needle`, and corpus-backed `char_lm`.
- Reproducibility and docs:
  - JSON configs for smoke and real-compute runs
  - runbook + config reference + implementation notes
  - `.gitignore` for generated artifacts.

## 2. Validation completed

All automated tests pass:

- `uv run pytest` -> `20 passed`

Smoke/integration runs completed successfully:

- `uv run python scripts/run_experiment.py --config configs/phase_a_smoke.json`
- `uv run python scripts/run_ablation_sweep.py --config configs/phase_a_smoke.json ...`
- `uv run python scripts/run_continual_eval.py --config configs/continual_smoke.json`
- `uv run python scripts/report_results.py --inputs ...`
- `uv run python scripts/phase_a_benchmark.py --task both --steps 10 --eval-steps 2 ...`

## 3. Bugs found and fixed

1. Autograd versioning failure from in-place memory mutation during training.
   - Fix: memory reads now use detached memory snapshots.
2. Replay unit test flakiness from duplicate sampled indices.
   - Fix: priority update assertions now validate tree leaf priorities directly.
3. Dataclass clone issue in benchmark code (`slots=True` with `__dict__`).
   - Fix: switched to `dataclasses.asdict`.
4. Checkpoint resume failure with newer PyTorch default `weights_only=True`.
   - Fix: explicit `torch.load(..., weights_only=False)` in trainer checkpoint loader.

## 4. What is still intentionally deferred

- REM latent replay is still not in the core trainer loop.
- Multi-GPU / multi-node distributed training is not implemented yet.
- External experiment tracker integration (for example W&B) is not added yet.

## 5. Memory requirements by experiment

## Assumptions

- Parameter counts are exact from current configs/model.
- GPU memory estimates are for training (forward + backward) on one process.
- For AMP runs, activations are assumed mostly fp16/bf16; optimizer states remain fp32.
- Replay buffer stores tensors on CPU in this codebase (`ReplayEntry` uses `.cpu()` copies).
- Actual usage can vary with backend/kernel implementation and fragmentation.

## Per-config memory budget

| Config | Params | GPU train memory (estimate) | Recommended GPU | Replay raw tensor memory (CPU) | Replay realistic CPU footprint | Checkpoint size (each) |
|---|---:|---:|---:|---:|---:|---:|
| `configs/phase_a_smoke.json` | 127,426 | ~0.2-0.6 GB (CPU default run) | CPU or 4 GB GPU | ~0.25 MB | ~2-10 MB | ~2-4 MB |
| `configs/continual_smoke.json` | 127,426 | ~0.2-0.6 GB (CPU default run) | CPU or 4 GB GPU | ~0.25 MB | ~2-10 MB | ~2-4 MB |
| `configs/phase_a_real_compute.json` | 4,513,282 | ~2.5-4.0 GB | >=8 GB (12 GB preferred) | ~51.2 MB | ~0.15-0.30 GB | ~60-90 MB |
| `configs/continual_real_compute.json` | 4,513,282 | ~2.0-3.5 GB | >=8 GB (12 GB preferred) | ~61.4 MB | ~0.18-0.36 GB | ~60-90 MB |
| `configs/char_lm_real_compute.json` | 14,473,218 | ~5.5-8.5 GB | >=16 GB (24 GB preferred) | ~307.2 MB | ~0.9-1.8 GB | ~180-280 MB |

Notes:

- `run_ablation_sweep.py` runs variants sequentially by default. If you parallelize variants yourself, multiply GPU/CPU needs by concurrency.
- Baseline/STM-only variants keep similar model parameter allocations in this implementation; replay/CPU pressure drops when replay is disabled.
- If you increase `replay_capacity`, host RAM usage grows roughly linearly.

## 6. Practical cluster planning guidance

- For `phase_a_real_compute`: provision `1 x 12 GB GPU` and `>=8 GB host RAM`.
- For `continual_real_compute`: provision `1 x 12 GB GPU` and `>=10 GB host RAM`.
- For `char_lm_real_compute`: provision `1 x 24 GB GPU` and `>=16 GB host RAM`.
- Disk:
  - allocate at least `20-50 GB` for long runs with periodic checkpoints and metrics.

## 7. Command quickstart

```bash
uv sync
uv run pytest
uv run python scripts/run_experiment.py --config configs/phase_a_smoke.json
uv run python scripts/run_ablation_sweep.py --config configs/phase_a_smoke.json
uv run python scripts/run_continual_eval.py --config configs/continual_smoke.json
```

Real-compute entrypoints:

```bash
uv run python scripts/run_experiment.py --config configs/phase_a_real_compute.json
uv run python scripts/run_experiment.py --config configs/char_lm_real_compute.json
uv run python scripts/run_continual_eval.py --config configs/continual_real_compute.json
```
