# Implementation Notes

This document captures what is implemented, what is intentionally deferred, and bug fixes made during development.

## Implemented components

- Core architecture:
  - `STM` (`EpisodicMemory`) with slot allocation and sparse top-k retrieval.
  - `LTM` (`SemanticMemory`) with delta-rule updates.
  - Replay buffer with prioritized and uniform sampling.
  - Memory-gated transformer with wake writes and NREM consolidation.
- Training infrastructure:
  - `Trainer` with:
    - gradient clipping
    - optional CUDA AMP
    - checkpointing and resume
    - periodic eval and metrics logging
  - JSONL metrics logger and summary output.
- Workflows:
  - single-run pipeline (`run_training_job`)
  - ablation sweep script
  - continual-learning script
  - metrics aggregation report script

## Intentional deferrals

- REM latent replay is not in the trainer loop yet.
- Distributed data parallel and multi-node orchestration are not included.
- External experiment trackers (for example W&B) are not integrated yet.
- Learned reducible-loss replay scoring is not implemented; replay uses priority and uniform modes.

## Bugs fixed during this development cycle

1. In-place autograd versioning issue during training:
   - Cause: memory buffers were mutated between forward and backward while read tensors still referenced those buffers.
   - Fix: memory reads now use detached snapshots of memory tensors.

2. Replay test flakiness:
   - Cause: duplicated indices from probabilistic sampling made strict post-update assertions unstable.
   - Fix: unit test now validates priority changes directly at tree leaves.

3. Dataclass copy bug in benchmark script:
   - Cause: using `__dict__` on dataclass with `slots=True`.
   - Fix: use `dataclasses.asdict`.

4. Mixed-precision dtype mismatch in `STM.read`:
   - Cause: under autocast, `weights` and `sparse_weights` could end up in different dtypes before `scatter_`.
   - Fix: explicit fp32 softmax weights and fp32 scatter destination in `EpisodicMemory.read`.

5. Mixed-precision dtype mismatch in NREM consolidation gate:
   - Cause: replay entries could be fp16 while consolidation gate weights are fp32, causing `linear` matmul dtype mismatch.
   - Fix: cast replay `keys`/`values` to the consolidation gate dtype before computing NREM scores.

6. NaN propagation into `STM.access_count` metadata:
   - Cause: low-magnitude or degenerate vectors under real-compute settings could introduce non-finite values during memory normalization and similarity computation.
   - Fix: normalize in fp32 with explicit `eps`, scrub non-finite deltas with `nan_to_num`, and sanitize metadata extraction in `_stage_experience`.

## Readiness checklist

- `uv run pytest` passes before each run.
- Smoke training script completes.
- Phase A benchmark script runs and writes metrics.
- Config-driven runs write:
  - resolved config
  - metrics JSONL
  - summary JSON
  - checkpoints
