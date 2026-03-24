# DreamFormer Experimental Validation Plan

This plan is designed to answer a narrow question:

> Does a dual-memory transformer with offline consolidation outperform simpler same-budget alternatives on recall, retention, and replay efficiency?

If the answer is "not clearly," the architecture should be reduced or abandoned.

## Primary hypotheses

### H1. Episodic memory helps exact long-gap retrieval

A model with `STM` should outperform a same-size transformer on tasks where a sparse fact must be recovered after a long distractor span.

### H2. Offline semantic consolidation improves retention

A model with `STM + LTM + NREM consolidation` should forget less than a same-budget baseline during sequential training.

### H3. Prioritized replay beats uniform replay at fixed replay budget

If replay is useful, then selective replay should outperform uniform replay when the number of replayed samples is held constant.

### H4. REM-style latent replay is optional, not assumed

`REM` should be treated as a high-risk extension. It only stays if it improves compositional generalization or robustness without harming core language modeling and retention.

## Model variants to compare

Every experiment should include parameter-matched or compute-matched baselines.

1. `Baseline Transformer`
2. `Transformer + larger context window`
3. `Transformer + STM`
4. `Transformer + LTM with offline updates only`
5. `Transformer + STM + LTM without prioritized replay`
6. `Full DreamFormer: STM + LTM + prioritized replay + NREM`
7. `Full DreamFormer + REM`

If compute is tight, keep `7` optional until `6` proves value.

## Evaluation ladder

Do not jump directly to full language modeling. Validate the mechanism stack from simplest to hardest.

### Phase A: component-level synthetic tasks

Purpose:

- Verify the memory systems work before spending large training budgets.

Tasks:

- Passkey retrieval
- Needle-in-a-haystack retrieval
- Associative recall
- Copy/reverse or ordered sequence recall
- Multi-step retrieval with distractors

Metrics:

- Retrieval accuracy vs distractor length
- Accuracy vs memory gap
- Slot reuse and collision rate
- Consolidation acceptance rate

Success condition:

- `STM` variants should show a clear gain over the baseline on long-gap exact retrieval.

### Phase B: small-scale sequence modeling

Purpose:

- Check whether the architecture helps real sequence modeling without collapsing throughput or stability.

Tasks:

- Character-level or byte-level long-context modeling
- Small language modeling corpus with long documents
- Topic-shifted sequential corpora

Metrics:

- Validation loss or perplexity
- Tokens/sec and memory overhead
- Attention-to-memory gate usage
- Replay efficiency measured as gain per replayed sample

Success condition:

- The full non-REM system should preserve or improve modeling quality while offering better recall or retention than the baseline.

### Phase C: continual learning

Purpose:

- Test the actual CLS-style promise: reduced forgetting under sequential exposure.

Protocol:

- Train on domains or tasks in sequence, not jointly.
- Evaluate after each phase on all previous phases.

Candidate setups:

- Sequential topic corpora
- Sequential algorithmic tasks
- Sequential retrieval tasks with changing distributions

Metrics:

- Average retained accuracy
- Backward transfer
- Forgetting score
- Area under the retention curve

Success condition:

- `STM + LTM + NREM` should reduce forgetting relative to both the plain transformer and the `STM`-only model.

### Phase D: compositional generalization and robustness

Purpose:

- Decide whether REM-style replay adds anything real.

Tasks:

- Held-out recombinations of seen patterns
- Noisy or partially masked retrieval prompts
- Schema transfer after consolidation

Metrics:

- Out-of-distribution accuracy
- Robustness to corruption or masking
- Regression in core language modeling metrics

Success condition:

- Keep REM only if it improves OOD or robustness with no material regression elsewhere.

## Ablation matrix

The following ablations are mandatory. Without them, the project cannot identify which mechanism matters.

1. `STM` on vs off
2. `LTM` on vs off
3. `LTM` online writes vs sleep-only writes
4. Uniform replay vs prioritized replay
5. Consolidation gate learned vs heuristic threshold
6. NREM only vs NREM + REM
7. Temporal links on vs off in `STM`
8. DNC-style deallocation/masking on vs off

## Metrics to log continuously

Learning metrics:

- Training loss
- Validation loss or perplexity
- Retention metrics after each training phase
- Retrieval accuracy at different context distances

Memory metrics:

- `STM` occupancy
- `STM` overwrite rate
- `STM` retrieval hit rate
- `LTM` norm growth and update magnitude
- Gate distribution between `STM` and `LTM`
- Replay priority histogram
- Consolidation rate

Systems metrics:

- Tokens/sec
- Peak memory usage
- Extra FLOPs per training token
- Replay cost per retained point of accuracy

## Failure modes and stop conditions

Stop or redesign the system if any of the following persist:

- The gate collapses almost entirely to one memory source.
- `STM` usage is high but retrieval gain is negligible.
- `LTM` updates grow without improving retained performance.
- Replay improves retention only by imposing unacceptable throughput cost.
- REM helps only synthetic tasks and harms real validation loss.

## Recommended experimental order

1. Baseline transformer on synthetic recall tasks.
2. Add `STM` and validate exact retrieval.
3. Add `LTM` with offline consolidation only.
4. Compare uniform vs prioritized replay.
5. Run sequential-training retention experiments.
6. Add REM-style latent replay only after the first five steps succeed.

## Minimum evidence needed to justify the architecture

DreamFormer is worth pursuing only if the experiments support all of the following:

- `STM` improves sparse long-gap retrieval.
- `LTM + NREM` reduces forgetting at fixed budget.
- Prioritized replay beats uniform replay at equal replay volume.
- Overall system cost is still defensible against simply using a stronger baseline.

If the architecture fails any of those checks, the next move is simplification, not more mechanisms.

## Expected deliverables

- A benchmark table comparing all variants
- Retention curves across sequential phases
- Retrieval accuracy vs context distance plots
- Replay-efficiency curves
- Gate-usage and memory-occupancy diagnostics
- A short go/no-go memo after each phase
