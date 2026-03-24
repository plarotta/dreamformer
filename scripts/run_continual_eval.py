from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from dreamformer import DreamFormerModel, ExperimentLogger
from dreamformer.experiments import apply_variant, make_model_config, resolve_device
from dreamformer.trainer import Trainer, TrainingConfig
from dreamformer.workflows import make_task_fn, make_training_config


def _phase_train_config(base: TrainingConfig, extra_steps: int, current_step: int) -> TrainingConfig:
    data = asdict(base)
    data["steps"] = current_step + extra_steps
    return TrainingConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential continual-learning phases.")
    parser.add_argument("--config", type=Path, required=True, help="Path to continual JSON config.")
    args = parser.parse_args()

    spec = json.loads(args.config.read_text(encoding="utf-8"))
    run_name = str(spec.get("run_name", "continual"))
    output_root = Path(spec.get("output_dir", "artifacts/runs")) / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(spec.get("seed", 42))
    device = resolve_device(str(spec.get("device", "auto")))
    phases: list[dict[str, Any]] = list(spec.get("phases", []))
    if not phases:
        raise ValueError("continual config requires non-empty 'phases'")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_cfg = make_model_config(spec.get("model_overrides"))
    model_cfg = apply_variant(model_cfg, str(spec.get("variant", "full_prioritized")))
    base_train_cfg = make_training_config(spec.get("train_overrides"))

    model = DreamFormerModel(model_cfg)
    logger = ExperimentLogger()
    trainer = Trainer(
        model=model,
        model_config=model_cfg,
        training_config=base_train_cfg,
        device=device,
        output_dir=output_root,
        logger=logger,
    )

    seen_tasks: dict[str, Any] = {}
    best_query_acc: dict[str, float] = {}
    phase_rows: list[dict[str, Any]] = []

    for phase_idx, phase in enumerate(phases, start=1):
        task_name = str(phase["task"])
        phase_steps = int(phase.get("steps", base_train_cfg.steps))
        corpus_path = phase.get("corpus_path", spec.get("corpus_path"))
        eval_corpus_path = phase.get("eval_corpus_path", corpus_path)

        phase_cfg = _phase_train_config(base_train_cfg, extra_steps=phase_steps, current_step=trainer.step)
        trainer.training_config = phase_cfg
        train_fn = make_task_fn(task_name, corpus_path=corpus_path)
        trainer.train(train_batch_fn=train_fn, eval_batch_fn=None, run_name=f"{run_name}_phase{phase_idx}")

        seen_tasks[task_name] = {
            "eval_fn": make_task_fn(task_name, corpus_path=eval_corpus_path),
            "corpus_path": eval_corpus_path,
        }

        eval_result: dict[str, dict[str, float]] = {}
        for seen_task, task_state in seen_tasks.items():
            metrics = trainer.evaluate(task_state["eval_fn"], num_batches=phase_cfg.eval_batches)
            eval_result[seen_task] = metrics
            qa = metrics.get("eval_query_acc")
            if qa is not None:
                best_query_acc[seen_task] = max(best_query_acc.get(seen_task, qa), qa)

        forgetting_terms = []
        for seen_task, metrics in eval_result.items():
            qa = metrics.get("eval_query_acc")
            if qa is None:
                continue
            forgetting_terms.append(max(0.0, best_query_acc[seen_task] - qa))
        forgetting = float(sum(forgetting_terms) / len(forgetting_terms)) if forgetting_terms else 0.0

        row = {
            "phase": phase_idx,
            "train_task": task_name,
            "total_step": trainer.step,
            "forgetting_query_acc": forgetting,
            "eval": eval_result,
        }
        phase_rows.append(row)
        print(
            f"phase={phase_idx} task={task_name} step={trainer.step} "
            f"forgetting_query_acc={forgetting:.4f}"
        )

    summary = {
        "run_name": run_name,
        "phases": phase_rows,
        "output_dir": str(output_root),
        "device": str(device),
    }
    summary_path = output_root / f"{run_name}_continual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.dump_jsonl(output_root / f"{run_name}_continual_metrics.jsonl")
    print(f"wrote_summary={summary_path}")


if __name__ == "__main__":
    main()
