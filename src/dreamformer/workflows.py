from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from .config import DreamFormerConfig
from .experiments import apply_variant, make_model_config, resolve_device
from .model import DreamFormerModel
from .tasks import CharCorpusSampler, TaskBatch, generate_needle_batch, generate_passkey_batch
from .trainer import BatchFn, Trainer, TrainingConfig


SUPPORTED_TASKS = ("passkey", "needle", "char_lm")


def make_training_config(overrides: dict[str, Any] | None = None) -> TrainingConfig:
    base = TrainingConfig()
    if overrides is None:
        return base
    data = asdict(base)
    data.update(overrides)
    return TrainingConfig(**data)


def make_task_fn(task: str, corpus_path: str | None = None) -> BatchFn:
    if task == "passkey":
        return generate_passkey_batch
    if task == "needle":
        return generate_needle_batch
    if task == "char_lm":
        if corpus_path is None:
            raise ValueError("task 'char_lm' requires 'corpus_path'")
        sampler = CharCorpusSampler.from_text_file(corpus_path)

        def _sample(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> TaskBatch:
            return sampler.sample_batch(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, device=device)

        return _sample
    raise ValueError(f"unsupported task '{task}'. options={SUPPORTED_TASKS}")


def run_training_job(spec: dict[str, Any]) -> dict[str, Any]:
    run_name = str(spec.get("run_name", "run"))
    variant = str(spec.get("variant", "full_prioritized"))
    task_name = str(spec.get("task", "passkey"))
    eval_task_name = str(spec.get("eval_task", task_name))
    output_dir = Path(spec.get("output_dir", "artifacts/runs")) / run_name
    seed = int(spec.get("seed", 42))
    device = resolve_device(str(spec.get("device", "auto")))
    resume_checkpoint = spec.get("resume_checkpoint")
    model_overrides = spec.get("model_overrides")
    train_overrides = spec.get("train_overrides")
    corpus_path = spec.get("corpus_path")
    eval_corpus_path = spec.get("eval_corpus_path", corpus_path)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = make_model_config(model_overrides)
    model_config = apply_variant(model_config, variant)
    training_config = make_training_config(train_overrides)

    train_fn = make_task_fn(task_name, corpus_path=corpus_path)
    eval_fn = make_task_fn(eval_task_name, corpus_path=eval_corpus_path) if eval_task_name else None

    model = DreamFormerModel(model_config)
    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        device=device,
        output_dir=output_dir,
    )

    if resume_checkpoint:
        trainer.load_checkpoint(resume_checkpoint)

    resolved = {
        "run_name": run_name,
        "variant": variant,
        "task": task_name,
        "eval_task": eval_task_name,
        "output_dir": str(output_dir),
        "seed": seed,
        "device": str(device),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "corpus_path": corpus_path,
        "eval_corpus_path": eval_corpus_path,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = trainer.train(train_batch_fn=train_fn, eval_batch_fn=eval_fn, run_name=run_name)
    summary["resolved_config"] = str(output_dir / "resolved_config.json")
    return summary
