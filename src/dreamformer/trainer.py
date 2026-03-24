from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch

from .config import DreamFormerConfig
from .metrics import ExperimentLogger
from .model import DreamFormerModel
from .tasks import TaskBatch, query_accuracy


BatchFn = Callable[[int, int, int, torch.device], TaskBatch]


@dataclass(slots=True)
class TrainingConfig:
    steps: int = 200
    batch_size: int = 32
    seq_len: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    log_every: int = 10
    eval_every: int = 50
    checkpoint_every: int = 100
    nrem_every: int = 5
    replay_beta_start: float = 0.4
    replay_beta_end: float = 1.0
    eval_batches: int = 20
    amp: bool = False
    compile_model: bool = False
    console_log: bool = True

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be > 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.log_every <= 0 or self.eval_every <= 0 or self.checkpoint_every <= 0:
            raise ValueError("log/eval/checkpoint intervals must be positive")
        if self.nrem_every <= 0:
            raise ValueError("nrem_every must be positive")
        if not 0.0 <= self.replay_beta_start <= 1.0:
            raise ValueError("replay_beta_start must be in [0, 1]")
        if not 0.0 <= self.replay_beta_end <= 1.0:
            raise ValueError("replay_beta_end must be in [0, 1]")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")


class Trainer:
    def __init__(
        self,
        model: DreamFormerModel,
        model_config: DreamFormerConfig,
        training_config: TrainingConfig,
        device: torch.device,
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or ExperimentLogger()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.step = 0
        self.best_eval_loss = float("inf")
        self.best_checkpoint_path: Path | None = None

        use_amp = training_config.amp and device.type == "cuda"
        self._use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self._train_start_time: float | None = None

        if training_config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_eval_loss": self.best_eval_loss,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_state"] = torch.cuda.get_rng_state_all()
        torch.save(state, checkpoint_path)
        self._emit(f"checkpoint_saved step={self.step} path={checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = int(checkpoint["step"])
        self.best_eval_loss = float(checkpoint.get("best_eval_loss", float("inf")))
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.random.set_rng_state(checkpoint["torch_state"])
        if torch.cuda.is_available() and "torch_cuda_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_state"])
        self._emit(
            "checkpoint_loaded "
            f"step={self.step} best_eval_loss={self.best_eval_loss:.6f} path={path}"
        )

    def train(
        self,
        train_batch_fn: BatchFn,
        eval_batch_fn: BatchFn | None = None,
        run_name: str = "run",
    ) -> dict[str, Any]:
        cfg = self.training_config
        final_eval: dict[str, float] | None = None
        self._train_start_time = time.monotonic()
        self._emit(
            "run_start "
            f"run_name={run_name} device={self.device} steps={cfg.steps} "
            f"batch_size={cfg.batch_size} seq_len={cfg.seq_len} "
            f"amp={self._use_amp} compile={cfg.compile_model}"
        )

        while self.step < cfg.steps:
            self.step += 1
            self.model.train()
            batch = train_batch_fn(cfg.batch_size, cfg.seq_len, self.model_config.vocab_size, self.device)
            autocast_ctx = (
                torch.autocast(device_type=self.device.type, enabled=True)
                if self._use_amp
                else nullcontext()
            )
            with autocast_ctx:
                out = self.model(batch.input_ids, targets=batch.targets, write_memory=True)
                if out.loss is None:
                    raise RuntimeError("model returned None loss while training")
                loss = out.loss

            self.optimizer.zero_grad(set_to_none=True)
            if self._use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                self.optimizer.step()

            beta = self._replay_beta(self.step)
            if self.step % cfg.nrem_every == 0:
                nrem = self.model.nrem_consolidation_step(batch_size=cfg.batch_size, beta=beta)
            else:
                nrem = {"sampled": 0.0, "selected": 0.0, "selection_rate": 0.0, "ltm_update_mse": 0.0}

            if self.step % cfg.log_every == 0:
                metrics = {
                    "loss": float(loss.item()),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "replay_beta": float(beta),
                    **out.memory_stats,
                    **{f"nrem_{k}": float(v) for k, v in nrem.items()},
                }
                qa = query_accuracy(batch, out.logits.detach())
                if qa is not None:
                    metrics["query_acc"] = qa
                self.logger.log(step=self.step, split="train", metrics=metrics, run=run_name)
                self._emit_train_progress(metrics)

            if eval_batch_fn is not None and self.step % cfg.eval_every == 0:
                final_eval = self.evaluate(eval_batch_fn, num_batches=cfg.eval_batches)
                self.logger.log(step=self.step, split="eval", metrics=final_eval, run=run_name)
                self._emit_eval_progress(final_eval)
                if final_eval["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = final_eval["eval_loss"]
                    self.best_checkpoint_path = self.save_checkpoint(
                        self.output_dir / f"{run_name}_checkpoint_best.pt"
                    )
                    self._emit(
                        "best_checkpoint_updated "
                        f"step={self.step} eval_loss={final_eval['eval_loss']:.6f} "
                        f"path={self.best_checkpoint_path}"
                    )

            if self.step % cfg.checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / f"{run_name}_checkpoint_step{self.step}.pt")

        last_checkpoint = self.save_checkpoint(self.output_dir / f"{run_name}_checkpoint_last.pt")

        if eval_batch_fn is not None and final_eval is None:
            final_eval = self.evaluate(eval_batch_fn, num_batches=cfg.eval_batches)
            self.logger.log(step=self.step, split="eval", metrics=final_eval, run=run_name)

        summary: dict[str, Any] = {
            "run_name": run_name,
            "final_step": self.step,
            "best_eval_loss": self.best_eval_loss,
            "last_checkpoint": str(last_checkpoint),
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
        }
        if final_eval is not None:
            summary.update(final_eval)

        summary_path = self.output_dir / f"{run_name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        metrics_path = self.output_dir / f"{run_name}_metrics.jsonl"
        self.logger.dump_jsonl(metrics_path)
        self._emit(
            "run_complete "
            f"run_name={run_name} final_step={self.step} "
            f"summary={summary_path} metrics={metrics_path} "
            f"best_checkpoint={self.best_checkpoint_path}"
        )
        return summary

    def evaluate(self, batch_fn: BatchFn, num_batches: int) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        query_accs: list[float] = []
        with torch.no_grad():
            for _ in range(num_batches):
                batch = batch_fn(
                    self.training_config.batch_size,
                    self.training_config.seq_len,
                    self.model_config.vocab_size,
                    self.device,
                )
                out = self.model(batch.input_ids, targets=batch.targets, write_memory=False)
                if out.loss is None:
                    raise RuntimeError("model returned None loss during evaluation")
                losses.append(float(out.loss.item()))
                qa = query_accuracy(batch, out.logits)
                if qa is not None:
                    query_accs.append(qa)

        metrics = {"eval_loss": float(sum(losses) / len(losses))}
        if query_accs:
            metrics["eval_query_acc"] = float(sum(query_accs) / len(query_accs))
        metrics.update(self.model.memory_stats())
        return metrics

    def _replay_beta(self, step: int) -> float:
        cfg = self.training_config
        if cfg.steps <= 1:
            return cfg.replay_beta_end
        t = (step - 1) / (cfg.steps - 1)
        return cfg.replay_beta_start + t * (cfg.replay_beta_end - cfg.replay_beta_start)

    def _emit(self, message: str) -> None:
        if self.training_config.console_log:
            print(message, flush=True)

    def _emit_train_progress(self, metrics: dict[str, float]) -> None:
        elapsed = 0.0
        steps_per_sec = 0.0
        eta_seconds = 0.0
        if self._train_start_time is not None:
            elapsed = max(0.0, time.monotonic() - self._train_start_time)
            steps_per_sec = self.step / elapsed if elapsed > 0 else 0.0
            remaining_steps = max(0, self.training_config.steps - self.step)
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0

        message = (
            "train "
            f"step={self.step}/{self.training_config.steps} "
            f"progress={100.0 * self.step / self.training_config.steps:.1f}% "
            f"loss={metrics['loss']:.6f} "
            f"lr={metrics['lr']:.2e} "
            f"replay_size={int(metrics['replay_size'])} "
            f"stm_live={int(metrics['stm_live_slots'])} "
            f"gate={metrics['memory_gate_mean']:.4f} "
            f"nrem_selected={metrics['nrem_selected']:.0f} "
            f"steps_per_sec={steps_per_sec:.2f} "
            f"elapsed_s={elapsed:.1f} "
            f"eta_s={eta_seconds:.1f}"
        )
        if "query_acc" in metrics:
            message += f" query_acc={metrics['query_acc']:.4f}"
        self._emit(message)

    def _emit_eval_progress(self, metrics: dict[str, float]) -> None:
        message = (
            "eval "
            f"step={self.step}/{self.training_config.steps} "
            f"eval_loss={metrics['eval_loss']:.6f} "
            f"replay_size={int(metrics['replay_size'])} "
            f"stm_live={int(metrics['stm_live_slots'])} "
            f"gate={metrics['memory_gate_mean']:.4f}"
        )
        if "eval_query_acc" in metrics:
            message += f" eval_query_acc={metrics['eval_query_acc']:.4f}"
        self._emit(message)
