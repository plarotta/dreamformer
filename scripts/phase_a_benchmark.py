from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from dreamformer import DreamFormerConfig, DreamFormerModel, ExperimentLogger


PAD = 0
KEY_MARK = 1
QUERY_MARK = 2


@dataclass(slots=True)
class TaskBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor
    answers: torch.Tensor


def build_targets(input_ids: torch.Tensor) -> torch.Tensor:
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = PAD
    return targets


def generate_passkey_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> TaskBatch:
    if seq_len < 8:
        raise ValueError("passkey sequence length must be >= 8")
    tokens = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    query_pos = seq_len - 2
    answer_pos = seq_len - 1

    keys = torch.randint(3, vocab_size, (batch_size,), device=device)
    tokens[:, 0] = KEY_MARK
    tokens[:, 1] = keys
    tokens[:, query_pos] = QUERY_MARK
    tokens[:, answer_pos] = keys

    return TaskBatch(
        input_ids=tokens,
        targets=build_targets(tokens),
        query_positions=torch.full((batch_size,), query_pos, device=device, dtype=torch.long),
        answers=keys,
    )


def generate_needle_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> TaskBatch:
    if seq_len < 12:
        raise ValueError("needle sequence length must be >= 12")
    tokens = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    query_pos = seq_len - 2
    answer_pos = seq_len - 1

    answers = torch.randint(3, vocab_size, (batch_size,), device=device)
    needle_pos = torch.randint(1, seq_len - 4, (batch_size,), device=device)
    for i in range(batch_size):
        tokens[i, needle_pos[i]] = answers[i]
    tokens[:, query_pos] = QUERY_MARK
    tokens[:, answer_pos] = answers

    return TaskBatch(
        input_ids=tokens,
        targets=build_targets(tokens),
        query_positions=torch.full((batch_size,), query_pos, device=device, dtype=torch.long),
        answers=answers,
    )


def query_accuracy(logits: torch.Tensor, query_positions: torch.Tensor, answers: torch.Tensor) -> float:
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    query_logits = logits[batch_indices, query_positions]
    pred = torch.argmax(query_logits, dim=-1)
    return float((pred == answers).float().mean().item())


def make_variant_config(
    base: DreamFormerConfig,
    variant: str,
) -> DreamFormerConfig:
    cfg = DreamFormerConfig(**asdict(base))
    if variant == "baseline":
        cfg.enable_stm = False
        cfg.enable_ltm = False
        cfg.enable_replay = False
        cfg.enable_nrem = False
    elif variant == "stm":
        cfg.enable_stm = True
        cfg.enable_ltm = False
        cfg.enable_replay = False
        cfg.enable_nrem = False
    elif variant == "full_uniform":
        cfg.enable_stm = True
        cfg.enable_ltm = True
        cfg.enable_replay = True
        cfg.enable_nrem = True
        cfg.replay_strategy = "uniform"
    elif variant == "full_prioritized":
        cfg.enable_stm = True
        cfg.enable_ltm = True
        cfg.enable_replay = True
        cfg.enable_nrem = True
        cfg.replay_strategy = "prioritized"
    else:
        raise ValueError(f"unknown variant: {variant}")
    return cfg


def train_and_eval_variant(
    variant: str,
    task_name: str,
    batch_fn: Callable[[int, int, int, torch.device], TaskBatch],
    steps: int,
    eval_steps: int,
    batch_size: int,
    seq_len: int,
    base_config: DreamFormerConfig,
    device: torch.device,
    logger: ExperimentLogger,
) -> dict[str, float]:
    config = make_variant_config(base_config, variant)
    model = DreamFormerModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1, steps + 1):
        model.train()
        batch = batch_fn(batch_size, seq_len, config.vocab_size, device)
        out = model(batch.input_ids, targets=batch.targets, write_memory=True)
        assert out.loss is not None

        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        if config.enable_nrem and step % 5 == 0:
            nrem = model.nrem_consolidation_step(batch_size=batch_size, beta=0.4)
        else:
            nrem = {"sampled": 0.0, "selected": 0.0, "selection_rate": 0.0, "ltm_update_mse": 0.0}

        if step % 10 == 0:
            train_acc = query_accuracy(out.logits.detach(), batch.query_positions, batch.answers)
            metrics = {
                "loss": float(out.loss.item()),
                "query_acc": train_acc,
                **out.memory_stats,
                **{f"nrem_{k}": float(v) for k, v in nrem.items()},
            }
            logger.log(
                step=step,
                split="train",
                metrics=metrics,
                variant=variant,
                task=task_name,
            )

    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(eval_steps):
            batch = batch_fn(batch_size, seq_len, config.vocab_size, device)
            out = model(batch.input_ids, targets=batch.targets, write_memory=False)
            assert out.loss is not None
            losses.append(float(out.loss.item()))
            accs.append(query_accuracy(out.logits, batch.query_positions, batch.answers))

    summary = {
        "eval_loss": float(sum(losses) / len(losses)),
        "eval_query_acc": float(sum(accs) / len(accs)),
    }
    logger.log(step=steps, split="eval", metrics=summary, variant=variant, task=task_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A benchmark runner for DreamFormer")
    parser.add_argument("--task", choices=["passkey", "needle", "both"], default="both")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "stm", "full_uniform", "full_prioritized"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase_a_metrics.jsonl"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    base_config = DreamFormerConfig(
        vocab_size=128,
        max_seq_len=max(64, args.seq_len),
        d_model=64,
        n_heads=4,
        n_layers=3,
        ffn_dim=128,
        memory_layer_index=2,
        num_stm_slots=64,
        memory_key_dim=32,
        memory_value_dim=32,
        replay_capacity=2048,
    )

    tasks: list[tuple[str, Callable[[int, int, int, torch.device], TaskBatch]]]
    if args.task == "passkey":
        tasks = [("passkey", generate_passkey_batch)]
    elif args.task == "needle":
        tasks = [("needle", generate_needle_batch)]
    else:
        tasks = [("passkey", generate_passkey_batch), ("needle", generate_needle_batch)]

    logger = ExperimentLogger()
    print("variant,task,eval_loss,eval_query_acc")
    for task_name, task_fn in tasks:
        for variant in args.variants:
            summary = train_and_eval_variant(
                variant=variant,
                task_name=task_name,
                batch_fn=task_fn,
                steps=args.steps,
                eval_steps=args.eval_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                base_config=base_config,
                device=device,
                logger=logger,
            )
            print(
                f"{variant},{task_name},{summary['eval_loss']:.4f},{summary['eval_query_acc']:.4f}"
            )

    logger.dump_jsonl(args.out)
    print(f"wrote_metrics={args.out}")


if __name__ == "__main__":
    main()
