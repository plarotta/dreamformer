from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


PAD = 0
KEY_MARK = 1
QUERY_MARK = 2
MIN_RANDOM_TOKEN = 3


@dataclass(slots=True)
class TaskBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor | None = None
    answers: torch.Tensor | None = None


def build_next_token_targets(input_ids: torch.Tensor, pad_token: int = PAD) -> torch.Tensor:
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = pad_token
    return targets


def generate_passkey_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> TaskBatch:
    if vocab_size <= MIN_RANDOM_TOKEN:
        raise ValueError("vocab_size must be greater than 3 for passkey batches")
    if seq_len < 8:
        raise ValueError("passkey sequence length must be >= 8")

    tokens = torch.randint(MIN_RANDOM_TOKEN, vocab_size, (batch_size, seq_len), device=device)
    query_pos = seq_len - 2
    answer_pos = seq_len - 1

    keys = torch.randint(MIN_RANDOM_TOKEN, vocab_size, (batch_size,), device=device)
    tokens[:, 0] = KEY_MARK
    tokens[:, 1] = keys
    tokens[:, query_pos] = QUERY_MARK
    tokens[:, answer_pos] = keys

    return TaskBatch(
        input_ids=tokens,
        targets=build_next_token_targets(tokens),
        query_positions=torch.full((batch_size,), query_pos, device=device, dtype=torch.long),
        answers=keys,
    )


def generate_needle_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> TaskBatch:
    if vocab_size <= MIN_RANDOM_TOKEN:
        raise ValueError("vocab_size must be greater than 3 for needle batches")
    if seq_len < 12:
        raise ValueError("needle sequence length must be >= 12")

    tokens = torch.randint(MIN_RANDOM_TOKEN, vocab_size, (batch_size, seq_len), device=device)
    query_pos = seq_len - 2
    answer_pos = seq_len - 1

    answers = torch.randint(MIN_RANDOM_TOKEN, vocab_size, (batch_size,), device=device)
    needle_pos = torch.randint(1, seq_len - 4, (batch_size,), device=device)
    batch_indices = torch.arange(batch_size, device=device)
    tokens[batch_indices, needle_pos] = answers
    tokens[:, query_pos] = QUERY_MARK
    tokens[:, answer_pos] = answers

    return TaskBatch(
        input_ids=tokens,
        targets=build_next_token_targets(tokens),
        query_positions=torch.full((batch_size,), query_pos, device=device, dtype=torch.long),
        answers=answers,
    )


class CharCorpusSampler:
    """Character-level sampler for language modeling experiments."""

    def __init__(self, token_ids: torch.Tensor, pad_token: int = PAD) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor")
        if token_ids.numel() < 4:
            raise ValueError("token_ids must have at least 4 elements")
        self.token_ids = token_ids.long()
        self.pad_token = pad_token

    @classmethod
    def from_text_file(cls, path: str | Path, offset: int = 3) -> "CharCorpusSampler":
        text = Path(path).read_text(encoding="utf-8")
        if not text:
            raise ValueError("text corpus is empty")
        # Offset by reserved markers so synthetic and corpus modes can share vocab space.
        token_ids = torch.tensor([min(255, ord(ch)) + offset for ch in text], dtype=torch.long)
        return cls(token_ids=token_ids)

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
    ) -> TaskBatch:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        max_token = int(self.token_ids.max().item())
        if vocab_size <= max_token:
            raise ValueError(
                f"vocab_size={vocab_size} is too small for corpus token max {max_token}; "
                "increase model vocab size or use a different corpus encoding."
            )
        if self.token_ids.numel() <= seq_len + 1:
            raise ValueError("corpus is too short for requested seq_len")

        max_start = self.token_ids.numel() - (seq_len + 1)
        starts = torch.randint(0, max_start, (batch_size,))
        input_rows = []
        target_rows = []
        for start in starts:
            s = int(start.item())
            window = self.token_ids[s : s + seq_len + 1]
            input_rows.append(window[:-1])
            target_rows.append(window[1:])

        input_ids = torch.stack(input_rows).to(device)
        targets = torch.stack(target_rows).to(device)
        return TaskBatch(input_ids=input_ids, targets=targets)


def query_accuracy(batch: TaskBatch, logits: torch.Tensor) -> float | None:
    if batch.query_positions is None or batch.answers is None:
        return None
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    query_logits = logits[batch_indices, batch.query_positions]
    pred = torch.argmax(query_logits, dim=-1)
    return float((pred == batch.answers).float().mean().item())
