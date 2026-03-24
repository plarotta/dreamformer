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
    key_copies: int = 1,
    query_gap_min: int = 8,
    answer_vocab_size: int | None = None,
    distractor_vocab_size: int | None = None,
) -> TaskBatch:
    if vocab_size <= MIN_RANDOM_TOKEN:
        raise ValueError("vocab_size must be greater than 3 for passkey batches")
    if seq_len < 8:
        raise ValueError("passkey sequence length must be >= 8")
    if key_copies <= 0:
        raise ValueError("key_copies must be positive")
    if query_gap_min < 0:
        raise ValueError("query_gap_min must be >= 0")

    answer_upper = _bounded_vocab_upper(vocab_size=vocab_size, limit=answer_vocab_size)
    distractor_upper = _bounded_vocab_upper(vocab_size=vocab_size, limit=distractor_vocab_size)
    tokens = torch.randint(MIN_RANDOM_TOKEN, distractor_upper, (batch_size, seq_len), device=device)
    query_pos = seq_len - 2
    answer_pos = seq_len - 1

    keys = torch.randint(MIN_RANDOM_TOKEN, answer_upper, (batch_size,), device=device)
    key_positions = _paired_marker_positions(
        query_pos=query_pos,
        copies=key_copies,
        min_query_gap=query_gap_min,
        device=device,
    )
    for pos in key_positions.tolist():
        tokens[:, pos] = KEY_MARK
        tokens[:, pos + 1] = keys
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
    needle_copies: int = 1,
    query_gap_min: int = 8,
    answer_vocab_size: int | None = None,
    distractor_vocab_size: int | None = None,
) -> TaskBatch:
    if vocab_size <= MIN_RANDOM_TOKEN:
        raise ValueError("vocab_size must be greater than 3 for needle batches")
    if seq_len < 12:
        raise ValueError("needle sequence length must be >= 12")
    if needle_copies <= 0:
        raise ValueError("needle_copies must be positive")
    if query_gap_min < 0:
        raise ValueError("query_gap_min must be >= 0")

    answer_upper = _bounded_vocab_upper(vocab_size=vocab_size, limit=answer_vocab_size)
    distractor_upper = _bounded_vocab_upper(vocab_size=vocab_size, limit=distractor_vocab_size)
    tokens = torch.randint(MIN_RANDOM_TOKEN, distractor_upper, (batch_size, seq_len), device=device)
    query_pos = seq_len - 2
    answer_pos = seq_len - 1

    answers = torch.randint(MIN_RANDOM_TOKEN, answer_upper, (batch_size,), device=device)
    candidate_end = query_pos - max(1, query_gap_min)
    candidate_positions = torch.arange(1, candidate_end, device=device)
    if candidate_positions.numel() < needle_copies:
        raise ValueError("sequence too short for requested needle_copies")
    batch_indices = torch.arange(batch_size, device=device)
    sampled_order = torch.rand(batch_size, candidate_positions.numel(), device=device).argsort(dim=1)
    needle_pos = candidate_positions[sampled_order[:, :needle_copies]]
    tokens[batch_indices.unsqueeze(1), needle_pos] = answers.unsqueeze(1)
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


def query_cross_entropy(batch: TaskBatch, logits: torch.Tensor) -> torch.Tensor | None:
    if batch.query_positions is None or batch.answers is None:
        return None
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    query_logits = logits[batch_indices, batch.query_positions]
    return torch.nn.functional.cross_entropy(query_logits, batch.answers)


def _bounded_vocab_upper(vocab_size: int, limit: int | None) -> int:
    if limit is None:
        return vocab_size
    upper = min(vocab_size, MIN_RANDOM_TOKEN + int(limit))
    if upper <= MIN_RANDOM_TOKEN:
        raise ValueError("bounded vocab limit is too small")
    return upper


def _paired_marker_positions(
    query_pos: int,
    copies: int,
    min_query_gap: int,
    device: torch.device,
) -> torch.Tensor:
    max_start = query_pos - min_query_gap - 2
    if max_start < 0:
        raise ValueError("sequence too short for requested query_gap_min")
    max_copies = max(1, (max_start + 2) // 2)
    copies = min(copies, max_copies)
    if copies == 1:
        return torch.tensor([0], device=device, dtype=torch.long)
    raw = torch.linspace(0, max_start, steps=copies, device=device)
    positions = []
    last = -2
    for value in raw.tolist():
        pos = min(int(round(value)), max_start)
        if pos <= last + 1:
            pos = last + 2
        if pos > max_start:
            pos = max_start
        positions.append(pos)
        last = pos
    return torch.tensor(positions, device=device, dtype=torch.long)
