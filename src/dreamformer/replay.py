from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class ReplayEntry:
    key: torch.Tensor
    value: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplayBatch:
    indices: list[int]
    keys: torch.Tensor
    values: torch.Tensor
    priorities: torch.Tensor
    weights: torch.Tensor
    metadata: list[dict[str, Any]]


class PrioritizedReplayBuffer:
    """Proportional prioritized replay backed by a binary sum tree."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-3) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0]")
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._entries: list[ReplayEntry | None] = [None] * capacity
        self._next = 0
        self._size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        priority: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        raw_priority = self._sanitize_priority(self._max_priority if priority is None else priority)
        scaled = self._to_scaled_priority(raw_priority)
        index = self._next
        self._entries[index] = ReplayEntry(
            key=key.detach().cpu().clone(),
            value=value.detach().cpu().clone(),
            metadata={} if metadata is None else dict(metadata),
        )
        self._set_priority(index, scaled)

        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._max_priority = max(self._max_priority, raw_priority)
        return index

    def sample(self, batch_size: int, beta: float = 0.4) -> ReplayBatch | None:
        if self._size == 0:
            return None
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        beta = float(np.clip(beta, 0.0, 1.0))

        batch_size = min(batch_size, self._size)
        total_priority = float(self._tree[1])
        if not math.isfinite(total_priority) or total_priority <= 0:
            return None

        segment = total_priority / batch_size
        sampled_indices: list[int] = []
        sampled_entries: list[ReplayEntry] = []
        sampled_probs: list[float] = []
        sampled_prios: list[float] = []

        for i in range(batch_size):
            left = segment * i
            right = segment * (i + 1)
            mass = random.uniform(left, right)
            idx = self._find_prefix_index(mass)
            entry = self._entries[idx]
            if entry is None:
                continue
            leaf_priority = self._sanitize_priority(float(self._tree[idx + self.capacity]))
            prob = leaf_priority / total_priority

            sampled_indices.append(idx)
            sampled_entries.append(entry)
            sampled_probs.append(prob)
            sampled_prios.append(leaf_priority)

        if not sampled_entries:
            return None

        probs = np.asarray(sampled_probs, dtype=np.float64)
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max() + 1e-12

        return ReplayBatch(
            indices=sampled_indices,
            keys=torch.stack([entry.key for entry in sampled_entries]),
            values=torch.stack([entry.value for entry in sampled_entries]),
            priorities=torch.tensor(sampled_prios, dtype=torch.float32),
            weights=torch.tensor(weights, dtype=torch.float32),
            metadata=[entry.metadata for entry in sampled_entries],
        )

    def sample_uniform(self, batch_size: int) -> ReplayBatch | None:
        if self._size == 0:
            return None
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batch_size = min(batch_size, self._size)
        candidates = list(range(self._size))
        sampled_indices = random.sample(candidates, k=batch_size)
        sampled_entries: list[ReplayEntry] = []
        sampled_prios: list[float] = []
        for idx in sampled_indices:
            entry = self._entries[idx]
            if entry is None:
                continue
            sampled_entries.append(entry)
            sampled_prios.append(float(self._tree[idx + self.capacity]))

        if not sampled_entries:
            return None

        return ReplayBatch(
            indices=sampled_indices,
            keys=torch.stack([entry.key for entry in sampled_entries]),
            values=torch.stack([entry.value for entry in sampled_entries]),
            priorities=torch.tensor(sampled_prios, dtype=torch.float32),
            weights=torch.ones(len(sampled_entries), dtype=torch.float32),
            metadata=[entry.metadata for entry in sampled_entries],
        )

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have equal lengths")
        for idx, prio in zip(indices, priorities):
            raw = self._sanitize_priority(prio)
            scaled = self._to_scaled_priority(raw)
            self._set_priority(idx, scaled)
            self._max_priority = max(self._max_priority, raw)

    def _to_scaled_priority(self, raw: float) -> float:
        return (raw + self.epsilon) ** self.alpha

    def _sanitize_priority(self, raw: float) -> float:
        value = float(abs(raw))
        if not math.isfinite(value):
            return max(self.epsilon, self._max_priority)
        return max(self.epsilon, value)

    def _set_priority(self, index: int, scaled_priority: float) -> None:
        tree_idx = index + self.capacity
        delta = scaled_priority - self._tree[tree_idx]
        while tree_idx >= 1:
            self._tree[tree_idx] += delta
            tree_idx //= 2

    def _find_prefix_index(self, mass: float) -> int:
        mass = float(np.clip(mass, 0.0, self._tree[1]))
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if mass <= self._tree[left]:
                idx = left
            else:
                mass -= self._tree[left]
                idx = left + 1
        return idx - self.capacity

    @property
    def total_priority(self) -> float:
        return float(self._tree[1])
