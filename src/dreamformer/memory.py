from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """Slot-based high-fidelity short-term memory."""

    def __init__(self, num_slots: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.register_buffer("keys", torch.zeros(num_slots, key_dim))
        self.register_buffer("values", torch.zeros(num_slots, value_dim))
        self.register_buffer("usage", torch.zeros(num_slots))
        self.register_buffer("alive", torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer("access_count", torch.zeros(num_slots))

    @property
    def occupancy(self) -> float:
        return float(self.alive.float().mean().item())

    @property
    def live_slots(self) -> int:
        return int(self.alive.sum().item())

    def clear(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        with torch.no_grad():
            self.keys[indices] = 0.0
            self.values[indices] = 0.0
            self.usage[indices] = 0.0
            self.access_count[indices] = 0.0
            self.alive[indices] = False

    def reset(self) -> None:
        with torch.no_grad():
            self.keys.zero_()
            self.values.zero_()
            self.usage.zero_()
            self.access_count.zero_()
            self.alive.zero_()

    def read(
        self, query: torch.Tensor, top_k: int = 4, temperature: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.size(-1) != self.key_dim:
            raise ValueError(f"query dim {query.size(-1)} != key_dim {self.key_dim}")

        batch = query.shape[0]
        if self.live_slots == 0:
            values = torch.zeros(batch, self.value_dim, device=query.device, dtype=query.dtype)
            weights = torch.zeros(batch, self.num_slots, device=query.device, dtype=query.dtype)
            return values, weights

        keys = F.normalize(self.keys.detach().to(query.device).clone(), dim=-1)
        query = F.normalize(query, dim=-1)

        similarity = query @ keys.t()
        similarity[:, ~self.alive.to(query.device)] = -1e9

        top_k = max(1, min(top_k, self.num_slots))
        if top_k < self.num_slots:
            top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)
            sparse_weights = torch.softmax(top_values * temperature, dim=-1)
            weights = torch.zeros_like(similarity)
            weights.scatter_(1, top_indices, sparse_weights)
        else:
            weights = torch.softmax(similarity * temperature, dim=-1)

        memory_values = self.values.detach().to(query.device).clone()
        values = weights @ memory_values

        with torch.no_grad():
            usage_delta = weights.detach().mean(dim=0).to(self.usage.device)
            self.usage += usage_delta * 0.05
            self.access_count += usage_delta

        return values, weights

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        update_threshold: float = 0.9,
        usage_decay: float = 0.995,
    ) -> torch.Tensor:
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values batch sizes must match")
        if keys.shape[1] != self.key_dim or values.shape[1] != self.value_dim:
            raise ValueError("keys/values shape mismatch with memory dimensions")

        with torch.no_grad():
            self.usage *= usage_decay
            keys = F.normalize(keys.to(self.keys.device), dim=-1)
            values = values.to(self.values.device)

            selected = []
            live_mask = self.alive
            for key, value in zip(keys, values):
                if int(live_mask.sum().item()) > 0:
                    live_indices = torch.where(live_mask)[0]
                    live_keys = self.keys[live_indices]
                    similarity = F.cosine_similarity(live_keys, key.unsqueeze(0), dim=-1)
                    best_value, best_idx = torch.max(similarity, dim=0)
                    if float(best_value.item()) >= update_threshold:
                        slot = int(live_indices[int(best_idx.item())].item())
                    else:
                        slot = self._allocate_slot()
                else:
                    slot = self._allocate_slot()

                self.keys[slot] = key
                self.values[slot] = value
                self.usage[slot] = 1.0
                self.alive[slot] = True
                self.access_count[slot] += 1.0
                selected.append(slot)

            return torch.tensor(selected, device=self.keys.device, dtype=torch.long)

    def _allocate_slot(self) -> int:
        dead = torch.where(~self.alive)[0]
        if dead.numel() > 0:
            return int(dead[0].item())
        return int(torch.argmin(self.usage).item())


class SemanticMemory(nn.Module):
    """Compressed long-term associative memory using a delta-rule update."""

    def __init__(self, key_dim: int, value_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.eps = eps
        self.register_buffer("matrix", torch.zeros(key_dim, value_dim))
        self.register_buffer("normalizer", torch.zeros(key_dim))

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def reset(self) -> None:
        with torch.no_grad():
            self.matrix.zero_()
            self.normalizer.zero_()

    def read(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.size(-1) != self.key_dim:
            raise ValueError(f"query dim {query.size(-1)} != key_dim {self.key_dim}")

        q = self._phi(query)
        normalizer = self.normalizer.detach().to(query.device).clone()
        matrix = self.matrix.detach().to(query.device).clone()
        denom = (q @ normalizer).unsqueeze(-1).clamp_min(self.eps)
        out = (q @ matrix) / denom
        return out

    def update(self, keys: torch.Tensor, values: torch.Tensor, lr: float = 1.0) -> float:
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values batch sizes must match")
        if keys.shape[1] != self.key_dim or values.shape[1] != self.value_dim:
            raise ValueError("keys/values shape mismatch with memory dimensions")

        with torch.no_grad():
            keys = keys.to(self.matrix.device)
            values = values.to(self.matrix.device)
            k = self._phi(keys)
            current = self.read(keys)
            delta = values - current
            scale = lr / max(1, keys.shape[0])
            self.matrix += scale * (k.t() @ delta)
            self.normalizer += scale * k.sum(dim=0)
            mse = float((delta.pow(2).mean()).item())
            return mse
