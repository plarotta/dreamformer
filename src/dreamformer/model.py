from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DreamFormerConfig
from .memory import EpisodicMemory, SemanticMemory
from .replay import PrioritizedReplayBuffer


@dataclass(slots=True)
class DreamFormerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    memory_stats: dict[str, float]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(h)
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)
        return x


class DreamFormerModel(nn.Module):
    """Minimal memory-augmented autoregressive transformer."""

    def __init__(self, config: DreamFormerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.memory_key_proj = nn.Linear(config.d_model, config.memory_key_dim)
        self.memory_write_proj = nn.Linear(config.d_model, config.memory_value_dim)
        self.memory_read_proj = nn.Linear(config.memory_value_dim, config.d_model)
        self.memory_gate = nn.Linear(config.d_model, 1)
        self.consolidation_gate = nn.Linear(config.memory_key_dim + config.memory_value_dim, 1)

        self.stm = EpisodicMemory(
            num_slots=config.num_stm_slots,
            key_dim=config.memory_key_dim,
            value_dim=config.memory_value_dim,
        )
        self.ltm = SemanticMemory(
            key_dim=config.memory_key_dim,
            value_dim=config.memory_value_dim,
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            alpha=config.replay_alpha,
            epsilon=config.replay_epsilon,
        )

        self._last_gate_mean = 0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        write_memory: bool = False,
    ) -> DreamFormerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds configured max_seq_len={self.config.max_seq_len}"
            )

        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        x = self.dropout(x)

        attn_mask = self._causal_attn_mask(seq_len, input_ids.device)
        memory_hidden = None
        for idx, block in enumerate(self.blocks):
            x = block(x, attn_mask=attn_mask)
            if idx == self.config.memory_layer_index:
                x = self._inject_memory(x)
                memory_hidden = x

        if memory_hidden is None:
            memory_hidden = x

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * seq_len, self.config.vocab_size),
                targets.reshape(batch * seq_len),
            )

        if write_memory:
            self._stage_experience(memory_hidden.detach(), logits.detach(), targets)

        return DreamFormerOutput(
            logits=logits,
            loss=loss,
            memory_stats=self.memory_stats(),
        )

    def nrem_consolidation_step(self, batch_size: int = 32, beta: float = 0.4) -> dict[str, float]:
        if not (self.config.enable_nrem and self.config.enable_replay and self.config.enable_ltm):
            return {
                "sampled": 0.0,
                "selected": 0.0,
                "selection_rate": 0.0,
                "ltm_update_mse": 0.0,
            }

        if self.config.replay_strategy == "uniform":
            sample = self.replay_buffer.sample_uniform(batch_size=batch_size)
        else:
            sample = self.replay_buffer.sample(batch_size=batch_size, beta=beta)
        if sample is None:
            return {
                "sampled": 0.0,
                "selected": 0.0,
                "selection_rate": 0.0,
                "ltm_update_mse": 0.0,
            }

        device = self.token_embedding.weight.device
        target_dtype = self.consolidation_gate.weight.dtype
        keys = sample.keys.to(device=device, dtype=target_dtype)
        values = sample.values.to(device=device, dtype=target_dtype)

        with torch.no_grad():
            gate_input = torch.cat([keys, values], dim=-1)
            scores = torch.sigmoid(self.consolidation_gate(gate_input)).squeeze(-1)
            selected_mask = scores >= self.config.nrem_threshold
            selected = int(selected_mask.sum().item())

            update_mse = 0.0
            if selected > 0:
                update_mse = self.ltm.update(
                    keys[selected_mask],
                    values[selected_mask],
                    lr=self.config.ltm_update_rate,
                )

            # Lower priorities for high-confidence consolidated entries.
            if self.config.replay_strategy == "prioritized":
                damp = torch.clamp(1.0 - 0.5 * scores.cpu(), min=0.05)
                new_priorities = (sample.priorities * damp + self.config.replay_epsilon).tolist()
                self.replay_buffer.update_priorities(sample.indices, new_priorities)

        return {
            "sampled": float(len(sample.indices)),
            "selected": float(selected),
            "selection_rate": float(selected / max(1, len(sample.indices))),
            "ltm_update_mse": float(update_mse),
        }

    def clear_memories(self) -> None:
        self.stm.reset()
        self.ltm.reset()

    def memory_stats(self) -> dict[str, float]:
        return {
            "stm_occupancy": self.stm.occupancy,
            "stm_live_slots": float(self.stm.live_slots),
            "ltm_norm_mean": float(self.ltm.normalizer.mean().item()),
            "ltm_norm_max": float(self.ltm.normalizer.max().item()),
            "replay_size": float(len(self.replay_buffer)),
            "replay_total_priority": float(self.replay_buffer.total_priority),
            "memory_gate_mean": float(self._last_gate_mean),
        }

    def _inject_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        key_query = self.memory_key_proj(hidden).reshape(batch * seq_len, -1)
        if self.config.enable_stm:
            stm_values, _ = self.stm.read(
                key_query,
                top_k=self.config.stm_top_k,
            )
        else:
            stm_values = torch.zeros(
                batch * seq_len,
                self.config.memory_value_dim,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        if self.config.enable_ltm:
            ltm_values = self.ltm.read(key_query)
        else:
            ltm_values = torch.zeros_like(stm_values)

        if self.config.enable_ltm and not self.config.enable_stm:
            gate = torch.ones(batch * seq_len, 1, device=hidden.device, dtype=hidden.dtype)
        elif self.config.enable_stm and not self.config.enable_ltm:
            gate = torch.zeros(batch * seq_len, 1, device=hidden.device, dtype=hidden.dtype)
        elif self.config.fixed_memory_gate is not None:
            gate = torch.full(
                (batch * seq_len, 1),
                float(self.config.fixed_memory_gate),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            gate = torch.sigmoid(self.memory_gate(hidden)).reshape(batch * seq_len, 1)
        mixed = gate * ltm_values + (1.0 - gate) * stm_values
        self._last_gate_mean = float(gate.mean().detach().cpu().item())

        injected = self.memory_read_proj(mixed).reshape(batch, seq_len, -1)
        return hidden + injected

    def _stage_experience(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor | None,
    ) -> None:
        if not (self.config.enable_stm or self.config.enable_replay):
            return

        keys = self.memory_key_proj(hidden[:, -1, :])
        values = self.memory_write_proj(hidden[:, -1, :])
        if self.config.enable_stm:
            slots = self.stm.write(
                keys=keys,
                values=values,
                update_threshold=self.config.stm_update_threshold,
                usage_decay=self.config.stm_usage_decay,
            )
        else:
            slots = torch.full((hidden.shape[0],), -1, device=hidden.device, dtype=torch.long)

        if not self.config.enable_replay:
            return

        if targets is not None:
            per_sample_priority = F.cross_entropy(
                logits[:, -1, :],
                targets[:, -1],
                reduction="none",
            ).detach()
        else:
            per_sample_priority = values.norm(dim=-1).detach()

        for i in range(hidden.shape[0]):
            slot = int(slots[i].item())
            self.replay_buffer.add(
                key=keys[i],
                value=values[i],
                priority=float(per_sample_priority[i].cpu().item()),
                metadata={
                    "stm_slot": slot,
                    "access_count": int(self.stm.access_count[slot].item()) if slot >= 0 else 0,
                },
            )

    @staticmethod
    def _causal_attn_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)
