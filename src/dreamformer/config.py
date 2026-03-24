from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DreamFormerConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1

    memory_layer_index: int = -1
    num_stm_slots: int = 128
    memory_key_dim: int = 64
    memory_value_dim: int = 64
    stm_top_k: int = 4
    stm_update_threshold: float = 0.9
    stm_usage_decay: float = 0.995

    replay_capacity: int = 4096
    replay_alpha: float = 0.6
    replay_epsilon: float = 1e-3

    ltm_update_rate: float = 1.0
    nrem_threshold: float = 0.5

    enable_stm: bool = True
    enable_ltm: bool = True
    enable_replay: bool = True
    enable_nrem: bool = True
    replay_strategy: str = "prioritized"
    fixed_memory_gate: float | None = None
    memory_gate_init: float = 0.30
    memory_gate_target: float = 0.25
    memory_gate_band: float = 0.10
    memory_gate_regularization_weight: float = 0.02
    normalize_memory_reads: bool = True
    memory_read_norm_eps: float = 1e-6
    stm_fusion_scale_init: float = 0.35
    ltm_fusion_scale_init: float = 1.00

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.memory_key_dim <= 0 or self.memory_value_dim <= 0:
            raise ValueError("memory dims must be positive")
        if self.num_stm_slots <= 0:
            raise ValueError("num_stm_slots must be positive")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if not 0.0 <= self.replay_alpha <= 1.0:
            raise ValueError("replay_alpha must be in [0.0, 1.0]")
        if self.replay_strategy not in {"prioritized", "uniform"}:
            raise ValueError("replay_strategy must be 'prioritized' or 'uniform'")
        if self.fixed_memory_gate is not None and not 0.0 <= self.fixed_memory_gate <= 1.0:
            raise ValueError("fixed_memory_gate must be in [0.0, 1.0]")
        if not 0.0 < self.memory_gate_init < 1.0:
            raise ValueError("memory_gate_init must be in (0.0, 1.0)")
        if not 0.0 <= self.memory_gate_target <= 1.0:
            raise ValueError("memory_gate_target must be in [0.0, 1.0]")
        if not 0.0 <= self.memory_gate_band <= 1.0:
            raise ValueError("memory_gate_band must be in [0.0, 1.0]")
        if self.memory_gate_regularization_weight < 0.0:
            raise ValueError("memory_gate_regularization_weight must be >= 0.0")
        if self.memory_read_norm_eps <= 0.0:
            raise ValueError("memory_read_norm_eps must be > 0.0")
        if self.stm_fusion_scale_init <= 0.0:
            raise ValueError("stm_fusion_scale_init must be > 0.0")
        if self.ltm_fusion_scale_init <= 0.0:
            raise ValueError("ltm_fusion_scale_init must be > 0.0")
        if self.memory_layer_index < 0:
            self.memory_layer_index = self.n_layers + self.memory_layer_index
        if not 0 <= self.memory_layer_index < self.n_layers:
            raise ValueError("memory_layer_index must target an existing transformer layer")
