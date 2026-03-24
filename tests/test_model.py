import torch

from dreamformer import DreamFormerConfig, DreamFormerModel


def _tiny_config() -> DreamFormerConfig:
    return DreamFormerConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        memory_layer_index=1,
        num_stm_slots=16,
        memory_key_dim=16,
        memory_value_dim=16,
        replay_capacity=64,
        stm_top_k=2,
    )


def test_model_forward_shape_and_loss() -> None:
    torch.manual_seed(7)
    config = _tiny_config()
    model = DreamFormerModel(config)

    tokens = torch.randint(0, config.vocab_size, (3, 12))
    targets = torch.randint(0, config.vocab_size, (3, 12))

    out = model(tokens, targets=targets, write_memory=False)
    assert out.logits.shape == (3, 12, config.vocab_size)
    assert out.loss is not None
    assert out.loss.ndim == 0


def test_model_wake_and_nrem_cycle() -> None:
    torch.manual_seed(11)
    config = _tiny_config()
    model = DreamFormerModel(config)

    tokens = torch.randint(0, config.vocab_size, (4, 10))
    targets = torch.randint(0, config.vocab_size, (4, 10))

    out = model(tokens, targets=targets, write_memory=True)
    assert out.memory_stats["stm_live_slots"] > 0.0
    assert out.memory_stats["replay_size"] > 0.0

    before_norm = float(model.ltm.normalizer.sum().item())
    stats = model.nrem_consolidation_step(batch_size=4, beta=0.4)
    after_norm = float(model.ltm.normalizer.sum().item())

    assert stats["sampled"] > 0.0
    assert stats["selection_rate"] >= 0.0
    assert after_norm >= before_norm


def test_model_ablation_baseline_disables_memory_systems() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    config.enable_stm = False
    config.enable_ltm = False
    config.enable_replay = False
    config.enable_nrem = False

    model = DreamFormerModel(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    out = model(tokens, targets=targets, write_memory=True)
    assert out.memory_stats["stm_live_slots"] == 0.0
    assert out.memory_stats["replay_size"] == 0.0

    nrem = model.nrem_consolidation_step(batch_size=4, beta=0.4)
    assert nrem["sampled"] == 0.0


def test_model_uniform_replay_path() -> None:
    torch.manual_seed(19)
    config = _tiny_config()
    config.replay_strategy = "uniform"
    model = DreamFormerModel(config)

    tokens = torch.randint(0, config.vocab_size, (4, 10))
    targets = torch.randint(0, config.vocab_size, (4, 10))
    _ = model(tokens, targets=targets, write_memory=True)

    stats = model.nrem_consolidation_step(batch_size=4, beta=0.4)
    assert stats["sampled"] >= 0.0


def test_model_nrem_handles_half_precision_replay_entries() -> None:
    torch.manual_seed(23)
    config = _tiny_config()
    model = DreamFormerModel(config)

    # Simulate fp16 samples in replay, as happens in mixed-precision training.
    for _ in range(8):
        key = torch.randn(config.memory_key_dim, dtype=torch.float16)
        value = torch.randn(config.memory_value_dim, dtype=torch.float16)
        model.replay_buffer.add(key=key, value=value, priority=1.0)

    stats = model.nrem_consolidation_step(batch_size=4, beta=0.4)
    assert stats["sampled"] > 0.0


def test_model_stage_experience_handles_zero_vectors_without_nan_metadata() -> None:
    torch.manual_seed(29)
    config = _tiny_config()
    model = DreamFormerModel(config)

    hidden = torch.zeros(3, 5, config.d_model)
    logits = torch.zeros(3, 5, config.vocab_size)
    targets = torch.zeros(3, 5, dtype=torch.long)

    model._stage_experience(hidden, logits, targets)

    assert torch.isfinite(model.stm.access_count).all()
    batch = model.replay_buffer.sample(batch_size=3, beta=0.4)
    assert batch is not None
    for meta in batch.metadata:
        assert isinstance(meta["access_count"], int)


def test_model_gate_initialization_and_regularization() -> None:
    torch.manual_seed(31)
    config = _tiny_config()
    config.memory_gate_init = 0.25
    config.memory_gate_target = 0.20
    config.memory_gate_band = 0.05
    config.memory_gate_regularization_weight = 0.1
    model = DreamFormerModel(config)

    bias_prob = torch.sigmoid(model.memory_gate.bias.detach()).mean().item()
    assert abs(bias_prob - 0.25) < 0.05

    tokens = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    out = model(tokens, targets=targets, write_memory=False)
    assert out.loss is not None
    assert out.memory_stats["memory_gate_reg_loss"] >= 0.0
    assert 0.0 <= out.memory_stats["memory_gate_min"] <= 1.0
    assert 0.0 <= out.memory_stats["memory_gate_max"] <= 1.0
