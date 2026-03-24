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
