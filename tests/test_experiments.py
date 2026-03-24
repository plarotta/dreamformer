import pytest

from dreamformer import DreamFormerConfig, apply_variant, resolve_device


def test_apply_variant_baseline_disables_all_memory_paths() -> None:
    cfg = apply_variant(DreamFormerConfig(), "baseline")
    assert cfg.enable_stm is False
    assert cfg.enable_ltm is False
    assert cfg.enable_replay is False
    assert cfg.enable_nrem is False


def test_apply_variant_full_prioritized_enables_all_memory_paths() -> None:
    cfg = apply_variant(DreamFormerConfig(), "full_prioritized")
    assert cfg.enable_stm is True
    assert cfg.enable_ltm is True
    assert cfg.enable_replay is True
    assert cfg.enable_nrem is True
    assert cfg.replay_strategy == "prioritized"


def test_resolve_device_cpu() -> None:
    device = resolve_device("cpu")
    assert str(device) == "cpu"


def test_apply_variant_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        _ = apply_variant(DreamFormerConfig(), "does_not_exist")
