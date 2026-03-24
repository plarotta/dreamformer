from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import torch

from .config import DreamFormerConfig


SUPPORTED_VARIANTS = (
    "baseline",
    "stm",
    "ltm_only",
    "full_uniform",
    "full_prioritized",
    "nrem_off",
)


def apply_variant(config: DreamFormerConfig, variant: str) -> DreamFormerConfig:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unknown variant '{variant}'. options={SUPPORTED_VARIANTS}")
    cfg = copy.deepcopy(config)

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
    elif variant == "ltm_only":
        cfg.enable_stm = False
        cfg.enable_ltm = True
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
    elif variant == "nrem_off":
        cfg.enable_stm = True
        cfg.enable_ltm = True
        cfg.enable_replay = True
        cfg.enable_nrem = False

    return cfg


def make_model_config(overrides: dict[str, Any] | None = None) -> DreamFormerConfig:
    base = DreamFormerConfig()
    if overrides is None:
        return base
    data = asdict(base)
    data.update(overrides)
    return DreamFormerConfig(**data)


def resolve_device(requested: str) -> torch.device:
    value = requested.lower().strip()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested device 'cuda' is unavailable on this machine")
        return torch.device("cuda")
    if value == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise ValueError("requested device 'mps' is unavailable on this machine")
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, mps, cpu")
