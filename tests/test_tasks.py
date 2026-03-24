from pathlib import Path

import torch

from dreamformer.tasks import CharCorpusSampler, generate_needle_batch, generate_passkey_batch


def test_passkey_batch_shapes_and_targets() -> None:
    batch = generate_passkey_batch(batch_size=4, seq_len=16, vocab_size=64, device=torch.device("cpu"))
    assert batch.input_ids.shape == (4, 16)
    assert batch.targets.shape == (4, 16)
    assert batch.query_positions is not None
    assert batch.answers is not None


def test_needle_batch_shapes_and_targets() -> None:
    batch = generate_needle_batch(batch_size=3, seq_len=20, vocab_size=64, device=torch.device("cpu"))
    assert batch.input_ids.shape == (3, 20)
    assert batch.targets.shape == (3, 20)
    assert batch.query_positions is not None
    assert batch.answers is not None


def test_char_sampler_from_text_and_batch(tmp_path: Path) -> None:
    corpus = tmp_path / "train.txt"
    corpus.write_text("hello world " * 20, encoding="utf-8")
    sampler = CharCorpusSampler.from_text_file(corpus)
    batch = sampler.sample_batch(
        batch_size=2,
        seq_len=12,
        vocab_size=512,
        device=torch.device("cpu"),
    )
    assert batch.input_ids.shape == (2, 12)
    assert batch.targets.shape == (2, 12)
