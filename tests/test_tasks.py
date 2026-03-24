from pathlib import Path

import torch

from dreamformer.tasks import (
    CharCorpusSampler,
    KEY_MARK,
    query_cross_entropy,
    generate_needle_batch,
    generate_passkey_batch,
)


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


def test_passkey_batch_supports_repeated_key_copies() -> None:
    batch = generate_passkey_batch(
        batch_size=2,
        seq_len=20,
        vocab_size=64,
        device=torch.device("cpu"),
        key_copies=3,
    )
    key_marks = (batch.input_ids == KEY_MARK).sum(dim=1)
    assert torch.equal(key_marks, torch.full_like(key_marks, 3))


def test_needle_batch_supports_multiple_needles() -> None:
    batch = generate_needle_batch(
        batch_size=2,
        seq_len=20,
        vocab_size=64,
        device=torch.device("cpu"),
        needle_copies=3,
    )
    assert batch.answers is not None
    answer_matches = (batch.input_ids == batch.answers.unsqueeze(1)).sum(dim=1)
    assert torch.all(answer_matches >= 4)


def test_query_cross_entropy_matches_answers() -> None:
    batch = generate_passkey_batch(batch_size=2, seq_len=16, vocab_size=64, device=torch.device("cpu"))
    assert batch.query_positions is not None
    assert batch.answers is not None
    logits = torch.zeros(2, 16, 64)
    logits[torch.arange(2), batch.query_positions, batch.answers] = 5.0
    loss = query_cross_entropy(batch, logits)
    assert loss is not None
    assert float(loss.item()) < 1.0


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
