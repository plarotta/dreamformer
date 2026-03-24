import torch

from dreamformer.replay import PrioritizedReplayBuffer


def test_replay_buffer_capacity_and_sampling() -> None:
    replay = PrioritizedReplayBuffer(capacity=8, alpha=0.6, epsilon=1e-3)

    for idx in range(12):
        replay.add(
            key=torch.tensor([float(idx), 0.0]),
            value=torch.tensor([0.0, float(idx)]),
            priority=float(idx + 1),
            metadata={"idx": idx},
        )

    assert len(replay) == 8

    batch = replay.sample(batch_size=4, beta=0.4)
    assert batch is not None
    assert len(batch.indices) == 4
    assert batch.keys.shape == (4, 2)
    assert batch.values.shape == (4, 2)
    assert batch.weights.shape == (4,)


def test_replay_priority_updates() -> None:
    replay = PrioritizedReplayBuffer(capacity=4, alpha=0.6, epsilon=1e-3)
    idx0 = replay.add(torch.tensor([1.0]), torch.tensor([1.0]), priority=1.0)
    idx1 = replay.add(torch.tensor([2.0]), torch.tensor([2.0]), priority=1.0)

    replay.update_priorities([idx0, idx1], [10.0, 0.1])

    p0 = replay._tree[replay.capacity + idx0]
    p1 = replay._tree[replay.capacity + idx1]
    assert p0 > p1


def test_uniform_sampling_returns_expected_batch() -> None:
    replay = PrioritizedReplayBuffer(capacity=6, alpha=0.6, epsilon=1e-3)
    for idx in range(4):
        replay.add(
            key=torch.tensor([float(idx)]),
            value=torch.tensor([float(idx)]),
            priority=float(idx + 1),
        )

    batch = replay.sample_uniform(batch_size=3)
    assert batch is not None
    assert len(batch.indices) == 3
    assert batch.weights.shape == (3,)
    assert float(batch.weights.mean().item()) == 1.0
