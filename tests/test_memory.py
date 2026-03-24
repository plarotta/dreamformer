import torch

from dreamformer.memory import EpisodicMemory, SemanticMemory


def test_stm_read_returns_nearest_value() -> None:
    stm = EpisodicMemory(num_slots=4, key_dim=3, value_dim=2)
    keys = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    stm.write(keys, values)

    query = torch.tensor([[0.95, 0.05, 0.0]])
    readout, weights = stm.read(query, top_k=1)

    assert readout.shape == (1, 2)
    assert weights.shape == (1, 4)
    assert readout[0, 0] > readout[0, 1]
    assert stm.live_slots == 2
    assert stm.occupancy > 0.0


def test_ltm_update_increases_read_signal() -> None:
    ltm = SemanticMemory(key_dim=3, value_dim=2)
    key = torch.tensor([[1.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 5.0]])

    before = ltm.read(key)
    ltm.update(key, value, lr=1.0)
    after = ltm.read(key)

    assert before.shape == after.shape == (1, 2)
    assert after[0, 1] > before[0, 1]
