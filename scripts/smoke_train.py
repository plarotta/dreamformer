from __future__ import annotations

import torch

from dreamformer import DreamFormerConfig, DreamFormerModel


def main() -> None:
    torch.manual_seed(42)
    config = DreamFormerConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_heads=4,
        n_layers=3,
        ffn_dim=128,
        memory_layer_index=2,
        num_stm_slots=32,
        memory_key_dim=32,
        memory_value_dim=32,
        replay_capacity=256,
    )
    model = DreamFormerModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1, 31):
        tokens = torch.randint(0, config.vocab_size, (8, 24))
        targets = torch.roll(tokens, shifts=-1, dims=1)

        out = model(tokens, targets=targets, write_memory=True)
        assert out.loss is not None

        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        if step % 5 == 0:
            nrem_stats = model.nrem_consolidation_step(batch_size=16, beta=0.4)
            print(
                f"step={step:02d} loss={out.loss.item():.4f} "
                f"stm_live={out.memory_stats['stm_live_slots']:.0f} "
                f"replay={out.memory_stats['replay_size']:.0f} "
                f"nrem_selected={nrem_stats['selected']:.0f}"
            )


if __name__ == "__main__":
    main()
