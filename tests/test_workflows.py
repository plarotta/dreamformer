from pathlib import Path

from dreamformer.workflows import run_training_job


def test_run_training_job_smoke(tmp_path: Path) -> None:
    spec = {
        "run_name": "wf_smoke",
        "variant": "full_prioritized",
        "task": "passkey",
        "eval_task": "passkey",
        "task_overrides": {
            "key_copies": 3
        },
        "seed": 123,
        "device": "cpu",
        "output_dir": str(tmp_path),
        "model_overrides": {
            "vocab_size": 64,
            "max_seq_len": 32,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "ffn_dim": 64,
            "memory_layer_index": 1,
            "num_stm_slots": 16,
            "memory_key_dim": 16,
            "memory_value_dim": 16,
            "replay_capacity": 128
        },
        "train_overrides": {
            "steps": 8,
            "batch_size": 4,
            "seq_len": 16,
            "log_every": 2,
            "eval_every": 4,
            "checkpoint_every": 4,
            "nrem_every": 2,
            "eval_batches": 2,
            "query_loss_weight": 2.0
        }
    }
    summary = run_training_job(spec)
    assert summary["final_step"] == 8
    assert "eval_query_loss" in summary
    run_dir = tmp_path / "wf_smoke"
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "wf_smoke_summary.json").exists()
