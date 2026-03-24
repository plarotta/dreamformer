from pathlib import Path

import torch

from dreamformer import DreamFormerConfig, DreamFormerModel
from dreamformer.tasks import generate_passkey_batch
from dreamformer.trainer import Trainer, TrainingConfig


def _tiny_model_config() -> DreamFormerConfig:
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
        replay_capacity=128,
    )


def _tiny_train_config() -> TrainingConfig:
    return TrainingConfig(
        steps=8,
        batch_size=4,
        seq_len=16,
        learning_rate=1e-3,
        log_every=2,
        eval_every=4,
        checkpoint_every=4,
        nrem_every=2,
        eval_batches=2,
    )


def test_trainer_runs_and_writes_outputs(tmp_path: Path) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_train_config()
    trainer = Trainer(
        model=DreamFormerModel(model_cfg),
        model_config=model_cfg,
        training_config=train_cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    summary = trainer.train(
        train_batch_fn=generate_passkey_batch,
        eval_batch_fn=generate_passkey_batch,
        run_name="tiny",
    )

    assert summary["final_step"] == train_cfg.steps
    assert (tmp_path / "tiny_summary.json").exists()
    assert (tmp_path / "tiny_metrics.jsonl").exists()
    assert (tmp_path / "tiny_checkpoint_last.pt").exists()


def test_trainer_resume_from_checkpoint(tmp_path: Path) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_train_config()
    trainer = Trainer(
        model=DreamFormerModel(model_cfg),
        model_config=model_cfg,
        training_config=train_cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    trainer.train(
        train_batch_fn=generate_passkey_batch,
        eval_batch_fn=generate_passkey_batch,
        run_name="resume",
    )

    ckpt = tmp_path / "resume_checkpoint_last.pt"
    resumed_cfg = TrainingConfig(
        steps=10,
        batch_size=train_cfg.batch_size,
        seq_len=train_cfg.seq_len,
        learning_rate=train_cfg.learning_rate,
        log_every=train_cfg.log_every,
        eval_every=train_cfg.eval_every,
        checkpoint_every=train_cfg.checkpoint_every,
        nrem_every=train_cfg.nrem_every,
        eval_batches=train_cfg.eval_batches,
    )
    resumed = Trainer(
        model=DreamFormerModel(model_cfg),
        model_config=model_cfg,
        training_config=resumed_cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    resumed.load_checkpoint(ckpt)
    assert resumed.step == 8

    summary = resumed.train(
        train_batch_fn=generate_passkey_batch,
        eval_batch_fn=generate_passkey_batch,
        run_name="resume2",
    )
    assert summary["final_step"] == 10


def test_trainer_emits_console_progress(tmp_path: Path, capsys) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = TrainingConfig(
        steps=4,
        batch_size=4,
        seq_len=16,
        learning_rate=1e-3,
        log_every=2,
        eval_every=2,
        checkpoint_every=4,
        nrem_every=2,
        eval_batches=1,
        console_log=True,
    )
    trainer = Trainer(
        model=DreamFormerModel(model_cfg),
        model_config=model_cfg,
        training_config=train_cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    trainer.train(
        train_batch_fn=generate_passkey_batch,
        eval_batch_fn=generate_passkey_batch,
        run_name="console",
    )

    captured = capsys.readouterr()
    assert "run_start" in captured.out
    assert "train step=" in captured.out
    assert "eval step=" in captured.out
    assert "run_complete" in captured.out
