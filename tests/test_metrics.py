from pathlib import Path

from dreamformer.metrics import ExperimentLogger


def test_experiment_logger_mean_and_dump(tmp_path: Path) -> None:
    logger = ExperimentLogger()
    logger.log(step=1, split="train", metrics={"loss": 2.0, "acc": 0.1}, variant="baseline")
    logger.log(step=2, split="train", metrics={"loss": 1.0, "acc": 0.4}, variant="baseline")
    logger.log(step=2, split="eval", metrics={"loss": 1.5, "acc": 0.3}, variant="baseline")

    assert abs(logger.mean("loss", split="train") - 1.5) < 1e-6
    assert logger.latest() is not None

    out = tmp_path / "metrics.jsonl"
    logger.dump_jsonl(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").count("\n") == 3
