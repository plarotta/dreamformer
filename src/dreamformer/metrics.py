from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentLogger:
    """Simple in-memory logger with jsonl export for experiment tracking."""

    history: list[dict[str, Any]] = field(default_factory=list)

    def log(self, step: int, split: str, metrics: dict[str, float], **extra: Any) -> None:
        record: dict[str, Any] = {"step": int(step), "split": split}
        record.update({k: float(v) for k, v in metrics.items()})
        record.update(extra)
        self.history.append(record)

    def latest(self) -> dict[str, Any] | None:
        if not self.history:
            return None
        return self.history[-1]

    def mean(self, key: str, split: str | None = None) -> float:
        values: list[float] = []
        for row in self.history:
            if split is not None and row.get("split") != split:
                continue
            if key in row:
                values.append(float(row[key]))
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def dump_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in self.history:
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")
