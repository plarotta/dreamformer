from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DreamFormer jsonl metrics into summary tables.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more metrics .jsonl files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/report_summary.json"),
        help="Summary JSON output path.",
    )
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for metrics_path in args.inputs:
        rows = _load_jsonl(metrics_path)
        for row in rows:
            run = str(row.get("run", "unknown"))
            split = str(row.get("split", "unknown"))
            grouped.setdefault((run, split), []).append(row)

    summary_rows = []
    print("run,split,mean_loss,mean_eval_loss,mean_query_acc,last_step")
    for (run, split), rows in sorted(grouped.items()):
        mean_loss = _mean([float(r["loss"]) for r in rows if "loss" in r])
        mean_eval_loss = _mean([float(r["eval_loss"]) for r in rows if "eval_loss" in r])
        mean_query_acc = _mean(
            [float(r["query_acc"]) for r in rows if "query_acc" in r]
            + [float(r["eval_query_acc"]) for r in rows if "eval_query_acc" in r]
        )
        last_step = int(max(float(r.get("step", 0.0)) for r in rows))
        summary_rows.append(
            {
                "run": run,
                "split": split,
                "mean_loss": mean_loss,
                "mean_eval_loss": mean_eval_loss,
                "mean_query_acc": mean_query_acc,
                "last_step": last_step,
            }
        )
        print(
            f"{run},{split},{mean_loss:.6f},{mean_eval_loss:.6f},{mean_query_acc:.6f},{last_step}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote_summary={args.out}")


if __name__ == "__main__":
    main()
