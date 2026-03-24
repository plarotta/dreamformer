from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamformer.workflows import run_training_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single DreamFormer experiment from JSON config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment JSON config.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_training_job(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
