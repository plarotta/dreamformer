from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dreamformer.experiments import SUPPORTED_VARIANTS
from dreamformer.workflows import run_training_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ablation sweep from a base JSON config.")
    parser.add_argument("--config", type=Path, required=True, help="Base experiment JSON config.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "stm", "full_uniform", "full_prioritized"],
        help=f"Variant names. options={SUPPORTED_VARIANTS}",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/ablation_summary.json"),
        help="Path to write aggregated sweep summary.",
    )
    args = parser.parse_args()

    base = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for variant in args.variants:
        spec = copy.deepcopy(base)
        run_name = str(spec.get("run_name", "run"))
        spec["run_name"] = f"{run_name}_{variant}"
        spec["variant"] = variant
        print(f"running variant={variant} run_name={spec['run_name']}")
        summary = run_training_job(spec)
        rows.append(
            {
                "variant": variant,
                "run_name": spec["run_name"],
                "eval_loss": summary.get("eval_loss"),
                "eval_query_acc": summary.get("eval_query_acc"),
                "best_eval_loss": summary.get("best_eval_loss"),
                "summary_path": str(
                    Path(spec.get("output_dir", "artifacts/runs")) / spec["run_name"] / f"{spec['run_name']}_summary.json"
                ),
            }
        )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote_summary={args.summary_out}")
    print("variant,eval_loss,eval_query_acc,best_eval_loss")
    for row in rows:
        print(
            f"{row['variant']},"
            f"{row.get('eval_loss', 'nan')},"
            f"{row.get('eval_query_acc', 'nan')},"
            f"{row.get('best_eval_loss', 'nan')}"
        )


if __name__ == "__main__":
    main()
