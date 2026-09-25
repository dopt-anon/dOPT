"""Run and incrementally save the synthetic switched-system comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from .compare_methods import run


METHODS = {
    "dOPT": "dOPT",
    "ffo_nonlifted": "FFOLayer-nonlifted",
    "ffo_lifted": "FFOLayer-lifted",
    "cvxpylayer": "CVXPYLayer",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[5, 10, 15, 20, 25, 30, 35, 40],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for n_state in args.sizes:
        for method in args.methods:
            method_name = METHODS[method]
            print(f"\n=== {method_name}: nx={n_state}, seed={args.seed} ===", flush=True)
            experiment_args = SimpleNamespace(
                n_state=n_state,
                n_modes=2,
                horizon=10,
                n_train=16,
                epochs=args.epochs,
                seed=args.seed,
                p_floor=0.0,
                target_margin=None,
                stability_weight=None,
                stability_loss_type="normalized_linear",
                normalized_margin_weight=1.0 / 12.0,
                selected_method=method_name,
            )
            result = run(experiment_args)[method_name]
            # Record the actual requested seed, rather than relying on defaults.
            history = result["history"]
            stem = f"{method_name}_nx{n_state}"
            summary_path = args.output_dir / f"{stem}_summary.json"
            history_path = args.output_dir / f"{stem}_epochs.csv"
            summary = {key: value for key, value in result.items() if key != "history"}
            summary.update({"method": method_name, "n_state": n_state, "seed": args.seed})
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            with history_path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f"Saved {summary_path} and {history_path}", flush=True)


if __name__ == "__main__":
    main()
