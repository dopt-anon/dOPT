"""Plot the final switched-system backward-time comparison from one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = (
    "dOPT",
    "FFOLayer (Modified)",
    "FFOLayer (Lifted)",
    "CVXPYLayers",
)
STYLES = {
    "dOPT": ("#2563EB", "o"),
    "FFOLayer (Modified)": ("#D97706", "^"),
    "FFOLayer (Lifted)": ("#F59E0B", "D"),
    "CVXPYLayers": ("#15803D", "s"),
}


def load_results(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    results = {method: [] for method in METHODS}
    with csv_path.open(newline="") as source:
        for row in csv.DictReader(source):
            method = row["method"]
            if method not in results:
                raise ValueError(f"unknown method in {csv_path}: {method}")
            results[method].append(
                {
                    "nx": int(row["nx"]),
                    "variables": int(row["variables"]),
                    "backward": float(row["backward"]),
                }
            )
    for rows in results.values():
        rows.sort(key=lambda row: row["variables"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "results" / "results.csv",
    )
    parser.add_argument(
        "output_stem",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "results" / "backward_time",
    )
    args = parser.parse_args()

    results = load_results(args.csv)
    fig, axis = plt.subplots(figsize=(5.25, 3.35))
    for method in METHODS:
        rows = results[method]
        if not rows:
            continue
        color, marker = STYLES[method]
        axis.plot(
            [row["variables"] for row in rows],
            [row["backward"] for row in rows],
            label=method,
            color=color,
            marker=marker,
            linewidth=1.75,
            markersize=4.8,
        )

    axis.set_yscale("log")
    # Keep the limits of the original paper figure. The nx=50 and nx=60
    # CVXPYLayers measurements lie above the visible upper bound by design.
    axis.set_ylim(0.0014656808552978857, 60.973761574537846)
    axis.set_xlim(0, 5000)
    axis.set_xticks(range(0, 5001, 1000))
    axis.set_xlabel("# variables")
    axis.set_ylabel("Backward time per training step (s)")
    axis.grid(True, which="major", linewidth=0.45, alpha=0.45)
    axis.tick_params(which="minor", left=False)
    axis.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_stem.with_suffix('.pdf')}")
    print(f"Saved {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
