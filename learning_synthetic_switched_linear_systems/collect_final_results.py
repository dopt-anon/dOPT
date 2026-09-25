"""Collect formal switched-system summary JSON files into the plotting CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DISPLAY_NAMES = {
    "dOPT": "dOPT",
    "FFOLayer-nonlifted": "FFOLayer (Modified)",
    "FFOLayer-lifted": "FFOLayer (Lifted)",
    "CVXPYLayer": "CVXPYLayers",
}
METHOD_ORDER = {name: index for index, name in enumerate(DISPLAY_NAMES.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.input_dir.glob("*_summary.json"):
        summary = json.loads(path.read_text())
        method = DISPLAY_NAMES.get(summary["method"], summary["method"])
        nx = int(summary["n_state"])
        rows.append(
            {
                "method": method,
                "nx": nx,
                "variables": nx**2,
                "forward": summary["median_forward_time"],
                "backward": summary["median_training_backward_time"],
                "t*": summary["final_margin"],
                "final loss": summary["final_loss"],
            }
        )

    if not rows:
        raise FileNotFoundError(f"no *_summary.json files found in {args.input_dir}")
    rows.sort(key=lambda row: (METHOD_ORDER.get(row["method"], len(METHOD_ORDER)), row["nx"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
