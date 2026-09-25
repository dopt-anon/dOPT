"""Plot available 10-vehicle learning loss and stability-margin curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


RESULTS = Path(__file__).resolve().parent / "results"
PAPER_FIGURE = RESULTS / "10vehicle_platoon_loss_t_star.pdf"
METHODS = (
    ("dOPT", "dOPT", "#2563EB"),
    ("ffolayer_nonlifted", "FFOLayer (Modified)", "#D97706"),
    ("ffolayer_lifted", "FFOLayer (Lifted)", "#F59E0B"),
    ("cvxpylayer", "CvxpyLayers", "#15803D"),
)


def load_results(results_dir: Path = RESULTS):
    rows = {}
    for path in sorted(results_dir.glob("10vehicle_*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            rows[row["method"]] = row

    available = {}
    for method, *_ in METHODS:
        if method not in rows:
            continue
        row = rows[method]
        if row["status"] != "completed":
            print(f"Skipping {method}: status={row['status']}")
            continue
        length = len(row["cumulative_epoch_time_s"])
        if len(row["learning_loss_per_epoch"]) != length or len(row["margin_per_epoch"]) != length:
            print(f"Skipping {method}: inconsistent epoch histories")
            continue
        available[method] = row
    if not available:
        raise FileNotFoundError(
            f"no completed 10-vehicle results found in {results_dir}"
        )
    return available


def plot(output: Path = PAPER_FIGURE, split_time: float = 450.0,
         results_dir: Path = RESULTS):
    rows = load_results(results_dir)
    end_time = max(row["cumulative_epoch_time_s"][-1] for row in rows.values())
    use_split_axis = end_time > split_time
    if use_split_axis:
        fig, axes = plt.subplots(
            2, 2, figsize=(9.1, 5.5), sharey="row", sharex="col",
            gridspec_kw={"width_ratios": (6, 1.35), "wspace": 0.045, "hspace": 0.13},
        )
    else:
        fig, single_axes = plt.subplots(
            2, 1, figsize=(7.6, 5.5), sharex=True,
            gridspec_kw={"hspace": 0.13},
        )
        axes = single_axes[:, None]

    for method, _label, color in METHODS:
        if method not in rows:
            continue
        row = rows[method]
        times = row["cumulative_epoch_time_s"]
        loss = row["learning_loss_per_epoch"]
        margin = row["margin_per_epoch"]
        for axis in axes[0]:
            axis.plot(times, loss, color=color, linewidth=1.8)
        for axis in axes[1]:
            axis.plot(times, margin, color=color, linewidth=1.8)

    for axis in axes.flat:
        axis.grid(color="0.85", linewidth=0.6, alpha=0.7)
        axis.tick_params(labelsize=10)
    for axis in axes[1]:
        axis.axhline(0.0, color="0.40", linestyle=":", linewidth=0.9)
    for left in (axes[0, 0], axes[1, 0]):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        left.yaxis.set_major_formatter(formatter)
        left.yaxis.set_offset_position("left")

    if use_split_axis:
        for left, right in axes:
            left.set_xlim(0.0, split_time)
            right.set_xlim(split_time, end_time)
            right.tick_params(axis="y", left=False, labelleft=False)
            left.spines["right"].set_visible(False)
            right.spines["left"].set_visible(False)
            diagonal = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=8,
                            linestyle="none", color="0.25", clip_on=False)
            left.plot([1, 1], [0, 1], transform=left.transAxes, **diagonal)
            right.plot([0, 0], [0, 1], transform=right.transAxes, **diagonal)
    else:
        for axis in axes[:, 0]:
            axis.set_xlim(0.0, end_time)

    axes[0, 0].set_ylabel("Learning loss", fontsize=12)
    axes[1, 0].set_ylabel(r"Stability margin $t^\star$", fontsize=12)
    if use_split_axis:
        axes[1, 0].set_xticks([0, 100, 200, 300, 400])
        axes[1, 1].set_xticks([split_time, end_time])
        axes[1, 1].set_xticklabels([f"{split_time:g}", f"{end_time / 1000:.1f}k"])
    fig.supxlabel("Training time [s]", y=0.05, fontsize=12)

    handles = [Line2D([0], [0], color=color, linewidth=2, label=label)
               for method, label, color in METHODS if method in rows]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=len(handles), frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.87)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight",
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PAPER_FIGURE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--split-time", type=float, default=450.0)
    arguments = parser.parse_args()
    plot(arguments.output, arguments.split_time, arguments.results_dir)
