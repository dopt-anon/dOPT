#!/usr/bin/env python3
"""Create one combined figure with the backward-time plot and learning metrics plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Georgia"],
    "mathtext.fontset": "stix",
})

ROOT = Path(__file__).resolve().parent
SWITCHED_CSV = (
    ROOT
    / "learning_synthetic_switched_linear_systems"
    / "results"
    / "reference_results.csv"
)
VEHICLE_RESULTS = (
    ROOT / "vehicle_platoon_learning" / "results" / "reference_results"
)
DEFAULT_OUTPUT_STEM = ROOT / "figure" / "combined_figure"

BACKWARD_METHODS = (
    "dOPT",
    "FFOLayer (Modified)",
    "FFOLayer (Lifted)",
    "CVXPYLayers",
)
BACKWARD_STYLES = {
    "dOPT": ("#2563EB", "o"),
    "FFOLayer (Modified)": ("#D97706", "^"),
    "FFOLayer (Lifted)": ("#F59E0B", "D"),
    "CVXPYLayers": ("#15803D", "s"),
}

VEHICLE_METHODS = (
    ("dOPT", "dOPT", "#2563EB"),
    ("ffolayer_nonlifted", "FFOLayer (Mod)", "#D97706"),
    ("ffolayer_lifted", "FFOLayer (Lifted)", "#F59E0B"),
    ("cvxpylayer", "CvxpyLayers", "#15803D"),
)


def load_backward_results(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    results = {method: [] for method in BACKWARD_METHODS}
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


def load_vehicle_results(results_dir: Path) -> dict[str, dict[str, object]]:
    joint = json.loads((results_dir / "10vehicle_dOPT_ffo_original_patch.json").read_text())
    cvxpy = json.loads((results_dir / "10vehicle_cvxpylayer.json").read_text())
    rows = {row["method"]: row for row in joint["results"] + cvxpy["results"]}
    for method, *_ in VEHICLE_METHODS:
        row = rows[method]
        if row["status"] != "completed":
            raise RuntimeError(f"{method} did not complete")
        length = len(row["cumulative_epoch_time_s"])
        if len(row["learning_loss_per_epoch"]) != length or len(row["margin_per_epoch"]) != length:
            raise ValueError(f"{method} has inconsistent epoch histories")
    return rows


def plot_backward_panel(axis: plt.Axes, csv_path: Path) -> None:
    results = load_backward_results(csv_path)
    for method in BACKWARD_METHODS:
        rows = results[method]
        if not rows:
            continue
        color, marker = BACKWARD_STYLES[method]
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
    axis.set_ylim(0.0014656808552978857, 500.0)
    axis.set_xlim(0, 5400)
    axis.set_xticks(range(0, 5001, 1000))
    axis.tick_params(axis="x", labelsize=16, pad=6)
    axis.tick_params(axis="y", labelsize=17, pad=6)
    axis.set_xlabel("# variables", fontsize=16)
    axis.xaxis.set_label_coords(0.5, -0.13)
    axis.set_ylabel("Backward time per training step (s)", fontsize=19)
    axis.grid(True, which="major", linewidth=0.45, alpha=0.45)
    axis.tick_params(which="minor", left=False)

    for row in results["dOPT"]:
        n = row["nx"]
        axis.annotate(rf"${n} \times {n}$", (row["variables"], row["backward"]),
                      xytext=(8-n/5, -6-n/8), textcoords="offset points",
                      fontsize=10, color="0.25")


def plot_vehicle_panel(axes: list[plt.Axes], results_dir: Path, split_time: float = 450.0) -> None:
    rows = load_vehicle_results(results_dir)
    end_time = rows["cvxpylayer"]["cumulative_epoch_time_s"][-1]

    for method, _label, color in VEHICLE_METHODS:
        row = rows[method]
        times = row["cumulative_epoch_time_s"]
        loss = row["learning_loss_per_epoch"]
        margin = row["margin_per_epoch"]
        for axis in axes[:2]:
            axis.plot(times, loss, color=color, linewidth=1.8)
        for axis in axes[2:]:
            axis.plot(times, margin, color=color, linewidth=1.8)

    for axis in axes:
        axis.grid(color="0.85", linewidth=0.6, alpha=0.7)
        axis.tick_params(axis="x", labelsize=16)
        axis.tick_params(axis="y", labelsize=14)
    for axis in axes[2:]:
        axis.axhline(0.0, color="0.40", linestyle=":", linewidth=0.9)
    for top in axes[:2]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        top.yaxis.set_major_formatter(formatter)
        top.yaxis.set_offset_position("left")
        top.tick_params(axis="x", labelbottom=False)
    for bottom in axes[2:]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        bottom.yaxis.set_major_formatter(formatter)
        bottom.yaxis.set_offset_position("left")

    for left, right in [(axes[0], axes[1]), (axes[2], axes[3])]:
        left.set_xlim(0.0, split_time)
        right.set_xlim(split_time, end_time)
        right.tick_params(axis="y", left=False, labelleft=False)
        left.spines["right"].set_visible(False)
        right.spines["left"].set_visible(False)
        diagonal = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=8,
                        linestyle="none", color="0.25", clip_on=False)
        left.plot([1, 1], [0, 1], transform=left.transAxes, **diagonal)
        right.plot([0, 0], [0, 1], transform=right.transAxes, **diagonal)

    axes[0].set_ylabel("Learning loss", fontsize=19)
    axes[2].set_ylabel(r"Stability margin $t^\star$", fontsize=19)

    axes[0].set_xticks(np.linspace(0, 400, 5))
    axes[1].set_xticks(np.linspace(500, 13000, 25))
    axes[2].set_xticks(np.linspace(0, 400, 5))
    axes[3].set_xticks(np.linspace(500, 13000, 25))

    tick_positions = axes[3].get_xticks()
    labels = ["" for _ in tick_positions]
    labels[0] = f"{tick_positions[0]:g}"
    labels[-1] = f"{tick_positions[-1] / 1000:.1f}k"
    axes[3].set_xticklabels(labels)

    for axis in axes[:2]:
        axis.xaxis.grid(True, which="major", color="0.85", linewidth=0.6, alpha=0.7)
    for axis in axes[2:]:
        axis.xaxis.grid(True, which="major", color="0.85", linewidth=0.6, alpha=0.7)

    axes[2].set_xlabel("Training time [s]", fontsize=16)
    # Center the shared label under both parts of the broken x-axis.
    axes[2].xaxis.set_label_coords(0.63, -0.26)


def make_combined_figure(output_stem: Path, csv_path: Path, results_dir: Path, split_time: float) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.15)

    backward_ax = fig.add_subplot(gs[0, 0])
    plot_backward_panel(backward_ax, csv_path)

    vehicle_gs = gs[0, 1].subgridspec(2, 2, width_ratios=[6, 1.35], wspace=0.045, hspace=0.13)
    vehicle_axes = [
        fig.add_subplot(vehicle_gs[0, 0]),
        fig.add_subplot(vehicle_gs[0, 1]),
        fig.add_subplot(vehicle_gs[1, 0]),
        fig.add_subplot(vehicle_gs[1, 1]),
    ]
    plot_vehicle_panel(vehicle_axes, results_dir, split_time=split_time)

    fig.legend(handles=[Line2D([0], [0], color=color, marker=marker, linewidth=2, label=method)
                       for method, (color, marker) in BACKWARD_STYLES.items()],
               loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=4, frameon=False, fontsize=19)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.87)

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine switched-system backward-time and vehicle-learning plots into one figure.")
    parser.add_argument("--csv", type=Path, default=SWITCHED_CSV, help="Path to the switched linear system CSV results.")
    parser.add_argument("--results-dir", type=Path, default=VEHICLE_RESULTS, help="Directory containing vehicle-platoon JSON results.")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM, help="Output stem for the combined PDF/PNG figure.")
    parser.add_argument("--split-time", type=float, default=450.0, help="Cutoff time used for the vehicle-learning panel.")
    args = parser.parse_args()

    make_combined_figure(args.output_stem, args.csv, args.results_dir, args.split_time)
    print(f"Saved {args.output_stem.with_suffix('.pdf')}")
    print(f"Saved {args.output_stem.with_suffix('.png')}")
