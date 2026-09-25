"""Compare direct and explicitly lifted switched-system stability layers."""

import argparse
import gc
import statistics
import time

import numpy as np
import torch

from .switched_linear_system import generate_common_lyapunov_modes, generate_sls_data
from .switched_linear_system_learning import (
    CommonLyapunovStabilityLayer,
    LiftedFFOLyapunovStabilityLayer,
    LearnedSwitchedLinearSystem,
    fit_modes_least_squares,
    train_switched_linear_system,
)


def fixed_sequence(length, n_modes):
    return [min(n_modes - 1, k * n_modes // length) for k in range(length)]


def serialize_history(history):
    return [
        {
            "epoch": epoch,
            "total_loss": record.total_loss,
            "trajectory_loss": record.trajectory_loss,
            "stability_margin": record.stability_margin,
            "stability_forward_time": record.stability_forward_time,
            "backward_time": record.backward_time,
        }
        for epoch, record in enumerate(history)
    ]


def summarize(history):
    warm = history[1:] if len(history) > 1 else history
    return {
        "initial_loss": history[0].total_loss,
        "final_loss": history[-1].total_loss,
        "final_trajectory_loss": history[-1].trajectory_loss,
        "final_margin": history[-1].stability_margin,
        "median_forward_time": statistics.median(
            record.stability_forward_time for record in warm
        ),
        "median_training_backward_time": statistics.median(
            record.backward_time for record in warm
        ),
    }


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dtype = torch.float64
    target_margin = (
        0.24 / args.n_state
        if args.target_margin is None
        else args.target_margin
    )
    stability_weight = (
        50.0 * args.n_state / 12.0
        if getattr(args, "stability_weight", None) is None
        else args.stability_weight
    )
    stability_loss_type = getattr(args, "stability_loss_type", "hinge")
    normalized_margin_weight = getattr(
        args, "normalized_margin_weight", 1.0 / 12.0
    )
    true_modes, _ = generate_common_lyapunov_modes(
        n_state=args.n_state, n_modes=args.n_modes,
        contraction=0.95, lyapunov_condition_number=5.0, seed=args.seed,
    )
    sequence = fixed_sequence(args.horizon, args.n_modes)
    data = generate_sls_data(
        n_state=args.n_state, T=args.horizon, N=args.n_train,
        modes=true_modes, mode_sequence=sequence, noise_std=0.03,
        seed=args.seed + 1,
    )
    trajectories = data["X"].to(dtype)
    initial = 1.25 * fit_modes_least_squares(
        trajectories, data["mode_seq"], args.n_modes, ridge=1e-5
    )

    # Construct and run one backend at a time so compiled problems are released.
    configuration_builders = {
        "dOPT": lambda: CommonLyapunovStabilityLayer(
            args.n_state, args.n_modes, backend="dsdp", dsdp_mode="dense",
            p_floor=getattr(args, "p_floor", 0.0),
            dsdp_settings={
                "solver": "MOSEK",
                "solver_args": {"eps": 1e-7},
            },
            dtype=dtype,
        ),
        "CVXPYLayer": lambda: CommonLyapunovStabilityLayer(
            args.n_state, args.n_modes, backend="cvxpylayer",
            p_floor=getattr(args, "p_floor", 0.0), dtype=dtype
        ),
        "FFOLayer-nonlifted": lambda: CommonLyapunovStabilityLayer(
            args.n_state, args.n_modes, backend="ffolayer",
            p_floor=getattr(args, "p_floor", 0.0), dtype=dtype
        ),
        "FFOLayer-lifted": lambda: LiftedFFOLyapunovStabilityLayer(
            args.n_state, args.n_modes,
            p_floor=getattr(args, "p_floor", 0.0), dtype=dtype,
        ),
    }
    selected_method = getattr(args, "selected_method", None)
    if selected_method is not None:
        if selected_method not in configuration_builders:
            raise ValueError(f"unknown selected method: {selected_method}")
        configuration_builders = {
            selected_method: configuration_builders[selected_method]
        }
    results = {}
    expected_formulation_signature = None
    for name, build_stability_layer in configuration_builders.items():
        print(f"\nRunning {name}...", flush=True)
        construction_start = time.perf_counter()
        stability_layer = build_stability_layer()
        construction_time = time.perf_counter() - construction_start
        if expected_formulation_signature is None:
            expected_formulation_signature = stability_layer.formulation_signature
        elif (
            name != "FFOLayer-lifted"
            and stability_layer.formulation_signature
            != expected_formulation_signature
        ):
            raise RuntimeError(
                f"unfair formulation for {name}: "
                f"{stability_layer.formulation_signature} != "
                f"{expected_formulation_signature}"
            )
        model = LearnedSwitchedLinearSystem(initial.clone())
        # Keep construction and the first solve separate. CvxpyLayer performs
        # substantial compilation in its constructor, while a plain CVXPY
        # Problem performs lazy canonicalization in its first solve.
        first_forward_start = time.perf_counter()
        with torch.no_grad():
            stability_layer(
                {
                    mode_idx: initial[mode_idx]
                    for mode_idx in range(args.n_modes)
                }
            )
        first_forward_time = time.perf_counter() - first_forward_start

        start = time.perf_counter()
        history = train_switched_linear_system(
            model, stability_layer, trajectories, data["mode_seq"],
            epochs=args.epochs, learning_rate=5e-3,
            stability_weight=stability_weight, target_margin=target_margin,
            stability_loss_type=stability_loss_type,
            normalized_margin_weight=normalized_margin_weight,
        )
        results[name] = summarize(history)
        results[name]["target_margin"] = (
            target_margin if stability_loss_type == "hinge" else None
        )
        results[name]["stability_weight"] = (
            stability_weight if stability_loss_type == "hinge" else None
        )
        results[name]["stability_loss_type"] = stability_loss_type
        results[name]["normalized_margin_weight"] = normalized_margin_weight
        results[name]["p_floor"] = getattr(args, "p_floor", 0.0)
        results[name]["history"] = serialize_history(history)
        results[name]["construction_time"] = construction_time
        results[name]["first_forward_time"] = first_forward_time
        results[name]["setup_plus_first_forward"] = (
            construction_time + first_forward_time
        )
        results[name]["formulation_signature"] = (
            stability_layer.formulation_signature
        )
        results[name]["training_time"] = time.perf_counter() - start
        print(f"Completed {name}.", flush=True)
        del history, model, stability_layer
        gc.collect()

    for name, metrics in results.items():
        print(f"\n{name}")
        for key, value in metrics.items():
            if key == "history":
                continue
            if isinstance(value, (int, float)):
                print(f"  {key:32s} {value:.8g}")
            else:
                print(f"  {key:32s} {value}")
    return results


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n-state", type=int, default=2)
    p.add_argument("--n-modes", type=int, default=2)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-train", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--p-floor", type=float, default=0.0)
    p.add_argument(
        "--target-margin", type=float, default=None,
        help="Stability target; defaults to the dimension-normalized 0.24/n_state.",
    )
    p.add_argument(
        "--stability-weight", type=float, default=None,
        help=(
            "Stability-loss weight; defaults to the dimension-scaled "
            "50*n_state/12."
        ),
    )
    p.add_argument(
        "--stability-loss-type",
        choices=["hinge", "normalized_linear"],
        default="hinge",
    )
    p.add_argument("--normalized-margin-weight", type=float, default=1.0 / 12.0)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
