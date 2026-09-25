"""Fair timing comparison of native stability layers on platoon learning.

All methods share data, initialization, optimization settings, and solver
tolerances. The explicit-slack FFOLayer is the sole lifted formulation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import cvxpy as cp
import torch
from torch.nn import functional as F

from third_party.FFOLayer.src.ffolayer import FFOLayer
from learning_synthetic_switched_linear_systems.lyapunov_block_sdp import build_common_lyapunov_block_sdp
from learning_synthetic_switched_linear_systems.switched_linear_system_learning import CommonLyapunovStabilityLayer

from .learning import (
    LatentSingleSwitchSystem,
    fit_pooled_least_squares,
    split_pooled_initialization,
)
from .platoon_system import (
    discretize_platoon, generate_learning_data, load_continuous_platoon,
    load_paper_three_vehicle_platoon,
)


DTYPE = torch.float64
SCS_ARGS = {"eps": 1e-7}


def dsdp_settings(solver: str):
    return {
        "solver": solver,
        "eps_active": 1e-7,
        "solver_args": (dict(SCS_ARGS) if solver == "SCS"
                        else {"eps": 1e-7}),
    }


class TimedLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.forward_times: list[float] = []

    def forward(self, modes):
        start = time.perf_counter()
        result = self.layer(modes)
        self.forward_times.append(time.perf_counter() - start)
        return result


class LiftedFFOStabilityLayer(torch.nn.Module):
    """Retain the original explicit-slack FFOLayer implementation."""

    def __init__(self, n_state: int, n_modes: int, *, t_bound: float = 1.0,
                 p_floor: float = 1e-5,
                 proximal_weight: float = 0.0,
                 solver: str = "SCS"):
        super().__init__()
        self.n_state, self.n_modes, self.t_bound = n_state, n_modes, t_bound
        self.p_floor = p_floor
        dummy = {i: torch.zeros(n_state, n_state, dtype=DTYPE) for i in range(n_modes)}
        _, a, b, layout = build_common_lyapunov_block_sdp(
            dummy, t_bound=t_bound, p_floor=p_floor
        )
        slices = [layout.p_slice, *(layout.slack_slices[k] for k in layout.mode_keys), layout.t_slice]
        columns = torch.tensor([
            row + col * layout.matrix_size
            for block in slices
            for col in range(block.start, block.stop)
            for row in range(block.start, block.stop)
        ], dtype=torch.long)
        self.register_buffer("block_columns", columns)

        variables = [cp.Variable((s.stop - s.start, s.stop - s.start), symmetric=True) for s in slices]
        c_parameter = cp.Parameter(len(columns))
        a_parameter = cp.Parameter((a.shape[0], len(columns)))
        b_parameter = cp.Parameter(b.shape)
        vector = cp.hstack([cp.vec(variable, order="F") for variable in variables])
        objective = c_parameter @ vector
        if proximal_weight:
            objective += .5 * proximal_weight * cp.sum_squares(vector)
        problem = cp.Problem(
            cp.Minimize(objective),
            [*(variable >> 0 for variable in variables), a_parameter @ vector == b_parameter],
        )
        if not problem.is_dpp():
            raise RuntimeError("FFO block problem must be DPP")
        self.ffo = FFOLayer(
            problem, parameters=[c_parameter, a_parameter, b_parameter], variables=variables,
            eps=SCS_ARGS["eps"], backward_eps=SCS_ARGS["eps"],
        )
        self.ffo_forward_times: list[float] = []
        self.proximal_weight = proximal_weight
        self.solver = solver
        q = n_state * (n_state + 1) // 2
        self.formulation_signature = {
            "variables": {
                "P_symmetric": q,
                "slack_symmetric": n_modes * q,
                "t_embedding_symmetric": 3,
            },
            "psd_cone_sizes": (n_state,) * (n_modes + 1) + (2,),
            "equality_count": a.shape[0],
            "explicit_slack_variables": n_modes,
            "definition_equalities": n_modes * q,
        }

    def reset_timings(self):
        self.ffo_forward_times.clear()

    def forward(self, modes):
        objective, a_matrix, b_vector, _ = build_common_lyapunov_block_sdp(
            modes, t_bound=self.t_bound, p_floor=self.p_floor
        )
        block_objective = objective.T.reshape(-1)[self.block_columns]
        block_a = a_matrix[:, self.block_columns]
        start = time.perf_counter()
        ffo_solver_args = {"solver": cp.SCS if self.solver == "SCS" else cp.MOSEK}
        if self.solver == "SCS":
            ffo_solver_args.update(SCS_ARGS)
        else:
            ffo_solver_args["eps"] = 1e-7
        blocks = self.ffo(
            # FFOLayer 0.1.2's conic backward assumes its internal leading
            # batch axis is explicit, even for a single problem.
            block_objective.unsqueeze(0), block_a.unsqueeze(0), b_vector.unsqueeze(0),
            solver_args=ffo_solver_args,
        )
        self.ffo_forward_times.append(time.perf_counter() - start)
        if self.solver == "MOSEK":
            self.ffo._solver_args_bwd.pop("max_iters", None)
            self.ffo._solver_args_bwd.update({"solver": cp.MOSEK, "eps": 1e-7})
        t_block = blocks[-1]
        if t_block.ndim == 3 and t_block.shape[0] == 1:
            t_block = t_block[0]
        t_ffo = t_block[0, 1]
        return {"t": t_ffo, "T": t_block}


def median(values, skip=5):
    kept = values[skip:] if len(values) > skip else values
    return statistics.median(kept) if kept else None


def serializable_solver_args(arguments):
    return {
        key: (value if isinstance(value, (str, int, float, bool, type(None))) else str(value))
        for key, value in arguments.items()
    }


def make_layer(
    method: str,
    n_state: int,
    *,
    solver: str = "SCS",
    p_floor: float = 1e-5,
):
    start = time.perf_counter()
    if method in {"ffolayer", "ffolayer_lifted"}:
        raw = LiftedFFOStabilityLayer(n_state, 2, solver=solver, p_floor=p_floor)
    else:
        backend = (
            "ffolayer" if method == "ffolayer_nonlifted"
            else "dsdp" if method == "dOPT" else method
        )
        ffo_forward_args = {
            "solver": cp.SCS if solver == "SCS" else cp.MOSEK,
            "eps": SCS_ARGS["eps"] if solver == "SCS" else 1e-7,
        }
        raw = CommonLyapunovStabilityLayer(
            n_state, 2, backend=backend, dsdp_mode="dense",
            p_floor=p_floor,
            dsdp_settings=dsdp_settings(solver), dtype=DTYPE,
            ffo_solver_args=ffo_forward_args if backend == "ffolayer" else None,
            cvxpy_solver_args=(
                {"solve_method": "SCS", **SCS_ARGS}
                if backend == "cvxpylayer" else None
            ),
        )
    construction = time.perf_counter() - start
    return TimedLayer(raw), construction


def train(model, layer, states, inputs, ground_truth_modes, args):
    backward_times, stage, margin_history, relative_error_history = [], [], [], []
    learning_loss_history, fit_loss_history = [], []
    epoch_times = []
    stable_streak = 0
    stopped_on_margin = False
    optimizer = torch.optim.Adam([
        {"params": [model.mode_matrices], "lr": args.learning_rate},
        {"params": [model.raw_switch_fraction], "lr": args.switch_learning_rate},
    ])
    for epoch in range(args.soft_epochs):
        epoch_start = time.perf_counter()
        progress = epoch / max(args.soft_epochs - 1, 1)
        temperature = args.initial_temperature * (args.final_temperature / args.initial_temperature) ** progress
        optimizer.zero_grad()
        prediction = model.rollout(states[:, 0], inputs, temperature=temperature)
        trajectory_loss = F.mse_loss(prediction, states)
        try:
            margin = layer(model.modes_dict())["t"]
        except Exception as exc:
            raise RuntimeError(f"soft epoch {epoch}: {exc}") from exc
        stability_loss = -discrete_state_dimension(model) * margin
        loss = trajectory_loss + args.normalized_margin_weight * stability_loss
        start = time.perf_counter(); loss.backward(); backward_times.append(time.perf_counter() - start)
        stage.append("soft")
        margin_history.append(float(margin.detach()))
        learning_loss_history.append(float(loss.detach()))
        fit_loss_history.append(float(trajectory_loss.detach()))
        relative_errors = (
            torch.linalg.matrix_norm(
                model.mode_matrices.detach() - ground_truth_modes, dim=(-2, -1)
            )
            / torch.linalg.matrix_norm(ground_truth_modes, dim=(-2, -1))
        ).tolist()
        relative_error_history.append(relative_errors)
        optimizer.step()
        epoch_times.append(time.perf_counter() - epoch_start)
        print(
            f"epoch={epoch + 1}/{args.soft_epochs + args.hard_epochs} "
            f"stage=soft stage_epoch={epoch + 1}/{args.soft_epochs} "
            f"loss={float(loss.detach()):.9e} "
            f"fit_loss={float(trajectory_loss.detach()):.9e} "
            f"t_star={float(margin.detach()):.9e} "
            f"epoch_time={epoch_times[-1]:.9e} "
            f"A0_rel={relative_errors[0]:.9e} "
            f"A1_rel={relative_errors[1]:.9e}",
            flush=True,
        )

    hard_modes = model.hard_mode_sequence().detach()
    optimizer = torch.optim.Adam([model.mode_matrices], lr=args.hard_learning_rate)
    for epoch in range(args.hard_epochs):
        epoch_start = time.perf_counter()
        optimizer.zero_grad()
        current, following = states[:, :-1], states[:, 1:]
        transition, input_matrix = model.mode_matrices[hard_modes], model.input_matrices[hard_modes]
        prediction = torch.einsum("tij,btj->bti", transition, current)
        prediction += torch.einsum("tij,btj->bti", input_matrix, inputs)
        one_step_loss = F.mse_loss(prediction, following)
        try:
            margin = layer(model.modes_dict())["t"]
        except Exception as exc:
            raise RuntimeError(f"hard epoch {epoch}: {exc}") from exc
        stability_loss = -discrete_state_dimension(model) * margin
        loss = one_step_loss + args.normalized_margin_weight * stability_loss
        start = time.perf_counter(); loss.backward(); backward_times.append(time.perf_counter() - start)
        stage.append("hard")
        margin_history.append(float(margin.detach()))
        learning_loss_history.append(float(loss.detach()))
        fit_loss_history.append(float(one_step_loss.detach()))
        relative_errors = (
            torch.linalg.matrix_norm(
                model.mode_matrices.detach() - ground_truth_modes, dim=(-2, -1)
            )
            / torch.linalg.matrix_norm(ground_truth_modes, dim=(-2, -1))
        ).tolist()
        relative_error_history.append(relative_errors)
        should_stop = False
        if args.stability_stop_threshold is not None:
            stable_streak = (
                stable_streak + 1
                if float(margin.detach()) >= args.stability_stop_threshold
                else 0
            )
            should_stop = stable_streak >= args.stability_stop_patience
        if not should_stop:
            optimizer.step()
        epoch_times.append(time.perf_counter() - epoch_start)
        print(
            f"epoch={args.soft_epochs + epoch + 1}/{args.soft_epochs + args.hard_epochs} "
            f"stage=hard stage_epoch={epoch + 1}/{args.hard_epochs} "
            f"loss={float(loss.detach()):.9e} "
            f"fit_loss={float(one_step_loss.detach()):.9e} "
            f"t_star={float(margin.detach()):.9e} "
            f"epoch_time={epoch_times[-1]:.9e} "
            f"A0_rel={relative_errors[0]:.9e} "
            f"A1_rel={relative_errors[1]:.9e}",
            flush=True,
        )
        if should_stop:
            stopped_on_margin = True
            print(
                f"early_stop=stability_threshold "
                f"threshold={args.stability_stop_threshold:.9e} "
                f"patience={args.stability_stop_patience}",
                flush=True,
            )
            break
    return (backward_times, stage, margin_history, learning_loss_history,
            fit_loss_history, epoch_times, float(trajectory_loss.detach()),
            relative_error_history, float(one_step_loss.detach()),
            float(margin.detach()), stopped_on_margin)


def discrete_state_dimension(model):
    return model.mode_matrices.shape[-1]


def run_method(method, initial, discrete, data, args):
    model = LatentSingleSwitchSystem(
        initial.clone(), discrete.inputs, horizon=args.horizon,
        initial_switch_time=args.initial_switch_time,
    )
    layer, construction = make_layer(
        method, discrete.modes.shape[-1], solver=args.solver, p_floor=args.p_floor
    )
    start = time.perf_counter(); first = layer(model.modes_dict())["t"]; first_forward = time.perf_counter() - start
    # Warm-up must not change the common initialization or enter training statistics.
    layer.forward_times.clear()
    if method in {"ffolayer", "ffolayer_lifted"}:
        layer.layer.reset_timings()
    wall_start = time.perf_counter()
    try:
        (backward, stages, margin_history, learning_loss_history, fit_loss_history,
         epoch_times, trajectory_loss, relative_error_history, one_step_loss, margin,
         stopped_on_margin) = train(
            model, layer, data["states"], data["inputs"], discrete.modes, args
        )
    except Exception as exc:
        training_wall = time.perf_counter() - wall_start
        result = {
            "method": method, "status": "failed", "error": str(exc),
            "construction_s": construction, "first_forward_s": first_forward,
            "first_margin": float(first.detach()), "training_wall_until_failure_s": training_wall,
            "completed_forward_calls": len(layer.forward_times),
            "median_forward_s_until_failure": median(layer.forward_times),
        }
        if method in {"ffolayer", "ffolayer_lifted"}:
            result.update({
                "median_ffo_forward_s_until_failure": median(layer.layer.ffo_forward_times),
                "forward_semantics": "FFOLayer native forward and backward",
                "ffo_auxiliary_proximal_weight": layer.layer.proximal_weight,
                "ffo_alpha": layer.layer.ffo.alpha,
                "ffo_forward_solver_args": serializable_solver_args(layer.layer.ffo._solver_args_fwd),
                "ffo_backward_solver_args": serializable_solver_args(layer.layer.ffo._solver_args_bwd),
            })
        return result
    training_wall = time.perf_counter() - wall_start
    soft_idx = [i for i, value in enumerate(stages) if value == "soft"]
    hard_idx = [i for i, value in enumerate(stages) if value == "hard"]
    result = {
        "method": method, "status": "completed", "construction_s": construction, "first_forward_s": first_forward,
        "formulation_signature": layer.layer.formulation_signature,
        "first_margin": float(first.detach()), "training_wall_s": training_wall,
        "median_forward_s": median(layer.forward_times), "median_backward_s": median(backward),
        "soft_median_forward_s": median([layer.forward_times[i] for i in soft_idx]),
        "soft_median_backward_s": median([backward[i] for i in soft_idx]),
        "hard_median_forward_s": median([layer.forward_times[i] for i in hard_idx]),
        "hard_median_backward_s": median([backward[i] for i in hard_idx]),
        "learned_switch_time": float(model.switch_time.detach()),
        "final_trajectory_loss": trajectory_loss, "final_one_step_loss": one_step_loss,
        "final_margin": margin,
        "stopped_on_stability_threshold": stopped_on_margin,
        "completed_soft_epochs": len(soft_idx),
        "completed_hard_epochs": len(hard_idx),
        "margin_per_epoch": margin_history,
        "learning_loss_per_epoch": learning_loss_history,
        "fit_loss_per_epoch": fit_loss_history,
        "stage_per_epoch": stages,
        "forward_time_per_epoch_s": layer.forward_times,
        "backward_time_per_epoch_s": backward,
        "epoch_time_per_epoch_s": epoch_times,
        "cumulative_epoch_time_s": list(itertools.accumulate(epoch_times)),
        "relative_frobenius_errors_per_epoch": relative_error_history,
        "frobenius_errors": torch.linalg.matrix_norm(model.mode_matrices.detach() - discrete.modes, dim=(-2, -1)).tolist(),
        "relative_frobenius_errors": (
            torch.linalg.matrix_norm(
                model.mode_matrices.detach() - discrete.modes, dim=(-2, -1)
            )
            / torch.linalg.matrix_norm(discrete.modes, dim=(-2, -1))
        ).tolist(),
        "learned_mode_matrices": model.mode_matrices.detach().tolist(),
    }
    if method == "ffolayer_nonlifted":
        result.update({
            "forward_semantics": "FFOLayer native forward and backward",
            "ffo_auxiliary_proximal_weight": 0.0,
            "ffo_alpha": layer.layer.ffo_layer.alpha,
            "ffo_forward_solver_args": serializable_solver_args(
                layer.layer.ffo_layer._solver_args_fwd
            ),
            "ffo_backward_solver_args": serializable_solver_args(
                layer.layer.ffo_layer._solver_args_bwd
            ),
        })
    if method in {"ffolayer", "ffolayer_lifted"}:
        result.update({
            "median_ffo_forward_s": median(layer.layer.ffo_forward_times),
            "forward_semantics": "FFOLayer native forward and backward",
            "ffo_auxiliary_proximal_weight": layer.layer.proximal_weight,
            "ffo_alpha": layer.layer.ffo.alpha,
            "ffo_forward_solver_args": serializable_solver_args(layer.layer.ffo._solver_args_fwd),
            "ffo_backward_solver_args": serializable_solver_args(layer.layer.ffo._solver_args_bwd),
        })
    return result


def run(args):
    global SCS_ARGS
    SCS_ARGS = {"eps": args.scs_eps}
    torch.manual_seed(args.seed)
    if args.benchmark in {"paper-3vehicle", "3vehicle"}:
        continuous = load_paper_three_vehicle_platoon()
    else:
        if args.xml is None:
            raise ValueError("--xml is required for an XML platoon benchmark")
        continuous = load_continuous_platoon(args.xml)
    discrete = discretize_platoon(continuous, args.dt)
    data = generate_learning_data(
        discrete, n_trajectories=args.n_train, horizon=args.horizon, seed=args.seed,
        observation_noise=args.observation_noise,
        relative_observation_noise=args.relative_observation_noise,
        switch_time=args.true_switch_time,
    )
    pooled = fit_pooled_least_squares(
        data["states"], data["inputs"], discrete.inputs, ridge=args.ridge
    )
    initial = split_pooled_initialization(
        pooled,
        perturbation_scale=args.init_perturbation,
        seed=args.seed + 1,
    )
    results = []
    for method in args.methods:
        print(f"running {method}...", flush=True)
        results.append(run_method(method, initial, discrete, data, args))
        printable = {k: v for k, v in results[-1].items() if k != "learned_mode_matrices"}
        print(json.dumps(printable, indent=2), flush=True)
    serialized_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key != "output"
    }
    payload = {
        "common_setup": serialized_args | {
            "solver_args": (SCS_ARGS if args.solver == "SCS"
                            else {"eps": 1e-7})
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["3vehicle", "5vehicle", "10vehicle", "paper-3vehicle", "xml-10vehicle"], default="3vehicle")
    p.add_argument("--xml", type=Path)
    p.add_argument("--methods", nargs="+", choices=[
        "dOPT", "cvxpylayer", "ffolayer_nonlifted", "ffolayer_lifted",
        "dsdp", "ffolayer",
    ], default=["dOPT", "ffolayer_nonlifted", "ffolayer_lifted", "cvxpylayer"])
    p.add_argument("--dt", type=float, default=.1); p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--observation-noise", type=float, default=0.0)
    p.add_argument("--relative-observation-noise", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=30); p.add_argument("--true-switch-time", type=int, default=14)
    p.add_argument("--initial-switch-time", type=float, default=20.3)
    p.add_argument("--ridge", type=float, default=1e-5); p.add_argument("--init-perturbation", type=float, default=1e-5)
    p.add_argument("--soft-epochs", type=int, default=50); p.add_argument("--hard-epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--hard-learning-rate", type=float, default=3e-3)
    p.add_argument("--switch-learning-rate", type=float, default=5e-2)
    p.add_argument("--initial-temperature", type=float, default=3.); p.add_argument("--final-temperature", type=float, default=.25)
    p.add_argument("--normalized-margin-weight", type=float, default=1.0 / 12.0)
    p.add_argument(
        "--stability-stop-threshold", type=float, default=None,
        help="stop hard training after this margin is reached",
    )
    p.add_argument("--stability-stop-patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--solver", choices=["SCS", "MOSEK"], default="SCS")
    p.add_argument("--scs-eps", type=float, default=1e-7)
    p.add_argument("--p-floor", type=float, default=1e-5)
    p.add_argument("--output", type=Path, default=Path("vehicle_platoon_learning/results/layer_comparison.json"))
    return p


if __name__ == "__main__":
    run(parser().parse_args())
