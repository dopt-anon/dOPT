"""Learning a switched linear system with a differentiable stability margin."""

from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Mapping, Optional

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from third_party.FFOLayer.src.ffolayer import FFOLayer
from third_party.FFOLayer.src.ffolayer.ffocp_eq_patch import (
    FFOLayer as PatchedFFOLayer,
)
from torch import nn
from torch.nn import functional as F

from src.dLya import dLyaLayer
from src.lyapunov_problem import build_common_lyapunov_problem
from .lyapunov_block_sdp import build_common_lyapunov_block_sdp


class LearnedSwitchedLinearSystem(nn.Module):
    """Globally shared trainable matrices for a switched linear system."""

    def __init__(self, initial_modes: torch.Tensor):
        super().__init__()
        if initial_modes.ndim != 3:
            raise ValueError(
                "initial_modes must have shape (n_modes, n_state, n_state)"
            )
        if initial_modes.shape[1] != initial_modes.shape[2]:
            raise ValueError("every mode matrix must be square")
        self.mode_matrices = nn.Parameter(initial_modes.clone())

    @property
    def n_modes(self) -> int:
        return self.mode_matrices.shape[0]

    @property
    def n_state(self) -> int:
        return self.mode_matrices.shape[1]

    def modes_dict(self) -> Dict[int, torch.Tensor]:
        return {
            mode_idx: self.mode_matrices[mode_idx]
            for mode_idx in range(self.n_modes)
        }

    def rollout(
        self,
        x0: torch.Tensor,
        mode_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out predicted states, including ``x0`` as the first state.

        ``mode_sequence[k]`` controls the transition from state ``k`` to
        state ``k + 1``. If a data generator stores one mode label per state,
        pass ``mode_sequence[:-1]``.
        """
        if x0.ndim != 2 or x0.shape[1] != self.n_state:
            raise ValueError("x0 must have shape (batch, n_state)")
        if mode_sequence.ndim != 1:
            raise ValueError("mode_sequence must be one-dimensional")
        if mode_sequence.numel() == 0:
            return x0[:, None, :]
        if mode_sequence.min() < 0 or mode_sequence.max() >= self.n_modes:
            raise ValueError("mode_sequence contains an invalid mode index")

        states = [x0]
        state = x0
        for mode_idx in mode_sequence:
            a_i = self.mode_matrices[mode_idx]
            state = state @ a_i.T
            states.append(state)
        return torch.stack(states, dim=1)


class LearnedSingleSwitchSystem(LearnedSwitchedLinearSystem):
    """Two global modes with one shared learnable switch time."""

    def __init__(
        self,
        initial_modes: torch.Tensor,
        *,
        horizon: int,
        initial_switch_time: float,
    ):
        super().__init__(initial_modes)
        if self.n_modes != 2:
            raise ValueError("single-switch learning currently requires 2 modes")
        if horizon < 3:
            raise ValueError("horizon must be at least 3")
        self.horizon = horizon

        # The switch boundary is constrained to (0, horizon - 1).
        normalized_time = torch.as_tensor(
            initial_switch_time / (horizon - 1),
            dtype=initial_modes.dtype,
            device=initial_modes.device,
        ).clamp(1e-4, 1.0 - 1e-4)
        raw_switch_time = torch.logit(normalized_time)
        self.raw_switch_time = nn.Parameter(raw_switch_time)

    @property
    def switch_time(self) -> torch.Tensor:
        return (self.horizon - 1) * torch.sigmoid(self.raw_switch_time)

    def rollout_soft_switch(
        self,
        x0: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        """Roll out using a differentiable transition from mode 0 to mode 1."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if x0.ndim != 2 or x0.shape[1] != self.n_state:
            raise ValueError("x0 must have shape (batch, n_state)")

        states = [x0]
        state = x0
        for time_idx in range(self.horizon - 1):
            time = torch.as_tensor(
                time_idx + 0.5,
                dtype=state.dtype,
                device=state.device,
            )
            mode_one_weight = torch.sigmoid(
                (time - self.switch_time) / temperature
            )
            transition = (
                (1.0 - mode_one_weight) * self.mode_matrices[0]
                + mode_one_weight * self.mode_matrices[1]
            )
            state = state @ transition.T
            states.append(state)
        return torch.stack(states, dim=1)

    def hard_mode_sequence(self) -> torch.Tensor:
        """Return the discrete transition labels induced by learned switch time."""
        transition_indices = torch.arange(
            self.horizon - 1,
            device=self.mode_matrices.device,
            dtype=self.mode_matrices.dtype,
        )
        return (transition_indices + 0.5 >= self.switch_time).to(torch.long)


def _direct_lyapunov_operators(modes: torch.Tensor) -> torch.Tensor:
    """Matrices representing vech(P - A_i.T P A_i) from vech(P)."""
    n = modes.shape[-1]
    rows, cols = torch.triu_indices(n, n, device=modes.device)
    q = len(rows)
    basis = torch.zeros((q, n, n), dtype=modes.dtype, device=modes.device)
    indices = torch.arange(q, device=modes.device)
    basis[indices, rows, cols] = 1.0
    basis[indices, cols, rows] = 1.0
    identity = torch.eye(q, dtype=modes.dtype, device=modes.device)
    return torch.stack([
        identity - (mode.T @ basis @ mode)[:, rows, cols].T
        for mode in modes
    ])


class CommonLyapunovStabilityLayer(nn.Module):
    """Common direct ``(P,t)`` formulation for all differentiation backends."""

    def __init__(
        self,
        n_state: int,
        n_modes: int,
        *,
        t_bound: float = 1.0,
        p_floor: float = 0.0,
        backend: str = "dsdp",
        dsdp_mode: str = "dense",
        dsdp_settings: Optional[dict] = None,
        ffo_solver_args: Optional[dict] = None,
        ffo_backward_solver_args: Optional[dict] = None,
        cvxpy_solver_args: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if n_state < 1 or n_modes < 1:
            raise ValueError("n_state and n_modes must be positive")
        if backend not in {"dsdp", "cvxpylayer", "ffolayer"}:
            raise ValueError("unknown Lyapunov backend")
        if dsdp_mode != "dense":
            raise ValueError("the native Lyapunov layer currently supports dense mode")
        self.n_state, self.n_modes = n_state, n_modes
        self.p_floor, self.backend = p_floor, backend
        q = n_state * (n_state + 1) // 2
        self.formulation_signature = {
            "variables": {"P_symmetric": q, "t_scalar": 1},
            "psd_cone_sizes": (n_state,) * (n_modes + 1),
            "equality_count": 1,
            "explicit_slack_variables": 0,
            "definition_equalities": 0,
        }

        settings = {
            "solver": "SCS",
            "solver_args": {"eps": 1e-7},
            "eps_active": 1e-5,
        }
        if dsdp_settings:
            settings.update(dsdp_settings)
        settings["solver_args"] = dict(settings.get("solver_args", {}))
        problem_data = build_common_lyapunov_problem(
            n_state, n_modes, p_floor
        )
        self.problem_data = problem_data
        if backend == "dsdp":
            self.native_layer = dLyaLayer(
                n_state, n_modes, problem=problem_data, settings=settings
            )
            self.cvxpy_layer = self.ffo_layer = None
            return

        problem = problem_data.problem
        operator_parameters = problem_data.parameters
        P, t = problem_data.variables
        self.operator_parameters = operator_parameters
        self.P_variable, self.t_variable = P, t
        self.cvxpy_solver_args = (
            dict(cvxpy_solver_args) if cvxpy_solver_args is not None
            else {"solve_method": "SCS", "eps": 1e-7}
        )
        if backend == "cvxpylayer":
            self.cvxpy_layer = CvxpyLayer(
                problem, parameters=operator_parameters, variables=[P, t]
            )
            self.ffo_layer = None
        else:
            self.ffo_layer = PatchedFFOLayer(
                problem,
                parameters=operator_parameters,
                variables=[P, t],
                eps=1e-7,
                backward_eps=1e-7,
            )
            self.cvxpy_layer = None
            self.ffo_solver_args = (
                dict(ffo_solver_args) if ffo_solver_args is not None
                else {"solver": cp.MOSEK, "eps": 1e-7}
            )
            self.ffo_backward_solver_args = (
                dict(ffo_backward_solver_args)
                if ffo_backward_solver_args is not None
                else dict(self.ffo_solver_args)
            )

    def forward(self, modes: Mapping[int, torch.Tensor]) -> Dict[str, object]:
        if len(modes) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} modes, got {len(modes)}")
        stacked_modes = torch.stack([modes[key] for key in sorted(modes)])
        if self.backend == "dsdp":
            result = self.native_layer(stacked_modes)
            return result | {"Z": None, "dual_psd": None, "dual_equalities": None}

        operators = tuple(_direct_lyapunov_operators(stacked_modes))
        if self.backend == "cvxpylayer":
            P, t = self.cvxpy_layer(
                *operators, solver_args=self.cvxpy_solver_args
            )
        else:
            batched = [operator.unsqueeze(0) for operator in operators]
            P, t = self.ffo_layer(
                *batched, solver_args=self.ffo_solver_args
            )
            # FFOLayer carries an SCS-only default into its perturbed solve;
            # MOSEK does not accept this option.
            if self.ffo_backward_solver_args.get("solver") == cp.MOSEK:
                self.ffo_layer._solver_args_bwd.pop("max_iters", None)
            self.ffo_layer._solver_args_bwd.update(
                self.ffo_backward_solver_args
            )
            if P.ndim == 3:
                P = P[0]
            if t.ndim:
                t = t.reshape(-1)[0]
        return {"P": P, "t": t, "Z": None,
                "dual_psd": None, "dual_equalities": None}


class LiftedFFOLyapunovStabilityLayer(nn.Module):
    """FFOLayer applied to the explicit-slack block lift of the same SDP."""

    def __init__(
        self,
        n_state: int,
        n_modes: int,
        *,
        t_bound: float = 1.0,
        p_floor: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_state, self.n_modes = n_state, n_modes
        self.t_bound, self.p_floor = t_bound, p_floor
        dummy = {
            index: torch.zeros(
                (n_state, n_state), dtype=dtype, device=device
            )
            for index in range(n_modes)
        }
        _, equality_matrix, rhs, layout = build_common_lyapunov_block_sdp(
            dummy, t_bound=t_bound, p_floor=p_floor
        )
        block_slices = [
            layout.p_slice,
            *(layout.slack_slices[key] for key in layout.mode_keys),
            layout.t_slice,
        ]
        block_columns = torch.as_tensor([
            row + col * layout.matrix_size
            for block in block_slices
            for col in range(block.start, block.stop)
            for row in range(block.start, block.stop)
        ], dtype=torch.long, device=device)
        self.register_buffer("block_columns", block_columns)

        variables = [
            cp.Variable(
                (block.stop - block.start, block.stop - block.start),
                symmetric=True,
            )
            for block in block_slices
        ]
        objective_parameter = cp.Parameter(len(block_columns))
        equality_parameter = cp.Parameter(
            (equality_matrix.shape[0], len(block_columns))
        )
        rhs_parameter = cp.Parameter(rhs.shape)
        block_vector = cp.hstack([
            cp.vec(variable, order="F") for variable in variables
        ])
        problem = cp.Problem(
            cp.Minimize(objective_parameter @ block_vector),
            [
                *(variable >> 0 for variable in variables),
                equality_parameter @ block_vector == rhs_parameter,
            ],
        )
        if not problem.is_dpp():
            raise RuntimeError("lifted FFOLayer formulation must be DPP")
        self.ffo_layer = FFOLayer(
            problem,
            parameters=[
                objective_parameter, equality_parameter, rhs_parameter
            ],
            variables=variables,
            eps=1e-7,
            backward_eps=1e-7,
        )
        self.solver_args = {"solver": cp.MOSEK, "eps": 1e-7}
        self.formulation_signature = {
            "variables": {
                "P_symmetric": n_state * (n_state + 1) // 2,
                "slack_symmetric": (
                    n_modes * n_state * (n_state + 1) // 2
                ),
                "t_embedding_symmetric": 3,
            },
            "psd_cone_sizes": (
                *((n_state,) * (n_modes + 1)), 2
            ),
            "equality_count": equality_matrix.shape[0],
            "explicit_slack_variables": n_modes,
            "definition_equalities": (
                n_modes * n_state * (n_state + 1) // 2
            ),
        }

    def forward(self, modes: Mapping[int, torch.Tensor]) -> Dict[str, object]:
        if len(modes) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} modes, got {len(modes)}")
        objective, equality_matrix, rhs, _ = build_common_lyapunov_block_sdp(
            modes, t_bound=self.t_bound, p_floor=self.p_floor
        )
        objective = objective.T.reshape(-1)[self.block_columns]
        equality_matrix = equality_matrix[:, self.block_columns]
        blocks = self.ffo_layer(
            objective.unsqueeze(0),
            equality_matrix.unsqueeze(0),
            rhs.unsqueeze(0),
            solver_args=self.solver_args,
        )
        self.ffo_layer._solver_args_bwd.pop("max_iters", None)
        self.ffo_layer._solver_args_bwd.update(self.solver_args)
        blocks = tuple(
            block[0] if block.ndim == 3 and block.shape[0] == 1 else block
            for block in blocks
        )
        t_block = blocks[-1]
        return {
            "P": blocks[0] + self.p_floor * torch.eye(
                self.n_state, dtype=blocks[0].dtype, device=blocks[0].device
            ),
            "t": t_block[0, 1],
            "Z": None,
            "dual_psd": None,
            "dual_equalities": None,
        }


def fit_modes_least_squares(
    trajectories: torch.Tensor,
    mode_sequence: torch.Tensor,
    n_modes: int,
    *,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Fit each mode independently from one-step state pairs."""
    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape (N, T, n_state)")
    if mode_sequence.ndim != 1:
        raise ValueError("mode_sequence must be one-dimensional")
    if mode_sequence.numel() < trajectories.shape[1] - 1:
        raise ValueError("mode_sequence does not cover every transition")

    n_state = trajectories.shape[2]
    identity = torch.eye(
        n_state,
        dtype=trajectories.dtype,
        device=trajectories.device,
    )
    estimates = []
    transition_modes = mode_sequence[: trajectories.shape[1] - 1]
    for mode_idx in range(n_modes):
        time_indices = torch.nonzero(
            transition_modes == mode_idx,
            as_tuple=True,
        )[0]
        if time_indices.numel() == 0:
            raise ValueError(f"mode {mode_idx} has no observed transitions")

        x_current = trajectories[:, time_indices, :].reshape(-1, n_state)
        x_next = trajectories[:, time_indices + 1, :].reshape(-1, n_state)
        gram = x_current.T @ x_current + ridge * identity
        # Solve X.T X B = X.T Y, where B = A.T.
        a_transpose = torch.linalg.solve(gram, x_current.T @ x_next)
        estimates.append(a_transpose.T)
    return torch.stack(estimates)


def trajectory_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared multi-step rollout error, excluding the shared x0."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target trajectories must have equal shape")
    return F.mse_loss(predicted[:, 1:, :], target[:, 1:, :])


def stability_hinge_loss(
    t_star: torch.Tensor,
    *,
    target_margin: float = 1e-3,
) -> torch.Tensor:
    """Zero penalty above the desired stability margin."""
    target = torch.as_tensor(
        target_margin,
        dtype=t_star.dtype,
        device=t_star.device,
    )
    return F.relu(target - t_star).square()


@dataclass
class TrainingRecord:
    epoch: int
    total_loss: float
    trajectory_loss: float
    stability_loss: float
    stability_margin: float
    stability_forward_time: float
    backward_time: float


@dataclass
class SingleSwitchTrainingRecord(TrainingRecord):
    switch_time: float
    temperature: float


def train_switched_linear_system(
    model: LearnedSwitchedLinearSystem,
    stability_layer: CommonLyapunovStabilityLayer,
    trajectories: torch.Tensor,
    mode_sequence: torch.Tensor,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-2,
    stability_weight: float = 10.0,
    target_margin: float = 1e-3,
    stability_loss_type: str = "hinge",
    normalized_margin_weight: float = 1.0 / 12.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> List[TrainingRecord]:
    """Train mode matrices using trajectory and dSDP stability losses."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    transition_modes = mode_sequence[: trajectories.shape[1] - 1]
    records = []
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        optimizer.zero_grad()
        predicted = model.rollout(trajectories[:, 0, :], transition_modes)
        loss_traj = trajectory_loss(predicted, trajectories)

        forward_start = time.perf_counter()
        stability_result = stability_layer(model.modes_dict())
        stability_forward_time = time.perf_counter() - forward_start
        t_star = stability_result["t"]
        if stability_loss_type == "hinge":
            loss_stability = stability_hinge_loss(
                t_star,
                target_margin=target_margin,
            )
            total_loss = loss_traj + stability_weight * loss_stability
        elif stability_loss_type == "normalized_linear":
            loss_stability = -model.n_state * t_star
            total_loss = (
                loss_traj + normalized_margin_weight * loss_stability
            )
        else:
            raise ValueError(
                "stability_loss_type must be 'hinge' or 'normalized_linear'"
            )
        backward_start = time.perf_counter()
        total_loss.backward()
        backward_time = time.perf_counter() - backward_start
        optimizer.step()

        record = TrainingRecord(
            epoch=epoch,
            total_loss=float(total_loss.detach()),
            trajectory_loss=float(loss_traj.detach()),
            stability_loss=float(loss_stability.detach()),
            stability_margin=float(t_star.detach()),
            stability_forward_time=stability_forward_time,
            backward_time=backward_time,
        )
        records.append(record)
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"loss={record.total_loss:.6g} "
            f"trajectory_loss={record.trajectory_loss:.6g} "
            f"t_star={record.stability_margin:.6g} "
            f"forward_s={record.stability_forward_time:.3f} "
            f"backward_s={record.backward_time:.3f} "
            f"epoch_s={time.perf_counter() - epoch_start:.3f}",
            flush=True,
        )
    return records


def train_trajectory_only(
    model: LearnedSwitchedLinearSystem,
    trajectories: torch.Tensor,
    mode_sequence: torch.Tensor,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-2,
    optimizer: Optional[torch.optim.Optimizer] = None,
    margin_evaluator: Optional[
        Callable[[Mapping[int, torch.Tensor]], float]
    ] = None,
) -> List[TrainingRecord]:
    """Train the same model without the common-Lyapunov regularizer."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    transition_modes = mode_sequence[: trajectories.shape[1] - 1]
    records = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted = model.rollout(trajectories[:, 0, :], transition_modes)
        loss_traj = trajectory_loss(predicted, trajectories)
        loss_traj.backward()
        if margin_evaluator is None:
            stability_margin = float("nan")
        else:
            stability_margin = margin_evaluator(model.modes_dict())
        optimizer.step()
        records.append(
            TrainingRecord(
                epoch=epoch,
                total_loss=float(loss_traj.detach()),
                trajectory_loss=float(loss_traj.detach()),
                stability_loss=0.0,
                stability_margin=stability_margin,
                stability_forward_time=0.0,
                backward_time=0.0,
            )
        )
    return records


def train_single_switch_system(
    model: LearnedSingleSwitchSystem,
    trajectories: torch.Tensor,
    *,
    epochs: int = 60,
    learning_rate: float = 5e-3,
    switch_time_learning_rate: Optional[float] = None,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    stability_layer: Optional[CommonLyapunovStabilityLayer] = None,
    margin_evaluator: Optional[
        Callable[[Mapping[int, torch.Tensor]], float]
    ] = None,
    stability_weight: float = 50.0,
    target_margin: float = 1e-3,
) -> List[SingleSwitchTrainingRecord]:
    """Jointly learn two mode matrices and their shared single switch time."""
    if trajectories.shape[1] != model.horizon:
        raise ValueError("trajectory horizon does not match model horizon")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if temperature_start <= 0 or temperature_end <= 0:
        raise ValueError("temperatures must be positive")

    if switch_time_learning_rate is None:
        switch_time_learning_rate = learning_rate
    optimizer = torch.optim.Adam(
        [
            {"params": [model.mode_matrices], "lr": learning_rate},
            {
                "params": [model.raw_switch_time],
                "lr": switch_time_learning_rate,
            },
        ]
    )
    records = []
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        temperature = temperature_start * (
            temperature_end / temperature_start
        ) ** progress

        optimizer.zero_grad()
        predicted = model.rollout_soft_switch(
            trajectories[:, 0, :],
            temperature=temperature,
        )
        loss_traj = trajectory_loss(predicted, trajectories)

        if stability_layer is None:
            if margin_evaluator is None:
                margin_value = float("nan")
            else:
                margin_value = margin_evaluator(model.modes_dict())
            t_star = torch.as_tensor(
                margin_value,
                dtype=trajectories.dtype,
                device=trajectories.device,
            )
            loss_stability = torch.zeros_like(loss_traj)
        else:
            stability_result = stability_layer(model.modes_dict())
            t_star = stability_result["t"]
            loss_stability = stability_hinge_loss(
                t_star,
                target_margin=target_margin,
            )

        total_loss = loss_traj + stability_weight * loss_stability
        total_loss.backward()
        optimizer.step()
        records.append(
            SingleSwitchTrainingRecord(
                epoch=epoch,
                total_loss=float(total_loss.detach()),
                trajectory_loss=float(loss_traj.detach()),
                stability_loss=float(loss_stability.detach()),
                stability_margin=float(t_star.detach()),
                stability_forward_time=0.0,
                backward_time=0.0,
                switch_time=float(model.switch_time.detach()),
                temperature=float(temperature),
            )
        )
    return records
