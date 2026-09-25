"""Load, extend, discretize, and simulate the ARCH/NFM platoon model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import torch


@dataclass(frozen=True)
class ContinuousPlatoon:
    connected: torch.Tensor
    disconnected: torch.Tensor
    leader_input: torch.Tensor
    state_names: tuple[str, ...]


@dataclass(frozen=True)
class DiscretePlatoon:
    modes: torch.Tensor
    inputs: torch.Tensor
    state_names: tuple[str, ...]
    dt: float


def load_paper_three_vehicle_platoon() -> ContinuousPlatoon:
    """Exact 9-state matrices printed in the ARCH15 benchmark paper."""
    connected = torch.tensor([
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0],
        [1.6050, 4.8680, -3.5754, -0.8198, 0.4270, -0.0450, -0.1942, 0.3626, -0.0946],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0, 0, 0],
        [0.8718, 3.8140, -0.0754, 1.1936, 3.6258, -3.2396, -0.5950, 0.1294, -0.0796],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, -1],
        [0.7132, 3.5730, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.0720, -3.1356],
    ], dtype=torch.float64)
    disconnected = torch.tensor([
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0],
        [1.6050, 4.8680, -3.5754, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 1.1936, 3.6258, -3.2396, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, -1],
        [0.7132, 3.5730, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.0720, -3.1356],
    ], dtype=torch.float64)
    leader_input = torch.zeros((9, 1), dtype=torch.float64)
    leader_input[1, 0] = 1.0
    names = tuple(name for i in range(1, 4) for name in (f"e{i}", f"e{i}prime", f"a{i}"))
    return ContinuousPlatoon(connected, disconnected, leader_input, names)


def _parse_linear_expression(expression: str, state_names: tuple[str, ...]) -> torch.Tensor:
    """Parse the simple signed sums used by the SpaceEx benchmark."""
    coefficients = torch.zeros(len(state_names), dtype=torch.float64)
    compact = re.sub(r"\s+", "", expression).replace("-", "+-")
    name_to_index = {name: index for index, name in enumerate(state_names)}
    for term in compact.split("+"):
        if not term:
            continue
        match = re.fullmatch(
            r"(?:(?P<coefficient>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\*)?"
            r"(?P<variable>[A-Za-z_]\w*)",
            term,
        )
        if match is None:
            raise ValueError(f"unsupported linear term {term!r} in {expression!r}")
        variable = match.group("variable")
        if variable not in name_to_index:
            # The archived XML declares an unused auxiliary variable b. It is
            # intentionally excluded from the physical 3N-state model.
            continue
        raw_coefficient = match.group("coefficient")
        coefficient = 1.0 if raw_coefficient is None else float(raw_coefficient)
        coefficients[name_to_index[variable]] += coefficient
    return coefficients


def load_connected_matrix_from_spaceex(xml_path: str | Path) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Extract the physical 3N-by-3N connected matrix from the SpaceEx XML."""
    root = ET.parse(xml_path).getroot()
    namespace = {"sx": "http://www-verimag.imag.fr/xml-namespaces/sspaceex"}
    parameters = [element.attrib["name"] for element in root.findall(".//sx:param", namespace)]
    state_names = tuple(name for name in parameters if name != "b")
    if len(state_names) % 3 != 0:
        raise ValueError("expected three physical states per controlled vehicle")
    flow = root.find(".//sx:location/sx:flow", namespace)
    if flow is None or flow.text is None:
        raise ValueError("SpaceEx file contains no location flow")

    equations = {}
    for equation in flow.text.split("&"):
        equation = equation.strip()
        if not equation:
            continue
        lhs, rhs = equation.split("==", maxsplit=1)
        derivative = lhs.strip()
        if derivative.endswith("'"):
            equations[derivative[:-1]] = rhs.strip()

    missing = set(state_names) - set(equations)
    if missing:
        raise ValueError(f"missing derivatives for {sorted(missing)}")
    matrix = torch.stack([
        _parse_linear_expression(equations[name], state_names)
        for name in state_names
    ])
    return matrix, state_names


def build_disconnected_matrix(
    connected: torch.Tensor,
) -> torch.Tensor:
    """Extend the ARCH communication-loss sparsification pattern.

    Vehicles 1,...,N-1 retain only their local feedback blocks, while the
    final vehicle's acceleration row remains dense, as in the 3-vehicle case.
    """
    if connected.ndim != 2 or connected.shape[0] != connected.shape[1]:
        raise ValueError("connected must be square")
    if connected.shape[0] % 3 != 0:
        raise ValueError("state dimension must be divisible by three")
    disconnected = connected.clone()
    n_vehicles = connected.shape[0] // 3
    for vehicle in range(n_vehicles - 1):
        acceleration_row = 3 * vehicle + 2
        own_block = slice(3 * vehicle, 3 * vehicle + 3)
        local_coefficients = connected[acceleration_row, own_block].clone()
        disconnected[acceleration_row, :] = 0.0
        disconnected[acceleration_row, own_block] = local_coefficients
    return disconnected


def load_continuous_platoon(
    xml_path: str | Path,
) -> ContinuousPlatoon:
    connected, state_names = load_connected_matrix_from_spaceex(xml_path)
    disconnected = build_disconnected_matrix(connected)
    leader_input = torch.zeros((connected.shape[0], 1), dtype=connected.dtype)
    leader_input[1, 0] = 1.0  # e1prime' = a_leader - a1
    return ContinuousPlatoon(connected, disconnected, leader_input, state_names)


def discretize_platoon(system: ContinuousPlatoon, dt: float) -> DiscretePlatoon:
    """Exact zero-order-hold discretization via an augmented matrix exponential."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    discrete_modes = []
    discrete_inputs = []
    for continuous_mode in (system.connected, system.disconnected):
        n_state, n_input = system.leader_input.shape
        augmented = torch.zeros(
            (n_state + n_input, n_state + n_input), dtype=continuous_mode.dtype
        )
        augmented[:n_state, :n_state] = continuous_mode
        augmented[:n_state, n_state:] = system.leader_input
        exponential = torch.matrix_exp(dt * augmented)
        discrete_modes.append(exponential[:n_state, :n_state])
        discrete_inputs.append(exponential[:n_state, n_state:])
    return DiscretePlatoon(
        modes=torch.stack(discrete_modes),
        inputs=torch.stack(discrete_inputs),
        state_names=system.state_names,
        dt=dt,
    )


def generate_learning_data(
    system: DiscretePlatoon,
    *,
    n_trajectories: int,
    horizon: int,
    seed: int,
    observation_noise: float = 0.0,
    relative_observation_noise: float = 0.0,
    switch_time: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate trajectories sharing one switch; labels are evaluation-only."""
    if n_trajectories < 1 or horizon < 2:
        raise ValueError("n_trajectories must be positive and horizon at least two")
    generator = torch.Generator().manual_seed(seed)
    n_state = system.modes.shape[-1]
    states = torch.empty(
        (n_trajectories, horizon, n_state), dtype=system.modes.dtype
    )
    inputs = torch.empty(
        (n_trajectories, horizon - 1, 1), dtype=system.modes.dtype
    )
    if switch_time is None:
        switch_time = (horizon - 1) // 2
    if not 1 <= switch_time < horizon - 1:
        raise ValueError("switch_time must lie strictly inside the transition horizon")
    ground_truth_modes = (torch.arange(horizon - 1) >= switch_time).long()
    states[:, 0, :] = 0.35 * torch.randn(
        (n_trajectories, n_state), generator=generator, dtype=system.modes.dtype
    )

    for trajectory in range(n_trajectories):
        # Piecewise-constant excitation is closer to a leader acceleration
        # command than independent white noise and still identifies B/A well.
        input_value = torch.empty((), dtype=system.modes.dtype).uniform_(
            -2.0, 1.0, generator=generator
        )
        for time_index in range(horizon - 1):
            if time_index % 5 == 0:
                input_value = torch.empty((), dtype=system.modes.dtype).uniform_(
                    -2.0, 1.0, generator=generator
                )
            mode = int(ground_truth_modes[time_index])
            inputs[trajectory, time_index, 0] = input_value
            states[trajectory, time_index + 1] = (
                system.modes[mode] @ states[trajectory, time_index]
                + system.inputs[mode, :, 0] * input_value
            )

    clean_states = states.clone()
    if observation_noise < 0 or relative_observation_noise < 0:
        raise ValueError("noise levels must be nonnegative")
    noise = torch.randn(states.shape, generator=generator, dtype=states.dtype)
    if observation_noise > 0:
        states += observation_noise * noise
    if relative_observation_noise > 0:
        state_scale = clean_states.std(dim=(0, 1), unbiased=False)
        states += relative_observation_noise * state_scale.view(1, 1, -1) * noise
    return {
        "states": states,
        "clean_states": clean_states,
        "inputs": inputs,
        # Never pass this field to initialization or the training loss.
        "ground_truth_modes": ground_truth_modes,
    }
