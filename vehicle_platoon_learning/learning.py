"""Latent-switch identification with known input matrices."""
from __future__ import annotations
import torch
from torch import nn

def fit_pooled_least_squares(states, inputs, known_input_matrices, *, ridge=1e-6):
    """Fit one A without using switch labels."""
    n = states.shape[-1]
    x = states[:, :-1].reshape(-1, n)
    y = states[:, 1:].reshape(-1, n)
    u = inputs.reshape(-1, inputs.shape[-1])
    y = y - u @ known_input_matrices.mean(dim=0).T
    eye = torch.eye(n, dtype=states.dtype, device=states.device)
    return torch.linalg.solve(x.T @ x + ridge * eye, x.T @ y).T

def split_pooled_initialization(pooled, *, perturbation_scale, seed):
    """Create two initial modes without consulting either GT mode."""
    generator = torch.Generator(device=pooled.device).manual_seed(seed)
    delta = torch.randn(pooled.shape, generator=generator, dtype=pooled.dtype, device=pooled.device)
    delta = perturbation_scale * torch.linalg.matrix_norm(pooled) * delta / torch.linalg.matrix_norm(delta)
    return torch.stack((pooled - delta, pooled + delta))


class LatentSingleSwitchSystem(nn.Module):
    def __init__(self, initial_modes, known_inputs, *, horizon, initial_switch_time):
        super().__init__()
        self.mode_matrices = nn.Parameter(initial_modes.clone())
        self.register_buffer("input_matrices", known_inputs.clone())
        self.horizon = horizon
        fraction = torch.tensor(initial_switch_time / (horizon - 1), dtype=initial_modes.dtype).clamp(1e-5, 1 - 1e-5)
        self.raw_switch_fraction = nn.Parameter(torch.logit(fraction))

    @property
    def switch_time(self):
        return (self.horizon - 1) * torch.sigmoid(self.raw_switch_fraction)

    def modes_dict(self):
        return {i: matrix for i, matrix in enumerate(self.mode_matrices)}

    def switch_probabilities(self, *, temperature):
        times = torch.arange(self.horizon - 1, dtype=self.mode_matrices.dtype, device=self.mode_matrices.device)
        return torch.sigmoid((times - self.switch_time) / temperature)

    def hard_mode_sequence(self):
        times = torch.arange(self.horizon - 1, device=self.mode_matrices.device)
        return (times >= self.switch_time).long()

    def rollout(self, initial_states, inputs, *, temperature):
        states, state = [initial_states], initial_states
        for t, probability in enumerate(self.switch_probabilities(temperature=temperature)):
            transition = (1 - probability) * self.mode_matrices[0] + probability * self.mode_matrices[1]
            input_matrix = (1 - probability) * self.input_matrices[0] + probability * self.input_matrices[1]
            state = state @ transition.T + inputs[:, t] @ input_matrix.T
            states.append(state)
        return torch.stack(states, dim=1)
