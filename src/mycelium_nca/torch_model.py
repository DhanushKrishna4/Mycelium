"""Differentiable BHWC transition, matching numpy_sim without a target lookup."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ALPHA_THRESHOLD, WEIGHT_SHAPES


def initial_weights(seed=42):
    rng = np.random.Generator(np.random.PCG64(seed))
    limit = math.sqrt(6 / (48 + 64))
    return {"W1": rng.uniform(-limit, limit, (48, 64)).astype(np.float32),
            "b1": np.zeros(64, np.float32), "W2": np.zeros((64, 16), np.float32),
            "b2": np.zeros(16, np.float32)}


def perceive(state):
    p = F.pad(state, (0, 0, 1, 1, 1, 1))
    gx = (p[:, :-2, 2:] - p[:, :-2, :-2]
          + 2 * (p[:, 1:-1, 2:] - p[:, 1:-1, :-2])
          + p[:, 2:, 2:] - p[:, 2:, :-2]) * 0.125
    gy = (p[:, 2:, :-2] - p[:, :-2, :-2]
          + 2 * (p[:, 2:, 1:-1] - p[:, :-2, 1:-1])
          + p[:, 2:, 2:] - p[:, :-2, 2:]) * 0.125
    return torch.cat((state, gx, gy), dim=-1)


def living(state):
    alpha = state[..., 3].unsqueeze(1)
    return F.max_pool2d(alpha, 3, stride=1, padding=1).squeeze(1) > ALPHA_THRESHOLD


class NCA(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        for key, value in initial_weights(seed).items():
            self.register_parameter(key, nn.Parameter(torch.from_numpy(value)))

    def perceive(self, state):
        return perceive(state)

    def network_delta(self, state):
        return F.relu(perceive(state) @ self.W1 + self.b1) @ self.W2 + self.b2

    def forward(self, state, mask, lesion=None):
        if lesion is not None:
            state = state.masked_fill(lesion[..., None], 0)
        pre_alive = living(state)
        if mask.ndim == state.ndim - 1:
            mask = mask.unsqueeze(-1)
        if mask.shape != (*state.shape[:-1], 1):
            raise ValueError("Mask must contain one value per cell, shared by all channels.")
        candidate = state + self.network_delta(state) * mask
        result = candidate * (pre_alive & living(candidate))[..., None]
        return result.masked_fill(lesion[..., None], 0) if lesion is not None else result

    def weights_numpy(self):
        return {k: getattr(self, k).detach().cpu().numpy().copy() for k in WEIGHT_SHAPES}

    def load_weights(self, weights):
        with torch.no_grad():
            for key, shape in WEIGHT_SHAPES.items():
                if tuple(weights[key].shape) != shape:
                    raise ValueError(f"Invalid {key} shape; expected {shape}.")
                param = getattr(self, key)
                param.copy_(torch.as_tensor(weights[key], dtype=param.dtype, device=param.device))
        return self
