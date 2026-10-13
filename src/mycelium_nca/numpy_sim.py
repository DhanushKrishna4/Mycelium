"""Vectorized HWC float32 inference. No target or training dependency exists here."""

import numpy as np

from .config import ALPHA_THRESHOLD, CHANNELS, GRID_SIZE, MAX_GRID_SIZE, MAX_STATE_ABS


def seed_state(size=GRID_SIZE):
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_GRID_SIZE:
        raise ValueError(f"Grid size must be an integer in [1, {MAX_GRID_SIZE}].")
    state = np.zeros((size, size, CHANNELS), dtype=np.float32)
    state[size // 2, size // 2, 3:] = 1.0
    return state


def perceive(state):
    """[state, horizontal Sobel, vertical Sobel], cross correlation, zero pad."""
    p = np.pad(state, ((1, 1), (1, 1), (0, 0)))
    gx = (p[:-2, 2:] - p[:-2, :-2] + 2 * (p[1:-1, 2:] - p[1:-1, :-2])
          + p[2:, 2:] - p[2:, :-2]) * np.float32(0.125)
    gy = (p[2:, :-2] - p[:-2, :-2] + 2 * (p[2:, 1:-1] - p[:-2, 1:-1])
          + p[2:, 2:] - p[:-2, 2:]) * np.float32(0.125)
    return np.concatenate((state, gx, gy), axis=-1)


def living(state):
    """Whether any cell in the 3x3 neighborhood has alpha strictly above 0.1."""
    a = np.pad(state[..., 3] > ALPHA_THRESHOLD, 1, constant_values=False)
    return (a[:-2, :-2] | a[:-2, 1:-1] | a[:-2, 2:]
            | a[1:-1, :-2] | a[1:-1, 1:-1] | a[1:-1, 2:]
            | a[2:, :-2] | a[2:, 1:-1] | a[2:, 2:])


def network_delta(state, weights):
    h = np.maximum(perceive(state) @ weights["W1"] + weights["b1"], 0)
    return h @ weights["W2"] + weights["b2"]


def step(state, weights, mask, lesion=None, return_delta=False):
    """Apply one explicit-mask transition; input arrays are never mutated.

    The optional returned delta is the actual state difference, including living
    and lesion constraints. A skipped learned update does not inhibit those rules.
    """
    if lesion is not None:
        state = np.where(np.asarray(lesion, dtype=bool)[..., None], 0, state)
    pre_alive = living(state)
    mask = np.asarray(mask)
    if mask.shape == state.shape[:2]:
        mask = mask[..., None]
    if mask.shape != (*state.shape[:2], 1):
        raise ValueError("Update mask must have shape (height,width) or (height,width,1).")
    candidate = state + network_delta(state, weights) * mask
    result = candidate * (pre_alive & living(candidate))[..., None]
    if lesion is not None:
        result = np.where(np.asarray(lesion, dtype=bool)[..., None], 0, result)
    result = np.asarray(result, dtype=np.float32)
    return (result, result - state) if return_delta else result


def check_numerical(state, max_abs=MAX_STATE_ABS):
    if not np.isfinite(state).all():
        raise FloatingPointError("Numerical failure: the cell state contains nonfinite values.")
    if np.max(np.abs(state), initial=0) > max_abs:
        raise FloatingPointError(f"Numerical failure: a cell state exceeds {max_abs:g}; simulation stopped.")


def render_rgba(state):
    """Convert learned premultiplied RGBA to straight-alpha bytes for Canvas/PNG."""
    alpha = np.clip(state[..., 3:4], 0, 1)
    rgb = np.divide(state[..., :3], alpha, out=np.zeros_like(state[..., :3]), where=alpha > 0)
    rgba = np.concatenate((np.clip(rgb, 0, 1), alpha), axis=-1)
    return np.rint(rgba * 255).astype(np.uint8)
