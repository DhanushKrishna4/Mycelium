"""Deterministic interventions applied before the next stochastic transition."""

import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def region_mask(shape, event):
    h, w = shape
    x = _integer(event.get("x", w // 2), "x", 0, w - 1)
    y = _integer(event.get("y", h // 2), "y", 0, h - 1)
    yy, xx = np.ogrid[:h, :w]
    if event.get("shape", "disk") == "rectangle":
        rw = _integer(event.get("width", 1), "width", 1, w)
        rh = _integer(event.get("height", 1), "height", 1, h)
        return (xx >= x) & (xx < x + rw) & (yy >= y) & (yy < y + rh)
    if event.get("shape", "disk") != "disk":
        raise ValueError("Supported intervention shapes are disk and rectangle.")
    radius = _integer(event.get("radius", 3), "radius", 0, max(h, w))
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2


def apply_event(state, lesions, event):
    """Return independent state/mask arrays; persistent lesions win over every event."""
    if not isinstance(event, dict):
        raise ValueError("An intervention must be a JSON object.")
    state, lesions = state.copy(), np.asarray(lesions, dtype=bool).copy()
    kind = event.get("type")
    if kind == "clear":
        state.fill(0)
    elif kind == "clear_lesions":
        lesions.fill(False)
    elif kind == "seed":
        h, w = state.shape[:2]
        x = _integer(event.get("x", w // 2), "x", 0, w - 1)
        y = _integer(event.get("y", h // 2), "y", 0, h - 1)
        state[y, x, :3] = 0
        state[y, x, 3:] = 1
    elif kind in ("erase", "lesion", "noise", "hidden_ablation"):
        region = region_mask(state.shape[:2], event)
        if kind == "erase":
            state[region] = 0
        elif kind == "lesion":
            lesions[region] = True
        elif kind == "hidden_ablation":
            channels = event.get("channels", list(range(4, 16)))
            if not isinstance(channels, list) or not 1 <= len(channels) <= 12:
                raise ValueError("Hidden ablation requires 1 to 12 channel indices.")
            for channel in channels:
                channel = _integer(channel, "channel", 4, 15)
                state[..., channel][region] = 0
        else:
            amplitude = event.get("amplitude", 0.1)
            if isinstance(amplitude, bool) or not isinstance(amplitude, (float, int)) or not 0 <= amplitude <= 1:
                raise ValueError("Noise amplitude must be finite and between zero and one.")
            channels = event.get("channels", list(range(16)))
            if not isinstance(channels, list) or not 1 <= len(channels) <= 16 or len(set(channels)) != len(channels):
                raise ValueError("Noise requires 1 to 16 unique channel indices.")
            channels = [_integer(c, "channel", 0, 15) for c in channels]
            seed = _integer(event.get("seed", 0), "noise seed", 0, 2**32 - 1)
            rng = np.random.Generator(np.random.PCG64(seed))
            # Draw the entire grid in documented HWC C-order, then select the region.
            noise = (rng.random((*state.shape[:2], len(channels)), dtype=np.float32) * 2 - 1) * amplitude
            for i, channel in enumerate(channels):
                state[..., channel][region] += noise[..., i][region]
    else:
        raise ValueError(f"Unknown intervention type: {kind!r}.")
    state[lesions] = 0
    return state, lesions
