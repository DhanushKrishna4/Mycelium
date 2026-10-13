"""Measurements are observers, never inputs to the learned transition."""
from __future__ import annotations

import numpy as np

QUALITY = {"maximum_relative_error": 0.35, "minimum_iou": 0.65}


def measure(state, target, alpha_threshold=0.1):
    state = np.asarray(state, dtype=np.float32)
    visible = state[..., :4]
    target = np.asarray(target, dtype=np.float32)
    if visible.shape != target.shape or target.ndim != 3 or target.shape[-1] != 4:
        raise ValueError("Measurement needs matching H×W×4 visible states")
    if not np.isfinite(state).all() or not np.isfinite(target).all():
        raise ValueError("Cannot measure nonfinite states")
    foreground = target[..., 3] > alpha_threshold
    predicted = visible[..., 3] > alpha_threshold
    cell_error = np.mean((visible.astype(np.float64) - target) ** 2, axis=-1)
    empty_error = np.mean(target.astype(np.float64) ** 2, axis=-1)
    def balanced(error):
        groups = [float(error[mask].mean()) for mask in (foreground, ~foreground) if mask.any()]
        return float(np.mean(groups))
    loss = balanced(cell_error)
    baseline = balanced(empty_error)
    union = int((predicted | foreground).sum())
    return {
        "raw_mse": float(cell_error.mean()),
        "balanced_mse": loss,
        "empty_baseline": baseline,
        "relative_error": loss / baseline if baseline > 0 else (0.0 if loss == 0 else None),
        "iou": float((predicted & foreground).sum() / union) if union else 1.0,
        "occupied": int(predicted.sum()),
        "max_abs_state": float(np.max(np.abs(state))),
    }


def acceptable(metrics, quality=None):
    q = quality or QUALITY
    r = metrics.get("relative_error")
    return bool(r is not None and np.isfinite(r) and 0 <= r <= q["maximum_relative_error"]
                and np.isfinite(metrics["iou"]) and q["minimum_iou"] <= metrics["iou"] <= 1)


def normalized_recovery(before, after, current, epsilon=1e-10):
    """No clipping: overshoot and deterioration are scientifically meaningful."""
    denominator = after - before
    return None if abs(denominator) <= epsilon else float((after - current) / denominator)


def repair_outcome(before, removed_fraction, curve, quality=None, min_removed=0.08, sustained=3):
    if isinstance(sustained, bool) or not isinstance(sustained, int) or sustained < 1:
        raise ValueError("Sustained recovery requires a positive integer sample count")
    started_acceptable = acceptable(before, quality)
    meaningful = bool(np.isfinite(removed_fraction) and min_removed <= removed_fraction <= 1)
    consecutive = 0
    recovered_at = None
    for sample in curve:
        consecutive = consecutive + 1 if acceptable(sample, quality) else 0
        if consecutive >= sustained and recovered_at is None:
            recovered_at = int(sample["elapsed"])
    success = bool(started_acceptable and meaningful and recovered_at is not None)
    later = [sample for sample in curve if recovered_at is not None and sample["elapsed"] > recovered_at]
    deteriorated = bool(success and any(not acceptable(sample, quality) for sample in later))
    return {"pre_damage_acceptable": started_acceptable, "meaningful_damage": meaningful,
            "success": success, "recovery_time": recovered_at if success else None,
            "retained_recovery": bool(success and not deteriorated),
            "deterioration_after_recovery": deteriorated,
            "final_acceptable": bool(curve and acceptable(curve[-1], quality))}
