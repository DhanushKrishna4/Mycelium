import numpy as np
import pytest
from mycelium_nca.metrics import measure, acceptable, normalized_recovery, repair_outcome


def target():
    t = np.zeros((32, 32, 4), np.float32)
    t[12:20, 12:20] = [0.3, 0.7, 0.4, 1]
    return t


def test_background_cannot_make_empty_a_success():
    t = target()
    m = measure(np.zeros_like(t), t)
    assert m["relative_error"] == pytest.approx(1)
    assert m["iou"] == 0
    assert not acceptable(m)


def test_exact_target():
    m = measure(target(), target())
    assert m["raw_mse"] == m["balanced_mse"] == m["relative_error"] == 0
    assert m["iou"] == 1
    assert acceptable(m)


def test_recovery_is_unclamped_and_zero_denominator_explicit():
    assert normalized_recovery(1, 1, 0) is None
    assert normalized_recovery(1, 3, 4) == -.5
    assert normalized_recovery(1, 3, 0) == 1.5


def test_repair_requires_mature_start_damage_and_sustained_recovery():
    good = measure(target(), target())
    bad = measure(np.zeros_like(target()), target())
    curve = [dict(good, elapsed=i * 16) for i in range(1, 4)]
    assert repair_outcome(good, .2, curve)["success"]
    assert repair_outcome(good, .2, curve)["recovery_time"] == 48
    assert not repair_outcome(bad, .2, curve)["success"]
    assert not repair_outcome(good, .01, curve)["success"]
    assert not repair_outcome(good, .2, curve[:2])["success"]
