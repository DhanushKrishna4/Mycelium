"""Run a frozen PyTorch-reference fixture using only the deployed NumPy runtime."""

import numpy as np

from .model_io import bounded_json, checked_array, validate_weights, weights_hash
from .numpy_sim import network_delta, perceive, step

PARITY_SCHEMA = "mycelium.parity.v1"


def run_fixture(source):
    """Return measured errors and pass/fail for the native or Pyodide NumPy path."""
    fixture = bounded_json(source, max_bytes=1_000_000)
    if fixture.get("schema") != PARITY_SCHEMA:
        raise ValueError("Unsupported numerical parity fixture.")
    size = fixture.get("grid_size")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 9:
        raise ValueError("Parity fixture grid must be between 1 and 9 cells.")
    shape = (size, size, 16)
    weights = validate_weights(fixture.get("weights"))
    if weights_hash(weights) != fixture.get("weights_hash"):
        raise ValueError("Parity fixture weight hash mismatch.")
    state = checked_array(fixture.get("state"), shape, "fixture state")
    masks = checked_array(fixture.get("masks"), (8, size, size), "fixture masks", max_abs=1)
    if not np.isin(masks, (0, 1)).all():
        raise ValueError("Parity fixture masks must contain zero/one only.")
    lesion_values = checked_array(fixture.get("lesions"), (size, size), "fixture lesions", max_abs=1)
    if not np.isin(lesion_values, (0, 1)).all():
        raise ValueError("Parity fixture lesions must contain zero/one only.")
    lesions = lesion_values.astype(bool)
    reference = fixture.get("expected")
    if not isinstance(reference, dict):
        raise ValueError("Parity fixture expected outputs are missing.")
    rtol, atol = 2e-5, 2e-6
    if fixture.get("rtol") != rtol or fixture.get("atol") != atol:
        raise ValueError("Parity fixture tolerance differs from the v1 contract.")
    checks = []

    def compare(name, actual, expected_shape):
        expected = checked_array(reference.get(name), expected_shape, name)
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        permitted = atol + rtol * np.abs(expected.astype(np.float64))
        checks.append({"name": name, "passed": bool(np.all(difference <= permitted)),
                       "max_absolute_error": float(difference.max(initial=0)),
                       "max_tolerance_ratio": float((difference / permitted).max(initial=0)),
                       "values": int(actual.size)})

    compare("perception", perceive(state), (size, size, 48))
    compare("network_delta", network_delta(state, weights), shape)
    current = state.copy()
    for i, mask in enumerate(masks):
        current = step(current, weights, mask, lesions)
        if i == 0:
            compare("one_step", current, shape)
    compare("rollout_8", current, shape)

    # Exact, isolated living-threshold fixture: no network arithmetic near threshold.
    threshold = np.zeros_like(state)
    threshold[0, 0, 3] = np.nextafter(np.float32(.1), np.float32(1))
    threshold[-1, -1, 3] = np.float32(.1)
    zero_weights = {key: np.zeros_like(value) for key, value in weights.items()}
    observed = step(threshold, zero_weights, np.ones((size, size), bool))
    expected = checked_array(reference.get("threshold_step"), shape, "threshold step")
    checks.append({"name": "threshold_step", "passed": bool(np.array_equal(observed, expected)),
                   "max_absolute_error": float(np.max(np.abs(observed - expected))), "values": int(observed.size)})
    return {"schema": PARITY_SCHEMA, "passed": all(check["passed"] for check in checks),
            "numpy_version": np.__version__, "reference": fixture.get("reference"),
            "weights_hash": fixture["weights_hash"], "rtol": rtol, "atol": atol, "checks": checks}
