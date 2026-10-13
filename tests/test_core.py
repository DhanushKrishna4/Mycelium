import json
import sys

import numpy as np
import pytest

from mycelium_nca.config import WEIGHT_SHAPES
from mycelium_nca.interventions import apply_event
from mycelium_nca.model_io import load_model, model_document, save_model, weights_hash
from mycelium_nca.numpy_sim import check_numerical, living, network_delta, perceive, render_rgba, seed_state, step


def zero_weights():
    return {name: np.zeros(shape, np.float32) for name, shape in WEIGHT_SHAPES.items()}


def test_seed_contract():
    state = seed_state()
    assert state.shape == (32, 32, 16) and state.dtype == np.float32
    assert not state[..., :3].any()
    assert np.array_equal(state[16, 16, 3:], np.ones(13))
    assert np.count_nonzero(state) == 13


def test_identity_sobel_order_sign_padding():
    state = np.zeros((5, 5, 16), np.float32)
    state[..., 0] = np.arange(5)[None, :]
    state[..., 1] = np.arange(5)[:, None]
    features = perceive(state)
    np.testing.assert_array_equal(features[..., :16], state)
    np.testing.assert_allclose(features[1:4, 1:4, 16], 1)
    np.testing.assert_allclose(features[1:4, 1:4, 33], 1)
    np.testing.assert_array_equal(features[1:4, 1:4, 17], 0)
    np.testing.assert_array_equal(features[1:4, 1:4, 32], 0)
    impulse = np.zeros_like(state)
    impulse[0, 0, 5] = 1
    observed = perceive(impulse)
    assert observed[0, 1, 21] == -0.25
    assert observed[1, 0, 37] == -0.25
    assert not observed[-1].any() and not observed[:, -1].any()


def test_explicit_dense_layout():
    state = seed_state(5)
    weights = zero_weights()
    weights["W1"][3, 7] = 2
    weights["W2"][7, 4] = 3
    delta = network_delta(state, weights)
    assert delta[2, 2, 4] == 6
    assert np.count_nonzero(delta) == 1


def test_zero_state_absorbing_even_with_bias():
    weights = zero_weights()
    weights["b2"].fill(1)
    result = step(np.zeros((5, 5, 16), np.float32), weights, np.ones((5, 5), bool))
    assert not result.any()


def test_pre_and_post_living_mask_and_strict_threshold():
    state = seed_state(5)
    state[2, 2, 3] = np.float32(0.1)
    assert not living(state).any()
    state[2, 2, 3] = 1
    assert living(state).sum() == 9
    weights = zero_weights()
    weights["b2"][3] = -2
    result = step(state, weights, np.ones((5, 5), bool))
    assert not result.any(), "Post-update alpha extinction must kill hidden state too."
    weights["b2"].fill(1)
    result = step(state, weights, np.ones((5, 5), bool))
    assert not result[0].any(), "New spontaneous living cells outside pre-alive neighborhood are rejected."


def test_mask_shared_across_channels_and_input_immutable():
    state = seed_state(5)
    state[..., 3] = 1
    before = state.copy()
    weights = zero_weights()
    weights["b2"].fill(0.25)
    mask = np.zeros((5, 5), bool)
    mask[2, 3] = True
    result = step(state, weights, mask)
    np.testing.assert_array_equal(state, before)
    np.testing.assert_allclose(result[2, 3] - state[2, 3], 0.25)
    np.testing.assert_array_equal(result[1, 1], state[1, 1])
    with pytest.raises(ValueError, match="mask"):
        step(state, weights, np.ones(state.shape))


def test_erase_lesion_seed_and_noise():
    state = np.ones((9, 9, 16), np.float32)
    lesions = np.zeros((9, 9), bool)
    erased, lesions = apply_event(state, lesions, {"type": "erase", "x": 4, "y": 4, "radius": 1})
    assert not erased[4, 4].any() and np.all(state == 1)
    damaged, lesions = apply_event(state, lesions, {"type": "lesion", "x": 4, "y": 4, "radius": 1})
    for _ in range(3):
        damaged = step(damaged, zero_weights(), np.ones((9, 9), bool), lesions)
        assert not damaged[lesions].any()
    seeded, _ = apply_event(damaged, lesions, {"type": "seed", "x": 4, "y": 4})
    assert not seeded[4, 4].any(), "Persistent lesion wins over seed insertion."
    event = {"type": "noise", "x": 3, "y": 3, "radius": 2, "seed": 7, "amplitude": 0.2, "channels": [4, 8]}
    first, _ = apply_event(state, lesions, event)
    second, _ = apply_event(state, lesions, event)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[~lesions, :4], state[~lesions, :4])
    assert np.max(np.abs((first - state)[~lesions])) <= 0.2 + 1e-6


def test_render_is_straight_alpha_and_does_not_mutate():
    state = np.zeros((1, 1, 16), np.float32)
    state[0, 0, :4] = [0.2, 0.1, -0.1, 0.5]
    before = state.copy()
    np.testing.assert_array_equal(render_rgba(state)[0, 0], [102, 51, 0, 128])
    np.testing.assert_array_equal(state, before)


def test_numerical_failure_detection():
    state = seed_state(4)
    state[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        check_numerical(state)
    state[0, 0, 0] = 1e7
    with pytest.raises(FloatingPointError, match="exceeds"):
        check_numerical(state)


def test_model_roundtrip_and_bounded_import(tmp_path):
    weights = zero_weights()
    weights["W1"][:] = np.random.default_rng(3).normal(size=(48, 64)).astype(np.float32)
    path = tmp_path / "model.json"
    saved = save_model(path, weights, {"regime": "growth", "target_hash": "abc"})
    loaded = load_model(path)
    assert loaded["metadata"] == saved["metadata"]
    assert weights_hash(loaded["weights"]) == saved["weights_hash"]
    for name in weights:
        np.testing.assert_array_equal(weights[name], loaded["weights"][name])
    saved["weights"]["W1"][0][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(saved)
    saved["architecture"]["channels"] = 17
    with pytest.raises(ValueError, match="Incompatible architecture"):
        load_model(saved)
    with pytest.raises(ValueError, match="byte limit"):
        load_model(b" " * 1_000_001)
    with pytest.raises(ValueError, match="finite JSON"):
        load_model('{"weights":NaN}')
    bad = model_document(weights)
    bad["weights"]["W1"] = [[0]]
    with pytest.raises(ValueError, match="shape"):
        load_model(bad)


def test_runtime_import_does_not_require_torch():
    import subprocess
    from pathlib import Path
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = f"import sys; sys.path.insert(0, {source!r}); import mycelium_nca; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
