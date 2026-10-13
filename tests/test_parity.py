import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mycelium_nca import numpy_sim
from mycelium_nca.torch_model import NCA, initial_weights, perceive

RTOL, ATOL = 2e-5, 2e-6


def fixture():
    rng = np.random.default_rng(381)
    state = rng.normal(0, 0.2, (9, 9, 16)).astype(np.float32)
    state[..., 3] = 0.8  # Keep this fixture safely away from living-mask discontinuities.
    weights = initial_weights(42)
    weights["W2"] = rng.normal(0, 0.03, (64, 16)).astype(np.float32)
    weights["b2"] = rng.normal(0, 0.01, (16,)).astype(np.float32)
    return rng, state, weights


def test_initialization_final_layer_and_seed_reproducibility():
    a, b = NCA(42), NCA(42)
    assert not a.W2.detach().count_nonzero() and not a.b2.detach().count_nonzero()
    for name, value in a.weights_numpy().items():
        np.testing.assert_array_equal(value, b.weights_numpy()[name])


def test_intermediate_and_twelve_step_numpy_torch_parity():
    rng, state, weights = fixture()
    model = NCA().load_weights(weights)
    ts = torch.from_numpy(state[None].copy())
    np.testing.assert_allclose(perceive(ts).detach().numpy()[0], numpy_sim.perceive(state), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(model.network_delta(ts).detach().numpy()[0], numpy_sim.network_delta(state, weights), rtol=RTOL, atol=ATOL)
    lesion = np.zeros((9, 9), bool)
    lesion[0, :3] = True
    with torch.no_grad():
        for _ in range(12):
            mask = rng.random((9, 9), dtype=np.float32) < 0.5
            state = numpy_sim.step(state, weights, mask, lesion)
            ts = model(ts, torch.from_numpy(mask[None]), torch.from_numpy(lesion[None]))
            np.testing.assert_allclose(ts.numpy()[0], state, rtol=RTOL, atol=ATOL)


def test_threshold_fixture_separate_from_smooth_parity():
    weights = initial_weights(7)
    state = np.zeros((5, 5, 16), np.float32)
    state[0, 0, 3] = np.nextafter(np.float32(.1), np.float32(1))
    state[-1, -1, 3] = np.float32(.1)
    mask = np.ones((5, 5), bool)
    result = numpy_sim.step(state, weights, mask)
    with torch.no_grad():
        actual = NCA(7)(torch.from_numpy(state[None]), torch.from_numpy(mask[None])).numpy()[0]
    np.testing.assert_array_equal(actual, result)
    assert result[0, 0, 3] > 0.1 and result[-1, -1, 3] == 0


def test_fixed_mask_rollout_gradients_finite_nonzero():
    rng, state, weights = fixture()
    model = NCA().load_weights(weights)
    current = torch.tensor(state[None])
    for _ in range(4):
        mask = torch.from_numpy((rng.random((1, 9, 9), dtype=np.float32) < 0.5))
        current = model(current, mask)
    current[..., :4].square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_small_gradient_check_away_from_relu_and_living_thresholds():
    model = NCA().double()
    with torch.no_grad():
        model.W1.fill_(0.005)
        model.b1.fill_(0.5)
        model.W2.fill_(0.002)
    state = torch.full((1, 2, 2, 16), 0.3, dtype=torch.double, requires_grad=True)
    mask = torch.ones((1, 2, 2, 1), dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x: model(x, mask), (state,), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True)
