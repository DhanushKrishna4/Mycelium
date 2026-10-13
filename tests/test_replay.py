import json

import numpy as np
import pytest

from mycelium_nca.config import WEIGHT_SHAPES
from mycelium_nca.replay import Simulation


def weights():
    rng = np.random.Generator(np.random.PCG64(42))
    result = {key: rng.normal(0, 0.02, shape).astype(np.float32) for key, shape in WEIGHT_SHAPES.items()}
    result["b2"][3] = 0.02
    return result


def test_pcg64_float32_stream_fixture():
    rng = np.random.Generator(np.random.PCG64(42))
    # Stream pins dtype, shape, algorithm and iteration order, not just seed.
    np.testing.assert_array_equal(rng.random((2, 3), dtype=np.float32),
                                  np.array([[0.08925092, 0.773956, 0.6545715],
                                            [0.43887842, 0.43301523, 0.8585979]], np.float32))


def test_chunking_and_render_speed_do_not_change_trajectory():
    from mycelium_nca.numpy_sim import render_rgba
    a, b = Simulation(weights(), seed=7, size=8), Simulation(weights(), seed=7, size=8)
    a.advance(24)
    for _ in range(24):
        b.advance(1)
        render_rgba(b.state)
    np.testing.assert_array_equal(a.state, b.state)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_event_order_export_import_reconstruction_and_continuation():
    a = Simulation(weights(), seed=19, size=8)
    a.advance(6)
    a.apply({"type": "erase", "x": 4, "y": 4, "radius": 1})
    a.apply({"type": "seed", "x": 4, "y": 4})
    a.apply({"type": "update_probability", "value": 0.8})
    a.advance(3)
    a.apply({"type": "noise", "x": 2, "y": 2, "radius": 2, "seed": 91})
    a.apply({"type": "lesion", "x": 0, "y": 0, "radius": 2})
    document = json.loads(json.dumps(a.export()))
    b = Simulation.from_export(document, weights())
    rebuilt = b.reconstruct()
    assert b.step_count == rebuilt.step_count == a.step_count == 9
    np.testing.assert_array_equal(a.state, b.state)
    np.testing.assert_array_equal(a.state, rebuilt.state)
    assert a.events == b.events == rebuilt.events
    a.advance(10)
    b.advance(10)
    rebuilt.advance(10)
    np.testing.assert_array_equal(a.state, b.state)
    np.testing.assert_array_equal(a.state, rebuilt.state)


def test_clear_is_extinction_until_explicit_seed():
    simulation = Simulation(weights(), size=8)
    simulation.advance(8)
    simulation.apply({"type": "clear"})
    simulation.advance(8)
    assert simulation.status == "extinct" and not simulation.state.any()
    simulation.apply({"type": "seed"})
    assert simulation.state[4, 4, 3] == 1


def test_import_rejects_bad_grid_event_order_hash_and_length():
    simulation = Simulation(weights(), size=8)
    simulation.apply({"type": "seed"})
    document = simulation.export()
    document["grid"] = [100000, 100000]
    with pytest.raises(ValueError, match="Grid size"):
        Simulation.from_export(document, weights())
    document = simulation.export()
    document["events"][0]["order"] = 1
    with pytest.raises(ValueError, match="order"):
        Simulation.from_export(document, weights())
    document = simulation.export()
    document["model_hash"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        Simulation.from_export(document, weights())
    with pytest.raises(ValueError, match="byte limit"):
        Simulation.from_export(b" " * 8_000_001, weights())


def test_bounded_history_remains_reconstructible(monkeypatch):
    import mycelium_nca.replay as replay
    monkeypatch.setattr(replay, "MAX_EVENTS", 2)
    simulation = Simulation(weights(), size=8)
    for _ in range(4):
        simulation.advance(2)
        simulation.apply({"type": "noise", "seed": 4})
    assert simulation.partial_history
    assert len(simulation.events) <= 2
    np.testing.assert_array_equal(simulation.state, simulation.reconstruct().state)


def test_import_rejects_an_unverified_numpy_stream_version():
    document = Simulation(weights(), size=8).export()
    document["numpy_version"] = "1.26.4"
    with pytest.raises(ValueError, match="Exact replay requires NumPy"):
        Simulation.from_export(document, weights())
