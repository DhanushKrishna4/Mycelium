"""Training contracts: pool identity, graph detachment, and exact CPU resumption."""
from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from mycelium_nca.targets import TARGET_NAMES, generate_target, premultiply, target_hash, write_targets
from mycelium_nca.training import StatePool, TrainConfig, damage_batch, make_seed, training_streams


def test_original_targets_are_reproducible_premultiplied_and_bounded(tmp_path):
    records = write_targets(tmp_path)
    for name, record in zip(TARGET_NAMES, records):
        target = generate_target(name)
        np.testing.assert_array_equal(target, np.load(tmp_path / f"{name}.npy", allow_pickle=False))
        assert target.shape == (32, 32, 4) and target.dtype == np.float32
        assert np.all(target[..., :3] <= target[..., 3:4])
        assert record["sha256"] == target_hash(target)
        assert np.count_nonzero(target[..., 3] >= .5) > 100
        assert not target[:5].any() and not target[-4:].any()
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_alpha_preprocessing_does_not_modify_input():
    original = np.array([[[.8, .4, .2, .5], [1, 1, 1, 0]]], np.float32)
    snapshot = original.copy()
    np.testing.assert_allclose(premultiply(original), [[[.4, .2, .1, .5], [0, 0, 0, 0]]])
    np.testing.assert_array_equal(original, snapshot)
    with pytest.raises(ValueError):
        premultiply(original*3)


def test_sorted_pool_identities_write_back_to_their_original_entries():
    pool = StatePool(make_seed(), 8)
    for identity in range(8):
        pool.states[identity, ..., 0] = identity/8
    original = pool.states.copy()
    identities, batch = pool.sample(4, np.random.default_rng(9), np.zeros((32, 32, 4), np.float32))
    assert np.all(identities[:-1] > identities[1:])
    batch[0] = make_seed()  # replacing the worst sample must preserve its ID
    pool.commit(identities, batch)
    np.testing.assert_array_equal(pool.states[identities[0]], make_seed())
    untouched = sorted(set(range(8))-set(identities))
    np.testing.assert_array_equal(pool.states[untouched], original[untouched])
    np.testing.assert_array_equal(pool.write_counts[identities], 1)


def test_pool_storage_detaches_tensors():
    torch = pytest.importorskip("torch")
    pool = StatePool(make_seed(), 4)
    tensor = torch.tensor(pool.states[:2], requires_grad=True)*2
    pool.commit(np.array([2, 3]), tensor)
    assert isinstance(pool.states, np.ndarray)
    np.testing.assert_array_equal(pool.states[2], make_seed()*2)
    tensor.sum().backward()
    assert pool.states.base is None


def test_damage_preserves_seed_and_some_living_cells():
    batch = np.ones((4, 32, 32, 16), np.float32)
    batch[0] = make_seed()
    before = batch[0].copy()
    records = damage_batch(batch, np.random.default_rng(11))
    np.testing.assert_array_equal(batch[0], before)
    assert any(record["applied"] for record in records)
    for record in records:
        if record["applied"]:
            damaged = batch[record["sample"]]
            assert np.count_nonzero(damaged[..., 3]) > 0
            assert np.all(damaged[damaged[..., 3] == 0] == 0)


def test_training_streams_are_independent():
    one, two = training_streams(12), training_streams(12)
    one["damage"].random(100)
    for name in ("pool", "updates", "unroll"):
        np.testing.assert_array_equal(one[name].random(20), two[name].random(20))


def test_native_checkpoint_resume_preserves_optimizer_pool_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    from mycelium_nca.training import run_training, _load_checkpoint
    base = TrainConfig(updates=4, batch_size=2, pool_size=4, unroll_min=2, unroll_max=3,
                       validation_every=0, checkpoint_every=1, threads=1, pilot=True,
                       output_dir=str(tmp_path / "continuous"), lr_decay_at=2)
    continuous = run_training(base)
    partial_config = replace(base, updates=2, output_dir=str(tmp_path / "resumed"))
    partial = run_training(partial_config)
    resumed = run_training(replace(partial_config, updates=4, resume=partial["checkpoint"]))
    expected = _load_checkpoint(torch, continuous["checkpoint"])
    actual = _load_checkpoint(torch, resumed["checkpoint"])
    assert expected["update"] == actual["update"] == 4
    for key in expected["model"]:
        torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)
    np.testing.assert_array_equal(expected["pool"], actual["pool"])
    np.testing.assert_array_equal(expected["pool_write_counts"], actual["pool_write_counts"])
    assert expected["random_states"] == actual["random_states"]
    assert expected["scheduler"] == actual["scheduler"]
    assert expected["total_cell_steps"] == actual["total_cell_steps"]
    for identity, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, actual["optimizer"]["state"][identity][key], rtol=0, atol=0)
    assert len(list((tmp_path / "resumed" / "checkpoints").glob("step-*.pt"))) == 2
    with pytest.raises(ValueError, match="resume would change learning_rate"):
        run_training(replace(partial_config, updates=5, resume=resumed["checkpoint"], learning_rate=.1))
