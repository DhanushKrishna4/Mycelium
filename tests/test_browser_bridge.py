import copy
import json
import numpy as np
import pytest
from mycelium_nca.browser import Laboratory
from mycelium_nca.model_io import model_document
from mycelium_nca.torch_model import initial_weights
from mycelium_nca.targets import generate_target


def lab():
    models={name:model_document(initial_weights(),{"target":"ring"}) for name in ("a","b")}
    manifest={"models":[{"id":name,"target":"ring","regime":"growth"} for name in models],"targets":[]}
    return Laboratory(manifest,models,{"ring":{"rgba":generate_target("ring").tolist()}})


def test_paired_clone_and_shared_masks():
    l=lab()
    l.dispatch({"type":"tick","steps":8})
    result=l.dispatch({"type":"fork"})
    assert result["divergence"]["visible"]==0
    l.dispatch({"type":"tick","steps":8})
    assert np.array_equal(l.simulations[0].state,l.simulations[1].state)
    l.dispatch({"type":"event","kind":"erase","x":16,"y":16,"radius":1})
    assert np.any(l.simulations[0].state)
    assert not np.any(l.simulations[1].state)
    l.dispatch({"type":"tick","steps":8})
    assert l.simulations[0].rng.bit_generator.state==l.simulations[1].rng.bit_generator.state


def test_compare_events_both_and_trajectory_survives_export():
    l=lab(); l.dispatch({"type":"reset","model":"a","secondModel":"b","mode":"compare"})
    l.dispatch({"type":"settings","probability":.25,"view":"hidden","channel":10})
    l.dispatch({"type":"event","kind":"noise","x":16,"y":16,"radius":1,"seed":12,"amplitude":.1})
    l.dispatch({"type":"tick","steps":8})
    exported=l.dispatch({"type":"exportExperiment"})["download"]["data"]
    other=lab(); other.dispatch({"type":"importExperiment","data":exported})
    for a,b in zip(l.simulations,other.simulations):
        a.advance(7); b.advance(7)
        assert np.array_equal(a.state,b.state)
    assert np.array_equal(l.simulations[0].state,l.simulations[1].state)


def test_clear_extinction_explicit_seed_and_lesion():
    l=lab()
    l.dispatch({"type":"event","kind":"clear"})
    assert l.dispatch({"type":"tick","steps":8})["status"]=="extinct"
    l.dispatch({"type":"event","kind":"lesion","x":16,"y":16,"radius":1})
    l.dispatch({"type":"event","kind":"seed","x":16,"y":16})
    assert not np.any(l.simulations[0].state)
    l.dispatch({"type":"event","kind":"clear_lesions"})
    assert l.dispatch({"type":"event","kind":"seed","x":16,"y":16})["status"]=="ready"


def test_bounded_commands():
    l=lab()
    with pytest.raises(ValueError): l.dispatch({"type":"tick","steps":5000})
    with pytest.raises(ValueError): l.dispatch({"type":"settings","channel":999})
    with pytest.raises(ValueError): l.dispatch({"type":"importModel","data":"{}"})
    export=json.loads(l.dispatch({"type":"exportExperiment"})["download"]["data"])
    export["mode"]="paired"
    with pytest.raises(ValueError): l.dispatch({"type":"importExperiment","data":export})


def test_compare_numerical_failure_rolls_back_both_branches(monkeypatch):
    l = lab()
    l.dispatch({"type":"reset", "model":"a", "secondModel":"b", "mode":"compare"})
    before = [(s.state.copy(), s.last_delta.copy(), s.step_count,
               copy.deepcopy(s.rng.bit_generator.state)) for s in l.simulations]
    second = l.simulations[1]
    original_advance = second.advance

    def fail_after_draw(steps):
        original_advance(steps)
        raise FloatingPointError("Deliberate failure after advancing the second branch")

    monkeypatch.setattr(second, "advance", fail_after_draw)
    result = l.dispatch({"type":"tick", "steps":8})
    assert result["status"] == "numerical_failure"
    for sim, (state, delta, count, rng) in zip(l.simulations, before):
        np.testing.assert_array_equal(sim.state, state)
        np.testing.assert_array_equal(sim.last_delta, delta)
        assert sim.step_count == count
        assert sim.rng.bit_generator.state == rng
    assert l.dispatch({"type":"tick", "steps":1})["step"] == 0
