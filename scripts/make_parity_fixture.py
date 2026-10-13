"""Freeze tiny nontrivial PyTorch outputs for the deployed Pyodide NumPy test."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from mycelium_nca.model_io import weights_hash
from mycelium_nca.parity import PARITY_SCHEMA, run_fixture
from mycelium_nca.torch_model import NCA, perceive


def generate():
    torch.set_num_threads(1)
    size = 5
    rng = np.random.Generator(np.random.PCG64(3719))
    model = NCA(42)
    weights = model.weights_numpy()
    weights["W2"] = rng.normal(0, .03, (64, 16)).astype(np.float32)
    weights["b2"] = rng.normal(0, .005, 16).astype(np.float32)
    model.load_weights(weights)
    state = rng.normal(0, .2, (size, size, 16)).astype(np.float32)
    state[..., 3] = .7
    state[0, 0, 4:] = .8  # Edge/corner perception exposes wraparound and kernel sign bugs.
    state[-1, 0, 5] = -.9
    lesions = np.zeros((size, size), bool)
    lesions[0, -1] = True
    masks = rng.random((8, size, size), dtype=np.float32) < .5
    tensor = torch.from_numpy(state[None].copy())
    with torch.no_grad():
        expected = {"perception": perceive(tensor)[0].numpy().tolist(),
                    "network_delta": model.network_delta(tensor)[0].numpy().tolist()}
        for index, mask in enumerate(masks):
            tensor = model(tensor, torch.from_numpy(mask[None]), torch.from_numpy(lesions[None]))
            if index == 0:
                expected["one_step"] = tensor[0].numpy().tolist()
        expected["rollout_8"] = tensor[0].numpy().tolist()
        threshold = np.zeros_like(state)
        threshold[0, 0, 3] = np.nextafter(np.float32(.1), np.float32(1))
        threshold[-1, -1, 3] = np.float32(.1)
        model.load_weights({key: np.zeros_like(value) for key, value in weights.items()})
        expected["threshold_step"] = model(torch.from_numpy(threshold[None]), torch.ones((1, size, size), dtype=torch.bool))[0].numpy().tolist()
    fixture = {"schema": PARITY_SCHEMA, "grid_size": size, "rtol": 2e-5, "atol": 2e-6,
               "reference": {"backend": "PyTorch CPU", "torch": torch.__version__,
                             "python": platform.python_version(), "platform": platform.platform(),
                             "description": "Frozen explicit masks; nonzero learned update fixture; not trained weights."},
               "weights_hash": weights_hash(weights),
               "weights": {key: value.tolist() for key, value in weights.items()},
               "state": state.tolist(), "masks": masks.tolist(), "lesions": lesions.tolist(),
               "expected": expected}
    native = run_fixture(fixture)
    if not native["passed"]:
        raise AssertionError(json.dumps(native, indent=2))
    fixture["native_numpy_measurement"] = native
    return fixture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="fixtures/parity-v1.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = generate()
    output.write_text(json.dumps(fixture, allow_nan=False, separators=(",", ":")) + "\n")
    print(json.dumps({"fixture": str(output), "bytes": output.stat().st_size,
                      "native_result": fixture["native_numpy_measurement"]}, indent=2))
