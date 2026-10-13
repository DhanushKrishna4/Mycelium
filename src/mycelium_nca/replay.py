"""Persistent simulation and bounded, data-only, step-stamped experiments."""

import copy
import json

import numpy as np

from .config import CHANNELS, GRID_SIZE, MAX_GRID_SIZE, RUNTIME_VERSION, UPDATE_PROBABILITY
from .interventions import apply_event
from .model_io import bounded_json, checked_array, validate_weights, weights_hash
from .numpy_sim import check_numerical, seed_state, step

REPLAY_SCHEMA = "mycelium.experiment.v1"
MAX_REPLAY_BYTES = 8_000_000
MAX_EVENTS = 4096
MAX_STEP = 100_000_000
RNG_CONVENTION = "PCG64; Generator.random((height,width),dtype=float32) < p; one call per transition; row-major"


def _count(value, name, maximum=MAX_STEP):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0,{maximum}].")
    return value


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("Update probability must be between zero and one.")
    return float(value)


def _rng(state):
    if not isinstance(state, dict) or state.get("bit_generator") != "PCG64":
        raise ValueError("Replay requires a PCG64 random state.")
    generator = np.random.Generator(np.random.PCG64(0))
    try:
        generator.bit_generator.state = copy.deepcopy(state)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError("Invalid PCG64 state.") from exc
    return generator


class Simulation:
    def __init__(self, weights, seed=0, size=GRID_SIZE, update_probability=UPDATE_PROBABILITY,
                 initial_state=None, model_hash="", target_hash=""):
        _count(seed, "Seed", 2**32 - 1)
        _count(size, "Grid size", MAX_GRID_SIZE)
        if size < 1:
            raise ValueError("Grid size must be positive.")
        self.weights = validate_weights(weights)
        self.model_hash = model_hash or weights_hash(weights)
        self.target_hash = target_hash
        self.state = seed_state(size) if initial_state is None else checked_array(initial_state, (size, size, CHANNELS), "initial_state")
        self.lesions = np.zeros((size, size), bool)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.update_probability = _probability(update_probability)
        self.step_count = 0
        self.events = []
        self.last_delta = np.zeros_like(self.state)
        self.partial_history = False
        self.status = "ready"
        self._origin = self._snapshot()

    def _snapshot(self):
        return {"step": self.step_count, "state": self.state.tolist(),
                "lesions": self.lesions.tolist(), "rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "update_probability": self.update_probability}

    def apply(self, event, *, record=True):
        event = bounded_json(event, max_bytes=16_384)
        if "step" in event and event["step"] != self.step_count:
            raise ValueError("Apply events at the current integer simulation-step boundary.")
        # Promote a complete snapshot before truncating history; never mislabel it as full replay.
        if record and len(self.events) >= MAX_EVENTS:
            self._origin = self._snapshot()
            self.events = []
            self.partial_history = True
        if event.get("type") == "update_probability":
            self.update_probability = _probability(event.get("value"))
        else:
            self.state, self.lesions = apply_event(self.state, self.lesions, event)
        check_numerical(self.state)
        self.last_delta.fill(0)
        if record:
            self.events.append({**copy.deepcopy(event), "step": self.step_count, "order": len(self.events)})
        self.status = "extinct" if not np.any(self.state) else "ready"
        return self.state

    def advance(self, steps=1, masks=None):
        _count(steps, "Chunk size", 256)
        if self.step_count + steps > MAX_STEP:
            raise ValueError("Maximum simulation step reached.")
        if masks is not None and len(masks) != steps:
            raise ValueError("Explicit masks must contain exactly one mask per requested step.")
        for i in range(steps):
            # Still draw one mask for explicit-mask runs, so paired branches keep aligned RNGs.
            generated = self.rng.random(self.state.shape[:2], dtype=np.float32) < self.update_probability
            mask = generated if masks is None else masks[i]
            candidate, delta = step(self.state, self.weights, mask, self.lesions, return_delta=True)
            try:
                check_numerical(candidate)
            except FloatingPointError:
                self.status = "numerical_failure"
                raise
            self.state, self.last_delta = candidate, delta
            self.step_count += 1
            if masks is not None and not np.array_equal(np.asarray(mask).reshape(generated.shape), generated):
                # External parity/paired masks are not serialized as enormous arrays.
                # Retain their complete resulting state as a transparent partial origin.
                self._origin = self._snapshot()
                self.events = []
                self.partial_history = True
        if self.status != "numerical_failure":
            self.status = "extinct" if not np.any(self.state) else "ready"
        return self.state

    def clone(self):
        return Simulation.from_export(self.export(), self.weights)

    def export(self, measurements=None):
        document = {"schema": REPLAY_SCHEMA, "runtime_version": RUNTIME_VERSION,
                    "numpy_version": np.__version__, "model_hash": self.model_hash,
                    "target_hash": self.target_hash, "grid": list(self.state.shape[:2]),
                    "channels": CHANNELS, "rng_convention": RNG_CONVENTION,
                    "origin": copy.deepcopy(self._origin), "current": self._snapshot(),
                    "events": copy.deepcopy(self.events), "partial_history": self.partial_history,
                    "measurements": measurements or []}
        bounded_json(document, MAX_REPLAY_BYTES)
        return document

    @classmethod
    def from_export(cls, source, weights, expected_model_hash=""):
        document = bounded_json(source, MAX_REPLAY_BYTES)
        if document.get("schema") != REPLAY_SCHEMA or document.get("runtime_version") != RUNTIME_VERSION:
            raise ValueError("Unsupported experiment schema/runtime version.")
        if document.get("numpy_version") != np.__version__:
            raise ValueError(f"Exact replay requires NumPy {np.__version__}; the experiment uses {document.get("numpy_version")!r}.")
        if not isinstance(document.get("partial_history"), bool):
            raise ValueError("Experiment partial_history must be a boolean.")
        if not isinstance(document.get("measurements", []), list) or len(document.get("measurements", [])) > 4096:
            raise ValueError("Experiment measurements must be a list of at most 4096 records.")
        target_hash = document.get("target_hash", "")
        if not isinstance(target_hash, str) or (target_hash and (len(target_hash) != 64 or any(c not in "0123456789abcdef" for c in target_hash))):
            raise ValueError("Experiment target hash must be empty or a lowercase SHA-256 digest.")
        if document.get("rng_convention") != RNG_CONVENTION:
            raise ValueError("Unsupported RNG sampling convention.")
        grid = document.get("grid")
        if not isinstance(grid, list) or len(grid) != 2 or grid[0] != grid[1]:
            raise ValueError("Experiment grid must be square.")
        size = _count(grid[0], "Grid size", MAX_GRID_SIZE)
        if size < 1 or document.get("channels") != CHANNELS:
            raise ValueError("Invalid grid or channel count.")
        actual_hash = weights_hash(validate_weights(weights))
        if document.get("model_hash") != actual_hash or (expected_model_hash and actual_hash != expected_model_hash):
            raise ValueError("Experiment model hash does not match the supplied weights.")
        events = document.get("events")
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise ValueError(f"Experiment supports at most {MAX_EVENTS} events.")

        def snapshot(value):
            if not isinstance(value, dict):
                raise ValueError("Missing experiment snapshot.")
            count = _count(value.get("step"), "Snapshot step")
            state = checked_array(value.get("state"), (size, size, CHANNELS), "snapshot state")
            lesion_values = checked_array(value.get("lesions"), (size, size), "lesions", max_abs=1)
            if not np.isin(lesion_values, (0, 1)).all():
                raise ValueError("Lesion mask must contain only booleans or zero/one.")
            lesions = lesion_values.astype(bool)
            if np.any(state[lesions]):
                raise ValueError("Snapshot violates its persistent lesions.")
            probability = _probability(value.get("update_probability"))
            generator = _rng(value.get("rng_state"))
            return count, state, lesions, probability, generator

        origin = snapshot(document.get("origin"))
        current = snapshot(document.get("current"))
        if origin[0] > current[0]:
            raise ValueError("History origin cannot occur after current step.")
        previous_step = origin[0]
        validation_state, validation_lesions = origin[1].copy(), origin[2].copy()
        for order, event in enumerate(events):
            if not isinstance(event, dict) or event.get("order") != order:
                raise ValueError("Experiment event order is invalid.")
            event_step = _count(event.get("step"), "Event step")
            if not previous_step <= event_step <= current[0]:
                raise ValueError("Experiment event steps must be ordered within the recorded history.")
            previous_step = event_step
            if event.get("type") == "update_probability":
                _probability(event.get("value"))
            else:
                validation_state, validation_lesions = apply_event(validation_state, validation_lesions, event)
        simulation = cls(weights, size=size, initial_state=current[1], model_hash=actual_hash,
                         target_hash=document.get("target_hash", ""), update_probability=current[3])
        simulation.step_count, simulation.lesions, simulation.rng = current[0], current[2], current[4]
        simulation._origin = copy.deepcopy(document["origin"])
        simulation.events = copy.deepcopy(events)
        simulation.partial_history = bool(document.get("partial_history", False))
        simulation.status = "extinct" if not np.any(simulation.state) else "ready"
        return simulation

    def reconstruct(self):
        """Re-run recorded events from the retained origin (bounded to 100k steps)."""
        start = self._origin
        if self.step_count - start["step"] > 100_000:
            raise ValueError("History reconstruction is limited to 100,000 transitions; resume the saved snapshot instead.")
        simulation = Simulation(self.weights, size=self.state.shape[0], initial_state=start["state"],
                                model_hash=self.model_hash, target_hash=self.target_hash,
                                update_probability=start["update_probability"])
        simulation.step_count = start["step"]
        simulation.rng = _rng(start["rng_state"])
        simulation.lesions = np.asarray(start["lesions"], dtype=bool)
        simulation._origin = copy.deepcopy(start)
        for event in self.events:
            while simulation.step_count < event["step"]:
                simulation.advance(min(256, event["step"] - simulation.step_count))
            simulation.apply(event)
        while simulation.step_count < self.step_count:
            simulation.advance(min(256, self.step_count - simulation.step_count))
        simulation.partial_history = self.partial_history
        return simulation


def export_experiment(simulation):
    return json.dumps(simulation.export(), allow_nan=False, separators=(",", ":"))


def import_experiment(source, weights):
    return Simulation.from_export(source, weights)
