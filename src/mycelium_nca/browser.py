"""Worker-owned laboratory. Only JSON data crosses the JavaScript boundary."""
from __future__ import annotations
import copy
import csv
import io
import json
import numpy as np
from .metrics import measure
from .model_io import bounded_json, checked_array, load_model, model_document
from .numpy_sim import render_rgba
from .replay import Simulation

_lab = None


def _dump(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def heatmap(values, scale=1, signed=True):
    """Fixed scales, a dark zero reference; clipping is display-only."""
    value = np.clip(values / scale, -1 if signed else 0, 1)
    base = np.array([18, 24, 22], np.float32)
    positive = np.array([190, 218, 134], np.float32)
    negative = np.array([135, 159, 231], np.float32)
    color = np.where((value >= 0)[..., None], positive, negative)
    rgb = base + np.abs(value)[..., None] * (color-base)
    return np.concatenate((rgb.astype(np.uint8), np.full((*value.shape,1), 255, np.uint8)), axis=-1)


class Laboratory:
    def __init__(self, manifest, models, targets):
        self.manifest = manifest
        self.models = {name: load_model(value) for name, value in models.items()}
        self.targets = {}
        for name, value in targets.items():
            rgba = value.get("rgba", value.get("data"))
            if rgba is not None:
                self.targets[name] = checked_array(rgba, (32,32,4), "target", max_abs=1)
        self.entries = {m["id"]: m for m in manifest.get("models", [])}
        self.simulations = []
        self.ids = []
        self.curves = []
        self.mode = "single"
        self.view = "rgba"
        self.channel = 4
        self.status = "ready"
        if not self.models:
            raise ValueError("No trained model artifacts were packaged.")
        self.reset({"model":next(iter(self.models))})

    def reset(self, command):
        first = command.get("model", self.ids[0] if self.ids else next(iter(self.models)))
        mode = command.get("mode", "single")
        if mode not in ("single", "compare", "paired"):
            raise ValueError("Unknown experiment mode")
        ids = [first]
        if mode in ("compare", "paired"):
            ids.append(command.get("secondModel", first) if mode == "compare" else first)
        if any(name not in self.models for name in ids):
            raise ValueError("Model is unavailable")
        if mode == "compare" and any(self.entries.get(name, {}).get("target") != "ring" for name in ids):
            raise ValueError("Training-method comparison uses the ring anchor models.")
        simulations = []
        for name in ids:
            model = self.models[name]
            simulations.append(Simulation(model["weights"], seed=command.get("seed", 123),
                                model_hash=model["weights_hash"], target_hash=model.get("metadata", {}).get("target_sha256", "")))
        self.ids, self.simulations, self.mode = ids, simulations, mode
        self.curves = []
        self.status = "ready"
        self.record()

    def metrics(self):
        result = []
        for name, sim in zip(self.ids, self.simulations):
            target = self.targets.get(self.entries.get(name, {}).get("target"))
            result.append(measure(sim.state, target) if target is not None else {"occupied":int((sim.state[...,3]>.1).sum())})
        return result

    def divergence(self):
        if len(self.simulations) != 2:
            return None
        difference = self.simulations[0].state - self.simulations[1].state
        return {"visible":float(np.mean(difference[...,:4].astype(np.float64)**2)),
                "hidden":float(np.mean(difference[...,4:].astype(np.float64)**2))}

    def record(self):
        m = self.metrics()
        sample = {"step":self.simulations[0].step_count, "errors":[v.get("relative_error") for v in m],
                  "ious":[v.get("iou") for v in m], "divergence":self.divergence()}
        self.curves.append(sample)
        self.curves = self.curves[-512:]

    def response(self):
        pixels = []
        for sim in self.simulations:
            if self.view == "rgba":
                image = render_rgba(sim.state)
            elif self.view == "alpha":
                image = heatmap(sim.state[...,3], 1, False)
            elif self.view == "hidden":
                image = heatmap(sim.state[...,self.channel], 2, True)
            else:
                image = heatmap(np.sqrt(np.mean(sim.last_delta**2, axis=-1)), .05, False)
            pixels.append(image.ravel().tolist())
        inspection = None
        if self.mode == "paired":
            diff = self.simulations[0].state-self.simulations[1].state
            inspection = {"visibleDifference":heatmap(np.sqrt(np.mean(diff[...,:4]**2, axis=-1)), 1, False).ravel().tolist(),
                          "hiddenDifference":heatmap(np.sqrt(np.mean(diff[...,4:]**2, axis=-1)), 1, False).ravel().tolist(),
                          "visibleScale":1, "hiddenScale":1}
        branches = []
        for name, sim in zip(self.ids, self.simulations):
            entry = self.entries.get(name, {})
            metadata = self.models[name].get("metadata", {})
            branches.append({"id":name,"regime":entry.get("regime", metadata.get("regime", "imported")),
                             "checkpoint":entry.get("checkpoint", metadata.get("checkpoint_update", "imported")),
                             "modelHash":sim.model_hash,"status":sim.status})
        status = self.status if self.status != "ready" else ("extinct" if all(s.status == "extinct" for s in self.simulations) else "ready")
        return {"step":self.simulations[0].step_count, "pixels":pixels, "metrics":self.metrics(), "curves":self.curves,
                "branches":branches, "mode":self.mode,"status":status, "divergence":self.divergence(), "inspection":inspection,
                "historyPartial":any(s.partial_history for s in self.simulations),"view":self.view,"channel":self.channel,
                "probability":self.simulations[0].update_probability,"gridSize":32}

    def dispatch(self, command):
        kind = command.get("type")
        extra = {}
        if kind == "reset":
            self.reset(command)
        elif kind == "tick":
            steps = command.get("steps", 1)
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
                raise ValueError("Browser chunks must be between one and eight steps")
            for _ in range(steps):
                if self.status != "ready":
                    break
                # Two branches commit together; numerical failure cannot leave
                # their clocks or RNG streams one transition apart.
                snapshots = [(s.state, s.last_delta, s.step_count, copy.deepcopy(s.rng.bit_generator.state), s.status) for s in self.simulations]
                try:
                    # Each branch starts with identical RNG state and draws exactly once
                    # per transition. Damage/noise have their own independent stream.
                    for sim in self.simulations:
                        sim.advance(1)
                except FloatingPointError:
                    for sim, snapshot in zip(self.simulations, snapshots):
                        sim.state, sim.last_delta, sim.step_count, rng_state, sim.status = snapshot
                        sim.rng.bit_generator.state = rng_state
                    self.status = "numerical_failure"
                    break
                if self.simulations[0].step_count % 8 == 0:
                    self.record()
        elif kind == "event":
            event = {k:v for k,v in command.items() if k not in ("type", "branch", "kind")}
            event["type"] = command.get("kind")
            indices = [1] if self.mode == "paired" else list(range(len(self.simulations)))
            for index in indices:
                self.simulations[index].apply(event)
            self.record()
        elif kind == "settings":
            if "view" in command:
                if command["view"] not in ("rgba", "alpha", "hidden", "delta"):
                    raise ValueError("Unknown inspection view")
                self.view = command["view"]
            if "channel" in command:
                if type(command["channel"]) is not int or not 4 <= command["channel"] <= 15:
                    raise ValueError("Hidden channel must be from 4 through 15")
                self.channel = command["channel"]
            if "probability" in command:
                for sim in self.simulations:
                    sim.apply({"type":"update_probability", "value":command["probability"]})
        elif kind == "fork":
            first = self.simulations[0]
            self.simulations = [first, first.clone()]
            self.ids = [self.ids[0]]*2
            self.mode = "paired"
            self.curves = []
            self.record()
        elif kind == "capture":
            extra["capturePixels"] = render_rgba(self.simulations[0].state).ravel().tolist()
        elif kind == "exportExperiment":
            document = {"schema":"mycelium.laboratory.v1","ids":self.ids,"mode":self.mode,
                        "branches":[s.export() for s in self.simulations], "measurements":self.curves}
            extra["download"] = {"name":"mycelium-experiment.json","mime":"application/json","data":_dump(document)}
        elif kind == "importExperiment":
            document = bounded_json(command.get("data"), 16_000_000)
            if document.get("schema") != "mycelium.laboratory.v1":
                raise ValueError("Expected Mycelium laboratory experiment v1")
            ids, branches = document.get("ids"), document.get("branches")
            mode = document.get("mode")
            if not isinstance(ids,list) or len(ids) not in (1,2) or not isinstance(branches,list) or len(branches)!=len(ids):
                raise ValueError("An experiment contains one or two branches")
            if mode not in ("single","compare","paired") or (mode == "single") != (len(ids)==1):
                raise ValueError("Experiment mode does not match its branch count")
            loaded=[]
            for name, branch in zip(ids,branches):
                if name not in self.models:
                    # Artifact identity is authoritative, rather than its friendly filename.
                    matches=[key for key,value in self.models.items() if value["weights_hash"]==branch.get("model_hash")]
                    if not matches:
                        raise ValueError("Import this experiment's trained model first")
                    name=matches[0]
                sim=Simulation.from_export(branch,self.models[name]["weights"])
                if sim.state.shape != (32,32,16):
                    raise ValueError("The browser release supports 32×32 experiments")
                loaded.append((name,sim))
            if len(loaded)==2:
                a,b=loaded[0][1],loaded[1][1]
                if a.step_count != b.step_count or a.rng.bit_generator.state != b.rng.bit_generator.state or a.update_probability != b.update_probability:
                    raise ValueError("Comparison branches must have synchronized steps, probabilities, and random streams")
                if a.target_hash != b.target_hash:
                    raise ValueError("Comparison branches must share a target")
                if mode == "paired" and a.model_hash != b.model_hash:
                    raise ValueError("A paired perturbation experiment requires identical model weights")
            for name,sim in loaded:
                if sim.target_hash != self.models[name].get("metadata",{}).get("target_sha256", ""):
                    raise ValueError("Experiment target hash differs from model provenance")
                target=self.targets.get(self.entries.get(name,{}).get("target"))
                if target is not None:
                    measure(sim.state,target)
            self.ids=[p[0] for p in loaded]; self.simulations=[p[1] for p in loaded]; self.mode=mode
            self.curves=[]; self.status="ready"; self.record()
        elif kind == "exportModel":
            model=self.models[self.ids[0]]
            extra["download"]={"name":self.ids[0]+".json","mime":"application/json",
                               "data":_dump(model_document(model["weights"],model.get("metadata")))}
        elif kind == "importModel":
            model=load_model(command.get("data"))
            self.models["imported"]=model
            target_hash=model.get("metadata",{}).get("target_sha256")
            target_id=next((t["id"] for t in self.manifest.get("targets",[]) if t.get("sha256")==target_hash),None)
            self.entries["imported"]={"id":"imported","name":"Imported model","regime":model.get("metadata",{}).get("regime","imported"),"target":target_id}
            self.reset({"model":"imported"})
            extra["importedModel"]=self.entries["imported"]
        elif kind == "measurements":
            if command.get("format", "json") == "csv":
                output=io.StringIO(); writer=csv.writer(output)
                writer.writerow(["step","branch","relative_error","iou","visible_divergence","hidden_divergence"])
                for sample in self.curves:
                    for index,error in enumerate(sample["errors"]):
                        d=sample.get("divergence") or {}
                        writer.writerow([sample["step"],index,error,sample["ious"][index],d.get("visible"),d.get("hidden")])
                extra["download"]={"name":"mycelium-measurements.csv","mime":"text/csv","data":output.getvalue()}
            else:
                extra["download"]={"name":"mycelium-measurements.json","mime":"application/json","data":_dump({"partial":len(self.curves)>=512,"samples":self.curves})}
        elif kind != "state":
            raise ValueError("Unknown worker command")
        return {**self.response(),**extra}


def initialize(manifest_json, models_json, targets_json):
    global _lab
    _lab=Laboratory(json.loads(manifest_json),json.loads(models_json),json.loads(targets_json))
    return _dump(_lab.response())


def dispatch(command_json):
    if _lab is None:
        raise ValueError("Runtime has not loaded its models")
    command=bounded_json(command_json,16_000_000)
    return _dump(_lab.dispatch(command))
