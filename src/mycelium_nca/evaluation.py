"""Frozen, exhaustive release benchmark; failures remain first-class trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from .metrics import measure, acceptable, normalized_recovery, repair_outcome
from .numpy_sim import step, seed_state, check_numerical
from .model_io import bounded_json, load_model
from .targets import target_hash


def _environment():
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "runtime": "native NumPy CPU",
            "rng": "PCG64; Generator.random((height,width),dtype=float32) < p; C-order"}


def _rss_mb():
    # ru_maxrss is a process-lifetime high-water mark, not an isolated model allocation.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2 if sys.platform == "darwin" else 1024))


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _aggregate(group):
    errors = [t["final"]["relative_error"] for t in group
              if t["final"] is not None and t["final"]["relative_error"] is not None]
    successes = sum(bool(t["success"]) for t in group)
    result = {"successes": successes, "failures": len(group) - successes, "trials": len(group),
              "success_rate": successes / len(group) if group else None,
              "numerical_failures": sum(t["numerical_failure"] is not None for t in group),
              "mean_relative_error": float(np.mean(errors)) if errors else None,
              "mean_error_samples": len(errors),
              "mean_error_note": "Last valid state per trial, including failures; failed horizons are explicitly recorded."}
    if group and group[0]["kind"] == "repair":
        conditional = [t for t in group if t.get("pre_damage_acceptable") and t.get("meaningful_damage")]
        recovery_times = [t["recovery_time"] for t in group if t.get("success")]
        result.update(pre_damage_failures=sum(not t.get("pre_damage_acceptable", False) for t in group),
                      negligible_damage_trials=sum(t.get("pre_damage_acceptable", False) and not t.get("meaningful_damage", False) for t in group),
                      conditional_trials=len(conditional),
                      conditional_success_rate=sum(t["success"] for t in conditional) / len(conditional) if conditional else None,
                      retained_recoveries=sum(bool(t.get("retained_recovery")) for t in group),
                      mean_recovery_time=float(np.mean(recovery_times)) if recovery_times else None)
    return result


def _plot_report(report, path):
    """Small dependency-free SVG: every seed contributes; no interpolated observations."""
    width, height = 1080, 380
    colors = ["#a9ce76", "#f1ab87", "#9cace0", "#e1ca77", "#a0d6cf", "#c899b7", "#b5b5b5", "#cadd9a", "#e68b74"]
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(report["model"])} recorded evaluation curves">',
              '<rect width="100%" height="100%" fill="#151c19"/>',
              '<g font-family="system-ui,sans-serif" font-size="11" fill="#ebece2">',
              f'<text x="24" y="24" font-size="16">{escape(report["model"])} · all frozen trials</text>']
    for column, kind in enumerate(("growth", "persistence", "repair")):
        group = [t for t in report["trials"] if t["kind"] == kind]
        x0, y0, plot_w, plot_h = 55 + column * 350, 74, 285, 220
        horizon = max((t["horizon"] for t in group), default=1)
        values = [s["relative_error"] for t in group for s in t["curve"] if s["relative_error"] is not None]
        upper = max(1.05, max(values, default=1) * 1.03)
        summary = report["aggregates"][kind]
        pieces.append(f'<text x="{x0}" y="50" font-size="14">{kind.title()} · {summary["successes"]}/{summary["trials"]} success</text>')
        for fraction in (0, .5, 1):
            y = y0 + plot_h * (1-fraction)
            pieces.extend([f'<path d="M{x0} {y:.2f}h{plot_w}" stroke="#34433b"/>',
                           f'<text x="{x0-7}" y="{y+4:.2f}" text-anchor="end">{upper*fraction:.2g}</text>'])
        threshold_y = y0 + plot_h * (1-report["config"]["quality"]["maximum_relative_error"]/upper)
        pieces.append(f'<path d="M{x0} {threshold_y:.2f}h{plot_w}" stroke="#eee9c9" stroke-dasharray="3 4"/>')
        for index, trial in enumerate(group):
            points = " ".join(f'{x0+s["step"]/horizon*plot_w:.2f},{y0+(1-s["relative_error"]/upper)*plot_h:.2f}'
                              for s in trial["curve"] if s["relative_error"] is not None)
            label = f'{trial["id"]}; success={trial["success"]}; completed={trial["completed_steps"]}/{horizon}'
            pieces.append(f'<polyline points="{points}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="1.4" opacity=".8"><title>{escape(label)}</title></polyline>')
        for fraction in (0, .5, 1):
            pieces.append(f'<text x="{x0+fraction*plot_w:.2f}" y="{y0+plot_h+20}" text-anchor="middle">{int(horizon*fraction)}</text>')
        pieces.append(f'<text x="{x0+plot_w/2}" y="{y0+plot_h+39}" text-anchor="middle">Simulation step</text>')
    pieces.extend(['<text x="15" y="215" transform="rotate(-90 15 215)">Error / empty baseline</text>',
                   '<text x="24" y="366">Each line is one recorded trial. Dashed line: error threshold; success also requires IoU and regime-specific criteria.</text>', '</g></svg>'])
    path.write_text("\n".join(pieces) + "\n")


def _write_csvs(report, destination):
    fields = ["trial", "kind", "seed", "damage_family", "step", "phase", "elapsed", "raw_mse", "balanced_mse",
              "empty_baseline", "relative_error", "iou", "occupied", "max_abs_state", "recovery", "success", "numerical_failure"]
    with (destination / f"{report['model']}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in report["trials"]:
            for sample in trial["curve"]:
                row = {**sample, "trial": trial["id"], "damage_family": trial.get("damage", {}).get("name"),
                       **{key: trial[key] for key in ("kind", "seed", "success", "numerical_failure")}}
                writer.writerow({key: row.get(key) for key in fields})
    fields = ["id", "kind", "seed", "success", "completed_steps", "horizon", "numerical_failure", "failure_step",
              "pre_damage_acceptable", "meaningful_damage", "removed_fraction", "damage_severity", "recovery_time",
              "retained_recovery", "deterioration_after_recovery", "final_acceptable", "persistence_failure"]
    with (destination / f"{report['model']}-trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: trial.get(key) for key in fields} for trial in report["trials"])


def evaluate_model(model_path, target_path, config_path="configs/evaluation-v1.json", output="reports"):
    started = time.perf_counter()
    config_bytes = Path(config_path).read_bytes()
    cfg = json.loads(config_bytes)
    if cfg.get("schema") != "mycelium.evaluation.v1" or cfg.get("grid_size") != 32:
        raise ValueError("Expected the Mycelium v1 32×32 evaluation suite.")
    artifact = load_model(model_path)
    weights = artifact["weights"]
    target = np.load(target_path, allow_pickle=False)
    if target.shape != (32, 32, 4) or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError("Evaluation target must be finite 32×32×4 float32 premultiplied RGBA.")
    if np.any(target < 0) or np.any(target > 1) or np.any(target[..., :3] > target[..., 3:4] + 1e-7):
        raise ValueError("Evaluation target is not valid premultiplied RGBA.")
    target_sha256 = target_hash(target)
    if artifact.get("metadata", {}).get("target_sha256") != target_sha256:
        raise ValueError("Evaluation target hash does not match the model's training target.")
    quality = cfg["quality"]
    trials = []

    def measurements(state):
        return measure(state, target, cfg["alpha_threshold"])

    def run(kind, seed, horizon, intervention=None, probability=.5):
        rng = np.random.Generator(np.random.PCG64(seed))
        state = seed_state()
        lesions = np.zeros((32, 32), bool)
        record = {"id": f"{kind}:{intervention['name'] + ':' if intervention else ''}{seed}",
                  "kind": kind, "seed": seed, "probability": probability, "horizon": horizon,
                  "curve": [], "events": [], "numerical_failure": None, "failure_step": None}
        if intervention:
            record["damage"] = intervention
        pre, after, removed = None, None, None
        damage_step = cfg["damage_step"]
        completed_steps = 0

        def sample(t, phase="rollout"):
            value = dict(step=t, phase=phase, **measurements(state))
            if after is not None:
                value.update(elapsed=t-damage_step,
                             recovery=normalized_recovery(pre["balanced_mse"], after["balanced_mse"], value["balanced_mse"]))
            record["curve"].append(value)
            return value

        try:
            for t in range(horizon+1):
                damage_now = intervention is not None and (t == damage_step or (intervention.get("repeat") and t in (352, 448)))
                if damage_now:
                    before_event = sample(t, "before_damage")
                    previous_occupied = state[..., 3] > cfg["alpha_threshold"]
                    x0, y0, x1, y1 = intervention["rect"]
                    state[y0:y1, x0:x1] = 0
                    if intervention.get("persistent"):
                        lesions[y0:y1, x0:x1] = True
                    fraction = float((previous_occupied & (state[..., 3] <= cfg["alpha_threshold"])).sum() / max(1, previous_occupied.sum()))
                    after_event = measurements(state)
                    record["events"].append({"step": t, "intervention": intervention, "before": before_event,
                                             "immediately_after": after_event, "removed_fraction": fraction})
                    if t == damage_step:
                        pre, after, removed = before_event, after_event, fraction
                        record.update(before_damage=pre, after_damage=after, removed_fraction=removed,
                                      damage=intervention,
                                      damage_severity="moderate" if removed <= cfg["moderate_maximum_removed_fraction"] else "severe")
                    sample(t, "immediately_after_damage")
                elif t % cfg["measurement_interval"] == 0 or t == horizon:
                    sample(t)
                if t == horizon:
                    break
                p = probability if t >= damage_step else .5
                mask = rng.random((32, 32), dtype=np.float32) < p
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = step(state, weights, mask, lesion=lesions)
                check_numerical(candidate)
                state = candidate
                completed_steps = t + 1
        except FloatingPointError as exc:
            record["numerical_failure"] = str(exc)
            record["failure_step"] = completed_steps + 1
            # Retain the last valid state, never a stale successful measurement.
            if not record["curve"] or record["curve"][-1]["step"] != completed_steps:
                sample(completed_steps, "last_valid_before_failure")
        record["completed_steps"] = completed_steps
        record["horizon_reached"] = completed_steps == horizon
        record["final"] = record["curve"][-1] if record["curve"] else None
        record["final_acceptable"] = bool(record["final"] and acceptable(record["final"], quality))
        record["success"] = bool(not record["numerical_failure"] and record["horizon_reached"] and record["final_acceptable"])
        if kind == "persistence":
            mature = [s for s in record["curve"] if s["step"] >= cfg["growth_horizon"]]
            record["persistence_failure"] = bool(record["numerical_failure"] or not mature or any(not acceptable(s, quality) for s in mature))
            record["success"] = not record["persistence_failure"]
        if intervention:
            # Pre-damage numerical failures remain end-to-end repair failures.
            if pre is None:
                record.update(pre_damage_acceptable=False, meaningful_damage=False, success=False,
                              recovery_time=None, retained_recovery=False, deterioration_after_recovery=False)
            else:
                repair_curve = [s for s in record["curve"] if s.get("elapsed", 0) > 0 and s["phase"] == "rollout"]
                record.update(repair_outcome(pre, removed, repair_curve, quality, cfg["minimum_removed_fraction"], cfg["sustained_samples"]))
                if record["numerical_failure"]:
                    record["success"] = False
                    record["recovery_time"] = None
                    record["retained_recovery"] = False
        if kind == "stress:total_erasure":
            record["extinction_expected"] = True
            record["remained_extinct"] = bool(after is not None and not record["numerical_failure"] and not np.any(state))
        trials.append(record)

    for seed in cfg["growth_seeds"]:
        run("growth", seed, cfg["growth_horizon"])
    for seed in cfg["persistence_seeds"]:
        run("persistence", seed, cfg["persistence_horizon"])
    for family in cfg["damage_families"]:
        for seed in cfg["repair_seeds"]:
            run("repair", seed, cfg["damage_step"]+cfg["repair_horizon"], family)
    stress = {
        "total_erasure": {"name": "total_erasure", "rect": [0, 0, 32, 32]},
        "large_cut": {"name": "large_cut", "rect": [0, 0, 20, 32]},
        "persistent_lesion": {"name": "persistent_lesion", "rect": [14, 5, 17, 27], "persistent": True},
        "repeated_damage": {"name": "repeated_damage", "rect": [16, 9, 23, 18], "repeat": True},
    }
    for name in cfg["stress"]:
        probability = float(name.split("_")[-1]) if name.startswith("probability_") else .5
        if not name.startswith("probability_") and name not in stress:
            raise ValueError(f"Unknown stress test {name!r}")
        run("stress:"+name, cfg["stress_seed"], cfg["damage_step"]+cfg["repair_horizon"], stress.get(name), probability)
    aggregates = {kind: _aggregate([t for t in trials if t["kind"] == kind]) for kind in ("growth", "persistence", "repair")}
    aggregates["repair"]["by_severity"] = {severity: _aggregate([t for t in trials if t["kind"] == "repair" and t.get("damage_severity") == severity])
                                                for severity in ("moderate", "severe")}
    aggregates["repair"]["by_family"] = {family["name"]: _aggregate([t for t in trials if t["kind"] == "repair" and t.get("damage", {}).get("name") == family["name"]])
                                              for family in cfg["damage_families"]}
    model_path = Path(model_path)
    model_id = model_path.parent.name if model_path.stem == "model" else model_path.stem
    report = {"schema": "mycelium.results.v1", "model": model_id,
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "weights_hash": artifact["weights_hash"], "target_sha256": target_sha256,
              "metadata": artifact.get("metadata", {}),
              "evaluation_sha256": hashlib.sha256(config_bytes).hexdigest(), "config": cfg,
              "environment": _environment(), "duration_seconds": time.perf_counter()-started,
              "peak_process_rss_mb": _rss_mb(), "memory_measurement": "Process-lifetime ru_maxrss; includes preceding evaluations in this process.",
              "completed_transitions": sum(t["completed_steps"] for t in trials),
              "recovery_time_convention": "Elapsed steps at confirmation of first 3 consecutive acceptable post-damage samples; retained_recovery separately records later deterioration.",
              "aggregates": aggregates, "trials": trials}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / f"{model_id}.json", report)
    _write_csvs(report, destination)
    _plot_report(report, destination / f"{model_id}.svg")
    return report


def run_evaluation(models_dir="models", output_dir="reports", config_path="configs/evaluation-v1.json"):
    """Evaluate every supplied artifact serially; return the public summary index."""
    source = Path(models_dir)
    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No trained model JSON artifacts were found.")
    plan = []
    for path in paths:
        # A directory may include a manifest, but unknown model-like files must fail loudly.
        document = bounded_json(path)
        if document.get("schema") != "mycelium.model.v1" and path.name == "manifest.json":
            continue
        artifact = load_model(document)
        metadata = artifact["metadata"]
        name = metadata.get("target")
        candidates = [path.parent / "target.npy"]
        if isinstance(name, str) and Path(name).name == name:
            candidates.extend([Path(config_path).resolve().parent.parent / "targets" / f"{name}.npy",
                               Path("targets") / f"{name}.npy"])
        target_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if target_path is None:
            raise ValueError(f"Cannot locate exact target.npy or targets/{name}.npy for {path}.")
        plan.append((path, target_path))
    if not plan:
        raise ValueError("No trained model JSON artifacts were found.")
    reports = []
    for path, target_path in plan:
        report = evaluate_model(path, target_path, config_path, output_dir)
        reports.append(report)
        print(f"Evaluated {report['model']}: {len(report['trials'])} trials in {report['duration_seconds']:.2f}s", file=sys.stderr, flush=True)
    first = reports[0]
    models = []
    for report in reports:
        model_id = report["model"]
        models.append({"id": model_id, "target": report["metadata"].get("target"), "regime": report["metadata"].get("regime"),
                       "model_sha256": report["model_sha256"], "weights_hash": report["weights_hash"],
                       "metadata": report["metadata"], "aggregates": report["aggregates"], **report["aggregates"],
                       "duration_seconds": report["duration_seconds"], "peak_process_rss_mb": report["peak_process_rss_mb"],
                       "report_json": f"{model_id}.json", "report_csv": f"{model_id}.csv",
                       "trials_csv": f"{model_id}-trials.csv", "plot": f"{model_id}.svg"})
    summary = {"schema": "mycelium.evaluation.summary.v1", "models": models,
               "description": "Frozen, exhaustive local NumPy evaluation of known training targets; every trial, including failures, is retained.",
               "limitations": ["One training run per regime is a case study, not independent training replication.",
                               "Repair success requires an acceptable mature state, meaningful damage, and 3 consecutive acceptable measurements; later deterioration is reported separately.",
                               "Growth and persistence use distinct frozen stochastic schedules. Transparent background is balanced against target foreground.",
                               "Severe damage, persistent lesions, update-probability changes and total erasure are stress tests, not promised supported recovery conditions.",
                               "Native timing measures this recorded environment; browser performance is measured separately."],
               "evaluation_sha256": first["evaluation_sha256"], "config": first["config"], "environment": first["environment"],
               "total_trials": sum(len(report["trials"]) for report in reports),
               "total_duration_seconds": sum(report["duration_seconds"] for report in reports),
               "peak_process_rss_mb": _rss_mb()}
    destination = Path(output_dir)
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "evaluation-config.json", first["config"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("target")
    parser.add_argument("--config", default="configs/evaluation-v1.json")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    result = evaluate_model(args.model, args.target, args.config, args.output)
    print(json.dumps({"model": result["model"], "aggregates": result["aggregates"], "seconds": result["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
