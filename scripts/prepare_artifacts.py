"""Validate/copy locally trained exports and preserve their complete compact logs.

This command never trains, selects a checkpoint, or changes an evaluation score.
It publishes only the training loop's already selected data-only model exports.
Run from the repository root after training; --allow-partial is for development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from mycelium_nca.model_io import load_model
from mycelium_nca.targets import target_hash

RELEASE_RUNS = (
    "ring-regeneration", "ring-growth", "ring-persistence",
    "leaf-regeneration", "rosette-regeneration",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def archive_run(source: Path, destination: Path) -> dict:
    """Keep every log record/field; remove only insignificant JSON whitespace."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "config.json", "environment.json", "target.npy"):
        shutil.copyfile(source / name, destination / name)
    records = 0
    last_update = 0
    log = destination / "training.jsonl"
    temporary = log.with_name(log.name + ".tmp")
    with (source / "training.jsonl").open() as incoming, temporary.open("w") as outgoing:
        for line in incoming:
            item = json.loads(line)
            if not isinstance(item.get("update"), int) or item["update"] <= last_update:
                raise ValueError(f"Non-increasing training log updates in {source}")
            last_update = item["update"]
            records += 1
            outgoing.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, log)
    summary = read_json(source / "summary.json")
    if last_update != summary["updates"]:
        raise ValueError(f"Log/summary update count mismatch in {source}")
    snapshot = np.load(destination / "target.npy", allow_pickle=False)
    if target_hash(snapshot) != summary["target_sha256"]:
        raise ValueError(f"Target snapshot hash mismatch in {source}")
    provenance = {
        "source_run": source.as_posix(),
        "records": records,
        "logs_complete": True,
        "note": "All recorded updates and fields retained; JSON whitespace compacted.",
        "files": {
            name: {"sha256": sha256(destination / name), "bytes": (destination / name).stat().st_size}
            for name in ("summary.json", "config.json", "environment.json", "target.npy", "training.jsonl")
        },
        "source_training_log_sha256": sha256(source / "training.jsonl"),
    }
    write_json(destination / "provenance.json", provenance)
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--targets", type=Path, default=Path("targets"))
    parser.add_argument("--reports", type=Path, default=Path("reports/training"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    available = [name for name in RELEASE_RUNS if (args.runs / name / "summary.json").exists() and (args.runs / name / "model.json").exists()]
    missing = [name for name in RELEASE_RUNS if name not in available]
    if missing and not args.allow_partial:
        parser.error("Missing finished trained exports: " + ", ".join(missing))
    if not available:
        parser.error("No finished trained exports are available")
    # Validate every selected model before writing the release manifest.
    selected = []
    for name in available:
        run = args.runs / name
        summary = read_json(run / "summary.json")
        model = load_model(run / "model.json")
        metadata = model["metadata"]
        target, regime = name.split("-", 1)
        if metadata.get("target") != target or metadata.get("regime") != regime:
            raise ValueError(f"Model identity does not match run {name}")
        if summary.get("best_update") != metadata.get("checkpoint_update"):
            raise ValueError(f"Model is not the selected validation checkpoint for {name}")
        expected_hash = read_json(args.targets / f"{target}.json")["sha256"]
        if metadata.get("target_sha256") != expected_hash or summary.get("target_sha256") != expected_hash:
            raise ValueError(f"Model target does not match released artwork for {name}")
        selected.append((name, run, summary, model))
    args.models.mkdir(parents=True, exist_ok=True)
    records = []
    run_summaries = []
    for name, run, summary, model in selected:
        provenance = archive_run(run, args.reports / name)
        destination = args.models / f"{name}.json"
        shutil.copyfile(run / "model.json", destination)
        metadata = model["metadata"]
        best_checkpoint = run / "checkpoints/best.pt"
        records.append({
            "id": name, "name": name.replace("-", " · ").title(),
            "target": metadata["target"], "regime": metadata["regime"],
            "file": destination.name,
            "bytes": destination.stat().st_size,
            "model_sha256": sha256(destination),
            "weights_hash": model["weights_hash"],
            "target_sha256": metadata["target_sha256"],
            "checkpoint_update": metadata["checkpoint_update"],
            "native_checkpoint_sha256": sha256(best_checkpoint) if best_checkpoint.exists() else None,
            "training_seed": metadata["training_seed"],
            "selection": metadata["selection"],
            "training_summary": (args.reports / name / "summary.json").as_posix(),
            "training_log": (args.reports / name / "training.jsonl").as_posix(),
            "provenance": (args.reports / name / "provenance.json").as_posix(),
            "training_log_sha256": provenance["files"]["training.jsonl"]["sha256"],
        })
        run_summaries.append({"id": name, **summary})
    pilots = []
    for run in sorted(args.runs.glob("pilot-*")):
        if not (run / "summary.json").exists():
            continue
        archive_run(run, args.reports / run.name)
        pilots.append({"id": run.name, **read_json(run / "summary.json")})
    target_names = sorted({record["target"] for record in records})
    manifest = {
        "schema": "mycelium.gallery.v1",
        "defaultTarget": "ring", "anchorTarget": "ring",
        "complete_model_set": not missing,
        "missing_models": missing,
        "quality_claim": "Artifact presence is not a quality gate. Read the frozen evaluation reports.",
        "models": records,
        "targets": [{**read_json(args.targets / f"{name}.json"), "id": name,
                     "json": f"targets/{name}.json", "preview": f"targets/{name}.png"}
                    for name in target_names],
    }
    write_json(args.models / "manifest.json", manifest)
    write_json(args.reports / "summary.json", {
        "schema": "mycelium.training-summary.v1", "runs": run_summaries, "pilots": pilots,
        "total_release_updates": sum(item["updates"] for item in run_summaries),
        "total_release_training_seconds": sum(item["training_seconds"] for item in run_summaries),
        "maximum_release_process_rss_mib": max(item["peak_process_rss_mb"] for item in run_summaries),
        "failed_pilot_note": "The original transparent-center ring produced empty states and zero gradients. Pilots measured resource cost; they did not produce usable models. Original target snapshots/hashes are preserved.",
    })
    print(json.dumps({"models": len(records), "missing": missing,
                      "manifest": str(args.models / "manifest.json"),
                      "training_reports": str(args.reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
