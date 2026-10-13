"""Command-line entry points; runtime-only commands do not import PyTorch."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mycelium", description="Train and investigate tiny local neural cellular automata.")
    commands = parser.add_subparsers(dest="command", required=True)
    targets = commands.add_parser("targets", help="reproduce original target arrays and PNGs")
    targets.add_argument("--output", default="targets")
    train = commands.add_parser("train", help="train exactly one bounded native model")
    train.add_argument("--config", help="JSON TrainConfig; flags override it")
    for name in ("target", "regime", "device", "output-dir", "resume"):
        train.add_argument("--"+name)
    for name in ("seed", "updates", "batch-size", "pool-size", "unroll-min", "unroll-max", "threads", "checkpoint-every", "validation-every", "log-every"):
        train.add_argument("--"+name, type=int)
    for name in ("wall-seconds", "memory-mb", "learning-rate"):
        train.add_argument("--"+name, type=float)
    pilot = commands.add_parser("pilot", help="measure a ~50-update run including the largest unroll")
    pilot.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    pilot.add_argument("--updates", type=int, default=50)
    pilot.add_argument("--threads", type=int, default=2)
    pilot.add_argument("--output-dir")
    pilot.add_argument("--memory-mb", type=float, default=3000)
    pilot.add_argument("--wall-seconds", type=float, default=600)
    export = commands.add_parser("export", help="export a trusted local native checkpoint to browser-safe JSON")
    export.add_argument("checkpoint")
    export.add_argument("output")
    evaluate = commands.add_parser("evaluate", help="run the frozen final benchmark")
    evaluate.add_argument("--models", default="models")
    evaluate.add_argument("--output", default="reports")
    evaluate.add_argument("--config", default="configs/evaluation-v1.json")
    args = parser.parse_args(argv)
    if args.command == "targets":
        from .targets import write_targets
        result = write_targets(args.output)
    elif args.command == "train":
        from .training import TrainConfig, run_training
        values = json.loads(Path(args.config).read_text()) if args.config else {}
        valid_names = {field.name for field in fields(TrainConfig)}
        for key, value in vars(args).items():
            if key in valid_names and value is not None:
                values[key] = value
        if "output_dir" not in values:
            values["output_dir"] = f"runs/{Path(values.get('target', 'ring')).stem}-{values.get('regime', 'regeneration')}"
        result = run_training(TrainConfig(**values))
    elif args.command == "pilot":
        from .training import run_pilot
        result = run_pilot(**{key: value for key, value in vars(args).items() if key != "command"})
    elif args.command == "export":
        from .training import export_checkpoint
        result = export_checkpoint(args.checkpoint, args.output)
    else:
        from .evaluation import run_evaluation
        result = run_evaluation(args.models, args.output, args.config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
