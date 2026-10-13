"""Sequential five-run release protocol; never parallelizes trainers.

Run the CPU/MPS pilots first, then explicitly choose a measured device. The
first anchor run must be inspected and evaluated before passing --gallery.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from mycelium_nca.training import TrainConfig, run_training

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train-default.json")
parser.add_argument("--device", choices=("cpu", "mps"), required=True)
parser.add_argument("--updates", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gallery", action="store_true", help="run controls/gallery after checking the anchor")
args = parser.parse_args()
config = TrainConfig(**json.loads(Path(args.config).read_text()))
config = replace(config, device=args.device, seed=args.seed)
if args.updates is not None:
    config = replace(config, updates=args.updates)
runs = [("ring", "regeneration")]
if args.gallery:
    anchor = Path("runs/ring-regeneration/summary.json")
    if not anchor.exists() or not Path("models/ring-regeneration.json").exists():
        raise SystemExit("Train, export, inspect, and evaluate the anchor first without --gallery.")
    runs = [("ring", "growth"), ("ring", "persistence"), ("leaf", "regeneration"), ("rosette", "regeneration")]
Path("models").mkdir(exist_ok=True)
for target, regime in runs:
    name = f"{target}-{regime}"
    result = run_training(replace(config, target=target, regime=regime, output_dir=f"runs/{name}"))
    if result["model"]:
        shutil.copyfile(result["model"], f"models/{name}.json")
    print(json.dumps(result, indent=2))
    if result["status"] != "update_budget_exhausted":
        raise SystemExit(f"Stopped sequential release after {name}: {result['status']}")
