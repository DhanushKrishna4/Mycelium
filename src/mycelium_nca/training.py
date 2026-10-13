"""Bounded native training. Torch is imported only when training is requested.

Native resumable checkpoints are trusted-local PyTorch files, never browser
imports. Browser model exports are validated, data-only JSON via model_io.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from .targets import load_target, target_hash

VALIDATION_SEEDS = (910_031, 910_037)
CHECKPOINT_SCHEMA = "mycelium-native-checkpoint-v1"


@dataclass
class TrainConfig:
    target: str = "ring"
    regime: str = "regeneration"
    seed: int = 42
    updates: int = 2500
    batch_size: int = 4
    pool_size: int = 128
    unroll_min: int = 48
    unroll_max: int = 64
    learning_rate: float = .002
    lr_decay_at: int = 2000
    lr_decay: float = .1
    update_probability: float = .5
    wall_seconds: float = 3600
    memory_mb: float = 3000
    device: str = "cpu"
    threads: int = 2
    checkpoint_every: int = 100
    validation_every: int = 100
    output_dir: str = "runs/ring-regeneration"
    resume: str | None = None
    # Exposed for a short resource pilot; full training leaves these false.
    pilot: bool = False
    log_every: int = 10

    def validate(self) -> None:
        if self.regime not in ("growth", "persistence", "regeneration"):
            raise ValueError("regime must be growth, persistence, or regeneration")
        if not 1 <= self.updates <= 100_000:
            raise ValueError("updates must be in 1..100000 (default release cap: 2500)")
        if not 1 <= self.batch_size <= 8 or not self.batch_size <= self.pool_size <= 256:
            raise ValueError("batch must be 1..8 and pool must be batch..256")
        if self.regime == "regeneration" and self.batch_size < 2:
            raise ValueError("regeneration needs >=2 examples to preserve a seed example")
        if not 1 <= self.unroll_min <= self.unroll_max <= 96:
            raise ValueError("bounded unroll must satisfy 1 <= min <= max <= 96")
        if not 0 < self.update_probability <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid update probability or learning rate")
        if self.wall_seconds <= 0 or self.memory_mb <= 0:
            raise ValueError("wall and memory budgets must be positive")
        if self.device not in ("cpu", "mps") or not 1 <= self.threads <= 8:
            raise ValueError("device must be cpu or mps, threads must be 1..8")
        if self.checkpoint_every < 1 or self.validation_every < 0 or self.log_every < 1:
            raise ValueError("invalid checkpoint, validation, or logging interval")


def make_seed(size: int = 32) -> np.ndarray:
    seed = np.zeros((size, size, 16), np.float32)
    seed[size//2, size//2, 3:] = 1
    return seed


class StatePool:
    """CPU storage has no autograd history; sorted identities travel with states."""

    def __init__(self, seed: np.ndarray, count: int):
        self.states = np.repeat(seed[None], count, axis=0)
        self.write_counts = np.zeros(count, np.int64)

    def sample(self, count: int, rng: np.random.Generator, target: np.ndarray):
        identities = rng.choice(len(self.states), count, replace=False)
        batch = self.states[identities].copy()
        losses = ((batch[..., :4]-target)**2).mean(axis=(1, 2, 3))
        order = np.argsort(-losses, kind="stable")
        return identities[order], batch[order]

    def commit(self, identities: np.ndarray, states) -> None:
        # A tensor passed by a caller is explicitly detached before storage.
        if hasattr(states, "detach"):
            states = states.detach().cpu().numpy()
        values = np.asarray(states, dtype=np.float32)
        if values.shape != self.states[identities].shape:
            raise ValueError("pool writeback shape mismatch")
        if len(np.unique(identities)) != len(identities):
            raise ValueError("pool identities must be distinct")
        self.states[identities] = values
        self.write_counts[identities] += 1


def training_streams(seed: int) -> dict[str, np.random.Generator]:
    """Independent, serialized PCG64 streams; no PyTorch/NumPy seed equivalence."""
    return {name: np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, offset])))
            for name, offset in (("pool", 101), ("damage", 211), ("updates", 307), ("unroll", 401))}


def damage_batch(batch: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Erase disks in the lowest-loss half, retaining the first seed example.

    Reject disks that remove all mature cells. During early training, damage
    may be skipped; the rejected attempt count is returned for provenance.
    """
    count = max(1, len(batch)//2)
    yy, xx = np.ogrid[:batch.shape[1], :batch.shape[2]]
    records = []
    for index in range(max(1, len(batch)-count), len(batch)):
        mature = batch[index, ..., 3] > .1
        occupied = np.argwhere(mature)
        if len(occupied) < 8:
            records.append({"sample": index, "applied": False, "reason": "insufficient_living_cells"})
            continue
        for attempt in range(8):
            cy, cx = occupied[int(rng.integers(len(occupied)))]
            radius = float(rng.uniform(2, 6))
            mask = (xx-cx)**2+(yy-cy)**2 <= radius**2
            removed = int((mature & mask).sum())
            fraction = removed / len(occupied)
            if .05 <= fraction <= .65:
                batch[index, mask] = 0
                records.append({"sample": index, "applied": True, "x": int(cx), "y": int(cy),
                                "radius": radius, "occupied_fraction_removed": fraction,
                                "rejected_attempts": attempt})
                break
        else:
            records.append({"sample": index, "applied": False, "reason": "no_safe_disk", "rejected_attempts": 8})
    return records


def peak_memory_mb() -> float:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum / (1024**2 if sys.platform == "darwin" else 1024)


def _sync(torch, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _memory(torch, device: str) -> dict:
    result = {"peak_process_rss_mb": peak_memory_mb()}
    if device == "mps":
        result.update(mps_allocated_mb=torch.mps.current_allocated_memory()/1024**2,
                      mps_driver_mb=torch.mps.driver_allocated_memory()/1024**2)
    return result


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def _atomic_checkpoint(torch, path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(torch, path: str | Path) -> dict:
    # These files are explicitly local-only; the browser never accepts pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("incompatible native training checkpoint")
    return checkpoint


def _validation(model, torch, target, device: str) -> dict:
    """Frozen checkpoint-selection suite, disjoint from final evaluation.

    Two seed schedules: grow128, persist256, erase a fixed small disk then
    recover128. Every regime is scored on precisely the same three phases.
    """
    states = torch.as_tensor(np.repeat(make_seed()[None], len(VALIDATION_SEEDS), axis=0), device=device)
    randoms = [np.random.Generator(np.random.PCG64(seed)) for seed in VALIDATION_SEEDS]
    yy, xx = np.ogrid[:32, :32]
    erase = torch.as_tensor(((xx-21)**2+(yy-15)**2 <= 3.5**2)[None, ..., None], device=device)
    foreground = target[..., 3] >= .5
    phase_scores = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for step in range(1, 385):
            if step == 257:
                states = states.masked_fill(erase, 0)
            masks = np.stack([rng.random((32, 32, 1), dtype=np.float32) < .5 for rng in randoms])
            states = model(states, torch.as_tensor(masks, device=device))
            if step in (128, 256, 384):
                if not bool(torch.isfinite(states).all()) or float(states.abs().max()) > 1000:
                    model.train(was_training)
                    return {"valid": False, "score": None, "reason": "numerical_failure", "seeds": list(VALIDATION_SEEDS)}
                errors = ((states[..., :4]-target)**2).mean(dim=-1)
                balanced = .5*errors[:, foreground].mean()+.5*errors[:, ~foreground].mean()
                phase_scores[{128: "growth", 256: "persistence", 384: "repair"}[step]] = float(balanced)
    model.train(was_training)
    return {"valid": True, "score": sum(phase_scores.values())/3, "balanced_errors": phase_scores,
            "seeds": list(VALIDATION_SEEDS), "suite": "validation-v1", "horizons": [128, 256, 384]}


def run_training(config: TrainConfig) -> dict:
    """Run one bounded training process and atomically preserve resumable state."""
    config.validate()
    import torch
    from .torch_model import NCA
    from .model_io import save_model

    torch.set_num_threads(config.threads)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; use --device cpu")
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if not config.resume and (destination / "training.jsonl").exists():
        raise ValueError("output directory already has a run; use --resume or choose another directory")
    target_numpy = load_target(config.target)
    sha = target_hash(target_numpy)
    target = torch.as_tensor(target_numpy, device=config.device)
    model = NCA(seed=config.seed).to(config.device)
    initial_weights = model.weights_numpy()
    initial_hash = hashlib.sha256(b"".join(initial_weights[key].astype("<f4").tobytes() for key in sorted(initial_weights))).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [config.lr_decay_at], gamma=config.lr_decay)
    seed = make_seed()
    pool = StatePool(seed, config.pool_size)
    streams = training_streams(config.seed)
    counter = 0
    total_cell_steps = 0
    accumulated_seconds = 0.0
    best_score = float("inf")
    best_update = None
    durations = []
    validation = None
    extinct_updates = 0
    if config.resume:
        previous = _load_checkpoint(torch, config.resume)
        # Resource budgets, logging and destination may change on a resume;
        # mathematical settings and training identity may not.
        ignored = {"updates", "wall_seconds", "memory_mb", "output_dir", "resume", "log_every", "checkpoint_every", "validation_every"}
        for name, value in asdict(config).items():
            if name not in ignored and previous["config"][name] != value:
                raise ValueError(f"resume would change {name}; start a separately named run")
        if previous["target_sha256"] != sha:
            raise ValueError("resume target changed")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        pool.states = previous["pool"].copy()
        pool.write_counts = previous["pool_write_counts"].copy()
        counter = previous["update"]
        total_cell_steps = previous["total_cell_steps"]
        accumulated_seconds = previous["elapsed_seconds"]
        best_score = previous["best_score"]
        best_update = previous["best_update"]
        durations = previous["durations"]
        extinct_updates = previous.get("consecutive_extinct_updates", 0)
        for key, state in previous["random_states"].items():
            streams[key].bit_generator.state = state
        torch.set_rng_state(previous["torch_random_state"])
        if config.device == "mps" and previous.get("mps_random_state") is not None:
            torch.mps.set_rng_state(previous["mps_random_state"])
    np.save(destination / "target.npy", target_numpy, allow_pickle=False)
    _atomic_json(destination / "config.json", asdict(config))
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "platform": platform.platform(), "machine": platform.machine(), "device": config.device,
                   "threads": config.threads, "mps_available": torch.backends.mps.is_available()}
    _atomic_json(destination / "environment.json", environment)
    started = time.monotonic()
    status = "update_budget_exhausted"
    failure = None
    last_loss = None
    last_damage = []

    def payload() -> dict:
        return {"schema": CHECKPOINT_SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "pool": pool.states, "pool_write_counts": pool.write_counts,
                "update": counter, "config": asdict(config), "target_sha256": sha,
                "initial_weights_sha256": initial_hash, "total_cell_steps": total_cell_steps,
                "elapsed_seconds": accumulated_seconds + time.monotonic()-started,
                "best_score": best_score, "best_update": best_update, "durations": durations,
                "consecutive_extinct_updates": extinct_updates,
                "random_states": {key: copy.deepcopy(rng.bit_generator.state) for key, rng in streams.items()},
                "torch_random_state": torch.get_rng_state(),
                "mps_random_state": torch.mps.get_rng_state() if config.device == "mps" else None}

    def checkpoint() -> Path:
        path = checkpoint_dir / f"step-{counter:06d}.pt"
        _atomic_checkpoint(torch, path, payload())
        for expired in sorted(checkpoint_dir.glob("step-*.pt"))[:-2]:
            expired.unlink()
        return path

    def select_export() -> None:
        nonlocal best_score, best_update, validation
        validation = _validation(model, torch, target, config.device)
        if validation["valid"] and validation["score"] < best_score:
            best_score = validation["score"]
            best_update = counter
            _atomic_checkpoint(torch, checkpoint_dir / "best.pt", payload())
            metadata = {"name": f"{Path(config.target).stem}-{config.regime}", "target": Path(config.target).stem,
                        "regime": config.regime, "target_sha256": sha, "training_seed": config.seed,
                        "checkpoint_update": counter, "validation": validation, "training_config": asdict(config),
                        "initial_weights_sha256": initial_hash, "optimizer_updates": counter,
                        "total_cell_steps": total_cell_steps,
                        "training_seconds": accumulated_seconds+time.monotonic()-started,
                        "selection": "minimum frozen validation-v1 balanced error; no final-evaluation selection"}
            export = destination / "model.json"
            temporary = destination / "model.tmp.json"
            save_model(temporary, model.weights_numpy(), metadata)
            os.replace(temporary, export)

    log_path = destination / "training.jsonl"
    with log_path.open("a", buffering=1) as log:
        while counter < config.updates:
            if time.monotonic()-started >= config.wall_seconds:
                status = "wall_budget_exhausted"
                break
            measured_memory = _memory(torch, config.device)
            if max(measured_memory.values()) >= config.memory_mb:
                status = "memory_budget_exhausted"
                break
            tick = time.monotonic()
            if config.regime == "growth":
                identities = None
                batch = np.repeat(seed[None], config.batch_size, axis=0)
            else:
                identities, batch = pool.sample(config.batch_size, streams["pool"], target_numpy)
                batch[0] = seed
            last_damage = damage_batch(batch, streams["damage"]) if config.regime == "regeneration" else []
            unroll = int(streams["unroll"].integers(config.unroll_min, config.unroll_max+1))
            if config.pilot and counter % 10 == 0:
                unroll = config.unroll_max
            masks = streams["updates"].random((unroll, config.batch_size, 32, 32, 1), dtype=np.float32) < config.update_probability
            masks = torch.as_tensor(masks, device=config.device)
            state = torch.as_tensor(batch, device=config.device)
            optimizer.zero_grad(set_to_none=True)
            for step in range(unroll):
                state = model(state, masks[step])
            objective = ((state[..., :4]-target)**2).mean()
            if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(state).all()) or float(state.detach().abs().max()) > 1000:
                status, failure = "numerical_failure", "nonfinite or excessive rollout state"
                break
            objective.backward()
            gradients_valid = True
            gradient_norms = []
            for parameter in model.parameters():
                if parameter.grad is not None:
                    norm = parameter.grad.norm()
                    if not bool(torch.isfinite(norm)):
                        gradients_valid = False
                        break
                    gradient_norms.append(float(norm))
                    parameter.grad.div_(norm+1e-8)
            if not gradients_valid:
                status, failure = "numerical_failure", "nonfinite gradient"
                break
            optimizer.step()
            scheduler.step()
            if identities is not None:
                pool.commit(identities, state)
            counter += 1
            total_cell_steps += unroll*config.batch_size*32*32
            _sync(torch, config.device)
            duration = time.monotonic()-tick
            durations.append(duration)
            last_loss = float(objective.detach())
            extinct_updates = extinct_updates + 1 if all(norm == 0 for norm in gradient_norms) and not bool(state[..., 3].any()) else 0
            record = {"update": counter, "unroll": unroll, "batch_size": config.batch_size,
                      "objective_rgba_mse": last_loss, "auxiliary_loss": 0.0,
                      "learning_rate": optimizer.param_groups[0]["lr"], "seconds": duration,
                      "elapsed_seconds": accumulated_seconds+time.monotonic()-started,
                      "gradient_norms_before_normalization": gradient_norms,
                      "max_abs_state": float(state.detach().abs().max()),
                      "consecutive_extinct_updates": extinct_updates,
                      "damage": last_damage, "pool_ids": identities.tolist() if identities is not None else None,
                      **_memory(torch, config.device)}
            if config.validation_every and counter % config.validation_every == 0:
                select_export()
                record["validation"] = validation
            log.write(json.dumps(record, allow_nan=False)+"\n")
            if counter % config.log_every == 0 or counter == 1:
                print(json.dumps({key: record[key] for key in ("update", "objective_rgba_mse", "seconds", "peak_process_rss_mb")}), flush=True)
            if counter % config.checkpoint_every == 0:
                checkpoint()
            # Release the previous graph before constructing the next one.
            del state, objective, masks
            if extinct_updates >= 10:
                status, failure = "extinct_training_stalled", "ten consecutive empty rollouts with zero parameter gradients"
                break
        if counter > 0 and not config.pilot and status != "numerical_failure":
            select_export()
        final_checkpoint = checkpoint()
        summary = {"status": status, "convergence_claim": False, "failure": failure,
                   "updates": counter, "requested_updates": config.updates, "regime": config.regime,
                   "target": config.target, "target_sha256": sha, "initial_weights_sha256": initial_hash,
                   "seed": config.seed, "device": config.device, "canvas": [32, 32], "channels": 16,
                   "batch_size": config.batch_size, "pool_size": config.pool_size,
                   "unroll_range": [config.unroll_min, config.unroll_max], "total_cell_steps": total_cell_steps,
                   "training_seconds": accumulated_seconds+time.monotonic()-started,
                   "steady_update_seconds_median": float(np.median(durations[5:] or durations)) if durations else None,
                   "steady_update_seconds_mean": float(np.mean(durations[5:] or durations)) if durations else None,
                   "last_objective_rgba_mse": last_loss, "best_validation_score": best_score if np.isfinite(best_score) else None,
                   "best_update": best_update, "checkpoint": str(final_checkpoint),
                   "model": str(destination / "model.json") if (destination / "model.json").exists() else None,
                   "environment": environment, **_memory(torch, config.device)}
        _atomic_json(destination / "summary.json", summary)
    return summary


def run_pilot(device: str = "cpu", updates: int = 50, output_dir: str | None = None,
              threads: int = 2, memory_mb: float = 3000, wall_seconds: float = 600) -> dict:
    return run_training(TrainConfig(device=device, updates=updates, pilot=True, validation_every=0,
                                   unroll_min=48, unroll_max=96, threads=threads,
                                   output_dir=output_dir or f"runs/pilot-{device}",
                                   wall_seconds=wall_seconds, memory_mb=memory_mb))


def export_checkpoint(checkpoint_path: str, output_path: str) -> dict:
    import torch
    from .torch_model import NCA
    from .model_io import save_model
    saved = _load_checkpoint(torch, checkpoint_path)
    model = NCA(seed=saved["config"]["seed"])
    model.load_state_dict(saved["model"])
    metadata = {"target": Path(saved["config"]["target"]).stem, "regime": saved["config"]["regime"],
                "checkpoint_update": saved["update"], "target_sha256": saved["target_sha256"],
                "training_config": saved["config"], "selection": "explicit local checkpoint export"}
    save_model(output_path, model.weights_numpy(), metadata)
    return metadata
