# Local training protocol

Only native training imports PyTorch. The package runtime, target generation,
NumPy simulator, evaluation, and browser do not require it. Install the pinned
native environment with `python3.12 -m venv .venv` followed by
`.venv/bin/pip install -e '.[train,test]'`.

Generate the original artwork with `mycelium targets`. Each 32 × 32 target is
drawn at 4× resolution and reduced by area averaging in premultiplied RGBA. PNGs
are unpremultiplied output copies. The target hash covers contiguous little-endian
float32 HWC data. Ring v2 is a teal ring with an opaque dark core; its outline is
a disk. The earlier transparent-hole ring caused absorbing extinction in the
initial resource pilots: it rewarded killing the central seed before a pattern
could grow. The failed pilot snapshots and logs retain that original target.

## Measure before running the gallery

```
mycelium pilot --device cpu --output-dir runs/pilot-cpu
mycelium pilot --device mps --output-dir runs/pilot-mps
```

The pilot performs 50 optimizer updates. It samples unrolls from 48–96 inclusive
and explicitly uses the largest unroll every tenth update. It excludes the first
five optimizer timings when reporting steady-state throughput, synchronizes MPS
before stopping the timer, and records peak process RSS (plus MPS allocated and
driver memory when applicable). `ru_maxrss` is a process-lifetime high-water
mark, not instantaneous or whole-machine memory. Run pilots in fresh processes,
one at a time. CPU is always available; MPS is chosen only after measurement.

## Bounded runs and fair controls

```
mycelium train --config configs/train-default.json --target ring \
  --regime regeneration --device cpu --output-dir runs/ring-regeneration
mycelium train --config runs/ring-regeneration/config.json \
  --resume runs/ring-regeneration/checkpoints/step-001000.pt --updates 2500
mycelium train --target local-pattern.png --regime regeneration \
  --output-dir runs/custom --updates 2500
```

A custom PNG is centered within 22 × 22 cells on the release canvas. It requires
a new local training run. The web app does not learn a newly imported target.

Default training uses four examples, a 128-entry CPU pool, 48–64 update steps per
optimizer step, Adam with learning rate 0.002, and a tenfold learning-rate drop
after optimizer update 2000. The release cap is 2500 optimizer updates per model,
3600 seconds per invocation, and 3000 MiB measured process memory. Flags configure
all three budgets; extending the optimizer cap is explicit. A resumed run retains
its optimizer-update count and schedules; its wall budget begins at invocation,
while reported duration accumulates across invocations. Checkpoint and validation
work can finish just after a wall budget; a budget is checked between updates.

Every parameter gradient is divided by its own L2 norm plus `1e-8`, following the
[authors' reference notebook](https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb).
The loss is raw mean squared error over the four premultiplied visible channels.
There are no auxiliary terms: the log records `auxiliary_loss: 0.0` explicitly.
States are never clipped during learning or simulation.

Growth starts all four examples from a seed. Persistence samples without
replacement from a detached pool, orders sampled states and their identities
together by descending loss, and replaces the worst sample with a seed.
Regeneration also erases full-state disks in the lowest-loss half. The seeded
example is preserved. A disk must erase 5–65% of currently mature cells; up to
eight attempts are allowed and skipped/rejected damage is recorded. Thus small
early states are not routinely erased completely. Rollout results return to the
same sorted pool identities from which each example was sampled.

Initialization, pool selection, disk damage, rollout length, and cell updates use
separate reproducible streams. All three anchor regimes use identical initial
weights, optimization, rollout schedules, training seed, architecture, and update
budget. Pool and damage behavior are the experimental differences. The CLI also
supports additional independent training seeds; a single training seed per method
is a case study, not a replicated study.

Ten consecutive empty rollout batches with exactly zero parameter gradients stop
training with `extinct_training_stalled`. Numerical failure stops and preserves
the run. `update_budget_exhausted` says only that the requested work finished; it
does not assert convergence or a passed quality gate.

## Checkpoint selection, resume, and exports

Checkpoint selection uses only the fixed `validation-v1` suite: two PCG64 schedules
with seeds 910031 and 910037. It measures growth at step 128, persistence at 256,
then recovery at 384 after a fixed disk erasure at 256. The score is the average
foreground/background-balanced RGBA error over these three phases. Every regime
uses the same score. Validation seeds and disk are distinct from frozen final
evaluation schedules and held-out rectangular damage.

`model.json` contains the best valid checkpoint selected by that score. Final
evaluation does not choose checkpoints. The training logs record objectives,
durations, peak memory, gradient norms, pool identities, applied damage, and every
scheduled validation score. Native checkpoints contain model, optimizer,
scheduler, complete detached pool and pool write counts, counters, configuration,
all used RNG states, and accumulated runtime. They are atomically replaced and
retain the last two updates plus `best.pt`.

Native `.pt` checkpoints use PyTorch's trusted-local serialization and must never
be accepted from the web app. `mycelium export checkpoint.pt model.json` converts
a trusted checkpoint into the bounded data-only model format. Browser imports use
only that validated JSON format.

For sequential release training, run
`python scripts/train_release.py --device cpu` for the anchor. Inspect and evaluate
it, then run `python scripts/train_release.py --device cpu --gallery` for the two
controls and two remaining targets. This script never runs trainers concurrently.
Do not treat an exported weight file as evidence of learned growth or repair;
check the frozen evaluation report and actual simulation.
