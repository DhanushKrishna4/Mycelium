# Mycelium progress — 2026-10-13

Implementation, five bounded training runs, frozen evaluation, and the local static application are delivered. Failed model gates and unverified platform checks are listed below. The local implementation and browser evidence were completed on 2026-10-11; public release work is underway on 2026-10-13.

## Public release underway

- Repository created: [DhanushKrishna4/Mycelium](https://github.com/DhanushKrishna4/Mycelium); Pages configured to use GitHub Actions.
- Deployment destination: [https://dhanushkrishna4.github.io/Mycelium/](https://dhanushkrishna4.github.io/Mycelium/). Hosted runtime verification remains pending.
- The workflow now deploys `main` pushes after checks and the static build pass. Pull requests are checks/builds only; manual dispatch retains an explicit `deploy` input.
- Fresh isolated Python packaging exposed an incompatible license field with the pinned build backend. The compatible license table fixed it, and the isolated wheel build passed. Native tests (41) and JavaScript tests (3) passed again before publication.
- Initial commit/push, deployment completion, and the hosted browser report are the remaining release steps. Model results below are unchanged.

## Completed milestones

- [x] Exact 32×32×16 float32 transition; identity/Sobel perception; shared 48→64→16 update; native PyTorch/NumPy parity and gradient fixtures.
- [x] Pool training, atomic resumable checkpoints, bounded history, measured CPU/MPS pilots, resource budgets, and preservation of failed attempts.
- [x] Five actual trained artifacts: ring growth, persistence and regeneration; leaf and rosette regeneration. All ran sequentially for 2,500 updates; validation alone selected exported checkpoints.
- [x] Frozen 135-trial evaluation with raw JSON/CSV, plots, checkpoint/target hashes, environment, and failed outcomes.
- [x] Actual Pyodide parity and interactive Python simulation in a persistent worker.
- [x] Grow, Repair, Compare, Inspect; shared masks; all-state interventions; paired divergence; bounded replay and data-only imports/exports.
- [x] Real-browser cold loading, model changes, pause/step/reset, extinction, paired comparison, inspection, canvas lesion/noise, import/export, runtime recreation and missing-asset retry.
- [x] Desktop/mobile screenshots, real rollout animation, documentation, attribution, static build and GitHub Actions workflow.
- [x] Native tests: 41 passed. JavaScript transport/coordinates: 3 passed. Local `/Mycelium/` production serving and asset hashes verified with `scripts/verify_build.py`.

## Chosen configuration and measured resources

Python 3.12, NumPy 2.0.2, PyTorch 2.6.0; browser Pyodide 0.27.7. Zero final-layer initialization, alpha threshold 0.1, update probability 0.5, centered seed alpha/hidden = 1. Batch 4, pool 128, 48–64-step release unrolls, Adam 0.002 with decay at 2,000, per-parameter gradient normalization, two CPU threads. Native checkpoints retain the last two plus best.

50-update pilots included unroll 96: CPU median 0.163 s/update, peak 725.97 MiB; MPS median 0.344 s/update, peak 442.5 MiB process + 243.09 MiB driver. CPU was faster for this workload and was selected. Five release runs took 1,832.34 seconds (30.5 minutes), max process RSS 633.25 MiB. The initial transparent-center ring collapsed to extinction; its pilot records and original target remain in training provenance. The release ring has a dark opaque center.

Actual Chromium worker benchmark: 418.16 steps/s, one 32×32 model, 16 warmup + 128 measured steps, chunks of 8; includes yields and excludes rendering/startup. Browser parity maximum error across recorded checks was 1.19e-7, within rtol 2e-5 / atol 2e-6. See `reports/browser-diagnostics.json`.

## Results and remaining limits

| Model | Growth /8 | Persistence /4 | Repair /9 |
|---|---:|---:|---:|
| Ring growth | 0 | 0 | 0 |
| Ring persistence | 8 | 4 | 9 |
| Ring regeneration | 8 | 4 | 9 |
| Leaf regeneration | 8 | 0 | 9 |
| Rosette regeneration | 8 | 0 | 9 |

Leaf and rosette fail the 1,024-step persistence gate. Growth-only ring fails the 256-step growth gate. One ring-persistence recovery later deteriorates (8/9 retained recoveries). These are single-training-seed case studies; no independent training replication or generalization to unseen images is claimed. Large cuts can fail; total erasure stays extinct until an explicit seed.

Actual background-tab pause, Safari/Firefox, physical touch, actual browser zoom, outside-canvas dragging, reduced-motion emulation/screen-reader operation, live hanging-worker termination, and hosted GitHub Pages remain unverified. Source handlers and relevant transport/coordinate tests exist, but are not substitutes for those device checks. `reports/browser-verification.json` separates passed observations from unverified checks.

## Exact commands

From this checkout, with the prepared environment and runtime cache:

```sh
.venv/bin/python -m pytest -q
npm run check:web
npm run build
.venv/bin/python scripts/verify_build.py
.venv/bin/python scripts/serve.py
# Open http://127.0.0.1:8732/Mycelium/
```

Fresh-environment setup is in `README.md`. To reproduce the frozen native evaluation without retraining:

```sh
PYTHONPATH=src .venv/bin/python -m mycelium_nca.cli evaluate
```

For new training, use a fresh output directory and an explicit update/wall/memory budget; do not overwrite the frozen release or treat a favorable retry as part of the original results. See `docs/TRAINING.md`. There is no further automatic training scheduled.

One managed-environment issue remains: the editable-install `.pth` file repeatedly carries macOS `UF_HIDDEN`, causing Python to ignore it. Clearing that flag restored imports for that invocation, but the flag recurred later. Use `PYTHONPATH=src` for direct source CLI execution in this environment. Tests configure the source path, and the build sets it explicitly; static serving is unaffected.
