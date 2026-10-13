# Mycelium transition contract, version 1

The same NumPy module (`src/mycelium_nca/numpy_sim.py`) runs natively and inside
Pyodide. `torch_model.py` is its differentiable training counterpart. A transition
never receives a target, absolute coordinates, a clock, or a global summary.

## Arrays and perception

Native/browser state is C-contiguous `(height, width, 16)` float32; PyTorch uses
`(batch, height, width, 16)`. Coordinates are integer `x=column`, `y=row`, with origin
at the upper left. Validated grids are 32 × 32; imports cap experimental square
grids at 64 × 64. RGB channels 0–2 are premultiplied by channel 3 (alpha); channels
4–15 are unconstrained learned hidden values. No inference clipping occurs.

Each channel is perceived with identity plus these zero-padded **cross
correlations** (kernels are not flipped):

```text
gx = [[-1, 0, 1],       gy = [[-1,-2,-1],
      [-2, 0, 2],             [ 0, 0, 0],
      [-1, 0, 1]] / 8        [ 1, 2, 1]] / 8
```

The 48 features are concatenated `[state[0:16], gx[0:16], gy[0:16]]`.
The horizontal response to an increasing left-to-right ramp is positive; the
vertical response to an increasing top-to-bottom ramp is positive. The domain
does not wrap. Outside cells contribute exact zero.

The dense layers act on row vectors:

```text
delta = relu(perception @ W1 + b1) @ W2 + b2
W1: (48,64), b1: (64,), W2: (64,16), b2: (16,)
```

Initialization uses NumPy `Generator(PCG64(training_seed))`, uniform Glorot W1
with limit `sqrt(6/(48+64))`, cast to float32; all biases and the entire final
weight matrix start at zero. The default seed is 42. This makes the initial rule
produce no changes without hiding training inside inference.

## One transition and interventions

At boundary `t`, apply recorded events in their original order, then:

1. Zero every channel in persistent lesion cells.
2. Compute `pre_alive`: the maximum alpha in each cell's 3 × 3 Moore neighborhood
   (including itself) is **strictly greater than 0.1**.
3. Compute local `delta` from this constrained state.
4. Sample one independent update decision per cell, shared across its 16
   channels: `Generator(PCG64).random((height,width),dtype=float32) < p`.
   The default probability is 0.5. One call is made on **every** transition,
   including p=0 and p=1; row-major output order is part of the contract.
5. Form `candidate = state + delta * mask[...,None]`.
6. Compute `post_alive` by the same threshold/neighborhood rule on `candidate`.
7. Set `next = candidate * (pre_alive & post_alive)[...,None]`.
8. Zero persistent lesions again; detect nonfinite values or absolute values
   exceeding 1,000,000 and stop with a numerical-failure status.
9. Increment the integer simulation step.

There is no alternate path that restores a desired image. The all-zero state
is absorbing even when biases become nonzero: its pre-alive mask is false.
Alpha is not clipped before computing life. A skipped update only inhibits the
learned increment; the living and lesion constraints still apply.

Events use integer cell coordinates and are bounded at import. The disk region
includes `(col-x)^2 + (row-y)^2 <= radius^2`; rectangles extend right/down from
their upper-left corner and are clipped by the domain.

* `erase`: set **all** channels to zero in the region.
* `lesion`: mark the region permanently constrained to zero until `clear_lesions`.
  Lesions also remove their contributions to neighborhood perception. They are
  forced constraints, not cells trained to represent walls.
* `seed`: replace exactly one cell by zero RGB, alpha 1, and all 12 hidden
  channels 1. `seed_state()` inserts it at `(size//2,size//2)` in an empty grid.
* `clear`: clear all cells without removing lesions. The result remains extinct
  until an explicit seed/noise intervention adds state.
* `noise`: add uniform noise in `[-amplitude,+amplitude)` to specified channels
  inside the region; amplitude is bounded to [0,1]. Its independent documented
  PCG64 event seed generates a full `(H,W,selected_channel_count)` float32 array
  before region selection. Default channels are all 16. This event does not
  consume the simulation update stream.
* `hidden_ablation`: zero only specified channels 4–15, retaining RGBA.
* `update_probability`: update p before the following transition.

Persistent lesions override every event, including seeds. The `return_delta`
inspection value is the **actual state difference** after living/lesion rules,
not merely the raw neural-network proposal.

## Rendering, serialization, and replay

Rendering produces an independent straight-alpha uint8 RGBA array: clip alpha
to [0,1], divide premultiplied RGB by nonzero alpha, clip display RGB to [0,1],
round to bytes. It never alters the float state. This is suitable for Canvas
ImageData and PNG; inspection reads the original float values.

`mycelium.model.v1` is JSON with exact architecture, named nested float32 weight
arrays, provenance metadata and a SHA-256 hash. Canonical hash input, in order
W1,b1,W2,b2, is each ASCII header `name:comma-separated-shape:float32le\n`
followed by C-order little-endian IEEE754 float32 bytes. Model import caps files
at 1,000,000 bytes and validates shapes, finite values, and hash. No pickle or
executable code is accepted.

`mycelium.experiment.v1` records model/target hashes, runtime and NumPy versions,
grid, the exact RNG contract and state, full state and lesions at a retained
origin and current boundary, p, and ordered integer-step events. Imports cap at
8,000,000 bytes and 4,096 events. If event history fills, a complete snapshot
becomes the new origin and `partial_history=true`; reconstruction remains valid
from that boundary. State snapshots cap at 64 × 64 × 16. Live work chunks cap
at 256 steps. History reconstruction caps at 100,000 transitions; larger saved
experiments can still resume their current snapshot. Event `order` disambiguates
multiple interventions at the same step.

Playback chunks and rendering frequency never change RNG call shape or count.
The optional explicit-mask Python testing hook still advances the update RNG;
if an externally supplied mask differs from its generated mask, the resulting
complete state becomes a new partial-history origin. Huge external mask streams
are not hidden in exports. Normal browser simulations use the recorded generator.
Exact replay is supported within the same pinned NumPy build, execution
environment, and runtime contract. The recorded NumPy version describes the
source environment; importing elsewhere does **not** establish exact numerical
replay. Cross-backend tests compare explicit masks with tolerances; training and
NumPy integer seeds alone do not establish matched stochastic schedules.

## Reference and deliberate differences

This implementation follows the local-rule/living-mask idea from [Growing Neural
Cellular Automata](https://distill.pub/2020/growing-ca/) by Alexander Mordvintsev,
Ettore Randazzo, Eyvind Niklasson, and Michael Levin (2020), and the [authors'
reference notebook](https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb).
The reference was inspected directly. Its normalized Sobel filters, zero final
layer, per-cell random mask, 3 × 3 alpha threshold, and pre/post intersection
are retained. Its depthwise perception interleaves filters within each channel;
Mycelium deliberately uses contiguous blocks `[state,gx,gy]`. Reference exported
weights therefore cannot be imported unchanged. Mycelium uses 64 hidden units
instead of 128, batch 4 instead of 8, a pool of 128 instead of 1,024, and a full
32 × 32 domain instead of a target-sized image plus margins. It uses explicit
NumPy-generated masks with `< p`, whereas the reference uses TensorFlow random
uniform and `<= fire_rate`. Noise, lesions, bounded imports, replay, and failure
detection are project additions.

NumPy documents the limits of stream portability in its [random compatibility
policy](https://numpy.org/doc/stable/reference/random/compatibility.html).
The parity suite uses `rtol=2e-5, atol=2e-6` for perception, increments, and short
rollouts away from threshold discontinuities; deliberately near-threshold
fixtures are checked separately. Long divergent trajectories across backends
are possible after small numerical differences change a life decision.
