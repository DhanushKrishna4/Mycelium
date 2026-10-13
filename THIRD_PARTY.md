# Attribution and third-party components

## Research foundation

Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. **Growing Neural Cellular Automata.** Distill, 2020. DOI [10.23915/distill.00023](https://distill.pub/2020/growing-ca/).

The authors' [Google Research reference notebook](https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb) is Copyright 2020 Google LLC, licensed under Apache License 2.0. It was consulted to verify perception, living masks, gradient normalization, state pools, and damage training conventions. Mycelium independently implements these ideas; differences are documented in `docs/transition.md`. No reference-trained weights, emoji artwork, or article figures are shipped.

## Dependencies

- [Pyodide 0.27.7](https://github.com/pyodide/pyodide/tree/0.27.7): Mozilla Public License 2.0; its bundled Python and upstream compiled components retain their notices. Pyodide distribution files are acquired unchanged and hashes recorded in `scripts/runtime-lock.json`.
- [CPython](https://docs.python.org/3/license.html): Python Software Foundation License and bundled-component notices.
- [NumPy 2.0.2](https://github.com/numpy/numpy/blob/v2.0.2/LICENSE.txt): BSD 3-Clause; binary distributions also include their bundled-component licenses in the wheel.
- [PyTorch 2.6.0](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE): BSD-style license and component notices; native training/test dependency only, not shipped to the browser.
- [Pillow](https://python-pillow.org/): HPND-style license; native PNG preparation/demo generation only.
- [pytest](https://docs.pytest.org/): MIT; native test dependency.
- [psutil](https://github.com/giampaolo/psutil): BSD 3-Clause; native resource tooling.

JavaScript, CSS, original procedural targets, exported trained weights, and Python implementation in this repository are provided under the repository MIT license. The site uses system fonts and requires no external font service or third-party analytics. Packages remain subject to their own licenses, which are not replaced by this repository's license.
