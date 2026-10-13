# Static deployment

Repository: [DhanushKrishna4/Mycelium](https://github.com/DhanushKrishna4/Mycelium). Pages destination: [dhanushkrishna4.github.io/Mycelium/](https://dhanushkrishna4.github.io/Mycelium/). The repository was published and the hosted Python runtime verified on 2026-10-13. [The verification record](../reports/deployment-verification.json) identifies the exact tested commit, successful workflow, served asset hashes, and browser observations.

The complete production output is `dist/`. It contains the page, module worker, shared Python runtime ZIP, five models, target data, research results, and only six required Pyodide/NumPy runtime assets (16,909,239 bytes for runtime dependencies; inspect the build manifest for exact bytes). There are no inference requests, keys, cross-origin isolation headers, SharedArrayBuffer requirements, or backend services. Hash routes and relative asset paths support `/Mycelium/` and other repository prefixes.

1. Follow the Python 3.12 virtual-environment setup in [the README](../README.md), then run `npm run runtime`, `npm run build`, and `.venv/bin/python scripts/verify_build.py`.
2. Run `.venv/bin/python scripts/serve.py` and open `http://127.0.0.1:8732/Mycelium/`. This deliberately rejects incorrectly cased `/mycelium/`.
3. The repository's **Settings → Pages → Build and deployment → Source** is configured as **GitHub Actions**. Select the same source when deploying a fork. Public GitHub Pages is available with GitHub Free.
4. Push to `main` to run the native and JavaScript checks, assemble and verify the static artifact, and deploy after the build succeeds. Pull requests perform checks/builds without deployment. To deploy manually, run the Pages workflow on `main` with its `deploy` input enabled; a manual run without it performs checks/builds only.
5. After the deployment succeeds, open the Pages destination and verify cold loading, live growth, damage, extinction, and browser numerical parity. A green build or asset-only HTTP check does not establish that the hosted Python worker runs. These checks passed for the initial release; repeat them after changes to the runtime or hosting configuration.

The workflow never trains. Commit `models/`, `targets/`, `reports/`, fixtures, sources, and `scripts/runtime-lock.json`; do not commit `.cache/`, `.venv/`, `dist/`, or `runs/` native checkpoints. Acquisition fetches pinned Pyodide 0.27.7 and NumPy 2.0.2 from its public distribution, verifies hashes, and caches locally. `npm run runtime -- --refresh-lock` is a maintainer operation for intentionally reviewing upstream bytes, never a normal build step.

`build-manifest.json` records artifact sizes and SHA-256 hashes. `verify_build.py` checks them, verifies the runtime ZIP excludes training modules, and serves every declared asset over HTTP under the exact repository subpath. Local UI/Pyodide checks are recorded separately in `reports/browser-verification.json`; CI does not pretend to run them when it only performs native/build checks. Repository or organization Pages permissions can still require owner action if deployment settings change.

Sources: [GitHub custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), [Pyodide static deployment](https://pyodide.org/en/0.27.7/usage/downloading-and-deploying.html).
