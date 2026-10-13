// This worker executes the project's Python transition, never a JavaScript duplicate.
let pyodide;
let dispatch;
const root = new URL('./', import.meta.url);
const asset = (path) => new URL(path, root).href;
async function fetchJSON(path, maxBytes = 8 * 1024 * 1024) {
  const response = await fetch(asset(path));
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status}). Check the static build and reload.`);
  const text = await response.text();
  if (text.length > maxBytes) throw new Error(`Asset ${path} exceeds the supported size limit.`);
  return JSON.parse(text);
}
function modelPath(item) {
  const file = item.file || item.path;
  if (!file) throw new Error(`Model ${item.id} has no artifact path.`);
  return file.startsWith('models/') ? file : `models/${file}`;
}
function progress(message) { self.postMessage({ progress: message }); }
async function initialize() {
  progress('Loading Python 3.12 · self-hosted runtime');
  const { loadPyodide } = await import('./runtime/pyodide.mjs');
  pyodide = await loadPyodide({ indexURL: asset('runtime/'), fullStdLib: false });
  progress('Loading NumPy 2.0.2');
  await pyodide.loadPackage('numpy');
  progress('Loading the shared Python simulator');
  const response = await fetch(asset('runtime/mycelium_runtime.zip'));
  if (!response.ok) throw new Error('The Python runtime package is missing. Run npm run build again.');
  pyodide.unpackArchive(await response.arrayBuffer(), 'zip', { extractDir: '/home/pyodide' });
  const manifest = await fetchJSON('models/manifest.json');
  if (!Array.isArray(manifest.models) || !manifest.models.length) throw new Error('No trained model artifacts are available in this build.');
  const models = {}, targets = {};
  for (const model of manifest.models) models[model.id] = await fetchJSON(modelPath(model));
  const targetItems = manifest.targets || [...new Set(manifest.models.map((m) => m.target))].map((id) => ({ id, json: `targets/${id}.json` }));
  for (const target of targetItems) targets[target.id] = await fetchJSON(target.json || `targets/${target.id}.json`);
  pyodide.runPython('from mycelium_nca.browser import initialize, dispatch');
  const initializePython = pyodide.globals.get('initialize');
  let result;
  try { result = JSON.parse(initializePython(JSON.stringify(manifest), JSON.stringify(models), JSON.stringify(targets))); }
  finally { initializePython.destroy(); }
  dispatch = pyodide.globals.get('dispatch');
  progress('Python ready · all simulation stays on your device');
  return { ...result, manifest, targets, runtime: { pyodide: pyodide.version, numpy: '2.0.2', python: '3.12.7' } };
}
async function parity() {
  const fixture = await fetchJSON('fixtures/parity-v1.json', 1000000);
  pyodide.runPython('import json; from mycelium_nca.parity import run_fixture');
  const run = pyodide.runPython('lambda source: json.dumps(run_fixture(source), allow_nan=False)');
  try { return { parity: JSON.parse(run(JSON.stringify(fixture))) }; }
  finally { run.destroy(); }
}
async function benchmark() {
  // Benchmark an independent complete clone. The visible experiment and RNG never advance.
  pyodide.runPython(`from mycelium_nca.browser import _lab
_benchmark = _lab.simulations[0].clone()
_benchmark.advance(16)
`);
  const advance = pyodide.runPython('lambda: _benchmark.advance(8) is not None');
  const started = performance.now();
  try {
    for (let i = 0; i < 16; i++) {
      advance();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const milliseconds = performance.now() - started;
    const detail = JSON.parse(pyodide.runPython(`import json, numpy as np
json.dumps({'grid':list(_benchmark.state.shape),'finite':bool(np.isfinite(_benchmark.state).all()),'model_hash':_benchmark.model_hash,'starting_step':_benchmark.step_count-144,'occupied_cells':int((_benchmark.state[...,3]>.1).sum())})`));
    return { benchmark: { schema: 'mycelium.browser-benchmark.v1', ...detail, pyodide: pyodide.version, numpy: '2.0.2', measured_steps: 128, warmup_steps: 16, branches: 1, chunk_steps: 8, milliseconds, steps_per_second: 128000 / milliseconds, includes: 'Worker computation and chunk yields; excludes pixel rendering and startup.', user_agent: navigator.userAgent } };
  } finally { advance.destroy(); pyodide.runPython('del _benchmark'); }
}
function transferPixels(result) {
  const transfer = [];
  const make = (pixels) => {
    const typed = new Uint8ClampedArray(pixels.flat ? pixels.flat(Infinity) : pixels);
    transfer.push(typed.buffer);
    return typed;
  };
  if (result.pixels) result.pixels = result.pixels.map(make);
  if (result.capturePixels) result.capturePixels = [make(result.capturePixels)];
  if (result.inspection?.visibleDifference) result.inspection.visibleDifference = make(result.inspection.visibleDifference);
  if (result.inspection?.hiddenDifference) result.inspection.hiddenDifference = make(result.inspection.hiddenDifference);
  return transfer;
}
self.onmessage = async ({ data: { requestId, generation, command } }) => {
  try {
    if (command.type !== 'init' && !dispatch) throw new Error('Python is still loading. Please wait.');
    const started = performance.now();
    const result = command.type === 'init' ? await initialize() : command.type === 'parity' ? await parity() : command.type === 'benchmark' ? await benchmark() : JSON.parse(dispatch(JSON.stringify(command)));
    if (result.error) throw new Error(result.error);
    result.workerMilliseconds = performance.now() - started;
    self.postMessage({ requestId, generation, result }, transferPixels(result));
  } catch (error) {
    self.postMessage({ requestId, generation, error: error.message || String(error) });
  }
};
