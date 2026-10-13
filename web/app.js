import { RuntimeClient, cellAt } from './runtime-client.js';

const $ = (id) => document.getElementById(id);
const state = { client: null, manifest: null, targets: {}, model: null, secondModel: null, target: null, mode: 'grow', paired: false, playing: false, ready: false, brush: 'erase', last: null, lastTick: 0, pendingSteps: 0, chunk: 4, stats: [], lastRender: 0, frames: [], noticeTimer: null, parity: null, benchmark: null };
const regimeName = (regime) => ({ growth: 'Growth only', persistence: 'Persistence', regeneration: 'Regeneration' }[regime] || regime || 'Imported model');
const explanations = { grow: 'A pattern begins with one cell.', repair: 'Intervene. Watch what the rule can rebuild.', compare: 'Same experiment. Different learned rules.', inspect: 'Look beneath the visible pattern.' };
const descriptions = { erase: 'Erase zeros all 16 channels. Drag on the canvas, or use a preset.', seed: 'Insert the documented seed: zero RGB, alpha and hidden channels set to one.', lesion: 'A persistent lesion keeps every channel zero until lesions are cleared.', noise: 'Seeded uniform noise in all channels, bounded by the selected amplitude.' };
const fmt = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';

function notice(message) {
  $('notice').textContent = message;
  $('notice').hidden = false;
  clearTimeout(state.noticeTimer);
  state.noticeTimer = setTimeout(() => { $('notice').hidden = true; }, 6000);
}
function setPlaying(playing) {
  state.playing = playing && state.ready;
  $('play').innerHTML = state.playing ? '<span aria-hidden="true">Ⅱ</span> Pause' : '<span aria-hidden="true">▶</span> Grow';
  $('play').setAttribute('aria-label', state.playing ? 'Pause simulation' : 'Play simulation');
  $('sim-live-dot').style.opacity = state.playing ? '1' : '.4';
  if (state.ready && state.last?.status === 'ready') $('state-value').textContent = state.playing ? 'Evolving' : 'Ready';
  state.lastTick = performance.now();
  state.pendingSteps = 0;
  if (!state.playing) { $('sps-value').textContent = '—'; $('fps-value').textContent = '—'; }
}
function runtimeFailure(error) {
  state.ready = false;
  setPlaying(false);
  $('runtime-label').textContent = 'Runtime unavailable';
  $('runtime-dot').className = 'status-dot error';
  $('loading-overlay').hidden = false;
  $('loading-title').textContent = 'The culture is on hold.';
  $('loading-message').textContent = error.message;
  $('retry').hidden = false;
  $('state-value').textContent = 'Unavailable';
}
async function command(request, { render = true } = {}) {
  if (!state.ready) return null;
  try {
    const result = await state.client.request(request);
    if (!result) return null;
    if (result.download) download(result.download.data, result.download.name, result.download.mime);
    if (render && result.pixels) update(result);
    return result;
  } catch (error) { notice(error.message); return null; }
}
function drawPixels(canvas, pixels) {
  if (!pixels || pixels.length !== 32 * 32 * 4) return;
  const array = pixels instanceof Uint8ClampedArray ? pixels : new Uint8ClampedArray(pixels);
  canvas.getContext('2d').putImageData(new ImageData(array, 32, 32), 0, 0);
}
function targetPixels(target) {
  const flat = (target.rgba || target.pixels || target.data || target).flat(Infinity);
  // Target exports contain premultiplied float RGBA. Canvas ImageData expects straight color.
  const result = new Uint8ClampedArray(4096);
  for (let i = 0; i < flat.length && i < 4096; i += 4) {
    const alpha = flat[i + 3];
    for (let c = 0; c < 3; c++) result[i + c] = alpha > 0 ? flat[i + c] / alpha * 255 : 0;
    result[i + 3] = alpha * 255;
  }
  return result;
}
function gallery() {
  $('pattern-list').replaceChildren();
  const targetItems = state.manifest.targets || Object.keys(state.targets).map((id) => ({ id, name: id }));
  for (const target of targetItems) {
    const button = document.createElement('button');
    button.className = 'pattern-button';
    button.dataset.target = target.id;
    button.setAttribute('aria-label', `Select ${target.name || target.id} target`);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 32;
    canvas.setAttribute('aria-hidden', 'true');
    if (state.targets[target.id]) drawPixels(canvas, targetPixels(state.targets[target.id]));
    const label = document.createElement('span');
    label.textContent = target.name || target.id;
    button.append(canvas, label);
    button.addEventListener('click', () => selectTarget(target.id));
    $('pattern-list').append(button);
  }
}
function fillModels(target) {
  const options = state.manifest.models.filter((model) => model.target === target);
  $('model-select').replaceChildren();
  $('second-model-select').replaceChildren();
  for (const model of options) {
    const option = new Option(model.id === 'imported' ? 'Imported model' : regimeName(model.regime), model.id);
    $('model-select').add(option);
    $('second-model-select').add(option.cloneNode(true));
  }
  state.model = options.find((model) => model.regime === 'regeneration')?.id || options[0]?.id;
  state.secondModel = options.find((model) => model.regime === 'growth')?.id || options[0]?.id;
  $('model-select').value = state.model;
  $('second-model-select').value = state.secondModel;
}
function selectTarget(target, shouldReset = true) {
  if (state.mode === 'compare') {
    const models = state.manifest.models.filter((model) => model.target === target);
    if (models.length < 2) { notice('Training-method comparisons use the anchor target with all three regimes.'); return; }
  }
  state.target = target;
  fillModels(target);
  for (const button of document.querySelectorAll('[data-target]')) {
    const active = button.dataset.target === target;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
  const meta = (state.manifest.targets || []).find((item) => item.id === target);
  $('target-name').textContent = meta?.name || target;
  if (state.targets[target]) drawPixels($('target-canvas'), targetPixels(state.targets[target]));
  else { $('target-canvas').getContext('2d').clearRect(0, 0, 32, 32); $('target-name').textContent = 'Unmatched'; }
  if (shouldReset) reset();
}
function layoutBranches(two) {
  $('culture-b').hidden = !two;
  $('branch-a').hidden = !two;
  $('legend-b').hidden = !two;
  $('difference-panel').hidden = !state.paired;
  document.body.dataset.paired = String(state.paired);
}
async function reset() {
  if (!state.ready || !state.model) return;
  const playing = state.playing;
  setPlaying(false);
  state.paired = false;
  layoutBranches(state.mode === 'compare');
  state.client.nextGeneration();
  state.stats = []; state.frames = [];
  const result = await command({ type: 'reset', model: state.model, secondModel: state.secondModel, mode: state.mode === 'compare' ? 'compare' : 'single', seed: Math.min(4294967295, Math.max(0, Math.floor(Number($('random-seed').value)) || 0)) });
  if (!result) return;
  await settings();
  if (playing) setPlaying(true);
}
function setMode(mode, { preserve = false } = {}) {
  if (!Object.hasOwn(explanations, mode)) mode = 'grow';
  const previous = state.mode;
  state.mode = mode;
  document.body.dataset.mode = mode;
  for (const button of document.querySelectorAll('[data-mode]')) {
    const active = button.dataset.mode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
  $('mode-explainer').textContent = explanations[mode];
  $('comparison-controls').hidden = mode !== 'compare';
  history.replaceState(null, '', `#${mode}`);
  if (mode === 'compare' && state.ready && !preserve) {
    const anchor = state.manifest.anchorTarget || state.manifest.models.find((model) => model.regime === 'growth')?.target;
    if (anchor) selectTarget(anchor, false);
    reset();
  } else if (previous === 'compare' && state.ready && !preserve) reset();
  if (mode !== 'inspect' && state.paired && !preserve) reset();
  $('canvas-hint').textContent = mode === 'compare' ? 'Both cultures receive the same interventions.' : state.paired ? 'Interventions affect branch B only.' : 'Grow a pattern, then drag to erase.';
}
async function settings() {
  return command({ type: 'settings', probability: Number($('probability').value), view: $('view').value, channel: Number($('channel').value) });
}
function update(result) {
  state.last = result;
  const now = performance.now();
  state.frames.push(now);
  state.frames = state.frames.filter((time) => time >= now - 1500);
  result.pixels.forEach((pixels, index) => drawPixels($(index ? 'canvas-b' : 'canvas-a'), pixels));
  layoutBranches(result.pixels.length === 2);
  $('step-value').textContent = String(result.step || 0).padStart(4, '0');
  const metrics = result.metrics?.[0];
  if (metrics) {
    $('metric-error').textContent = fmt(metrics.balanced_mse ?? metrics.balanced_error);
    $('metric-iou').textContent = fmt(metrics.iou, 3);
    $('metric-occupied').textContent = String(metrics.occupied ?? metrics.occupied_cells ?? '—');
    $('metric-relative').textContent = fmt(metrics.relative_error, 3);
  }
  const status = result.status || 'ready';
  const extinct = status === 'extinct';
  const failed = /fail|nonfinite|unstable/i.test(status);
  $('state-value').textContent = failed ? 'Unstable' : extinct ? 'Extinct' : state.playing ? 'Evolving' : 'Ready';
  $('state-value').style.color = failed ? '#e9a489' : extinct ? 'var(--amber)' : 'var(--accent)';
  if (extinct || failed) {
    setPlaying(false);
    $('canvas-hint').textContent = failed ? 'Numerical failure. Inspect the saved state or restart.' : 'Extinct. Plant a seed to begin again.';
  } else {
    $('canvas-hint').textContent = result.mode === 'compare' ? 'Both cultures receive the same interventions.' : result.mode === 'paired' ? 'Interventions affect branch B only.' : 'Grow a pattern, then drag to erase.';
  }
  if (result.branches) {
    result.branches.forEach((branch, index) => { $(index ? 'branch-b' : 'branch-a').textContent = `${index ? 'B' : 'A'} / ${regimeName(branch.regime)} · #${branch.checkpoint}${state.paired ? (index ? ' · perturbed' : ' · control') : ''}`; });
    const first = result.branches[0];
    if (first) {
      $('checkpoint-name').textContent = `${regimeName(first.regime)} · ${state.manifest.models.find((model) => model.id === first.id)?.checkpoint_update ?? first.checkpoint ?? first.id}`;
      $('checkpoint-hash').textContent = first.modelHash ? `SHA256 ${first.modelHash.slice(0, 16)}…` : first.id;
    }
  }
  if (result.inspection) {
    drawPixels($('visible-difference'), result.inspection.visibleDifference);
    drawPixels($('hidden-difference'), result.inspection.hiddenDifference);
    $('divergence-value').textContent = `VISIBLE ${fmt(result.divergence?.visible ?? result.divergence, 5)} / HIDDEN ${fmt(result.divergence?.hidden ?? result.hiddenDivergence, 5)}`;
  }
  if (result.historyPartial) $('experiment-note').textContent = 'The bounded event history is partial. Export includes the complete current state and random stream for exact continuation in this runtime.';
  drawChart(result.curves || []);
  if (state.paired) drawChart((result.curves || []).map((point) => ({ step: point.step, errors: [point.divergence?.visible, point.divergence?.hidden] })), 'divergence-chart');
  state.lastRender = now;
}
function drawChart(curves, canvasId = 'error-chart') {
  const canvas = $(canvasId);
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 3);
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  const context = canvas.getContext('2d');
  context.scale(dpr, dpr);
  const width = rect.width, height = rect.height;
  const left = 40, right = width - 8, top = 10, bottom = height - 20;
  const values = curves.flatMap((point) => point.errors || []).filter(Number.isFinite);
  if (canvasId === 'error-chart') $('chart-empty').hidden = curves.length > 1;
  const max = values.length ? Math.max(...values, .00001) * 1.08 : 1;
  const minStep = curves[0]?.step || 0;
  const maxStep = Math.max(curves.at(-1)?.step || 0, minStep + 1);
  context.font = '8px ui-monospace, monospace';
  context.lineWidth = 1;
  for (let i = 0; i < 3; i++) {
    const y = top + i * (bottom - top) / 2;
    context.strokeStyle = '#36432f'; context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke();
    context.fillStyle = '#8f9e82'; context.textAlign = 'right'; context.fillText((max * (1 - i / 2)).toFixed(max < .01 ? 4 : 3), left - 7, y + 3);
  }
  context.textAlign = 'left'; context.fillText(String(minStep), left, height - 3);
  context.textAlign = 'right'; context.fillText(String(maxStep), right, height - 3);
  const branches = Math.max(...curves.map((point) => point.errors?.length || 0), 1);
  for (let branch = 0; branch < branches; branch++) {
    context.strokeStyle = branch ? '#8bb6c2' : '#c2d392'; context.lineWidth = 1.7; context.beginPath();
    let started = false;
    for (const point of curves) {
      const value = point.errors?.[branch];
      if (!Number.isFinite(value)) continue;
      const x = left + (point.step - minStep) / (maxStep - minStep) * (right - left);
      const y = bottom - value / max * (bottom - top);
      if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
    }
    context.stroke();
  }
}
async function animate(now) {
  requestAnimationFrame(animate);
  if (!state.playing || !state.ready || document.hidden) return;
  const elapsed = Math.min((now - state.lastTick) / 1000, .25);
  state.lastTick = now;
  state.pendingSteps = Math.min(state.pendingSteps + elapsed * Number($('speed').value), 16);
  if (state.client.busy || state.pendingSteps < 1 || now - state.lastRender < 33) return;
  const steps = Math.max(1, Math.min(8, state.chunk, Math.floor(state.pendingSteps)));
  state.pendingSteps -= steps;
  const result = await command({ type: 'tick', steps });
  if (!result) return;
  const duration = result.workerMilliseconds;
  if (duration > 70) state.chunk = Math.max(1, state.chunk - 1);
  else if (duration < 25) state.chunk = Math.min(8, state.chunk + 1);
  const measured = performance.now();
  state.stats.push({ time: measured, steps });
  state.stats = state.stats.filter((sample) => sample.time > measured - 1500);
  if (state.playing && state.stats.length >= 2) {
    const seconds = (measured - state.stats[0].time) / 1000;
    $('sps-value').textContent = (state.stats.slice(1).reduce((sum, sample) => sum + sample.steps, 0) / Math.max(.01, seconds)).toFixed(0);
    $('fps-value').textContent = ((state.frames.length - 1) / Math.max(.01, (measured - state.frames[0]) / 1000)).toFixed(0);
  }
}
function download(data, filename, mime = 'application/json') {
  const blob = data instanceof Blob ? data : new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], { type: mime });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob); link.download = filename || 'mycelium-export.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 2000);
}
let importKind = 'importExperiment';
function openImport(kind) { importKind = kind; $('file-input').value = ''; $('file-input').click(); }
async function boot() {
  state.client?.terminate();
  state.ready = false; setPlaying(false); state.paired = false;
  $('loading-overlay').hidden = false; $('retry').hidden = true;
  $('loading-title').textContent = 'A little Python. A living world.';
  $('loading-message').textContent = 'Loading the local simulation engine…';
  $('runtime-dot').className = 'status-dot loading'; $('runtime-label').textContent = 'Preparing Python';
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  state.client = new RuntimeClient(worker, { onProgress: (message) => { $('loading-message').textContent = message; }, onError: runtimeFailure });
  try {
    const result = await state.client.request({ type: 'init' }, { timeout: 120000 });
    if (!result) return;
    state.manifest = result.manifest; state.targets = result.targets;
    state.ready = true; gallery();
    selectTarget(state.manifest.defaultTarget || state.manifest.models.find((model) => model.regime === 'regeneration')?.target || state.manifest.models[0].target, false);
    $('loading-overlay').hidden = true;
    $('runtime-dot').className = 'status-dot'; $('runtime-label').textContent = 'Python / local';
    for (const id of ['play', 'step', 'restart', 'model-select']) $(id).disabled = false;
    update(result);
    setMode(location.hash.slice(1) || 'grow');
    await reset();
    // Respect reduced motion: the initial state is a real seed in either case.
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches && !document.hidden) setPlaying(true);
  } catch (error) { runtimeFailure(error); }
}

for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click', () => setMode(button.dataset.mode));
for (const button of document.querySelectorAll('[data-brush]')) button.addEventListener('click', () => {
  state.brush = button.dataset.brush;
  for (const item of document.querySelectorAll('[data-brush]')) { const active = item === button; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); }
  $('noise-controls').hidden = state.brush !== 'noise'; $('brush-help').textContent = descriptions[state.brush];
});
$('play').addEventListener('click', () => setPlaying(!state.playing));
$('step').addEventListener('click', () => { setPlaying(false); command({ type: 'tick', steps: 1 }); });
$('restart').addEventListener('click', reset);
$('model-select').addEventListener('change', () => { state.model = $('model-select').value; reset(); });
$('second-model-select').addEventListener('change', () => { state.secondModel = $('second-model-select').value; reset(); });
$('random-seed').addEventListener('change', reset);
$('speed').addEventListener('input', () => { $('speed-value').textContent = `${$('speed').value} steps/s`; });
$('radius').addEventListener('input', () => { $('radius-value').textContent = `${$('radius').value} cells`; });
$('noise').addEventListener('input', () => { $('noise-value').textContent = Number($('noise').value).toFixed(2); });
$('probability').addEventListener('input', () => { $('probability-value').textContent = Number($('probability').value).toFixed(2); });
$('probability').addEventListener('change', settings);
$('view').addEventListener('change', () => {
  $('channel-controls').hidden = $('view').value !== 'hidden';
  $('view-help').hidden = $('view').value === 'hidden';
  $('view-help').textContent = { rgba: 'Visible channels are premultiplied RGB and alpha. Display clipping leaves internal states unchanged.', alpha: 'Alpha is shown on a fixed scale from 0 (dark) to 1 (lime). Living masks use the documented 0.1 threshold.', delta: 'Root-mean-square magnitude of the most recent local update. Fixed display scale 0–0.05.' }[$('view').value] || '';
  settings();
});
$('channel').addEventListener('input', () => { $('channel-value').textContent = $('channel').value; });
$('channel').addEventListener('change', settings);
const branch = () => state.paired ? 1 : undefined;
$('cut').addEventListener('click', () => command({ type: 'event', kind: 'erase', shape: 'rectangle', x: 16, y: 0, width: 16, height: 32, branch: branch() }));
$('center-damage').addEventListener('click', () => command({ type: 'event', kind: 'erase', x: 16, y: 16, radius: 5, branch: branch() }));
$('seed-center').addEventListener('click', () => command({ type: 'event', kind: 'seed', x: 16, y: 16 }));
$('clear').addEventListener('click', () => command({ type: 'event', kind: 'clear', branch: branch() }));
$('clear-lesions').addEventListener('click', () => command({ type: 'event', kind: 'clear_lesions', branch: branch() }));
$('fork').addEventListener('click', async () => {
  setPlaying(false);
  const result = await command({ type: 'fork' }, { render: false });
  if (!result) return;
  state.paired = true; setMode('inspect', { preserve: true }); update(result);
  notice('The complete state is now paired. Damage B to measure how the branches diverge.');
});
for (const id of ['canvas-a', 'canvas-b']) {
  const canvas = $(id);
  let drawing = false, previous = '', lastEvent = 0;
  function paint(event, force = false) {
    const rect = canvas.getBoundingClientRect();
    const { x, y } = cellAt(event.clientX, event.clientY, rect);
    if (id === 'canvas-a') {
      const cursor = $('brush-cursor'); cursor.hidden = false;
      cursor.style.left = `${(x + .5) / 32 * 100}%`; cursor.style.top = `${(y + .5) / 32 * 100}%`;
      cursor.style.width = cursor.style.height = `${(Number($('radius').value) * 2 + 1) / 32 * 100}%`;
    }
    if (!drawing || !state.ready || (state.paired && id === 'canvas-a')) return;
    const key = `${x},${y}`;
    if ((!force && performance.now() - lastEvent < 35) || previous === key) return;
    previous = key; lastEvent = performance.now();
    command({ type: 'event', kind: state.brush, x, y, radius: Number($('radius').value), amplitude: Number($('noise').value), seed: Math.max(0, Math.min(4294967295, Math.floor(Number($('random-seed').value)) || 0)), branch: branch() });
  }
  canvas.addEventListener('pointerdown', (event) => { if (event.button !== 0) return; drawing = true; previous = ''; canvas.setPointerCapture(event.pointerId); paint(event, true); });
  canvas.addEventListener('pointermove', (event) => paint(event));
  canvas.addEventListener('pointerup', (event) => { paint(event, true); drawing = false; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); });
  canvas.addEventListener('pointercancel', () => { drawing = false; });
  canvas.addEventListener('pointerleave', () => { if (!drawing) $('brush-cursor').hidden = true; });
  canvas.addEventListener('lostpointercapture', () => { drawing = false; $('brush-cursor').hidden = true; });
}
$('capture').addEventListener('click', async () => {
  const result = await command({ type: 'capture' }, { render: false });
  if (!result?.capturePixels) return;
  const canvas = document.createElement('canvas'); canvas.width = 32 * result.capturePixels.length; canvas.height = 32;
  result.capturePixels.forEach((pixels, i) => canvas.getContext('2d').putImageData(new ImageData(pixels, 32, 32), i * 32, 0));
  const filename = `mycelium-step-${state.last?.step || 0}.png`;
  canvas.toBlob((blob) => {
    if (blob) download(blob, filename, 'image/png');
    else notice('The browser could not encode this PNG.');
  }, 'image/png');
});
$('export-experiment').addEventListener('click', () => command({ type: 'exportExperiment' }, { render: false }));
$('export-model').addEventListener('click', () => command({ type: 'exportModel' }, { render: false }));
$('export-csv').addEventListener('click', () => command({ type: 'measurements', format: 'csv' }, { render: false }));
$('export-json').addEventListener('click', () => command({ type: 'measurements', format: 'json' }, { render: false }));
$('import-experiment').addEventListener('click', () => openImport('importExperiment'));
$('import-model').addEventListener('click', () => openImport('importModel'));
$('file-input').addEventListener('change', async () => {
  const file = $('file-input').files[0]; if (!file) return;
  if (file.size > 8 * 1024 * 1024) { notice('This file exceeds the 8 MiB import limit.'); return; }
  setPlaying(false);
  let data;
  try { data = JSON.parse(await file.text()); } catch { notice('This file is not valid JSON. Import a Mycelium data-only model or experiment.'); return; }
  state.client.nextGeneration();
  const result = await command({ type: importKind, data }, { render: false });
  if (result?.pixels) {
    if (result.importedModel) {
      state.manifest.models = state.manifest.models.filter((model) => model.id !== 'imported');
      state.manifest.models.push(result.importedModel);
    }
    const first = state.manifest.models.find((model) => model.id === result.branches[0].id);
    if (first) {
      const previousMode = state.mode; state.mode = 'grow';
      selectTarget(first.target, false); state.mode = previousMode;
      state.model = first.id; $('model-select').value = first.id;
    }
    state.secondModel = result.branches[1]?.id || state.secondModel;
    $('second-model-select').value = state.secondModel;
    $('probability').value = result.probability; $('probability-value').textContent = Number(result.probability).toFixed(2);
    $('view').value = result.view; $('channel').value = result.channel; $('channel-value').textContent = result.channel;
    $('channel-controls').hidden = result.view !== 'hidden'; $('view-help').hidden = result.view === 'hidden';
    state.paired = result.mode === 'paired';
    if (result.mode === 'compare') setMode('compare', { preserve: true });
    else setMode(state.paired ? 'inspect' : 'grow', { preserve: true });
    update(result); notice('Imported successfully. The simulation is paused.');
  }
});
async function runDiagnostic(kind) {
  if (!state.ready) return;
  setPlaying(false);
  const button = $(kind === 'parity' ? 'run-parity' : 'run-benchmark');
  button.disabled = true; $('diagnostic-result').textContent = kind === 'parity' ? 'Checking the frozen PyTorch fixture in browser NumPy…' : 'Measuring 128 steps on an independent clone…';
  try {
    const result = await command({ type: kind }, { render: false });
    if (!result?.[kind]) { $('diagnostic-result').textContent = 'The check did not complete. See the error notice.'; return; }
    state[kind] = result[kind];
    $('diagnostic-result').textContent = JSON.stringify(result[kind], null, 2);
    $('export-diagnostics').disabled = false;
  } finally { button.disabled = false; }
}
$('run-parity').addEventListener('click', () => runDiagnostic('parity'));
$('run-benchmark').addEventListener('click', () => runDiagnostic('benchmark'));
$('export-diagnostics').addEventListener('click', () => download({ parity: state.parity, benchmark: state.benchmark }, 'mycelium-browser-diagnostics.json'));
$('retry').addEventListener('click', boot);
$('recreate-worker').addEventListener('click', boot);
document.addEventListener('visibilitychange', () => { if (document.hidden) setPlaying(false); });
window.addEventListener('pagehide', () => state.client?.terminate());
window.addEventListener('hashchange', () => setMode(location.hash.slice(1)));
new ResizeObserver(() => { if (state.last) drawChart(state.last.curves || []); }).observe($('error-chart').parentElement);
for (const button of document.querySelectorAll('[data-close]')) button.addEventListener('click', () => $(button.dataset.close).close());
for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', (event) => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } });
$('about-open').addEventListener('click', () => $('about-dialog').showModal());
$('results-open').addEventListener('click', async () => {
  $('results-dialog').showModal();
  const holder = $('results-content'); holder.textContent = 'Loading recorded results…';
  try {
    const response = await fetch(new URL('./reports/summary.json', import.meta.url));
    if (!response.ok) throw new Error('Recorded evaluation is not available in this build. No quality claim is inferred from the live view.');
    const report = await response.json(); renderResults(report, holder);
  } catch (error) { holder.textContent = error.message; }
});
function renderResults(report, holder) {
  holder.replaceChildren();
  const intro = document.createElement('p'); intro.textContent = report.description || report.note || `Evaluation schema ${report.schema_version || report.schemaVersion || '1'}. All published values come from saved native trials.`; holder.append(intro);
  const items = Array.isArray(report.models) ? report.models : report.models ? Object.entries(report.models).map(([id, model]) => ({ id, ...model })) : Array.isArray(report.results) ? report.results : [];
  if (!items.length) {
    const pre = document.createElement('pre'); pre.textContent = JSON.stringify(report, null, 2); pre.style.cssText = 'white-space:pre-wrap;font-size:10px;overflow-wrap:anywhere'; holder.append(pre); return;
  }
  const wrap = document.createElement('div'); wrap.className = 'results-table-wrap';
  const table = document.createElement('table');
  const head = document.createElement('thead'); const row = document.createElement('tr');
  for (const label of ['Model', 'Growth', 'Persistence', 'Repair']) { const cell = document.createElement('th'); cell.textContent = label; row.append(cell); }
  head.append(row); table.append(head);
  const body = document.createElement('tbody');
  const summarize = (value) => {
    if (typeof value === 'number') return fmt(value, 3);
    if (!value) return 'Not reported';
    if (value.successes !== undefined && value.trials !== undefined) return `${value.successes}/${value.trials} successful`;
    if (value.success_rate !== undefined) return `${(value.success_rate * 100).toFixed(1)}% success`;
    if (value.failures !== undefined) return `${value.failures} failures / ${value.trials ?? value.n ?? '—'} trials`;
    return Object.entries(value).filter(([, v]) => typeof v === 'number').slice(0, 2).map(([k, v]) => `${k.replaceAll('_', ' ')}: ${fmt(v, 3)}`).join('; ') || 'See raw report';
  };
  for (const item of items) {
    const row = document.createElement('tr');
    for (const value of [item.name || item.id || item.model_id, summarize(item.growth), summarize(item.persistence), summarize(item.repair)]) { const cell = document.createElement('td'); cell.textContent = value || '—'; row.append(cell); }
    body.append(row);
  }
  table.append(body); wrap.append(table); holder.append(wrap);
  for (const item of items) {
    const details = document.createElement('details'); details.className = 'result-card'; const summary = document.createElement('summary'); summary.textContent = `${item.name || item.id || item.model_id} · full aggregate`;
    const pre = document.createElement('pre'); pre.textContent = JSON.stringify(item, null, 2); details.append(summary, pre);
    const links = document.createElement('p');
    for (const [field, label] of [['report_json', 'All trials · JSON'], ['report_csv', 'All trials · CSV'], ['plot', 'Recorded curves']]) {
      if (typeof item[field] !== 'string') continue;
      const path = item[field].startsWith('reports/') ? item[field] : `reports/${item[field]}`;
      if (!/^reports\/[a-zA-Z0-9_./-]+$/.test(path) || path.includes('..')) continue;
      const link = document.createElement('a'); link.href = new URL(path, import.meta.url); link.textContent = label; link.style.marginRight = '16px'; links.append(link);
    }
    details.append(links); holder.append(details);
  }
  if (report.limitations) { const note = document.createElement('p'); note.className = 'results-warning'; note.textContent = Array.isArray(report.limitations) ? report.limitations.join(' ') : report.limitations; holder.append(note); }
}
// Read-only diagnostics for repeatable browser verification, never an alternate simulator.
Object.defineProperty(window, 'mycelium', { value: Object.freeze({ get ready() { return state.ready; }, get snapshot() { return state.last; }, get diagnostics() { return { parity: state.parity, benchmark: state.benchmark }; }, get transport() { return { busy: state.client?.busy, generation: state.client?.generation, queued: state.client?.queue.length }; } }) });
setMode(location.hash.slice(1) || 'grow');
requestAnimationFrame(animate);
boot();
