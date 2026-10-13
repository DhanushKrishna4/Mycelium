/** Serial worker transport. Generation changes invalidate both queued and in-flight views. */
export class RuntimeClient {
  constructor(worker, { onProgress = () => {}, onError = () => {}, timeout = 20000 } = {}) {
    this.worker = worker;
    this.onProgress = onProgress;
    this.onError = onError;
    this.timeout = timeout;
    this.generation = 0;
    this.sequence = 0;
    this.pending = null;
    this.queue = [];
    this.closed = false;
    worker.onmessage = ({ data }) => this.receive(data);
    worker.onerror = (event) => this.fail(new Error(event.message || 'The Python worker stopped unexpectedly.'));
  }
  get busy() { return Boolean(this.pending) || this.queue.length > 0; }
  nextGeneration() {
    this.generation++;
    for (const queued of this.queue) queued.resolve(null);
    this.queue.length = 0;
    return this.generation;
  }
  request(command, { timeout = this.timeout } = {}) {
    if (this.closed) return Promise.reject(new Error('Runtime unavailable. Use Reload runtime.'));
    if (this.queue.length >= 128) return Promise.reject(new Error('Too many pending events. Please let the current step finish.'));
    return new Promise((resolve, reject) => {
      this.queue.push({ command, resolve, reject, timeout, generation: this.generation, requestId: ++this.sequence });
      this.pump();
    });
  }
  pump() {
    if (this.pending || this.closed || !this.queue.length) return;
    const item = this.queue.shift();
    this.pending = item;
    item.timer = setTimeout(() => this.fail(new Error('The Python runtime is not responding. Recreate the worker to start again.')), item.timeout);
    this.worker.postMessage({ command: item.command, generation: item.generation, requestId: item.requestId });
  }
  receive(message) {
    if (message.progress) { this.onProgress(message.progress); return; }
    const item = this.pending;
    if (!item || message.requestId !== item.requestId) return;
    clearTimeout(item.timer);
    this.pending = null;
    if (item.generation !== this.generation || message.generation !== this.generation) item.resolve(null);
    else if (message.error) item.reject(new Error(message.error));
    else item.resolve(message.result);
    this.pump();
  }
  fail(error) {
    if (this.closed) return;
    this.closed = true;
    this.worker.terminate();
    if (this.pending) { clearTimeout(this.pending.timer); this.pending.reject(error); this.pending = null; }
    for (const item of this.queue) item.reject(error);
    this.queue.length = 0;
    this.onError(error);
  }
  terminate() {
    this.closed = true;
    this.worker.terminate();
    if (this.pending) { clearTimeout(this.pending.timer); this.pending.resolve(null); this.pending = null; }
    for (const item of this.queue) item.resolve(null);
    this.queue.length = 0;
  }
}

/** Actual CSS bounds, independent of DPR, browser zoom, and backing-store size. */
export function cellAt(clientX, clientY, rect, size = 32) {
  return {
    x: Math.max(0, Math.min(size - 1, Math.floor((clientX - rect.left) / rect.width * size))),
    y: Math.max(0, Math.min(size - 1, Math.floor((clientY - rect.top) / rect.height * size))),
  };
}
