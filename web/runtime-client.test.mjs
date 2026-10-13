import test from 'node:test';
import assert from 'node:assert/strict';
import { RuntimeClient, cellAt } from './runtime-client.js';

class FakeWorker {
  sent = [];
  postMessage(message) { this.sent.push(message); }
  terminate() { this.terminated = true; }
  answer(index, result) { const { requestId, generation } = this.sent[index]; this.onmessage({ data: { requestId, generation, result } }); }
}
test('requests are serial and a reset discards in-flight and queued old generations', async () => {
  const worker = new FakeWorker();
  const client = new RuntimeClient(worker);
  const tick = client.request({ type: 'tick', steps: 8 });
  const erase = client.request({ type: 'event', kind: 'erase' });
  assert.equal(worker.sent.length, 1);
  client.nextGeneration();
  const reset = client.request({ type: 'reset', model: 'new' });
  assert.equal(worker.sent.length, 1);
  worker.answer(0, { step: 8 });
  assert.equal(await tick, null);
  assert.equal(await erase, null);
  assert.equal(worker.sent.length, 2);
  worker.answer(1, { step: 0 });
  assert.deepEqual(await reset, { step: 0 });
  assert.equal(client.busy, false);
  client.terminate();
});
test('unresponsive workers terminate and reject queued requests', async () => {
  const worker = new FakeWorker();
  const client = new RuntimeClient(worker, { timeout: 10 });
  const a = assert.rejects(client.request({ type: 'tick' }), /not responding/);
  const b = assert.rejects(client.request({ type: 'reset' }), /not responding/);
  await Promise.all([a, b]);
  assert.equal(worker.terminated, true);
  assert.equal(client.closed, true);
});
test('pointer mapping handles offsets, non-integer CSS dimensions, zoom and capture outside', () => {
  const rect = { left: 20.5, top: 80, width: 321.2, height: 321.2 };
  assert.deepEqual(cellAt(181.1, 240.6, rect), { x: 16, y: 16 });
  assert.deepEqual(cellAt(-100, 900, rect), { x: 0, y: 31 });
});
