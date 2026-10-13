/** Fetch only Mycelium's pinned browser dependencies; normal use verifies committed hashes. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
const root = fileURLToPath(new URL('../', import.meta.url));
export const runtimeVersion = '0.27.7';
export const runtimeBase = `https://cdn.jsdelivr.net/pyodide/v${runtimeVersion}/full/`;
export const runtimeDirectory = resolve(root, '.cache', `pyodide-${runtimeVersion}`);
export const lockPath = resolve(root, 'scripts', 'runtime-lock.json');
export const digest = (buffer) => createHash('sha256').update(buffer).digest('hex');
export async function readLock() {
  let lock;
  try { lock = JSON.parse(await readFile(lockPath, 'utf8')); }
  catch { throw new Error('Missing scripts/runtime-lock.json. Maintainers must acquire and review the pinned runtime with npm run runtime -- --refresh-lock.'); }
  if (lock.pyodide !== runtimeVersion || lock.baseURL !== runtimeBase || lock.files?.length !== 6) throw new Error('The runtime lock does not match the supported Pyodide release.');
  for (const entry of lock.files) {
    if (!/^[a-zA-Z0-9._-]+$/.test(entry.file) || !/^[a-f0-9]{64}$/.test(entry.sha256) || !Number.isInteger(entry.bytes) || entry.bytes <= 0) throw new Error('Invalid runtime lock entry.');
  }
  return lock;
}
async function fetchAsset(file) {
  const response = await fetch(new URL(file, runtimeBase), { signal: AbortSignal.timeout(120000) });
  if (!response.ok) throw new Error(`Cannot download ${file}: HTTP ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}
async function save(file, bytes) {
  await mkdir(runtimeDirectory, { recursive: true });
  const destination = resolve(runtimeDirectory, file);
  await writeFile(`${destination}.partial`, bytes);
  await rename(`${destination}.partial`, destination);
}
async function main() {
  if (process.argv.includes('--refresh-lock')) {
    const indexBytes = await fetchAsset('pyodide-lock.json');
    const index = JSON.parse(indexBytes);
    const numpy = index.packages.numpy;
    if (numpy.version !== '2.0.2' || !numpy.file_name.endsWith('.whl') || (numpy.depends || []).length) throw new Error('Unexpected NumPy dependency contract. Review this release manually.');
    const names = ['pyodide.mjs', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json', numpy.file_name];
    const files = [];
    for (const file of names) {
      const bytes = file === 'pyodide-lock.json' ? indexBytes : await fetchAsset(file);
      const sha256 = digest(bytes);
      if (file === numpy.file_name && numpy.sha256 !== sha256) throw new Error('NumPy wheel differs from the upstream lock hash.');
      await save(file, bytes); files.push({ file, sha256, bytes: bytes.length });
      console.log(`Acquired ${file}: ${sha256}`);
    }
    await writeFile(lockPath, `${JSON.stringify({ schema: 'mycelium.runtime-lock.v1', pyodide: runtimeVersion, numpy: numpy.version, baseURL: runtimeBase, files }, null, 2)}\n`);
    console.log('Runtime lock refreshed. Review and commit scripts/runtime-lock.json with the release.');
    return;
  }
  const lock = await readLock();
  for (const entry of lock.files) {
    let bytes;
    try { bytes = await readFile(resolve(runtimeDirectory, entry.file)); } catch { /* acquire below */ }
    if (!bytes || bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) {
      bytes = await fetchAsset(entry.file);
      if (bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) throw new Error(`Integrity check failed for ${entry.file}. The lock has not been changed.`);
      await save(entry.file, bytes);
    }
    console.log(`Verified ${entry.file}`);
  }
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
