/** Offline static assembly. Models are verified and copied, never trained by this command. */
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve, join, relative } from 'node:path';
import { readLock, runtimeDirectory, digest } from './acquire-runtime.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(root, 'dist');
const temporary = resolve(root, '.cache', 'site-build');
const python = process.env.MYCELIUM_PYTHON || resolve(root, '.venv', 'bin', 'python');
const pythonSource = String.raw`
import hashlib, json, pathlib, sys, zipfile
import numpy as np
from mycelium_nca.model_io import load_model
root, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((root/'models/manifest.json').read_text())
entries = manifest.get('models', [])
if not entries: raise ValueError('No trained models: export actual training artifacts before building.')
for entry in entries:
    filename = entry.get('file', entry.get('path'))
    if not isinstance(filename,str) or pathlib.PurePosixPath(filename).name != filename:
        raise ValueError('Manifest model file must be a basename inside models/.')
    model = load_model((root/'models'/filename).read_bytes())
    expected = entry.get('weights_hash')
    if expected and expected != model['weights_hash']: raise ValueError('Manifest weight hash mismatch: '+filename)
    if entry.get('model_sha256') and hashlib.sha256((root/'models'/filename).read_bytes()).hexdigest() != entry['model_sha256']:
        raise ValueError('Manifest artifact hash mismatch: '+filename)
    metadata = model.get('metadata', {})
    if not metadata or not (metadata.get('update', metadata.get('updates', metadata.get('checkpoint_update', 0)))):
        raise ValueError('A model lacks a trained checkpoint counter: '+filename)
for target in manifest.get('targets', []):
    name = target['id']
    if not name.isidentifier(): raise ValueError('Invalid target name')
    array = np.load(root/'targets'/(name+'.npy'), allow_pickle=False)
    if array.shape != (32,32,4) or array.dtype != np.float32 or not np.isfinite(array).all(): raise ValueError('Invalid target array: '+name)
    sha = hashlib.sha256(array.astype('<f4').tobytes()).hexdigest()
    if sha != target['sha256']: raise ValueError('Target hash mismatch: '+name)
    metadata = json.loads((root/'targets'/(name+'.json')).read_text())
    (output/'targets'/(name+'.json')).write_text(json.dumps({**metadata,'rgba':array.tolist()},separators=(',',':'),allow_nan=False)+'\n')
modules = ['__init__','config','numpy_sim','metrics','model_io','interventions','replay','browser','parity']
with zipfile.ZipFile(output/'runtime/mycelium_runtime.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for module in sorted(modules):
        name = 'mycelium_nca/'+module+'.py'
        info = zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info,(root/'src'/name).read_bytes(),compresslevel=9)
print('Validated '+str(len(entries))+' trained models; packaged shared NumPy source.')
`;
async function main() {
  const lock = await readLock();
  // Validate acquisition before replacing any previous build.
  for (const entry of lock.files) {
    let bytes;
    try { bytes = await readFile(join(runtimeDirectory, entry.file)); }
    catch { throw new Error(`Missing ${entry.file}. Run npm run runtime first.`); }
    if (bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) throw new Error(`Runtime integrity check failed: ${entry.file}. Run npm run runtime.`);
  }
  await readFile(join(root, 'models/manifest.json'));
  await rm(temporary, { recursive: true, force: true });
  await mkdir(join(temporary, 'runtime'), { recursive: true });
  await mkdir(join(temporary, 'targets'), { recursive: true });
  for (const file of ['index.html', 'style.css', 'app.js', 'runtime-client.js', 'worker.js', 'icon.svg']) await cp(join(root, 'web', file), join(temporary, file));
  for (const file of ['LICENSE', 'THIRD_PARTY.md']) await cp(join(root, file), join(temporary, file));
  for (const directory of ['models', 'reports', 'fixtures']) {
    try { await cp(join(root, directory), join(temporary, directory), { recursive: true }); }
    catch (error) { if (error.code !== 'ENOENT' || directory !== 'reports') throw error; }
  }
  for (const entry of lock.files) await cp(join(runtimeDirectory, entry.file), join(temporary, 'runtime', entry.file));
  const result = spawnSync(python, ['-c', pythonSource, root, temporary], { cwd: root, encoding: 'utf8', env: { ...process.env, PYTHONPATH: join(root, 'src') } });
  if (result.error) throw new Error(`Python build dependency is unavailable: ${result.error.message}. Create .venv or set MYCELIUM_PYTHON.`);
  if (result.status !== 0) throw new Error(result.stderr || 'Python artifact validation failed.');
  console.log(result.stdout.trim());
  await writeFile(join(temporary, '.nojekyll'), '');
  const files = [];
  async function inventory(directory) {
    for (const entry of (await readdir(directory, { withFileTypes: true })).sort((a,b) => a.name.localeCompare(b.name))) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await inventory(path);
      else { const bytes = await readFile(path); files.push({ file: relative(temporary, path), bytes: bytes.length, sha256: digest(bytes) }); }
    }
  }
  await inventory(temporary);
  await writeFile(join(temporary, 'build-manifest.json'), `${JSON.stringify({ schema: 'mycelium.static-build.v1', runtime: lock.pyodide, files }, null, 2)}\n`);
  await rm(output, { recursive: true, force: true });
  await cp(temporary, output, { recursive: true });
  console.log(`Static build: ${files.length} verified files in dist/. Serve over HTTP; repository subpaths are supported.`);
}
main().catch((error) => { console.error(error.message); process.exitCode = 1; });
