"""Verify bytes, package boundaries, and HTTP serving at a Pages subpath."""
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import urllib.request
import urllib.error
import zipfile

root=Path(__file__).resolve().parent.parent
dist=root/'dist'
manifest=json.loads((dist/'build-manifest.json').read_text())
for entry in manifest['files']:
    path=dist/entry['file']
    assert path.resolve().is_relative_to(dist.resolve()) and not path.is_symlink(), entry['file']
    content=path.read_bytes()
    assert len(content)==entry['bytes'], entry['file']
    assert hashlib.sha256(content).hexdigest()==entry['sha256'], entry['file']
with zipfile.ZipFile(dist/'runtime/mycelium_runtime.zip') as archive:
    names=archive.namelist()
    assert 'mycelium_nca/numpy_sim.py' in names
    assert all(not any(word in name for word in ('training','torch_model','targets')) for name in names)
    for name in names:
        assert archive.read(name)==(root/'src'/name).read_bytes(),name

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith('/Mycelium/'):
            self.send_error(404);return
        self.path='/'+self.path[len('/Mycelium/'):]
        super().do_GET()
    def log_message(self,*args):pass

server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(dist)))
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
base=f'http://127.0.0.1:{server.server_port}'
try:
    for entry in manifest['files']:
        with urllib.request.urlopen(base+'/Mycelium/'+entry['file']) as response:
            assert response.status==200
            assert hashlib.sha256(response.read()).hexdigest()==entry['sha256']
    try:
        urllib.request.urlopen(base+'/mycelium/index.html')
        raise AssertionError('Incorrect case should fail')
    except urllib.error.HTTPError as error:
        assert error.code==404
finally:
    server.shutdown();server.server_close()
print(f"Verified {len(manifest['files'])} hashes and HTTP assets at /Mycelium/; shared Python source matches exactly.")
