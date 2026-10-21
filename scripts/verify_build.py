"""Verify bytes, package boundaries, and HTTP serving at a Pages subpath."""
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path
import struct
import threading
import urllib.request
import urllib.error
from urllib.parse import urlsplit
import zipfile
import zlib

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


class StaticHead(HTMLParser):
    """Read crawler-visible metadata without evaluating application scripts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_head = False
        self.meta = {}
        self.canonical = []

    def handle_starttag(self, tag, attrs):
        if tag == 'head':
            self.in_head = True
        if not self.in_head:
            return
        attributes = dict(attrs)
        if tag == 'meta':
            key = attributes.get('property') or attributes.get('name')
            if key:
                self.meta.setdefault(key, []).append(attributes.get('content', ''))
        elif tag == 'link' and 'canonical' in attributes.get('rel', '').lower().split():
            self.canonical.append(attributes.get('href', ''))

    def handle_endtag(self, tag):
        if tag == 'head':
            self.in_head = False


canonical_url = 'https://dhanushkrishna4.github.io/Mycelium/'
image_url = canonical_url + 'og-v1.png'
head = StaticHead()
head.feed((dist/'index.html').read_text(encoding='utf-8'))
assert head.canonical == [canonical_url], 'Expected one canonical HTTPS Mycelium URL'
required_metadata = (
    'og:type', 'og:site_name', 'og:title', 'og:description', 'og:url',
    'og:image', 'og:image:type', 'og:image:width', 'og:image:height', 'og:image:alt',
    'twitter:card', 'twitter:title', 'twitter:description', 'twitter:image',
    'twitter:image:alt',
)
metadata = {}
for key in required_metadata:
    values = head.meta.get(key, [])
    assert len(values) == 1 and values[0].strip(), f'Missing or duplicate static {key}'
    metadata[key] = values[0]
assert metadata['og:type'] == 'website'
assert metadata['og:site_name'] == 'Mycelium'
assert metadata['og:url'] == canonical_url
assert metadata['og:image'] == image_url
assert metadata['og:image:type'] == 'image/png'
assert metadata['og:image:width'] == '1200' and metadata['og:image:height'] == '630'
assert metadata['twitter:card'] == 'summary_large_image'
for field in ('title', 'description', 'image', 'image:alt'):
    assert metadata['og:' + field] == metadata['twitter:' + field], f'Conflicting {field}'
for optional in ('og:image:secure_url', 'og:image:url'):
    if optional in head.meta:
        assert head.meta[optional] == [image_url], f'Conflicting {optional}'

image_path = urlsplit(image_url).path.removeprefix('/Mycelium/')
image_file = dist/image_path
assert image_file.resolve().is_relative_to(dist.resolve()) and not image_file.is_symlink()
assert any(entry['file'] == image_path for entry in manifest['files']), 'OG image missing from build manifest'
png = image_file.read_bytes()
assert png.startswith(b'\x89PNG\r\n\x1a\n'), 'OG image must be actual PNG bytes'
chunks = {}
offset = 8
chunk_order = []
while offset < len(png):
    assert offset + 12 <= len(png), 'Truncated PNG chunk header'
    length, kind = struct.unpack('>I4s', png[offset:offset + 8])
    end = offset + 12 + length
    assert end <= len(png), 'Truncated PNG chunk data'
    data = png[offset + 8:offset + 8 + length]
    crc = struct.unpack('>I', png[offset + 8 + length:end])[0]
    assert zlib.crc32(kind + data) & 0xffffffff == crc, f'Invalid PNG {kind!r} checksum'
    chunks.setdefault(kind, []).append(data)
    chunk_order.append(kind)
    offset = end
    if kind == b'IEND':
        assert length == 0 and offset == len(png), 'Invalid PNG end marker'
        break
assert chunk_order[0] == b'IHDR' and chunk_order[-1] == b'IEND'
assert len(chunks[b'IHDR']) == 1 and len(chunks[b'IHDR'][0]) == 13
width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', chunks[b'IHDR'][0])
assert (width, height, depth, color) == (1200, 630, 8, 2), 'OG PNG must be 1200×630, 8-bit truecolor RGB'
assert compression == 0 and filtering == 0 and interlace in (0, 1)
assert b'IDAT' in chunks and b'tRNS' not in chunks, 'OG PNG must be opaque'
if b'sRGB' in chunks:
    assert chunks[b'sRGB'] == [b'\x00'], 'Expected one sRGB chunk with perceptual intent 0'
else:
    assert len(chunks.get(b'iCCP', [])) == 1, 'OG PNG requires an sRGB declaration or ICC profile'
    profile_name, profile_data = chunks[b'iCCP'][0].split(b'\x00', 1)
    assert 1 <= len(profile_name) <= 79 and profile_data[0] == 0, 'Invalid PNG ICC profile'
    profile = zlib.decompress(profile_data[1:])
    assert len(profile) >= 128 and profile[16:20] == b'RGB ' and profile[36:40] == b'acsp', 'Invalid RGB ICC profile'
    assert struct.unpack('>I', profile[:4])[0] == len(profile), 'Truncated ICC profile'

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
            if entry['file'] == image_path:
                assert response.headers.get_content_type() == 'image/png', 'OG image served with incorrect content type'
            assert hashlib.sha256(response.read()).hexdigest()==entry['sha256']
    try:
        urllib.request.urlopen(base+'/mycelium/index.html')
        raise AssertionError('Incorrect case should fail')
    except urllib.error.HTTPError as error:
        assert error.code==404
finally:
    server.shutdown();server.server_close()
print(f"Verified {len(manifest['files'])} hashes and HTTP assets at /Mycelium/; shared Python source matches exactly; static OG/Twitter metadata and opaque 1200×630 RGB PNG pass.")
