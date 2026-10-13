"""Serve the production bundle at an explicit, case-sensitive Pages subpath."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8732);p.add_argument('--base',default='/Mycelium/');p.add_argument('--directory',default='dist');a=p.parse_args()
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path=urlsplit(self.path).path
        if path == a.base.rstrip('/'):
            self.send_response(301);self.send_header('Location',a.base);self.end_headers();return
        if not path.startswith(a.base):
            self.send_error(404,'Use the exact repository subpath '+a.base);return
        self.path='/'+self.path[len(a.base):]
        return super().do_GET()
    def end_headers(self):
        self.send_header('Cache-Control','no-cache');super().end_headers()
server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(Path(a.directory).resolve())))
print(f'http://127.0.0.1:{a.port}{a.base}',flush=True)
server.serve_forever()
