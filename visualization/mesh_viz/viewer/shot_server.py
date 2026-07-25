"""Serve the viewer and accept a POSTed canvas capture, so screenshots need no user action."""
import base64
import http.server
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("ascii", "ignore")
        if "," in body:
            body = body.split(",", 1)[1]
        name = self.path.strip("/") or "shot.png"
        with open(os.path.join(ROOT, name), "wb") as fh:
            fh.write(base64.b64decode(body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", 8740), Handler).serve_forever()
