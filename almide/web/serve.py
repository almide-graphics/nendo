#!/usr/bin/env python3
"""Threaded no-cache dev server for the nendo studio (default port 8772)."""
import http.server, socketserver, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8772
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()
    def log_message(self, *a): pass
class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
with S(("127.0.0.1", PORT), H) as s:
    s.serve_forever()
