"""End-to-end tests for ollama-over-dns.

These exercise the whole stack together, not one layer at a time:

* ``TestSocketE2E`` runs the real ``Server`` over a real UDP socket against a
  stub backend over real HTTP, sends a real DNS query on the wire, and asserts
  on the parsed response. This is deterministic and needs no external tools.
* ``TestDigE2E`` drives a real ``server.py`` subprocess with the actual ``dig``
  client. Skipped automatically if ``dig`` is not installed.

Run with:  python -m unittest test_e2e
"""

import http.server
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
import warnings

# This module spawns background threads and real UDP sockets; silence the
# ResourceWarning that Python emits at finalization for those sockets.
warnings.filterwarnings("ignore", category=ResourceWarning)

from server import (build_query, build_opt, parse_message, TYPE_TXT,
                    OllamaBackend, Server)


def _decode_txt(rdata):
    """Decode length-prefixed TXT character-strings back into a string."""
    out = []
    i = 0
    while i < len(rdata):
        n = rdata[i]
        i += 1
        out.append(rdata[i:i + n].decode("utf-8"))
        i += n
    return "".join(out)


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_stub(chunks):
    """Build a stub /api/chat handler that yields ``chunks`` as NDJSON."""

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            lines = [
                json.dumps({"message": {"role": "assistant", "content": c},
                            "done": False}).encode() + b"\n"
                for c in chunks
            ]
            lines.append(json.dumps({"message": {"role": "assistant", "content": ""},
                                     "done": True}).encode() + b"\n")
            body = b"".join(lines)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True  # don't hold the connection open

        def log_message(self, *args):
            pass

    return Stub


def _query_with_opt(qname):
    """Build a query that carries an EDNS0 OPT record (for the echo test)."""
    body = bytearray(build_query(qname, qtype=TYPE_TXT))
    body[10], body[11] = 0, 1  # arcount = 1
    return bytes(body) + build_opt()


def _serve(sock, server, requests):
    """Handle ``requests`` DNS packets on ``sock`` (used by the socket test)."""
    for _ in range(requests):
        data, addr = sock.recvfrom(4096)
        header, questions, _answers, _additional, opt = parse_message(data)
        response = server.handle(header, questions, opt)
        if response:
            sock.sendto(response, addr)


class TestSocketE2E(unittest.TestCase):
    """Real UDP socket + real Server + stub backend over real HTTP."""

    def _run(self, chunks, page_bytes, queries):
        """Start stub + server, answer ``queries``, return list of responses."""
        stub_cls = _make_stub(chunks)
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), stub_cls)
        stub_port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        backend = OllamaBackend(f"http://127.0.0.1:{stub_port}", "test",
                                max_tokens=50, timeout=5)
        server = Server(backend, zone="e.com", page_bytes=page_bytes, wait=3.0)

        # A real client has its own socket; the server socket is separate so a
        # response never gets re-read as a query.
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_sock.bind(("127.0.0.1", 0))
        uport = server_sock.getsockname()[1]
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_sock.bind(("127.0.0.1", 0))
        server_sock.settimeout(5)  # don't block forever if a query is lost
        t = threading.Thread(target=_serve, args=(server_sock, server,
                                                   len(queries)), daemon=True)
        t.start()
        try:
            queries = [_query_with_opt(q) for q in queries]
            responses = []
            for query in queries:
                client_sock.sendto(query, ("127.0.0.1", uport))
                client_sock.settimeout(5)
                data, _ = client_sock.recvfrom(65535)
                responses.append(data)
            return responses
        finally:
            t.join(timeout=5)
            server_sock.close()
            client_sock.close()
            httpd.shutdown()
            httpd.server_close()

    def test_full_stack(self):
        responses = self._run(["hello ", "world ", "!!!"], 600, ["hello.e.com"])
        header, questions, answers, additional, opt = parse_message(responses[0])
        self.assertEqual(questions[0]["name"], "hello.e.com")
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["type"], TYPE_TXT)
        self.assertTrue(header["aa"])  # AA flag set
        self.assertIsNotNone(opt)  # EDNS0 OPT echoed
        self.assertLess(len(responses[0]), 1232)  # stays under EDNS buffer
        text = _decode_txt(answers[0]["rdata"])
        self.assertIn("hello world !!!", text)
        self.assertIn("[done]", text)

    def test_paging(self):
        # "ABCDEF" with 5-byte pages -> page0 "ABCDE" (full), page1 "F" (last).
        responses = self._run(["ABC", "DEF"], 5, ["pg.e.com", "p1.pg.e.com"])
        _, _, a0, _, _ = parse_message(responses[0])
        t0 = _decode_txt(a0[0]["rdata"])
        self.assertIn("[more: dig p1.pg.e.com TXT]", t0)  # chains off core name

        _, _, a1, _, _ = parse_message(responses[1])
        t1 = _decode_txt(a1[0]["rdata"])
        self.assertIn("[done]", t1)


class TestDigE2E(unittest.TestCase):
    """Drive a real server.py subprocess with the real dig client."""

    def test_dig_paging(self):
        dig = shutil.which("dig")
        if not dig:
            self.skipTest("dig not available")

        # A long answer (700 bytes) spans two 600-byte pages: page0 -> [more],
        # page1 -> [done]. This exercises the real paging hint over the wire.
        stub_cls = _make_stub(["x" * 700])
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), stub_cls)
        stub_port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            port = _free_port()
            proc = subprocess.Popen(
                [sys.executable, "server.py", "--backend", "ollama",
                 "--url", f"http://127.0.0.1:{stub_port}", "--model", "test",
                 "--port", str(port), "--wait", "3"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                time.sleep(1.5)  # let the server bind
                p0 = subprocess.run(
                    [dig, "@127.0.0.1", "-p", str(port), "hello.e.com", "TXT"],
                    capture_output=True, text=True, timeout=15)
                self.assertIn("[more: dig p1.hello.e.com TXT]", p0.stdout)

                p1 = subprocess.run(
                    [dig, "@127.0.0.1", "-p", str(port), "p1.hello.e.com", "TXT"],
                    capture_output=True, text=True, timeout=15)
                self.assertIn("[done]", p1.stdout)
            finally:
                proc.terminate()
                try:
                    proc.communicate(timeout=5)  # closes the stdout PIPE
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
