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
import struct
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
                    OllamaBackend, Server, serve)


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


def _make_stub(chunks, delay=0.0):
    """Build a stub /api/chat handler that yields ``chunks`` as NDJSON.

    With ``delay``, each chunk is written and flushed ``delay`` seconds apart,
    like a model generating in real time.
    """

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
            for line in lines:
                if delay:
                    time.sleep(delay)
                self.wfile.write(line)
                self.wfile.flush()
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


class _ServerProcess:
    """A real ``server.py`` subprocess pointed at a stub ollama over HTTP."""

    def __init__(self, chunks, *extra_args, delay=0.0):
        self.chunks = chunks
        self.extra_args = list(extra_args)
        self.delay = delay

    def __enter__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                     _make_stub(self.chunks, self.delay))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = _free_port()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "server.py", "--backend", "ollama",
             "--url", f"http://127.0.0.1:{self.httpd.server_address[1]}",
             "--model", "test", "--host", "127.0.0.1", "--port", str(self.port)]
            + self.extra_args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        # Wait for the "listening" line rather than sleeping a fixed time, but
        # never block forever if the server dies or never prints it.
        lines = []
        reader = threading.Thread(target=lambda: lines.append(self.proc.stdout.readline()),
                                  daemon=True)
        reader.start()
        reader.join(timeout=10)
        line = lines[0] if lines else ""
        if "listening" not in line:
            self.__exit__(None, None, None)
            raise RuntimeError(f"server.py did not start: {line!r}")
        return self

    def query(self, packet, timeout=5):
        """Send raw bytes, return the raw reply (or raise socket.timeout)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(timeout)
            s.sendto(packet, ("127.0.0.1", self.port))
            data, _ = s.recvfrom(65535)
            return data
        finally:
            s.close()

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.output = self.proc.communicate(timeout=5)[0]
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.output = self.proc.communicate()[0]
        self.httpd.shutdown()
        self.httpd.server_close()


class TestHostilePacketsE2E(unittest.TestCase):
    """A real server.py must survive malformed packets and keep answering."""

    def test_pointer_loop_packet_does_not_stop_the_server(self):
        loop = struct.pack(">HHHHHH", 1, 0, 1, 0, 0, 0) + b"\xc0\x0c" + b"\x00\x10\x00\x01"
        with _ServerProcess(["still here"]) as srv:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(loop, ("127.0.0.1", srv.port))
            s.close()
            reply = srv.query(build_query("alive.e.com"))
            _, _, answers, _, _ = parse_message(reply)
            self.assertIn("still here", _decode_txt(answers[0]["rdata"]))
        self.assertIn("dropping malformed packet", srv.output)
        self.assertIn("compression pointer loop", srv.output)

    def test_response_packet_gets_no_reply(self):
        pkt = bytearray(build_query("hi.e.com"))
        pkt[2] |= 0x80  # QR=1
        with _ServerProcess(["x"]) as srv:
            with self.assertRaises(socket.timeout):
                srv.query(bytes(pkt), timeout=1)
            # ...and a normal query right after is still answered.
            parse_message(srv.query(build_query("hi.e.com")))


class TestZoneApexE2E(unittest.TestCase):
    """NS/SOA for the apex over a real socket (and dig, if installed)."""

    def test_apex_ns_and_soa_have_no_txt_answer(self):
        with _ServerProcess(["x"]) as srv:
            for qtype in (2, 6):  # NS, SOA
                with self.subTest(qtype=qtype):
                    header, questions, answers, _, _ = parse_message(
                        srv.query(build_query("e.com", qtype=qtype)))
                    self.assertEqual(header["rcode"], 0)
                    self.assertEqual(questions[0]["qtype"], qtype)
                    self.assertEqual(answers, [])
            _, _, answers, _, _ = parse_message(srv.query(build_query("e.com")))
            self.assertIn("ollama-over-dns", _decode_txt(answers[0]["rdata"]))

    def test_apex_ns_with_dig(self):
        dig = shutil.which("dig")
        if not dig:
            self.skipTest("dig not available")
        with _ServerProcess(["x"]) as srv:
            out = subprocess.run([dig, "@127.0.0.1", "-p", str(srv.port), "e.com", "NS"],
                                 capture_output=True, text=True, timeout=15).stdout
            self.assertIn("status: NOERROR", out)
            self.assertIn("ANSWER: 0", out)
            self.assertNotIn("TXT", out.split("ANSWER: 0", 1)[1])


class _Held:
    """Backend that yields ``text`` and then holds the generation open until
    ``release`` is set, so a query for it waits the server's full ``--wait``."""

    def __init__(self, text="partial"):
        self.text = text
        self.release = threading.Event()

    def stream(self, prompt):
        yield self.text
        self.release.wait(10)


class _Serving:
    """Run ``serve()`` on a real UDP socket in a background thread."""

    def __init__(self, server, **kw):
        self.server, self.kw = server, kw

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.addr = self.sock.getsockname()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=serve, args=(self.sock, self.server),
                                       kwargs=dict(stop=self.stop, **self.kw), daemon=True)
        self.thread.start()
        return self

    def ask(self, qname, results, key=None, timeout=5):
        """Query ``qname``; store (seconds, text-or-None) in ``results[key]``."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        start = time.monotonic()
        try:
            s.sendto(build_query(qname), self.addr)
            data, _ = s.recvfrom(65535)
            _, _, answers, _, _ = parse_message(data)
            text = _decode_txt(answers[0]["rdata"]) if answers else ""
        except socket.timeout:
            text = None
        finally:
            s.close()
        results[key or qname] = (time.monotonic() - start, text)

    def ask_all(self, qnames, timeout=5):
        results = {}
        threads = [threading.Thread(target=self.ask, args=(q, results, q, timeout))
                   for q in qnames]
        [t.start() for t in threads]
        [t.join() for t in threads]
        return results

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sock.close()


class TestConcurrentServing(unittest.TestCase):
    """serve() handles each packet on its own thread."""

    def test_simultaneous_slow_queries_all_answer_within_one_wait(self):
        backend = _Held()
        server = Server(backend, wait=1.0, max_gen=8)
        names = [f"q{i}.e.com" for i in range(8)]
        with _Serving(server) as srv:
            start = time.monotonic()
            results = srv.ask_all(names)
            elapsed = time.monotonic() - start
        backend.release.set()
        for name in names:
            seconds, text = results[name]
            self.assertEqual(text, "partial [thinking... repeat the same query]", name)
            self.assertGreaterEqual(seconds, 0.9)  # each one really waited...
        self.assertLess(elapsed, 2.0)  # ...but in parallel (serial would be 8 s)

    def test_fast_query_not_delayed_by_a_waiting_one(self):
        backend = _Held()
        with _Serving(Server(backend, wait=2.0)) as srv:
            results = {}
            slow = threading.Thread(target=srv.ask, args=("slow.e.com", results))
            slow.start()
            time.sleep(0.2)  # the slow query is now waiting inside handle()
            srv.ask("e.com", results)  # apex usage string: no generation
            slow.join()
        backend.release.set()
        self.assertLess(results["e.com"][0], 0.5)
        self.assertIn("ollama-over-dns", results["e.com"][1])
        self.assertGreaterEqual(results["slow.e.com"][0], 1.9)

    def test_malformed_packet_does_not_delay_valid_one(self):
        with _Serving(Server(_Held(), wait=0)) as srv:
            junk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for pkt in (b"", b"\x00", struct.pack(">HHHHHH", 1, 0, 1, 0, 0, 0) + b"\xc0\x0c\x00\x10\x00\x01"):
                junk.sendto(pkt, srv.addr)
            junk.close()
            results = {}
            srv.ask("e.com", results)
        self.assertLess(results["e.com"][0], 0.5)
        self.assertIn("ollama-over-dns", results["e.com"][1])

    def test_inflight_cap_drops_excess_then_recovers(self):
        backend = _Held()
        with _Serving(Server(backend, wait=1.0, max_gen=8), max_inflight=2) as srv:
            results = {}
            threads = [threading.Thread(target=srv.ask, args=(f"c{i}.e.com", results, None, 2))
                       for i in range(3)]
            for t in threads:
                t.start()
                time.sleep(0.05)  # arrive in order: c0, c1, then c2 over the cap
            [t.join() for t in threads]
            answered = sorted(k for k, (_, text) in results.items() if text is not None)
            self.assertEqual(answered, ["c0.e.com", "c1.e.com"])
            self.assertIsNone(results["c2.e.com"][1])
            # Both slots were released: a new query is answered.
            later = {}
            srv.ask("e.com", later)
            self.assertIsNotNone(later["e.com"][1])
        backend.release.set()

    def test_handler_exception_releases_its_slot(self):
        class Exploding(Server):
            def handle(self, header, questions, opt):
                if questions and questions[0]["name"].startswith("boom"):
                    raise RuntimeError("kaboom")
                return super().handle(header, questions, opt)

        with _Serving(Exploding(_Held(), wait=0), max_inflight=1) as srv:
            results = {}
            for i in range(3):
                srv.ask(f"boom{i}.e.com", results, timeout=0.5)  # no reply, slot freed
            srv.ask("e.com", results)
        self.assertIsNone(results["boom0.e.com"][1])
        self.assertIn("ollama-over-dns", results["e.com"][1])

    def test_stop_returns_promptly(self):
        srv = _Serving(Server(_Held())).__enter__()
        start = time.monotonic()
        srv.__exit__(None, None, None)
        self.assertFalse(srv.thread.is_alive())
        self.assertLess(time.monotonic() - start, 1.0)

    def test_concurrent_queries_against_server_process(self):
        # Real server.py, stub model streaming 10 chunks 0.15 s apart (~1.5 s
        # per answer). Three different prompts at once: each finishes in about
        # one generation time, not three back to back.
        chunks = [f"w{i} " for i in range(10)]
        with _ServerProcess(chunks, "--max-gen", "3", delay=0.15) as proc:
            results = {}
            def ask(name):
                start = time.monotonic()
                _, _, answers, _, _ = parse_message(proc.query(build_query(name), timeout=10))
                results[name] = (time.monotonic() - start, _decode_txt(answers[0]["rdata"]))
            threads = [threading.Thread(target=ask, args=(f"p{i}x.e.com",)) for i in range(3)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        for name, (seconds, text) in results.items():
            self.assertEqual(text, "".join(chunks) + " [done]", name)
            self.assertLess(seconds, 3.0, name)  # serial would be ~1.6 / 3.2 / 4.8 s


def _long_name(n):
    """A valid name of exactly ``n`` characters under e.com."""
    labels, room = [], n - len(".e.com")
    while room > 0:
        size = min(63, room - 2 if room - 63 == 1 else room)
        labels.append("a" * size)
        room -= size + 1
    name = ".".join(labels) + ".e.com"
    assert len(name) == n, len(name)
    return name


class TestReplySizeE2E(unittest.TestCase):
    """Reply size against a real server.py, over a socket and with dig."""

    NAME = _long_name(250)  # its next-page hint is exactly 253 characters

    def test_longest_pageable_name_fits_1232(self):
        with _ServerProcess(["x" * 700]) as srv:
            reply = srv.query(_query_with_opt(self.NAME))
        header, _, answers, _, opt = parse_message(reply)
        self.assertLessEqual(len(reply), 1232)
        self.assertFalse(header["tc"])
        self.assertIn(f"[more: dig p1.{self.NAME} TXT]", _decode_txt(answers[0]["rdata"]))
        self.assertEqual((opt["class"], opt["rdata"]), (1232, b""))

    def _dig(self, port, *flags):
        dig = shutil.which("dig")
        if not dig:
            self.skipTest("dig not available")
        return subprocess.run([dig, "@127.0.0.1", "-p", str(port), self.NAME, "TXT", *flags],
                              capture_output=True, text=True, timeout=15).stdout

    @staticmethod
    def _flags(out):
        line = next(l for l in out.splitlines() if l.startswith(";; flags:"))
        return line.split(";")[2].split(":")[1].split()

    def test_dig_longest_name(self):
        with _ServerProcess(["x" * 700]) as srv:
            out = self._dig(srv.port)
        size = int(out.split("MSG SIZE  rcvd: ")[1].split()[0])
        self.assertLessEqual(size, 1232)
        self.assertNotIn("tc", self._flags(out))
        self.assertIn("[more: dig p1.", out)
        self.assertIn("udp: 1232", out)  # our buffer size, not an echo of dig's
        self.assertNotIn("COOKIE", out)  # dig's cookie is not echoed back
        self.assertNotIn("Truncated", out)

    def test_dig_small_buffer_and_no_edns_get_tc(self):
        with _ServerProcess(["x" * 700]) as srv:
            # +ignore: report the truncated UDP reply instead of retrying over TCP.
            for flags in (("+bufsize=512", "+ignore"), ("+noedns", "+ignore")):
                with self.subTest(flags=flags):
                    out = self._dig(srv.port, *flags)
                    self.assertIn("tc", self._flags(out))
                    self.assertIn("ANSWER: 0", out)


def _lan_address():
    """A non-loopback IPv4 address of this machine, or None."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1; nothing is sent
        addr = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if addr.startswith("127.") else addr


class TestDefaultsE2E(unittest.TestCase):
    """server.py with no --host/--port: loopback only, port 5300."""

    def test_default_bind(self):
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.bind(("127.0.0.1", 5300))
            probe.close()
        except OSError:
            self.skipTest("port 5300 is in use on this machine")
        proc = subprocess.Popen(
            [sys.executable, "-u", "server.py", "--backend", "llamacpp",
             "--url", "http://127.0.0.1:1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            self.assertIn("listening on udp 127.0.0.1:5300", proc.stdout.readline())
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(5)
            s.sendto(build_query("e.com"), ("127.0.0.1", 5300))
            _, _, answers, _, _ = parse_message(s.recvfrom(4096)[0])
            self.assertIn("dig @127.0.0.1 -p 5300 +short", _decode_txt(answers[0]["rdata"]))
            lan = _lan_address()
            if lan:  # not reachable from the network by default
                s.settimeout(1)
                s.sendto(build_query("e.com"), (lan, 5300))
                with self.assertRaises((socket.timeout, ConnectionRefusedError)):
                    s.recvfrom(4096)
            s.close()
        finally:
            proc.terminate()
            proc.communicate(timeout=5)


import client  # noqa: E402


class _Paced:
    """Streams ``text`` in small chunks with a delay, like a real model."""

    def __init__(self, text, chunk=3, delay=0.004, fail=None):
        self.text, self.chunk, self.delay, self.fail = text, chunk, delay, fail

    def stream(self, prompt):
        for i in range(0, len(self.text), self.chunk):
            time.sleep(self.delay)
            yield self.text[i:i + self.chunk]
        if self.fail:
            raise RuntimeError(self.fail)


def _record_statuses(server):
    """Wrap fetch_page to record the status of every reply the server sends."""
    seen = []
    original = server.manager.fetch_page

    def fetch_page(*a, **kw):
        reply = original(*a, **kw)
        seen.append(client.split_status(reply)[1])
        return reply

    server.manager.fetch_page = fetch_page
    return seen


class TestClientE2E(unittest.TestCase):
    """client.py against a real server: the answer must come back exactly."""

    # Quotes, a backslash, curly punctuation, 2/3/4-byte characters, a newline,
    # and words that a 37-byte page splits at spaces.
    TEXT = ('He said "hello" \\ it\u2019s 42\u00b0C in \u6771\u4eac \U0001F999 \u2014 fine.\n'
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu ") * 4

    def test_answer_is_byte_for_byte_exact_through_thinking_and_paging(self):
        # ~150 chars/s against a 0.15 s wait: a 37-byte page takes ~0.25 s to
        # fill, so the client sees [thinking] and has to repeat.
        server = Server(_Paced(self.TEXT, delay=0.02), page_bytes=37, wait=0.15)
        statuses = _record_statuses(server)
        with _Serving(server) as srv:
            answer, error = client.fetch_answer("exact.e.com", *srv.addr, timeout=30)
        self.assertIsNone(error)
        self.assertEqual(answer, self.TEXT)
        # The run really exercised both paths.
        self.assertIn("done", statuses)
        self.assertGreater(sum(1 for st in statuses if st.startswith("thinking")), 0)
        self.assertGreater(sum(1 for st in statuses if st.startswith("more")), 5)

    def test_page_ending_in_a_space_keeps_it(self):
        text = "abcdefghi jklmnopqr stuvwxyz"  # page 0 is "abcdefghi " (10 bytes)
        server = Server(_Paced(text), page_bytes=10, wait=1.0)
        with _Serving(server) as srv:
            answer, error = client.fetch_answer("space.e.com", *srv.addr)
        self.assertEqual((answer, error), (text, None))

    def test_cli_without_dig_on_path(self):
        # The CLI as a subprocess with an empty PATH: no dig, same exact answer.
        server = Server(_Paced(self.TEXT), page_bytes=200, wait=0.3)
        with _Serving(server) as srv:
            out = subprocess.run(
                [sys.executable, "client.py", "--port", str(srv.addr[1]),
                 "--prompt", "what is this"],
                capture_output=True, timeout=60, env={"PATH": "", "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.decode("utf-8"), self.TEXT + "\n")

    def test_busy_is_retried_until_a_slot_frees(self):
        hold = _Held()
        server = Server(_RoutingBackend(hold, _Paced("finally mine")), max_gen=1, wait=0.5)
        server.manager.get_or_start("other")  # takes the only slot
        threading.Timer(1.0, hold.release.set).start()
        statuses = _record_statuses(server)
        with _Serving(server) as srv:
            start = time.monotonic()
            answer, error = client.fetch_answer("mine.e.com", *srv.addr, timeout=20)
        self.assertEqual((answer, error), ("finally mine", None))
        self.assertGreaterEqual(time.monotonic() - start, 0.9)

    def test_mid_stream_error_prints_partial_and_exits_1(self):
        server = Server(_Paced("partial answer ", fail="backend died"), wait=1.0)
        with _Serving(server) as srv:
            out = subprocess.run(
                [sys.executable, "client.py", "--port", str(srv.addr[1]), "--prompt", "x"],
                capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 1)
        self.assertEqual(out.stdout, "partial answer \n")
        self.assertEqual(out.stderr, "error: backend died\n")

    def test_wrong_zone_reports_refused(self):
        with _Serving(Server(_Paced("x"))) as srv:
            answer, error = client.fetch_answer("x.wrong.org", *srv.addr)
        self.assertEqual(answer, "")
        self.assertIn("rcode 5", error)

    def test_nothing_listening(self):
        answer, error = client.fetch_answer("x.e.com", "127.0.0.1", _free_port(), timeout=5)
        self.assertEqual(answer, "")
        self.assertRegex(error, "nothing is listening|no reply")

    def test_silent_server_times_out(self):
        sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sink.bind(("127.0.0.1", 0))
        try:
            start = time.monotonic()
            answer, error = client.fetch_answer("x.e.com", *sink.getsockname(),
                                                timeout=1.0, query_timeout=0.3)
            self.assertLess(time.monotonic() - start, 3)
        finally:
            sink.close()
        self.assertEqual(answer, "")
        self.assertRegex(error, "no reply|timed out")

    def _fake_server(self, respond):
        """A raw UDP responder: respond(query_bytes) -> list of reply packets."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(5)

        def run():
            try:
                data, addr = sock.recvfrom(4096)
                for reply in respond(data):
                    sock.sendto(reply, addr)
            except OSError:
                pass

        threading.Thread(target=run, daemon=True).start()
        return sock

    def test_truncated_reply_is_an_error(self):
        from server import assemble_response

        def respond(data):
            header, questions, _, _, _ = parse_message(data)
            return [assemble_response(header["id"], questions, [], tc=True)]

        sock = self._fake_server(respond)
        try:
            _, error = client.fetch_answer("x.e.com", *sock.getsockname(), timeout=5)
        finally:
            sock.close()
        self.assertIn("truncated", error)

    def test_reply_with_wrong_id_is_ignored(self):
        from server import assemble_response, build_txt_rdata

        def respond(data):
            header, questions, _, _, _ = parse_message(data)
            rdata = build_txt_rdata(["right [done]"])
            ans = b"\xc0\x0c" + struct.pack(">HHIH", TYPE_TXT, 1, 0, len(rdata)) + rdata
            wrong = assemble_response((header["id"] + 1) & 0xFFFF, questions, [ans])
            right = assemble_response(header["id"], questions, [ans])
            return [wrong, b"garbage", right]

        sock = self._fake_server(respond)
        try:
            self.assertEqual(client.fetch_answer("x.e.com", *sock.getsockname(), timeout=5),
                             ("right", None))
        finally:
            sock.close()

    def test_max_pages(self):
        server = Server(_Paced("x" * 200, chunk=200, delay=0), page_bytes=10, wait=1.0)
        with _Serving(server) as srv:
            answer, error = client.fetch_answer("many.e.com", *srv.addr, max_pages=3)
        self.assertEqual(answer, "x" * 30)
        self.assertIn("--max-pages 3", error)


class _RoutingBackend:
    """Sends the prompt "other" to one backend and everything else to another."""

    def __init__(self, other, mine):
        self.other, self.mine = other, mine

    def stream(self, prompt):
        return (self.other if prompt == "other" else self.mine).stream(prompt)


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
