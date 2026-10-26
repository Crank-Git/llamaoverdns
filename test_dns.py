"""Phase 1: DNS wire-format parse/build tests.

Run with:  python -m unittest test_dns
"""

import struct
import unittest

import json
import http.server
import socketserver
import threading
import time

import client  # noqa: E402  (pure-logic tests below, no dig needed)

from server import (
    TYPE_OPT, TYPE_TXT, CLASS_IN, RCODE_NOERROR, RCODE_REFUSED,
    FLAG_QR, FLAG_AA, FLAG_RD,
    decode_name, encode_name, parse_header, parse_message,
    build_query, build_txt_rdata, build_answer, build_opt, assemble_response,
    query_to_prompt, opt_to_bytes, split_txt, strip_thinking, _paginate,
    Server, GenerationManager, OllamaBackend, LlamaCppBackend,
    TYPE_A, DNSError,
)


class RoundTripQuery(unittest.TestCase):
    def test_query_roundtrip(self):
        qname = "what.does.bravurva.mean.e.com"
        msg = build_query(qname, qtype=TYPE_TXT)
        header, questions, answers, additional, opt = parse_message(msg)
        self.assertEqual(header["id"], 0x1234)
        self.assertFalse(header["qr"])  # a query has QR=0
        self.assertTrue(header["rd"])
        self.assertEqual(len(questions), 1)
        q = questions[0]
        self.assertEqual(q["name"], qname)
        self.assertEqual(q["qtype"], TYPE_TXT)
        self.assertEqual(q["qclass"], CLASS_IN)


class NameDecoding(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        for name in ["e.com", "a.b.c", "", "www.example.com"]:
            wire = encode_name(name)
            decoded, end = decode_name(wire, 0)
            self.assertEqual(decoded, name)
            self.assertEqual(end, len(wire))

    def test_decode_compressed_name(self):
        # "www" is read first, then a pointer jumps back to "example.com" which
        # appears later. The pointer offset is relative to the message start, so
        # we prepend a 12-byte header: "example.com" lands at offset 12.
        #   [12-byte header]
        #   03 'www' C0 0C   -> read "www", jump to offset 12
        #   07 'example' 03 'com' 00
        body = b"\x00" * 12 + b"\x03www\xc0\x12\x07example\x03com\x00"
        name, end = decode_name(body, 12)
        self.assertEqual(name, "www.example.com")
        self.assertEqual(end, 18)  # just past the 2-byte pointer (offset 16)

    def test_root_name(self):
        name, end = decode_name(b"\x00", 0)
        self.assertEqual(name, "")
        self.assertEqual(end, 1)


class TruncatedMessages(unittest.TestCase):
    def test_short_header_rejected(self):
        with self.assertRaises(DNSError):
            parse_header(b"\x00\x01\x02")

    def test_truncated_question_rejected(self):
        # Valid header claims 1 question but the question data is cut short.
        msg = b"\x00\x00" + struct.pack(">HHHHHH", 1, 0, 1, 0, 0, 0) + b"\x03abc"
        with self.assertRaises(DNSError):
            parse_message(msg)

    def test_truncated_rdata_rejected(self):
        # Header says 1 answer with rdlen=10 but no rdata bytes follow.
        name = encode_name("x.com")
        partial = name + struct.pack(">HHIH", TYPE_TXT, CLASS_IN, 0, 10)
        msg = struct.pack(">HHHHHH", 1, FLAG_QR, 0, 1, 0, 0) + partial
        with self.assertRaises(DNSError):
            parse_message(msg)


class QueryToPrompt(unittest.TestCase):
    # (name, zone) -> (prompt, page) or None (not under zone)
    CASES = [
        ("what.does.bravurva.mean.e.com", "e.com", ("what does bravurva mean", 0)),
        ("why-is-the-sky-blue.e.com", "e.com", ("why is the sky blue", 0)),
        ("p1.what.does.bravurva.mean.e.com", "e.com", ("what does bravurva mean", 1)),
        ("p12.foo.e.com", "e.com", ("foo", 12)),
        ("p0.e.com", "e.com", ("", 0)),
        ("FOO.Bar.e.com", "e.com", ("foo bar", 0)),
        ("with-dash.and_underscore.e.com", "e.com", ("with dash and_underscore", 0)),
        ("what.e.com.e.com", "e.com", ("what e com", 0)),  # only trailing zone stripped
        ("e.com", "e.com", ("", 0)),  # the zone itself -> empty prompt
        ("what.e.com.", "e.com", ("what", 0)),  # trailing dot
        # not under the zone -> None
        ("mye.com", "e.com", None),
        ("a.b.c.d", "e.com", None),
        ("what.does.bravurva.mean", "e.com", None),
        ("e.com.e", "e.com", None),
    ]

    def test_cases(self):
        for name, zone, expected in self.CASES:
            with self.subTest(name=name):
                self.assertEqual(query_to_prompt(name, zone), expected)


class ResponseBuilding(unittest.TestCase):
    def test_txt_answer_roundtrip(self):
        qid = 0xABCD
        qname = "q.e.com"
        q = build_query(qname, qid=qid)
        header, questions, _, _, opt = parse_message(q)

        rdata = build_txt_rdata(["hello", "world"])
        ans = build_answer(qname, TYPE_TXT, rdata, ttl=0)
        resp = assemble_response(qid, questions, [ans], opt=build_opt())
        rheader, rquestions, ranswers, _, ropt = parse_message(resp)

        self.assertTrue(rheader["qr"])
        self.assertTrue(rheader["aa"])
        self.assertEqual(rheader["id"], qid)
        self.assertEqual(rquestions[0]["name"], qname)
        self.assertEqual(len(ranswers), 1)
        self.assertEqual(ranswers[0]["type"], TYPE_TXT)
        # RDATA is two length-prefixed strings.
        self.assertEqual(ranswers[0]["rdata"], rdata)
        self.assertIsNotNone(ropt)  # OPT record echoed

    def test_rcode_refused(self):
        resp = assemble_response(1, [], [], rcode=RCODE_REFUSED)
        header, _, _, _, _ = parse_message(resp)
        self.assertEqual(header["rcode"], RCODE_REFUSED)

    def test_opt_buffer_size_echoed(self):
        opt = build_opt(buffer_size=4096)
        # parse a hand-assembled message carrying the OPT record
        header = struct.pack(">HHHHHH", 1, FLAG_QR, 0, 0, 0, 1)
        resp = header + opt
        _, _, _, _, ropt = parse_message(resp)
        self.assertEqual(ropt["class"], 4096)  # OPT class carries buffer size


# --- Text slicing helpers (Phase 3) ------------------------------------------
class SliceTests(unittest.TestCase):
    def test_split_txt_char_strings(self):
        chunks = split_txt("ab" * 200)  # 400 chars -> 255 + 145
        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks[0]), 255)
        self.assertLessEqual(len(chunks[1]), 255)
        self.assertEqual(b"".join(chunks).decode("utf-8"), "ab" * 200)

    def test_multibyte_not_split(self):
        text = "a" * 8 + "\u20ac" + "c" * 20  # 8 + 3 + 20 = 31 bytes
        pages = _paginate(text, 10)
        for p in pages:
            p.decode("utf-8")  # each page must be valid UTF-8
        # the euro sign stays whole, on page 1
        self.assertEqual(pages[1].decode("utf-8"), "\u20acccccccc")
        self.assertEqual(b"".join(pages), text.encode("utf-8"))

    def test_strip_thinking_closed(self):
        self.assertEqual(
            strip_thinking("before<think>think</think>after"),
            "beforeafter",
        )

    def test_strip_thinking_open(self):
        self.assertEqual(strip_thinking("hi <think>still thinking"), "hi ")


# --- Server + generation (Phase 3) -------------------------------------------
class FakeBackend:
    """Canned backend: yields preset chunks per prompt, counting stream calls."""

    def __init__(self, chunks_by_prompt=None, chunk_delay=0.0):
        self.chunks = chunks_by_prompt or {}
        self.chunk_delay = chunk_delay
        self.stream_calls = 0

    def stream(self, prompt):
        self.stream_calls += 1
        for chunk in self.chunks.get(prompt, []):
            if self.chunk_delay:
                time.sleep(self.chunk_delay)
            yield chunk


class SlowBackend:
    """Yields one small chunk, then sleeps (simulates a thinking model)."""

    def __init__(self, delay):
        self.delay = delay
        self.stream_calls = 0

    def stream(self, prompt):
        self.stream_calls += 1
        yield "partial"
        time.sleep(self.delay)


def _make_question(name, qtype=TYPE_TXT):
    msg = build_query(name, qtype=qtype)
    header, questions, _, _, _ = parse_message(msg)
    return header, questions[0]


def _extract_txt(response):
    _, _, answers, _, _ = parse_message(response)
    rdata = answers[0]["rdata"]
    out = []
    i = 0
    while i < len(rdata):
        n = rdata[i]
        i += 1
        out.append(rdata[i:i + n].decode("utf-8"))
        i += n
    return "".join(out)


class ServerTests(unittest.TestCase):
    def test_zone_root_returns_usage(self):
        server = Server(FakeBackend())
        header, q = _make_question("e.com")
        resp = server.handle(header, [q], None)
        text = _extract_txt(resp)
        self.assertIn("ollama-over-dns", text)

    def test_not_under_zone_refused(self):
        server = Server(FakeBackend())
        header, q = _make_question("notmine.org")
        resp = server.handle(header, [q], None)
        rhdr, _, answers, _, _ = parse_message(resp)
        self.assertEqual(rhdr["rcode"], RCODE_REFUSED)
        self.assertEqual(answers, [])

    def test_non_txt_in_zone_empty_noerror(self):
        server = Server(FakeBackend())
        header, q = _make_question("foo.e.com", qtype=TYPE_A)
        resp = server.handle(header, [q], None)
        rhdr, _, answers, _, _ = parse_message(resp)
        self.assertEqual(rhdr["rcode"], RCODE_NOERROR)
        self.assertEqual(answers, [])

    def test_pagination_exact_multiples(self):
        backend = FakeBackend({"multi": ["a" * 600, "b" * 600]})
        server = Server(backend, page_bytes=600, wait=1.0)
        state, _ = server.manager.get_or_start("multi")
        r0 = server.manager.fetch_page(state, 0, 600, 1.0, "z.e.com")
        self.assertEqual(r0[:600], "a" * 600)  # page content...
        self.assertTrue(r0.endswith("[more: dig p1.z.e.com TXT]"))  # ...then status
        r1 = server.manager.fetch_page(state, 1, 600, 1.0, "z.e.com")
        self.assertEqual(r1[:600], "b" * 600)
        self.assertTrue(r1.endswith("[done]"))

    def test_page_forward_hint_uses_core_name(self):
        # Regression: paging must forward from the core name (page prefix
        # stripped), not the raw queried name, or it would start a new
        # generation. p1. -> p2.<core>.e.com, never p2.p1.<core>.e.com.
        backend = FakeBackend({"multi": ["a" * 600, "b" * 600, "c" * 600]})
        server = Server(backend, page_bytes=600, wait=1.0)
        header, q = _make_question("p1.multi.e.com")
        resp = server.handle(header, [q], None)
        text = _extract_txt(resp)
        self.assertIn("[more: dig p2.multi.e.com TXT]", text)
        self.assertNotIn("p1.p1", text)
        # page 0 of the bare prompt also forwards from the core name
        header0, q0 = _make_question("multi.e.com")
        resp0 = server.handle(header0, [q0], None)
        self.assertIn("[more: dig p1.multi.e.com TXT]", _extract_txt(resp0))

    def test_empty_output(self):
        backend = FakeBackend({"empty": []})
        server = Server(backend, page_bytes=600, wait=1.0)
        state, _ = server.manager.get_or_start("empty")
        reply = server.manager.fetch_page(state, 0, 600, 1.0, "z.e.com")
        self.assertEqual(reply, "[done]")

    def test_multibyte_page_edge_end_to_end(self):
        backend = FakeBackend({"u": ["a" * 8 + "\u20ac" + "c" * 20]})
        server = Server(backend, page_bytes=10, wait=1.0)
        state, _ = server.manager.get_or_start("u")
        server.manager.fetch_page(state, 0, 10, 1.0, "z.e.com")  # page 0
        pages = _paginate(state.text, 10)
        for p in pages:
            p.decode("utf-8")  # no page splits a character
        self.assertEqual(pages[1].decode("utf-8"), "\u20acccccccc")

    def test_wait_timeout_thinking(self):
        backend = SlowBackend(delay=0.5)
        server = Server(backend, page_bytes=600, wait=0.2)
        state, _ = server.manager.get_or_start("slow")
        reply = server.manager.fetch_page(state, 0, 600, 0.2, "z.e.com")
        self.assertTrue(reply.startswith("partial "))  # partial page content...
        self.assertTrue(reply.endswith("[thinking... repeat the same query]"))  # ...then status

    def test_repeat_query_does_not_respawn(self):
        backend = FakeBackend({"hello": ["world"]})
        server = Server(backend, wait=1.0)
        header, q = _make_question("hello.e.com")
        first = server.handle(header, [q], None)  # starts generation
        self.assertIn("[done]", _extract_txt(first))
        # second query for the same prompt reuses the cached generation
        _, q2 = _make_question("hello.e.com")
        second = server.handle(header, [q2], None)
        self.assertIn("[done]", _extract_txt(second))
        self.assertEqual(backend.stream_calls, 1)

    def test_busy_at_limit(self):
        class BusyBackend:
            def __init__(self):
                self.calls = 0
            def stream(self, prompt):
                self.calls += 1
                time.sleep(0.3)  # hold the slot
        occupy = BusyBackend()
        server = Server(occupy, wait=1.0, max_gen=1)
        state, err = server.manager.get_or_start("p")  # occupies the only slot
        self.assertIsNone(err)
        header, q = _make_question("other.e.com")
        resp = server.handle(header, [q], None)
        self.assertIn("[error: busy, try again]", _extract_txt(resp))
        # clean up the still-running generation
        server.manager.fetch_page(state, 0, 600, 0.2, "z.e.com")


# --- Real backends against stub HTTP servers (Phase 4) -----------------------
class _StubServer:
    """Minimal HTTP server that records the POST body and returns canned bytes."""

    def __init__(self, response_bytes):
        self.response_bytes = response_bytes
        self.received = None
        self.payload = None
        self.httpd = None
        self.thread = None

    def start(self):
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                server.received = self.rfile.read(length)
                server.payload = json.loads(server.received)
                self.send_response(200)
                self.send_header("Content-Length", str(len(server.response_bytes)))
                self.end_headers()
                self.wfile.write(server.response_bytes)

            def log_message(self, *args):
                pass

        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.httpd.shutdown()


class BackendTests(unittest.TestCase):
    def test_ollama_backend_ndjson(self):
        stub = _StubServer(
            b'{"message":{"role":"assistant","content":"hello "}}\n'
            b'{"message":{"role":"assistant","content":"world"}}\n'
        )
        url = stub.start()
        try:
            backend = OllamaBackend(url, "llama3", max_tokens=42)
            chunks = list(backend.stream("what is it"))
            self.assertEqual("".join(chunks), "hello world")
            self.assertEqual(stub.payload["model"], "llama3")
            self.assertIn("plain text", stub.payload["system"])
            self.assertTrue(stub.payload["stream"])
            self.assertEqual(stub.payload["options"]["max_tokens"], 42)
        finally:
            stub.stop()

    def test_llamacpp_backend_sse(self):
        stub = _StubServer(
            b'data: {"choices":[{"delta":{"content":"foo"}}]}\n'
            b'data: {"choices":[{"delta":{"content":"bar"}}]}\n'
            b'data: [DONE]\n'
        )
        url = stub.start()
        try:
            backend = LlamaCppBackend(url, model="qwen", max_tokens=7)
            chunks = list(backend.stream("do it"))
            self.assertEqual("".join(chunks), "foobar")
            self.assertEqual(stub.payload["max_tokens"], 7)
            roles = [m["role"] for m in stub.payload["messages"]]
            self.assertEqual(roles, ["system", "user"])
        finally:
            stub.stop()

    def test_backend_error_is_raised(self):
        # A non-2xx / connection failure should surface as an exception so the
        # manager records it as [error: ...] rather than crashing.
        backend = OllamaBackend("http://127.0.0.1:1", "x", timeout=1)
        with self.assertRaises(Exception):
            list(backend.stream("anything"))


# --- client.py pure logic (Phase 5) ------------------------------------------
class ClientParsing(unittest.TestCase):
    def test_split_status_with_more(self):
        content, status = client.split_status("hello world [more: dig p1.foo.e.com TXT]")
        self.assertEqual(content, "hello world")
        self.assertEqual(status, "more: dig p1.foo.e.com TXT")

    def test_split_status_done(self):
        content, status = client.split_status("[done]")
        self.assertEqual(content, "")
        self.assertEqual(status, "done")

    def test_split_status_no_status(self):
        content, status = client.split_status("just a usage string")
        self.assertEqual(content, "just a usage string")
        self.assertIsNone(status)

    def test_more_re_extracts_next_page(self):
        m = client._MORE_RE.search("more: dig p2.what.does.bravurva.mean.e.com TXT")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "p2.what.does.bravurva.mean.e.com")

    def test_empty_prompt_rejected(self):
        # An empty prompt would build a malformed name (empty label); the
        # client should refuse it cleanly rather than crashing.
        self.assertEqual(client.main(["--prompt", ""]), 2)

    def test_prompt_becomes_labels(self):
        # spaces -> dots; hyphens preserved (server turns them into spaces)
        qname = "what does bravurva mean".replace(" ", ".").strip(".") + ".e.com"
        self.assertEqual(qname, "what.does.bravurva.mean.e.com")
        qname = "why-is-the-sky-blue".replace(" ", ".").strip(".") + ".e.com"
        self.assertEqual(qname, "why-is-the-sky-blue.e.com")


if __name__ == "__main__":
    unittest.main()
