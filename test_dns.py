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
    TYPE_OPT, TYPE_TXT, CLASS_IN, RCODE_NOERROR, RCODE_REFUSED, RCODE_NOTIMP,
    FLAG_QR, FLAG_AA, FLAG_RD,
    decode_name, encode_name, parse_header, parse_message,
    build_query, build_txt_rdata, build_answer, build_opt, assemble_response,
    query_to_prompt, opt_to_bytes, split_txt, strip_thinking, _paginate,
    Server, GenerationManager, OllamaBackend, LlamaCppBackend,
    SYSTEM_PROMPT, normalize_url, make_backend,
    TYPE_A, TYPE_NS, TYPE_SOA, TYPE_AAAA, TYPE_MX, TYPE_CNAME, DNSError,
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

    def test_trailing_dot_encodes_like_bare_name(self):
        for name in ("e.com", "a.b.c", "x"):
            with self.subTest(name=name):
                self.assertEqual(encode_name(name + "."), encode_name(name))
        self.assertEqual(encode_name("."), b"\x00")

    def test_empty_inner_label_rejected(self):
        with self.assertRaises(DNSError):
            encode_name("a..b")

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


def _header(qdcount=1, flags=0, qid=1):
    return struct.pack(">HHHHHH", qid, flags, qdcount, 0, 0, 0)


class ParserHardening(unittest.TestCase):
    """Malformed names must raise DNSError quickly, never hang or crash."""

    def assertRejectedFast(self, msg):
        start = time.monotonic()
        with self.assertRaises(DNSError):
            parse_message(msg)
        self.assertLess(time.monotonic() - start, 0.1)

    def test_self_pointer_loop_rejected(self):
        # The name at offset 12 is a pointer to offset 12.
        self.assertRejectedFast(_header() + b"\xc0\x0c" + b"\x00\x10\x00\x01")

    def test_two_hop_pointer_loop_rejected(self):
        # offset 12 -> pointer to 14, offset 14 -> pointer to 12.
        self.assertRejectedFast(_header() + b"\xc0\x0e\xc0\x0c" + b"\x00\x10\x00\x01")

    def test_label_then_pointer_back_to_itself_rejected(self):
        # "a" then a pointer back to the start of "a": an infinite "a.a.a...".
        self.assertRejectedFast(_header() + b"\x01a\xc0\x0c" + b"\x00\x10\x00\x01")

    def test_forward_pointer_chain_still_decodes(self):
        # Legitimate (if unusual) chain of pointers ending in a real name.
        body = _header() + b"\xc0\x0e" + b"\xc0\x10" + b"\x01x\x00" + b"\x00\x10\x00\x01"
        _, questions, _, _, _ = parse_message(body)
        self.assertEqual(questions[0]["name"], "x")

    def test_name_over_255_bytes_rejected(self):
        name = b"".join(b"\x3f" + b"a" * 63 for _ in range(4)) + b"\x00"  # 257 bytes
        self.assertRejectedFast(_header() + name + b"\x00\x10\x00\x01")

    def test_name_of_exactly_255_bytes_accepted(self):
        # 3 x (1+63) + (1+61) + root = 255
        name = b"".join(b"\x3f" + b"a" * 63 for _ in range(3)) + b"\x3d" + b"b" * 61 + b"\x00"
        self.assertEqual(len(name), 255)
        _, questions, _, _, _ = parse_message(_header() + name + b"\x00\x10\x00\x01")
        self.assertEqual(len(encode_name(questions[0]["name"])), 255)

    def test_reserved_label_types_rejected(self):
        for prefix in (0x40, 0x80):
            with self.subTest(prefix=hex(prefix)):
                # Enough bytes follow that, read as a plain length (65 or
                # 129), the label would fit, so only the type check rejects it.
                length = prefix | 1
                self.assertRejectedFast(_header() + bytes([length]) + b"a" * length
                                        + b"\x00\x00\x10\x00\x01")

    def test_label_past_end_rejected(self):
        self.assertRejectedFast(_header() + b"\x05ab")

    def test_unrepresentable_label_bytes_rejected(self):
        # A '.' inside a label, a control byte, and a non-ASCII byte would all
        # change the name when it is echoed back, so they are refused.
        for bad in (b"a.b", b"a\x00b", b"a\xffb"):
            with self.subTest(label=bad):
                self.assertRejectedFast(_header() + bytes([len(bad)]) + bad + b"\x00\x00\x10\x00\x01")

    def test_space_in_label_accepted(self):
        # dig 'what\032is.e.com' sends a space inside one label.
        _, questions, _, _, _ = parse_message(_header() + b"\x07what is\x01e\x03com\x00\x00\x10\x00\x01")
        self.assertEqual(query_to_prompt(questions[0]["name"], "e.com"), ("what is", 0))

    def test_fuzz_never_hangs_or_raises_unexpected(self):
        # Seeded so a failure is reproducible. Mutates real queries (bit flips,
        # truncation, splices) and mixes in pure noise; every packet must either
        # parse and be handled, or raise DNSError, within a few milliseconds.
        import random
        rng = random.Random(1234)
        server = Server(FakeBackend(), wait=0, max_gen=10_000)
        seeds = [
            build_query("what.does.bravurva.mean.e.com"),
            build_query("p3.foo.e.com"),
            build_query("e.com", qtype=TYPE_A),
            build_query("notmine.org"),
        ]
        handled = rejected = 0
        for i in range(5000):
            kind = i % 4
            if kind == 0:
                pkt = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 80)))
            else:
                pkt = bytearray(rng.choice(seeds))
                for _ in range(rng.randrange(1, 6)):
                    pkt[rng.randrange(len(pkt))] = rng.getrandbits(8)
                if kind == 2:
                    pkt = pkt[:rng.randrange(len(pkt) + 1)]
                if kind == 3:
                    pkt += bytes(rng.getrandbits(8) for _ in range(rng.randrange(1, 20)))
                pkt = bytes(pkt)
            start = time.monotonic()
            try:
                header, questions, _, _, opt = parse_message(pkt)
            except DNSError:
                rejected += 1
            else:
                resp = server.handle(header, questions, opt)
                if resp is not None:
                    parse_message(resp)  # whatever we send back must itself parse
                handled += 1
            self.assertLess(time.monotonic() - start, 0.05, f"packet {i}: {pkt.hex()}")
        # Both paths were actually exercised.
        self.assertGreater(handled, 100)
        self.assertGreater(rejected, 100)


class QueryKinds(unittest.TestCase):
    """Only standard queries are answered."""

    def test_response_packets_dropped(self):
        msg = bytearray(build_query("hi.e.com"))
        msg[2] |= 0x80  # QR=1
        header, questions, _, _, opt = parse_message(bytes(msg))
        self.assertIsNone(Server(FakeBackend()).handle(header, questions, opt))

    def test_non_query_opcode_notimp(self):
        for opcode in (1, 2, 4, 5):  # IQUERY, STATUS, NOTIFY, UPDATE
            with self.subTest(opcode=opcode):
                msg = bytearray(build_query("hi.e.com"))
                msg[2] = (msg[2] & 0x87) | (opcode << 3)
                header, questions, _, _, opt = parse_message(bytes(msg))
                backend = FakeBackend()
                resp = Server(backend).handle(header, questions, opt)
                rhdr, _, answers, _, _ = parse_message(resp)
                self.assertEqual(rhdr["rcode"], RCODE_NOTIMP)
                self.assertEqual(rhdr["opcode"], opcode)
                self.assertTrue(rhdr["qr"])
                self.assertEqual(answers, [])
                self.assertEqual(backend.stream_calls, 0)


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


# --- Streamed thinking output -------------------------------------------------
class _SnapshotBackend:
    """Yields ``chunks`` and records the manager's ``state.text`` after each one.

    The generator only resumes after ``_run`` has stored the previous chunk, so
    each snapshot is exactly what a DNS query would have been served then.
    """

    def __init__(self, chunks, fail_after=False):
        self.chunks = chunks
        self.fail_after = fail_after
        self.snapshots = []
        self.manager = None

    def stream(self, prompt):
        state = self.manager._states[self.manager._key(prompt)]
        for chunk in self.chunks:
            yield chunk
            with state.lock:
                self.snapshots.append(state.text)
        if self.fail_after:
            raise RuntimeError("backend died")


def _stream_through_manager(chunks, fail_after=False):
    backend = _SnapshotBackend(chunks, fail_after)
    manager = GenerationManager(backend)
    backend.manager = manager
    state, _ = manager.get_or_start("p")
    deadline = time.monotonic() + 5
    while not state.done and time.monotonic() < deadline:
        time.sleep(0.005)
    return backend.snapshots, state


class StreamedThinking(unittest.TestCase):
    RAW = "<think>\nSECRET plan: check that x < y\n</think>\n\nThe answer is 42 <3"
    ANSWER = "The answer is 42 <3"

    def assertNeverLeaks(self, snapshots, final):
        for i, snap in enumerate(snapshots):
            self.assertNotIn("SECRET", snap, f"snapshot {i}")
            self.assertNotIn("think", snap, f"snapshot {i}")
            # Served text only ever grows: a page sent earlier stays valid.
            self.assertTrue(final.startswith(snap), f"snapshot {i} {snap!r} not a prefix of {final!r}")

    def test_one_character_per_chunk(self):
        snapshots, state = _stream_through_manager(list(self.RAW))
        self.assertEqual(len(snapshots), len(self.RAW))
        self.assertEqual(state.text, self.ANSWER)
        self.assertNeverLeaks(snapshots, state.text)

    def test_tokenized_like_a_real_model(self):
        # Roughly how a tokenizer splits it: the tags arrive as their own tokens.
        chunks = ["<think>", "\n", "SECRET", " plan", ":", " check", " that", " x", " <", " y",
                  "\n", "</think>", "\n\n", "The", " answer", " is", " 42", " <", "3"]
        self.assertEqual("".join(chunks), self.RAW)
        snapshots, state = _stream_through_manager(chunks)
        self.assertEqual(state.text, self.ANSWER)
        self.assertNeverLeaks(snapshots, state.text)

    def test_every_two_and_three_way_split(self):
        raw = "<think>SECRET</think>ok <b"
        for i in range(1, len(raw)):
            for j in range(i + 1, len(raw)):
                chunks = [raw[:i], raw[i:j], raw[j:]]
                with self.subTest(chunks=chunks):
                    snapshots, state = _stream_through_manager(chunks)
                    self.assertEqual(state.text, "ok <b")
                    self.assertNeverLeaks(snapshots, state.text)

    def test_partial_tag_is_never_served(self):
        snapshots, state = _stream_through_manager(["Hello <th", "ink>SECRET</think> world"])
        self.assertEqual(snapshots[0], "Hello ")
        self.assertEqual(state.text, "Hello  world")

    def test_partial_tag_at_end_is_kept(self):
        # Text that only looks like the start of a tag must survive once
        # generation is over.
        for tail in ("<", "<t", "<thin", "<think"):
            with self.subTest(tail=tail):
                _, state = _stream_through_manager(["a ", tail])
                self.assertEqual(state.text, "a " + tail)

    def test_unclosed_block_at_end_is_dropped(self):
        snapshots, state = _stream_through_manager(["ok ", "<think>", "SECRET", " still going"])
        self.assertEqual(state.text, "ok ")
        self.assertNeverLeaks(snapshots, state.text)

    def test_several_blocks(self):
        chunks = list("<think>SECRET1</think>one <think>SECRET2</think>two")
        snapshots, state = _stream_through_manager(chunks)
        self.assertEqual(state.text, "one two")
        self.assertNeverLeaks(snapshots, state.text)

    def test_error_mid_block_keeps_only_clean_text(self):
        _, state = _stream_through_manager(["Hi ", "<think>", "SECRET"], fail_after=True)
        self.assertEqual(state.text, "Hi ")
        self.assertEqual(state.error, "backend died")

    def test_empty_chunks_ignored(self):
        _, state = _stream_through_manager(["", "a", "", "b", ""])
        self.assertEqual(state.text, "ab")

    def test_served_pages_never_contain_thinking(self):
        # Poll the real Server.handle while a slow, character-by-character
        # thinking stream is generating, as repeated dig queries would.
        raw = self.RAW

        class Trickle:
            def stream(self, prompt):
                for c in raw:
                    time.sleep(0.003)
                    yield c

        server = Server(Trickle(), wait=0)
        header, q = _make_question("leak.e.com")
        replies = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            reply = _extract_txt(server.handle(header, [q], None))
            replies.append(reply)
            if "[done]" in reply:
                break
            time.sleep(0.002)
        self.assertIn("[done]", replies[-1])
        self.assertGreater(len(replies), 10)  # really polled mid-stream
        for r in replies:
            self.assertNotIn("SECRET", r)
            self.assertNotIn("think>", r)
        self.assertTrue(replies[-1].startswith(self.ANSWER + " "))

    def test_strip_thinking_holdback_only_when_not_final(self):
        self.assertEqual(strip_thinking("x <thi", final=False), "x ")
        self.assertEqual(strip_thinking("x <thi", final=True), "x <thi")
        self.assertEqual(strip_thinking("x <thi"), "x <thi")  # final is the default

    def test_strip_thinking_drops_leading_whitespace(self):
        self.assertEqual(strip_thinking("<think>a</think>\n\n  Answer\n"), "Answer\n")


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

    def test_apex_non_txt_empty_noerror(self):
        # NS/SOA/etc. for the apex (or the apex with a page prefix) must never
        # get the TXT usage string back.
        for name in ("e.com", "E.COM.", "p0.e.com", "p7.e.com"):
            for qtype in (TYPE_NS, TYPE_SOA, TYPE_A, TYPE_AAAA, TYPE_MX, TYPE_CNAME, 255):
                with self.subTest(name=name, qtype=qtype):
                    backend = FakeBackend()
                    header, q = _make_question(name, qtype=qtype)
                    resp = Server(backend).handle(header, [q], None)
                    rhdr, rq, answers, _, _ = parse_message(resp)
                    self.assertEqual(rhdr["rcode"], RCODE_NOERROR)
                    self.assertTrue(rhdr["aa"])
                    self.assertEqual(answers, [])
                    self.assertEqual(rq[0]["qtype"], qtype)  # question echoed as asked
                    self.assertEqual(backend.stream_calls, 0)

    def test_apex_txt_still_returns_usage(self):
        for name in ("e.com", "p0.e.com"):
            with self.subTest(name=name):
                header, q = _make_question(name)
                _, _, answers, _, _ = parse_message(Server(FakeBackend()).handle(header, [q], None))
                self.assertEqual(answers[0]["type"], TYPE_TXT)
                self.assertIn("ollama-over-dns", _extract_txt(Server(FakeBackend()).handle(header, [q], None)))

    def test_non_txt_below_apex_never_starts_generation(self):
        backend = FakeBackend({"foo": ["bar"]})
        server = Server(backend)
        for qtype in (TYPE_NS, TYPE_SOA, TYPE_A, TYPE_AAAA):
            header, q = _make_question("foo.e.com", qtype=qtype)
            server.handle(header, [q], None)
        self.assertEqual(backend.stream_calls, 0)

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


# --- Generation cache and paging status -------------------------------------
class _Gate:
    """Backend that yields ``before``, blocks until released, then yields
    ``after`` (or raises). Lets a test hold a generation open exactly as long as
    it needs, with no sleeps that guess at timing."""

    def __init__(self, before, after=(), fail=None):
        self.before, self.after, self.fail = before, after, fail
        self.release = threading.Event()
        self.stream_calls = 0

    def stream(self, prompt):
        self.stream_calls += 1
        yield from self.before
        self.release.wait(10)
        yield from self.after
        if self.fail:
            raise RuntimeError(self.fail)


def _wait_until(pred, timeout=5):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


class CacheAndPaging(unittest.TestCase):
    def test_failed_generation_is_not_reused(self):
        class Flaky:
            calls = 0
            def stream(self, prompt):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise ConnectionRefusedError("backend down")
                yield "back up"
        server = Server(Flaky(), wait=1.0)
        header, q = _make_question("hi.e.com")
        self.assertEqual(_extract_txt(server.handle(header, [q], None)), "[error: backend down]")
        self.assertEqual(_extract_txt(server.handle(header, [q], None)), "back up [done]")
        self.assertEqual(Flaky.calls, 2)
        # ...and the good result is cached as usual.
        server.handle(header, [q], None)
        self.assertEqual(Flaky.calls, 2)

    def test_retry_after_error_is_still_bounded_by_max_gen(self):
        hold = _Gate(["x"])
        mgr = GenerationManager(hold, max_gen=1)
        failed = _PromptStateFailed("p")
        mgr._states[mgr._key("p")] = failed
        busy, _ = mgr.get_or_start("other")           # takes the only slot
        state, err = mgr.get_or_start("p")            # retry of the failed prompt
        self.assertIsNone(state)
        self.assertEqual(err, "busy, try again")
        hold.release.set()

    def test_multibyte_page_edge_reports_more_without_waiting(self):
        # Page 0 is "aaaaaaaaa" (9 bytes): the euro sign moved to page 1. The
        # generation is still running; page 0 must still be reported full.
        gate = _Gate(["a" * 9 + "\u20ac" + "b" * 10])
        mgr = GenerationManager(gate)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.text)
        start = time.monotonic()
        reply = mgr.fetch_page(state, 0, 10, 3.0, "p.e.com")
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(reply, "aaaaaaaaa [more: dig p1.p.e.com TXT]")
        gate.release.set()

    def test_exactly_full_page_reports_more_while_generating(self):
        gate = _Gate(["a" * 10])
        mgr = GenerationManager(gate)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.text)
        start = time.monotonic()
        self.assertEqual(mgr.fetch_page(state, 0, 10, 3.0, "p.e.com"),
                         "a" * 10 + " [more: dig p1.p.e.com TXT]")
        self.assertLess(time.monotonic() - start, 0.5)
        gate.release.set()

    def test_short_page_waits_then_reports_thinking(self):
        gate = _Gate(["abc"])
        mgr = GenerationManager(gate)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.text)
        start = time.monotonic()
        self.assertEqual(mgr.fetch_page(state, 0, 10, 0.3, "p.e.com"),
                         "abc [thinking... repeat the same query]")
        self.assertGreaterEqual(time.monotonic() - start, 0.3)
        gate.release.set()

    def test_page_fills_while_waiting(self):
        # The page is short when the query arrives and fills during the wait.
        gate = _Gate(["abc"], after=["defghijklmn"])
        mgr = GenerationManager(gate)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.text)
        threading.Timer(0.1, gate.release.set).start()
        self.assertEqual(mgr.fetch_page(state, 0, 10, 3.0, "p.e.com"),
                         "abcdefghij [more: dig p1.p.e.com TXT]")

    def test_partial_failure_shows_error_on_last_page(self):
        gate = _Gate(["a" * 15], fail="backend died")
        gate.release.set()
        mgr = GenerationManager(gate)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.done)
        self.assertEqual(mgr.fetch_page(state, 0, 10, 1, "p.e.com"),
                         "a" * 10 + " [more: dig p1.p.e.com TXT]")
        self.assertEqual(mgr.fetch_page(state, 1, 10, 1, "p.e.com"),
                         "aaaaa [error: backend died]")
        self.assertEqual(mgr.fetch_page(state, 2, 10, 1, "p.e.com"),
                         "[error: backend died]")

    def test_failure_before_any_text(self):
        gate = _Gate([], fail="model not found")
        gate.release.set()
        mgr = GenerationManager(gate)
        state, _ = mgr.get_or_start("p")
        self.assertEqual(mgr.fetch_page(state, 0, 10, 1, "p.e.com"), "[error: model not found]")

    def test_successful_last_page_and_past_the_end(self):
        backend = FakeBackend({"p": ["a" * 15]})
        mgr = GenerationManager(backend)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.done)
        self.assertEqual(mgr.fetch_page(state, 1, 10, 1, "p.e.com"), "aaaaa [done]")
        self.assertEqual(mgr.fetch_page(state, 5, 10, 1, "p.e.com"), "[done]")

    def test_ttl_expires_finished_entries(self):
        backend = FakeBackend({"p": ["x"]})
        mgr = GenerationManager(backend, cache_ttl=0.05)
        state, _ = mgr.get_or_start("p")
        _wait_until(lambda: state.done)
        self.assertIs(mgr.get_or_start("p")[0], state)  # still fresh
        time.sleep(0.1)
        fresh, _ = mgr.get_or_start("p")
        self.assertIsNot(fresh, state)
        _wait_until(lambda: fresh.done)
        self.assertEqual(backend.stream_calls, 2)

    def test_ttl_never_expires_in_flight_work(self):
        gate = _Gate(["x"])
        mgr = GenerationManager(gate, cache_ttl=0)
        state, _ = mgr.get_or_start("p")
        time.sleep(0.05)
        self.assertIs(mgr.get_or_start("p")[0], state)
        self.assertEqual(gate.stream_calls, 1)
        gate.release.set()

    def test_entry_cap_evicts_oldest_finished(self):
        backend = FakeBackend({k: ["x"] for k in "abc"})
        mgr = GenerationManager(backend, max_entries=2)
        for k in "abc":
            st, _ = mgr.get_or_start(k)
            _wait_until(lambda: st.done)
        mgr.get_or_start("c")  # triggers eviction
        self.assertEqual(len(mgr._states), 2)
        self.assertNotIn(mgr._key("a"), mgr._states)

    def test_entry_cap_never_evicts_in_flight(self):
        gate = _Gate(["x"])
        mgr = GenerationManager(gate, max_entries=1, max_gen=5)
        states = [mgr.get_or_start(k)[0] for k in "abc"]
        mgr.get_or_start("a")
        self.assertEqual(len(mgr._states), 3)  # over the cap, but all running
        self.assertEqual(gate.stream_calls, 3)
        gate.release.set()
        _wait_until(lambda: all(s.done for s in states))
        mgr.get_or_start("a")
        self.assertLessEqual(len(mgr._states), 1)

    def test_concurrent_requests_for_one_prompt_start_one_generation(self):
        gate = _Gate(["x"])
        mgr = GenerationManager(gate, max_gen=1)
        barrier = threading.Barrier(20)
        results = []

        def go():
            barrier.wait()
            results.append(mgr.get_or_start("same"))

        threads = [threading.Thread(target=go) for _ in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(gate.stream_calls, 1)
        self.assertEqual({id(st) for st, err in results}, {id(results[0][0])})
        self.assertTrue(all(err is None for _, err in results))
        gate.release.set()

    def test_slot_released_after_success_and_failure(self):
        ok = _Gate(["x"]); ok.release.set()
        bad = _Gate([], fail="boom"); bad.release.set()
        for backend in (ok, bad):
            with self.subTest(fail=backend.fail):
                mgr = GenerationManager(backend, max_gen=1)
                st, _ = mgr.get_or_start("p")
                _wait_until(lambda: st.done)
                _wait_until(lambda: mgr._active == 0)
                self.assertIsNone(mgr.get_or_start("q")[1])


class _PromptStateFailed:
    """A finished, failed cache entry (as _run leaves one)."""

    def __init__(self, prompt):
        self.prompt = prompt
        self.text = ""
        self.done = True
        self.error = "backend down"
        self.finished_time = time.monotonic()
        self.lock = threading.Lock()


# --- Real backends against stub HTTP servers (Phase 4) -----------------------
class _StubServer:
    """Minimal HTTP server that records the POST body and returns canned bytes."""

    def __init__(self, response_bytes, status=200):
        self.response_bytes = response_bytes
        self.status = status
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
                self.send_response(server.status)
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


def _ndjson(*objs):
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def _sse(*objs, done=True):
    out = b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs)
    return out + (b"data: [DONE]\n\n" if done else b"")


def _stream_from_stub(make_backend_for, body, status=200):
    """Run a backend against a stub returning ``body``; return (chunks|exc, payload)."""
    stub = _StubServer(body, status=status)
    url = stub.start()
    try:
        try:
            result = list(make_backend_for(url).stream("what is it"))
        except Exception as e:  # the test inspects it
            result = e
        return result, stub.payload
    finally:
        stub.stop()


class BackendTests(unittest.TestCase):
    # --- ollama -------------------------------------------------------------
    def test_ollama_request_payload(self):
        chunks, payload = _stream_from_stub(
            lambda u: OllamaBackend(u, "llama3", max_tokens=42),
            _ndjson({"message": {"role": "assistant", "content": "hi"}, "done": True}))
        self.assertEqual(chunks, ["hi"])
        self.assertEqual(payload, {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "what is it"},
            ],
            "stream": True,
            "think": False,
            "options": {"num_predict": 42},
        })

    def test_ollama_stream_like_a_real_server(self):
        # Shape of a real /api/chat stream: empty contents, a thinking field
        # (ignored), and a final done object with stats.
        body = _ndjson(
            {"model": "m", "message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
            {"model": "m", "message": {"role": "assistant", "content": "hello "}, "done": False},
            {"model": "m", "message": {"role": "assistant", "content": "world"}, "done": False},
            {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True,
             "done_reason": "stop", "eval_count": 3},
        ) + b"\n"
        chunks, _ = _stream_from_stub(lambda u: OllamaBackend(u, "m"), body)
        self.assertEqual(chunks, ["hello ", "world"])

    def test_ollama_error_line_mid_stream_raises(self):
        body = _ndjson({"message": {"content": "partial "}},
                       {"error": "an error was encountered while running the model"})
        result, _ = _stream_from_stub(lambda u: OllamaBackend(u, "m"), body)
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(str(result), "an error was encountered while running the model")

    def test_ollama_http_error_body_is_surfaced(self):
        result, _ = _stream_from_stub(lambda u: OllamaBackend(u, "nope"),
                                      b'{"error":"model \'nope\' not found"}', status=404)
        self.assertEqual(str(result), "HTTP 404: model 'nope' not found")

    def test_ollama_mid_stream_error_reaches_dns_reply(self):
        stub = _StubServer(_ndjson({"message": {"content": "partial answer"}},
                                   {"error": "out of memory"}))
        url = stub.start()
        try:
            server = Server(OllamaBackend(url, "m"), wait=2.0)
            header, q = _make_question("oom.e.com")
            self.assertEqual(_extract_txt(server.handle(header, [q], None)),
                             "partial answer [error: out of memory]")
        finally:
            stub.stop()

    # --- llama.cpp ----------------------------------------------------------
    def test_llamacpp_request_payload(self):
        body = _sse({"choices": [{"delta": {"content": "x"}}]})
        _, payload = _stream_from_stub(lambda u: LlamaCppBackend(u, model="qwen", max_tokens=7), body)
        self.assertEqual(payload, {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "what is it"},
            ],
            "stream": True,
            "max_tokens": 7,
            "chat_template_kwargs": {"enable_thinking": False},
        })

    def test_llamacpp_model_omitted_when_not_given(self):
        body = _sse({"choices": [{"delta": {"content": "x"}}]})
        _, payload = _stream_from_stub(lambda u: LlamaCppBackend(u), body)
        self.assertNotIn("model", payload)

    def test_llamacpp_stream_like_a_real_server(self):
        # Copied from a real llama-server stream: a role-only first chunk with
        # "content": null, reasoning_content, a finish chunk with an empty
        # delta, and a usage chunk with no choices.
        body = _sse(
            {"choices": [{"finish_reason": None, "index": 0,
                          "delta": {"role": "assistant", "content": None}}],
             "object": "chat.completion.chunk"},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"index": 0, "delta": {"content": "foo"}}]},
            {"choices": [{"index": 0, "delta": {"content": "bar"}}]},
            {"choices": [{"finish_reason": "stop", "index": 0, "delta": {}}]},
            {"choices": [], "usage": {"completion_tokens": 2}},
        )
        chunks, _ = _stream_from_stub(lambda u: LlamaCppBackend(u), body)
        self.assertEqual(chunks, ["foo", "bar"])

    def test_llamacpp_stream_without_done_marker(self):
        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, done=False)
        chunks, _ = _stream_from_stub(lambda u: LlamaCppBackend(u), body)
        self.assertEqual(chunks, ["ok"])

    def test_llamacpp_error_event_raises(self):
        body = _sse({"choices": [{"delta": {"content": "a"}}]},
                    {"error": {"code": 500, "message": "context full", "type": "server_error"}})
        result, _ = _stream_from_stub(lambda u: LlamaCppBackend(u), body)
        self.assertEqual(str(result), "context full")

    def test_llamacpp_http_error_body_is_surfaced(self):
        body = b'{"error":{"code":400,"message":"bad request: messages empty","type":"invalid_request_error"}}'
        result, _ = _stream_from_stub(lambda u: LlamaCppBackend(u), body, status=400)
        self.assertEqual(str(result), "HTTP 400: bad request: messages empty")

    # --- both ---------------------------------------------------------------
    def test_http_error_without_json_uses_reason(self):
        result, _ = _stream_from_stub(lambda u: OllamaBackend(u, "m"), b"<html>oops</html>", status=502)
        self.assertEqual(str(result), "HTTP 502: Bad Gateway")

    def test_unreachable_backend_message(self):
        backend = OllamaBackend("http://127.0.0.1:1", "x", timeout=1)
        with self.assertRaises(RuntimeError) as cm:
            list(backend.stream("anything"))
        self.assertIn("cannot reach http://127.0.0.1:1/api/chat", str(cm.exception))


class BackendConfig(unittest.TestCase):
    URLS = [
        # (given, default port, expected)
        ("localhost", 11434, "http://localhost:11434"),
        ("0.0.0.0:11434", 11434, "http://0.0.0.0:11434"),
        ("192.168.1.215", 11434, "http://192.168.1.215:11434"),
        ("192.168.1.215:9999", 11434, "http://192.168.1.215:9999"),
        ("[::1]", 11434, "http://[::1]:11434"),
        ("[::1]:8080", 11434, "http://[::1]:8080"),
        ("http://localhost:11434", 11434, "http://localhost:11434"),
        ("http://localhost:11434/", 11434, "http://localhost:11434"),
        ("https://ollama.example.com", 11434, "https://ollama.example.com"),
        ("  localhost:8080 ", 8080, "http://localhost:8080"),
    ]

    def test_normalize_url(self):
        for given, port, expected in self.URLS:
            with self.subTest(given=given):
                self.assertEqual(normalize_url(given, port), expected)

    def test_normalize_url_rejects_bad_port(self):
        with self.assertRaises(SystemExit):
            normalize_url("localhost:notaport", 11434)

    def _make(self, argv, env=None):
        import argparse
        from unittest import mock
        p = argparse.ArgumentParser()
        p.add_argument("--backend", default="ollama")
        p.add_argument("--url")
        p.add_argument("--model")
        p.add_argument("--max-tokens", type=int, default=256)
        with mock.patch.dict("os.environ", env or {}, clear=True):
            return make_backend(p.parse_args(argv))

    def test_make_backend_ollama_host_forms(self):
        cases = [
            ({}, "http://localhost:11434"),
            ({"OLLAMA_HOST": "0.0.0.0"}, "http://0.0.0.0:11434"),
            ({"OLLAMA_HOST": "brain:11434"}, "http://brain:11434"),
            ({"OLLAMA_HOST": "http://brain:11434"}, "http://brain:11434"),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                b = self._make(["--model", "m"], env)
                self.assertIsInstance(b, OllamaBackend)
                self.assertEqual(b.url, expected)

    def test_make_backend_url_overrides_env(self):
        b = self._make(["--model", "m", "--url", "otherhost"], {"OLLAMA_HOST": "brain"})
        self.assertEqual(b.url, "http://otherhost:11434")

    def test_make_backend_ollama_requires_model(self):
        with self.assertRaises(SystemExit):
            self._make([])

    def test_make_backend_llamacpp(self):
        b = self._make(["--backend", "llamacpp", "--max-tokens", "9"])
        self.assertIsInstance(b, LlamaCppBackend)
        self.assertEqual(b.url, "http://localhost:8080")
        self.assertIsNone(b.model)
        self.assertEqual(b.max_tokens, 9)
        self.assertEqual(self._make(["--backend", "llamacpp", "--url", "box:9000"]).url,
                         "http://box:9000")


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
