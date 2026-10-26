"""ollama-over-dns: answer TXT queries with LLM output.

This file is organized in layers, added one build phase at a time:
  1. DNS wire-format parse/build (this module).
  2. query-name -> prompt/page parsing.
  3. Server loop + per-prompt generation/caching.
  4. Real backends (ollama / llamacpp).

Only the Python standard library is used.
"""

import struct

# --- Resource record types (IANA) -------------------------------------------
TYPE_A = 1
TYPE_NS = 2
TYPE_CNAME = 5
TYPE_SOA = 6
TYPE_PTR = 12
TYPE_MX = 15
TYPE_TXT = 16
TYPE_AAAA = 28
TYPE_OPT = 41  # EDNS0 pseudo-record

# --- Response codes (RCODE) -------------------------------------------------
RCODE_NOERROR = 0
RCODE_FORMERR = 1
RCODE_SERVFAIL = 2
RCODE_NXDOMAIN = 3
RCODE_NOTIMP = 4
RCODE_REFUSED = 5

# --- Flag bit masks within the 16-bit flags field ----------------------------
FLAG_QR = 1 << 15  # 1 = response
FLAG_AA = 1 << 10  # authoritative answer
FLAG_TC = 1 << 9  # truncated
FLAG_RD = 1 << 8  # recursion desired (echo from query)
FLAG_RA = 1 << 7  # recursion available

# Class IN (internet)
CLASS_IN = 1


class DNSError(ValueError):
    """Raised for malformed or truncated DNS messages."""


# --- Name decoding -----------------------------------------------------------
def decode_name(data, offset):
    """Decode a DNS domain name at ``offset`` (with compression support).

    Returns ``(name, next_offset)`` where ``name`` is a dotted string and
    ``next_offset`` is the position just past the name in the original stream
    (i.e. after any compression pointer, not after the pointer's target).
    """
    labels = []
    pos = offset
    jumped = False
    end = None
    while True:
        if pos >= len(data):
            raise DNSError("name ran past end of message")
        length = data[pos]
        if length & 0xC0 == 0xC0:
            # Compression pointer: 2 bytes, low 14 bits are an offset.
            if pos + 1 >= len(data):
                raise DNSError("truncated compression pointer")
            pointer = ((length & 0x3F) << 8) | data[pos + 1]
            if not jumped:
                # First pointer seen: remember where to resume after following it.
                jumped = True
                end = pos + 2
                pos = pointer
            else:
                pos = pointer
            continue
        # Normal label. A zero-length label is the root and terminates the name.
        if length == 0:
            if not jumped:
                end = pos + 1
            break
        labels.append(data[pos + 1:pos + 1 + length])
        pos += length + 1
    # ``end`` is set on every normal exit above (root label, or the pointer we
    # jumped from); nothing to fix up here.
    name = ".".join(l.decode("ascii", "replace") for l in labels)
    return name, end


def encode_name(name):
    """Encode a dotted name into wire format (no compression)."""
    if isinstance(name, str):
        name = name.encode("ascii")
    if name == b"":
        return b"\x00"  # root name
    out = b""
    for label in name.split(b".") if isinstance(name, bytes) else name.split("."):
        label = label.encode("ascii") if isinstance(label, str) else label
        if len(label) > 63:
            raise DNSError("label too long (>63 bytes)")
        out += bytes([len(label)]) + label
    return out + b"\x00"


# --- Header & record parsing -------------------------------------------------
def parse_header(data):
    if len(data) < 12:
        raise DNSError("truncated header (<12 bytes)")
    qid, flags, qd, an, ns, ar = struct.unpack(">HHHHHH", data[:12])
    return {
        "id": qid,
        "flags": flags,
        "qr": bool(flags & FLAG_QR),
        "opcode": (flags >> 11) & 0xF,
        "aa": bool(flags & FLAG_AA),
        "tc": bool(flags & FLAG_TC),
        "rd": bool(flags & FLAG_RD),
        "rcode": flags & 0xF,
        "qdcount": qd,
        "ancount": an,
        "nscount": ns,
        "arcount": ar,
    }


def parse_question(data, offset):
    """Parse a question record (QNAME + QTYPE + QCLASS; no rdata)."""
    name, offset = decode_name(data, offset)
    if offset + 4 > len(data):
        raise DNSError("truncated question")
    qtype, qclass = struct.unpack(">HH", data[offset:offset + 4])
    offset += 4
    return {"name": name, "qtype": qtype, "qclass": qclass}, offset


def parse_record(data, offset):
    """Parse an answer/authority/additional record. Returns ``(record, next_offset)``."""
    name, offset = decode_name(data, offset)
    if offset + 10 > len(data):
        raise DNSError("truncated record header")
    rtype, rclass, ttl, rdlen = struct.unpack(">HHIH", data[offset:offset + 10])
    offset += 10
    if offset + rdlen > len(data):
        raise DNSError("truncated record rdata")
    rdata = data[offset:offset + rdlen]
    offset += rdlen
    return {"name": name, "type": rtype, "class": rclass, "ttl": ttl, "rdata": rdata}, offset


def parse_message(data):
    """Parse a full DNS message.

    Returns ``(header, questions, answers, additional, opt)``. ``opt`` is the
    EDNS0 OPT record (as a raw record dict) or ``None``.
    """
    header = parse_header(data)
    offset = 12
    questions = []
    for _ in range(header["qdcount"]):
        q, offset = parse_question(data, offset)
        questions.append(q)
    answers = []
    for _ in range(header["ancount"]):
        rec, offset = parse_record(data, offset)
        answers.append(rec)
    additional = []
    opt = None
    for _ in range(header["arcount"]):
        rec, offset = parse_record(data, offset)
        if rec["type"] == TYPE_OPT:
            opt = rec
        else:
            additional.append(rec)
    return header, questions, answers, additional, opt


# --- Query -> prompt / page (Phase 2) ---------------------------------------
import re

_PAGE_PREFIX = re.compile(r"p[0-9]+")


def _strip_zone(name, zone):
    """Return the labels below ``zone`` (lowercased), or ``None`` if not under it."""
    name = name.rstrip(".").lower()
    zone = zone.rstrip(".").lower()
    name_labels = name.split(".") if name else []
    zone_labels = zone.split(".") if zone else []
    if len(name_labels) < len(zone_labels):
        return None
    if name_labels[-len(zone_labels):] != zone_labels:
        return None
    return name_labels[: len(name_labels) - len(zone_labels)]


def query_to_prompt(name, zone):
    """Turn a queried name into ``(prompt, page)``.

    The zone suffix is stripped, the remaining labels are joined with spaces
    (a hyphen inside a label counts as a space), and the result is lowercased.
    A leading ``pN`` label selects page N (default 0). Names not under the zone
    return ``None`` (the server answers those with REFUSED).
    """
    remaining = _strip_zone(name, zone)
    if remaining is None:
        return None
    page = 0
    labels = list(remaining)
    if labels and _PAGE_PREFIX.fullmatch(labels[0]):
        page = int(labels[0][1:])
        labels = labels[1:]
    text = " ".join(label.replace("-", " ") for label in labels)
    prompt = " ".join(text.split()).lower()  # collapse any doubled spaces
    return prompt, page


def core_name(name, zone):
    """Dotted name below ``zone`` with any leading ``pN`` page prefix removed.

    Used to build a forward-paging command that resolves to the *same*
    generation. Without stripping the page prefix here, paging ``p1.`` -> ``p2.``
    would leave a stale ``p1`` label in the middle and start a new generation.
    """
    remaining = _strip_zone(name, zone)
    if remaining is None:
        return None
    if remaining and _PAGE_PREFIX.fullmatch(remaining[0]):
        remaining = remaining[1:]
    return ".".join(remaining)


# --- Building ----------------------------------------------------------------
def build_query(qname, qtype=TYPE_TXT, qclass=CLASS_IN, qid=0x1234, rd=True):
    """Build a minimal query message (used by tests)."""
    flags = 0  # opcode 0 (query)
    if rd:
        flags |= FLAG_RD
    header = struct.pack(">HHHHHH", qid, flags, 1, 0, 0, 0)
    body = encode_name(qname) + struct.pack(">HH", qtype, qclass)
    return header + body


def build_txt_rdata(chunks):
    """Encode TXT RDATA: one-or-more length-prefixed character-strings."""
    out = b""
    for chunk in chunks:
        b = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        if len(b) > 255:
            raise DNSError("TXT character-string >255 bytes")
        out += bytes([len(b)]) + b
    return out


def build_answer(name, rtype, rdata, ttl=0, qclass=CLASS_IN):
    """Build a raw answer-record byte string (name not compressed)."""
    return encode_name(name) + struct.pack(">HHIH", rtype, qclass, ttl, len(rdata)) + rdata


def build_opt(buffer_size=1232, do=True):
    """Build an EDNS0 OPT record byte string.

    The DO (DNSSEC OK) flag is set when ``do`` is true so resolvers know we
    accept large responses.
    """
    rdata = b"\x00\x00\x00\x00"  # option code 0, option length 0
    flags = 0x8000 if do else 0  # DO bit in the low 16 bits of the TTL field
    # name=root(0x00), type=41, class=buffer size, ttl=ext-rcode|ver|flags
    return b"\x00" + struct.pack(">HHIH", TYPE_OPT, buffer_size, flags, len(rdata)) + rdata


def encode_question(q):
    return encode_name(q["name"]) + struct.pack(">HH", q["qtype"], q["qclass"])


def assemble_response(qid, questions, answers, *, opt=None, rcode=RCODE_NOERROR,
                      aa=True, rd=True, opcode=0):
    """Assemble a full response message from its parts."""
    flags = (opcode << 11)
    flags |= FLAG_QR
    if aa:
        flags |= FLAG_AA
    if rd:
        flags |= FLAG_RD
    flags |= (rcode & 0xF)
    ar = 1 if opt is not None else 0
    header = struct.pack(">HHHHHH", qid, flags, len(questions), len(answers), 0, ar)
    body = b"".join(encode_question(q) for q in questions)
    body += b"".join(answers)
    if opt is not None:
        body += opt
    return header + body


# --- Text slicing (Phase 3) --------------------------------------------------
# DNS TXT character-strings and generation pages must never split a multi-byte
# UTF-8 character, so slicing decodes-triangles back to a char boundary.


def _paginate(text, page_bytes):
    """Split ``text`` into ``page_bytes``-sized UTF-8 pages at char boundaries.

    A multi-byte character straddling a page edge stays whole on the later
    page (its leading byte is pulled back into the preceding page rather than
    dropped).
    """
    encoded = text.encode("utf-8")
    pages = []
    i = 0
    n = len(encoded)
    while i < n:
        end = min(i + page_bytes, n)
        chunk = encoded[i:end]
        while chunk:
            try:
                chunk.decode("utf-8")
                break
            except UnicodeDecodeError:
                chunk = chunk[:-1]
        pages.append(chunk)
        i += len(chunk)  # advance past only what we emitted
    return pages


def split_txt(text, max_len=255):
    """Split ``text`` into <=255-byte UTF-8 character-strings for a TXT RR."""
    return _paginate(text, max_len)


# --- Thinking-block stripping (Phase 3) --------------------------------------
# Thinking models (e.g. Qwen3) emit <think>...</think> blocks. We never forward
# those to DNS, even when generation is cut off mid-block.

_THINK_START = "<think>"
_THINK_END = "</think>"


def strip_thinking(text):
    """Remove closed <think> blocks; truncate at an open one."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(_THINK_START, i):
            close = text.find(_THINK_END, i + len(_THINK_START))
            if close == -1:
                break  # open block: drop everything from here on
            i = close + len(_THINK_END)
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


# --- Generation manager (Phase 3) --------------------------------------------
import hashlib
import time
import threading
from collections import OrderedDict


class _PromptState:
    """Per-prompt generation state, shared across pages of that prompt."""

    def __init__(self, prompt):
        self.prompt = prompt
        self.text = ""          # cleaned (thinking-stripped) generated text
        self.done = False
        self.error = None
        self.finished_time = None
        self.lock = threading.Lock()


class GenerationManager:
    """Owns per-prompt generation, caching, and concurrency limits."""

    def __init__(self, backend, *, cache_ttl=600, max_entries=64, max_gen=2):
        self.backend = backend
        self.cache_ttl = cache_ttl
        self.max_entries = max_entries
        self.max_gen = max_gen
        self._states = OrderedDict()   # hash(prompt) -> _PromptState
        self._active = 0               # in-flight generations
        self._lock = threading.Lock()

    @staticmethod
    def _key(prompt):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get_or_start(self, prompt):
        """Return ``(state, None)`` or ``(None, error_msg)`` if at the limit."""
        with self._lock:
            self._evict_locked()
            key = self._key(prompt)
            state = self._states.get(key)
            if state is not None:
                self._states.move_to_end(key)
                return state, None
            if self._active >= self.max_gen:
                return None, "busy, try again"
            state = _PromptState(prompt)
            self._states[key] = state
            self._active += 1
        t = threading.Thread(target=self._run, args=(state,), daemon=True)
        t.start()
        return state, None

    def _run(self, state):
        try:
            for chunk in self.backend.stream(state.prompt):
                state.text = strip_thinking(state.text + chunk)
            with state.lock:
                state.done = True
                state.finished_time = time.monotonic()
        except Exception as e:  # never let a backend error take us down
            with state.lock:
                state.error = str(e) or type(e).__name__
                state.done = True
                state.finished_time = time.monotonic()
        finally:
            with self._lock:
                self._active -= 1

    def _evict_locked(self):
        now = time.monotonic()
        for key in list(self._states):
            st = self._states[key]
            if st.done and st.finished_time is not None and now - st.finished_time > self.cache_ttl:
                del self._states[key]
        while len(self._states) > self.max_entries:
            # evict oldest finished entry; never evict in-flight work
            for key in list(self._states):
                if self._states[key].done:
                    del self._states[key]
                    break
            else:
                break

    def fetch_page(self, state, page, page_bytes, wait, page_base):
        """Wait for ``page`` to fill or generation to finish, then return status.

        ``page_base`` is the dotted name below the zone with the page prefix
        already removed (e.g. ``what.does.bravurva.mean.e.com``). The ``[more:``
        hint pages forward from it (``p{page+1}.{page_base}``), so paging chains
        through the same generation instead of starting a fresh one.
        """
        deadline = time.monotonic() + wait
        while True:
            with state.lock:
                text = state.text
                done = state.done
                error = state.error
            pages = _paginate(text, page_bytes)
            if page < len(pages):
                page_text = pages[page]
                is_last = page == len(pages) - 1
            else:
                page_text = b""
                is_last = False

            if done or len(page_text) == page_bytes or time.monotonic() >= deadline:
                break
            time.sleep(0.02)

        # The reply is the requested page's text with a final status string
        # appended (the status tells the client whether to page forward, wait,
        # or that generation finished).
        page_text_str = page_text.decode("utf-8", "replace")
        if error and not page_text:
            status = f"[error: {error}]"
        elif done:
            if is_last or not page_text:
                status = "[done]"
            else:
                status = f"[more: dig p{page + 1}.{page_base} TXT]"
        elif len(page_text) == page_bytes:
            status = f"[more: dig p{page + 1}.{page_base} TXT]"
        else:
            status = "[thinking... repeat the same query]"
        if page_text_str:
            return page_text_str + " " + status
        return status


# --- Server (Phase 3) --------------------------------------------------------
import socket
import sys
import argparse

SYSTEM_PROMPT = "Answer in plain text, under 600 characters, no markdown, no preamble."

USAGE_STRING = (
    "ollama-over-dns: dig @127.0.0.1 -p 5353 +short <prompt>.e.com TXT; "
    "labels before .e.com are the prompt; prefix p<N>. (e.g. p1.) to page."
)


class Server:
    """Turn parsed DNS questions into responses via the backend."""

    def __init__(self, backend, *, zone="e.com", page_bytes=600, wait=3.5,
                 max_gen=2, cache_ttl=600, max_entries=64,
                 system_prompt=SYSTEM_PROMPT, usage_string=USAGE_STRING):
        self.backend = backend
        self.zone = zone
        self.page_bytes = page_bytes
        self.wait = wait
        self.manager = GenerationManager(backend, cache_ttl=cache_ttl,
                                         max_entries=max_entries, max_gen=max_gen)
        self.system_prompt = system_prompt
        self.usage_string = usage_string

    def _txt_response(self, question, qid, opt, text):
        chunks = split_txt(text)
        rdata = build_txt_rdata(chunks)
        ans = build_answer(question["name"], TYPE_TXT, rdata, ttl=0)
        return assemble_response(qid, [question], [ans], opt=opt, aa=True)

    def handle(self, header, questions, opt):
        """Handle one parsed message. Returns response bytes, or None to drop."""
        opt_bytes = opt_to_bytes(opt)
        if not questions:
            return None
        q = questions[0]
        result = query_to_prompt(q["name"], self.zone)
        if result is None:
            # Not under our zone -> REFUSED.
            return assemble_response(header["id"], [q], [], opt=opt_bytes, rcode=RCODE_REFUSED)
        prompt, page = result
        if prompt == "":
            # The zone itself -> one-line usage string.
            return self._txt_response(q, header["id"], opt_bytes, self.usage_string)
        if q["qtype"] != TYPE_TXT:
            # Some other record type in our zone -> empty NOERROR.
            return assemble_response(header["id"], [q], [], opt=opt_bytes, rcode=RCODE_NOERROR)
        state, err = self.manager.get_or_start(prompt)
        if err is not None:
            return self._txt_response(q, header["id"], opt_bytes, f"[error: {err}]")
        # Page forward from the core name (page prefix + zone already stripped) so
        # paging chains through the same cached generation.
        core = core_name(q["name"], self.zone)
        page_base = f"{core}.{self.zone}" if core else self.zone
        reply = self.manager.fetch_page(state, page, self.page_bytes, self.wait, page_base)
        return self._txt_response(q, header["id"], opt_bytes, reply)


def run_udp_server(server, host, port):
    """Serve DNS over UDP. Malformed packets are logged and dropped."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"ollama-over-dns listening on udp {host}:{port} (zone {server.zone})")
    while True:
        data, addr = sock.recvfrom(2048)
        try:
            header, questions, _answers, _additional, opt = parse_message(data)
        except DNSError as e:
            print(f"dropping malformed packet from {addr}: {e}", file=sys.stderr)
            continue
        try:
            response = server.handle(header, questions, opt)
        except Exception as e:
            print(f"error handling {addr}: {e!r}", file=sys.stderr)
            continue
        if response is None:
            continue
        try:
            sock.sendto(response, addr)
        except Exception as e:
            print(f"error replying to {addr}: {e!r}", file=sys.stderr)


def opt_to_bytes(opt):
    """Rebuild the raw bytes of an EDNS0 OPT record for echoing back."""
    if opt is None:
        return None
    return b"\x00" + struct.pack(">HHIH", opt["type"], opt["class"],
                                 opt["ttl"], len(opt["rdata"])) + opt["rdata"]


# --- Backends (Phase 4) ------------------------------------------------------
# Both implement the one interface: stream(prompt) -> iterator of text chunks.
# The DNS/caching code never needs to know which one is in use.
import json
import urllib.request


class OllamaBackend:
    """ollama: POST /api/chat with stream:true, NDJSON output."""

    def __init__(self, url, model, max_tokens=256, timeout=30):
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def stream(self, prompt):
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"max_tokens": self.max_tokens},
        }
        req = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj.get("message", {}).get("content", "")


class LlamaCppBackend:
    """llama-server: POST /v1/chat/completions with stream:true, SSE output."""

    def __init__(self, url, model=None, max_tokens=256, timeout=30):
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def stream(self, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                yield obj["choices"][0].get("delta", {}).get("content", "")


def make_backend(args):
    """Construct the backend from parsed CLI args."""
    if args.backend == "ollama":
        if not args.model:
            raise SystemExit("ollama backend requires --model")
        import os
        url = args.url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        return OllamaBackend(url, args.model, max_tokens=args.max_tokens)
    url = args.url or "http://localhost:8080"
    return LlamaCppBackend(url, model=args.model, max_tokens=args.max_tokens)


def main(argv=None):
    p = argparse.ArgumentParser(description="Answer TXT DNS queries with LLM output.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5353)
    p.add_argument("--zone", default="e.com")
    p.add_argument("--backend", choices=["ollama", "llamacpp"], default="ollama")
    p.add_argument("--url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--wait", type=float, default=3.5)
    p.add_argument("--page-bytes", type=int, default=600)
    p.add_argument("--max-gen", type=int, default=2)
    args = p.parse_args(argv)

    backend = make_backend(args)
    server = Server(backend, zone=args.zone, page_bytes=args.page_bytes,
                    wait=args.wait, max_gen=args.max_gen)
    run_udp_server(server, args.host, args.port)


if __name__ == "__main__":
    main()
