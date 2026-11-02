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

# Largest UDP reply we ever send, and the buffer size we advertise: 1232 bytes
# avoids IP fragmentation on almost every path (DNS Flag Day 2020). Clients
# without EDNS get at most 512 (RFC 1035).
MAX_UDP = 1232
CLASSIC_UDP = 512

# Longest name allowed in presentation form (without the trailing dot).
MAX_NAME_TEXT = 253

# The widest UTF-8 character; no page or TXT string can be smaller than this.
MIN_PAGE_BYTES = 4

# Error reasons from backends can be long (HTTP bodies); keep replies small.
MAX_ERROR_CHARS = 200

# Opcode 0 is a standard query; everything else (IQUERY, STATUS, NOTIFY,
# UPDATE, ...) gets NOTIMP.
OPCODE_QUERY = 0

# A name is at most 255 bytes on the wire (RFC 1035 §2.3.4), so it has at most
# 127 labels. More pointer hops than that can only be a loop.
_MAX_NAME_WIRE = 255
_MAX_POINTER_HOPS = 127


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
    hops = 0
    wire_len = 1  # the terminating root byte
    while True:
        if pos >= len(data):
            raise DNSError("name ran past end of message")
        length = data[pos]
        if length & 0xC0 == 0xC0:
            # Compression pointer: 2 bytes, low 14 bits are an offset.
            if pos + 1 >= len(data):
                raise DNSError("truncated compression pointer")
            hops += 1
            if hops > _MAX_POINTER_HOPS:
                raise DNSError("compression pointer loop")
            pointer = ((length & 0x3F) << 8) | data[pos + 1]
            if not jumped:
                # First pointer seen: remember where to resume after following it.
                jumped = True
                end = pos + 2
            pos = pointer
            continue
        if length & 0xC0:
            raise DNSError("reserved label type")
        # Normal label. A zero-length label is the root and terminates the name.
        if length == 0:
            if not jumped:
                end = pos + 1
            break
        if pos + 1 + length > len(data):
            raise DNSError("label ran past end of message")
        label = data[pos + 1:pos + 1 + length]
        if any(b < 0x20 or b > 0x7E or b == 0x2E for b in label):
            raise DNSError("label has a non-printable, non-ASCII or '.' byte")
        wire_len += length + 1
        if wire_len > _MAX_NAME_WIRE:
            raise DNSError("name longer than 255 bytes")
        labels.append(label)
        pos += length + 1
    # ``end`` is set on every normal exit above (root label, or the pointer we
    # jumped from); nothing to fix up here.
    name = ".".join(l.decode("ascii") for l in labels)
    return name, end


def encode_name(name):
    """Encode a dotted name into wire format (no compression)."""
    if isinstance(name, str):
        name = name.encode("ascii")
    if name.endswith(b"."):
        name = name[:-1]  # a fully-qualified "e.com." is the same name as "e.com"
    if name == b"":
        return b"\x00"  # root name
    out = b""
    for label in name.split(b".") if isinstance(name, bytes) else name.split("."):
        label = label.encode("ascii") if isinstance(label, str) else label
        if not label:
            raise DNSError("empty label")  # a zero byte here would end the name early
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


def build_opt(buffer_size=MAX_UDP, do=False, ext_rcode=0):
    """Build an EDNS0 OPT record byte string with no options.

    ``do`` sets the DNSSEC OK flag. ``ext_rcode`` is the upper 8 bits of a
    12-bit RCODE (1 = BADVERS).
    """
    flags = 0x8000 if do else 0  # DO bit in the low 16 bits of the TTL field
    ttl = (ext_rcode << 24) | flags  # ext-rcode | version 0 | flags
    # name=root(0x00), type=41, class=buffer size, ttl, rdlength=0
    return b"\x00" + struct.pack(">HHIH", TYPE_OPT, buffer_size, ttl, 0)


def encode_question(q):
    return encode_name(q["name"]) + struct.pack(">HH", q["qtype"], q["qclass"])


def assemble_response(qid, questions, answers, *, opt=None, rcode=RCODE_NOERROR,
                      aa=True, rd=True, opcode=0, tc=False):
    """Assemble a full response message from its parts."""
    flags = (opcode << 11)
    flags |= FLAG_QR
    if aa:
        flags |= FLAG_AA
    if tc:
        flags |= FLAG_TC
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
    if page_bytes < MIN_PAGE_BYTES:
        # A page must hold the widest UTF-8 character, or a character that
        # doesn't fit would produce an empty page and never advance.
        raise ValueError(f"page_bytes must be at least {MIN_PAGE_BYTES}")
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


def strip_thinking(text, final=True):
    """Remove closed <think> blocks; truncate at an open one.

    ``text`` must be the *raw* model output accumulated so far, never output
    that was already stripped: stripping an open block drops its ``<think>``
    tag, so stripping again would expose the thinking text after it.

    While generation is still running (``final=False``), a trailing partial
    ``<think>`` tag (``"<"``, ``"<th"``, ...) is held back because the next
    chunk may complete it. When generation is over (``final=True``) it is kept,
    so an answer that really ends in ``"<"`` is not cut short.

    Leading whitespace is dropped: models usually put a blank line after
    ``</think>``.
    """
    out = []
    i = 0
    while True:
        start = text.find(_THINK_START, i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        close = text.find(_THINK_END, start + len(_THINK_START))
        if close == -1:
            break  # open block: drop everything from here on
        i = close + len(_THINK_END)
    result = "".join(out).lstrip()
    if not final:
        for k in range(len(_THINK_START) - 1, 0, -1):
            if result.endswith(_THINK_START[:k]):
                result = result[:-k]
                break
    return result


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
            if state is not None and state.error is None:
                self._states.move_to_end(key)
                return state, None
            # No entry, or a failed one: a failure is never reused, so a
            # backend that comes back is used on the very next query.
            if self._active >= self.max_gen:
                return None, "busy, try again"
            self._states.pop(key, None)  # re-insert at the newest end
            state = _PromptState(prompt)
            self._states[key] = state
            self._active += 1
        t = threading.Thread(target=self._run, args=(state,), daemon=True)
        t.start()
        return state, None

    def _run(self, state):
        raw = ""  # unstripped output; state.text is always derived from this
        try:
            for chunk in self.backend.stream(state.prompt):
                if not chunk:
                    continue
                raw += chunk
                text = strip_thinking(raw, final=False)
                with state.lock:
                    state.text = text
            with state.lock:
                state.text = strip_thinking(raw, final=True)
                state.done = True
                state.finished_time = time.monotonic()
        except Exception as e:  # never let a backend error take us down
            with state.lock:
                state.text = strip_thinking(raw, final=True)
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
            page_text = pages[page] if page < len(pages) else b""
            more_after = page < len(pages) - 1
            # A page is final once text exists past it. A page of exactly
            # page_bytes is final too: nothing can be added to it. (A page cut
            # short to keep a multi-byte character whole is final only once
            # the next page exists.)
            full = more_after or len(page_text) == page_bytes

            if done or full or time.monotonic() >= deadline:
                break
            time.sleep(0.02)

        # The reply is the requested page's text with a final status string
        # appended (the status tells the client whether to page forward, wait,
        # or that generation finished).
        page_text_str = page_text.decode("utf-8", "replace")
        next_name = f"p{page + 1}.{page_base}"
        if len(next_name) <= MAX_NAME_TEXT:
            more = f"[more: dig {next_name} TXT]"
        else:
            # The next page's name would not be a valid DNS name.
            more = "[error: prompt too long to page; shorten it]"
        if error:
            error = error[:MAX_ERROR_CHARS]
        if done:
            if more_after:
                status = more
            elif error:
                # The last page of a failed generation: show what was produced,
                # then the error, so a truncated answer never looks complete.
                status = f"[error: {error}]"
            else:
                status = "[done]"
        elif full:
            status = more
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


def max_page_bytes(limit=MAX_UDP):
    """Largest --page-bytes for which every possible reply fits ``limit``.

    Worst case: a 255-byte question name, the longest status string (a
    ``[more: ...]`` hint naming a 253-character name), and our OPT record.
    """
    fixed = 12 + (255 + 4) + (2 + 10) + 11  # header, question, answer RR, OPT
    status = len("[more: dig  TXT]") + MAX_NAME_TEXT
    page = limit
    while page > 0:
        text = page + 1 + status  # page, one space, status
        if fixed + text + -(-text // 255) <= limit:  # + one length byte per string
            return page
        page -= 1
    return 0


class Server:
    """Turn parsed DNS questions into responses via the backend."""

    def __init__(self, backend, *, zone="e.com", page_bytes=600, wait=3.5,
                 max_gen=2, cache_ttl=600, max_entries=64,
                 system_prompt=SYSTEM_PROMPT, usage_string=USAGE_STRING):
        if not MIN_PAGE_BYTES <= page_bytes <= max_page_bytes():
            raise ValueError(f"page_bytes must be {MIN_PAGE_BYTES}..{max_page_bytes()} "
                             f"so every reply fits {MAX_UDP} bytes")
        self.backend = backend
        self.zone = zone
        self.page_bytes = page_bytes
        self.wait = wait
        self.manager = GenerationManager(backend, cache_ttl=cache_ttl,
                                         max_entries=max_entries, max_gen=max_gen)
        self.system_prompt = system_prompt
        self.usage_string = usage_string

    def _txt_response(self, question, qid, opt, text, limit):
        rdata = build_txt_rdata(split_txt(text))
        # The owner name is a compression pointer to the question's name, which
        # always starts right after the 12-byte header.
        ans = b"\xc0\x0c" + struct.pack(">HHIH", TYPE_TXT, CLASS_IN, 0, len(rdata)) + rdata
        resp = assemble_response(qid, [question], [ans], opt=opt, aa=True)
        if len(resp) > limit:
            # Too big for what the client accepts: an empty, truncated reply
            # (RFC 2181 §9). Only clients without EDNS, or with a buffer under
            # 1232, can get here; max_page_bytes() guarantees the rest fit.
            return assemble_response(qid, [question], [], opt=opt, aa=True, tc=True)
        return resp

    def handle(self, header, questions, opt):
        """Handle one parsed message. Returns response bytes, or None to drop."""
        if header["qr"]:
            # A response, not a query. Answering it could start a loop with
            # another DNS speaker, so drop it.
            return None
        # Our own OPT, never an echo of the client's (its cookie, DO bit and
        # buffer size are its own); only sent when the query used EDNS.
        if opt is None:
            opt_bytes, limit = None, CLASSIC_UDP
        else:
            limit = max(CLASSIC_UDP, min(opt["class"], MAX_UDP))
            if (opt["ttl"] >> 16) & 0xFF != 0:
                # Only EDNS version 0 exists: BADVERS (RFC 6891 §6.1.3).
                return assemble_response(header["id"], questions[:1], [],
                                         opt=build_opt(ext_rcode=1))
            opt_bytes = build_opt()
        if header["opcode"] != OPCODE_QUERY:
            return assemble_response(header["id"], questions[:1], [], opt=opt_bytes,
                                     rcode=RCODE_NOTIMP, opcode=header["opcode"])
        if not questions:
            return None
        q = questions[0]
        result = query_to_prompt(q["name"], self.zone)
        if result is None:
            # Not under our zone -> REFUSED.
            return assemble_response(header["id"], [q], [], opt=opt_bytes, rcode=RCODE_REFUSED)
        prompt, page = result
        if q["qtype"] != TYPE_TXT:
            # Any other record type in our zone, the apex included -> empty
            # NOERROR. This check comes first so an NS/SOA question for the
            # apex never gets a TXT answer.
            return assemble_response(header["id"], [q], [], opt=opt_bytes, rcode=RCODE_NOERROR)
        if prompt == "":
            # TXT for the zone itself -> one-line usage string.
            return self._txt_response(q, header["id"], opt_bytes, self.usage_string, limit)
        state, err = self.manager.get_or_start(prompt)
        if err is not None:
            return self._txt_response(q, header["id"], opt_bytes, f"[error: {err}]", limit)
        # Page forward from the core name (page prefix + zone already stripped) so
        # paging chains through the same cached generation.
        core = core_name(q["name"], self.zone)
        page_base = f"{core}.{self.zone}" if core else self.zone
        reply = self.manager.fetch_page(state, page, self.page_bytes, self.wait, page_base)
        return self._txt_response(q, header["id"], opt_bytes, reply, limit)


# At most this many packets are handled at once. Each handler can block for up
# to --wait seconds, so this bounds the threads a flood of queries can create;
# packets beyond it are dropped (the client retries).
MAX_INFLIGHT = 64


def serve(sock, server, *, max_inflight=MAX_INFLIGHT, stop=None):
    """Answer DNS packets arriving on ``sock`` until ``stop`` is set.

    Each packet is handled on its own thread, so a query waiting on a slow
    generation never delays any other query. Malformed packets are logged and
    dropped.
    """
    slots = threading.BoundedSemaphore(max_inflight)
    stop = stop or threading.Event()
    sock.settimeout(0.2)  # wake up regularly to check ``stop``
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            return  # socket closed under us
        if not slots.acquire(blocking=False):
            print(f"dropping packet from {addr}: {max_inflight} queries in flight",
                  file=sys.stderr)
            continue
        threading.Thread(target=_handle_packet, args=(sock, server, data, addr, slots),
                         daemon=True).start()


def _handle_packet(sock, server, data, addr, slots):
    try:
        try:
            header, questions, _answers, _additional, opt = parse_message(data)
        except DNSError as e:
            print(f"dropping malformed packet from {addr}: {e}", file=sys.stderr)
            return
        try:
            response = server.handle(header, questions, opt)
        except Exception as e:
            print(f"error handling {addr}: {e!r}", file=sys.stderr)
            return
        if response is None:
            return
        try:
            sock.sendto(response, addr)
        except OSError as e:
            print(f"error replying to {addr}: {e!r}", file=sys.stderr)
    finally:
        slots.release()


def run_udp_server(server, host, port):
    """Bind a UDP socket and serve DNS on it forever."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    # Flushed so the line shows up immediately when stdout is a pipe or a log.
    print(f"ollama-over-dns listening on udp {host}:{port} (zone {server.zone})", flush=True)
    serve(sock, server)


# --- Backends (Phase 4) ------------------------------------------------------
# Both implement the one interface: stream(prompt) -> iterator of text chunks.
# The DNS/caching code never needs to know which one is in use.
import json
import os
import urllib.error
import urllib.parse
import urllib.request

# Seconds to wait to connect and between streamed lines. Generous, because the
# first request after a model is loaded can stall while it loads (tens of
# seconds on small boards).
BACKEND_TIMEOUT = 120

OLLAMA_DEFAULT_PORT = 11434


def normalize_url(url, default_port):
    """Accept ``host``, ``host:port`` or a full URL, the forms OLLAMA_HOST uses.

    Without a scheme, ``http://`` is assumed and a missing port becomes
    ``default_port`` (``localhost`` -> ``http://localhost:11434``), matching
    Ollama's own client. A URL with a scheme is used as given.
    """
    url = url.strip().rstrip("/")
    if "://" in url:
        return url
    try:
        port = urllib.parse.urlsplit("http://" + url).port
    except ValueError:
        raise SystemExit(f"invalid backend address: {url!r}")
    if port is None:
        url = f"{url}:{default_port}"
    return "http://" + url


def _error_text(obj):
    """Pull a message out of an error object (Ollama: str; OpenAI: dict)."""
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        err = err.get("message")
    return str(err) if err else None


def _open(req, timeout):
    """urlopen that turns HTTP and connection failures into short messages."""
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read(4096)
        try:
            detail = _error_text(json.loads(body))
        except ValueError:
            detail = None
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {req.full_url}: {e.reason}") from None


class OllamaBackend:
    """ollama: POST /api/chat with stream:true, NDJSON output."""

    def __init__(self, url, model, max_tokens=256, timeout=BACKEND_TIMEOUT):
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
            # Thinking models would otherwise spend the token budget on
            # reasoning (returned separately in message.thinking) and leave
            # the answer empty. Models that can't think accept this too.
            "think": False,
            "options": {"num_predict": self.max_tokens},
        }
        req = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _open(req, self.timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                err = _error_text(obj)
                if err:
                    raise RuntimeError(err)
                content = (obj.get("message") or {}).get("content")
                if content:
                    yield content


class LlamaCppBackend:
    """llama-server: POST /v1/chat/completions with stream:true, SSE output."""

    def __init__(self, url, model=None, max_tokens=256, timeout=BACKEND_TIMEOUT):
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def stream(self, prompt):
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": self.max_tokens,
            # Qwen3-style templates read this to skip the reasoning block;
            # templates that don't use it ignore it.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.model:
            payload["model"] = self.model
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _open(req, self.timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                err = _error_text(obj)
                if err:
                    raise RuntimeError(err)
                # The first chunk carries only the role ("content": null) and a
                # usage chunk can have no choices at all; skip both. Reasoning
                # sent as delta.reasoning_content is ignored.
                choices = obj.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content


def make_backend(args):
    """Construct the backend from parsed CLI args."""
    if args.backend == "ollama":
        if not args.model:
            raise SystemExit("ollama backend requires --model")
        url = args.url or os.environ.get("OLLAMA_HOST") or "localhost"
        return OllamaBackend(normalize_url(url, OLLAMA_DEFAULT_PORT), args.model,
                             max_tokens=args.max_tokens)
    url = normalize_url(args.url or "localhost", 8080)
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
    if not MIN_PAGE_BYTES <= args.page_bytes <= max_page_bytes():
        p.error(f"--page-bytes must be {MIN_PAGE_BYTES}..{max_page_bytes()} so every reply fits "
                f"{MAX_UDP} bytes")

    backend = make_backend(args)
    server = Server(backend, zone=args.zone, page_bytes=args.page_bytes,
                    wait=args.wait, max_gen=args.max_gen)
    run_udp_server(server, args.host, args.port)


if __name__ == "__main__":
    main()
