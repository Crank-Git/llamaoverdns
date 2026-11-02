#!/usr/bin/env python3
"""Convenience client for ollama-over-dns.

``dig`` is all you need to talk to the server. This script automates the
paging loop: it asks for page 0, repeats the query while the server says
``[thinking...]``, follows each ``[more: dig p<N+1>...]`` hint until
``[done]``, and prints the assembled answer. It sends the DNS queries itself
(standard library only), so it does not need ``dig``.

Usage:
    python3 client.py --prompt "what does bravurva mean"
    python3 client.py --host 127.0.0.1 --port 5300 \\
        --prompt "what does bravurva mean"

Exit status: 0 for a complete answer, 1 for an error or timeout (any partial
answer is still printed), 2 for a prompt that can't be sent.
"""

import argparse
import random
import re
import socket
import sys
import time

from server import (CLASSIC_UDP, DEFAULT_PORT, MAX_NAME_TEXT, MAX_UDP, RCODE_NOERROR,
                    DNSError, build_opt, build_query, parse_message)

# Inside a "[more: dig <qname> TXT]" hint, grab the next page's name.
_MORE_RE = re.compile(r"dig (\S+)\s+TXT")
# The status the server appends to every reply. Matched by its prefix, because
# an error reason may itself contain brackets.
_STATUS_PREFIXES = ("[done]", "[more: ", "[thinking", "[error: ")


class ClientError(Exception):
    """A query that could not be answered."""


def split_status(text):
    """Split a reply into ``(content, status_or_None)``.

    The server sends ``<page text> <status>`` or just ``<status>``. ``status``
    is returned without its brackets (``done``, ``more: dig ... TXT``, ...).
    ``content`` is the page text exactly as generated: only the single space
    the server put before the status is removed, so whitespace at the end of
    a page (a space between two words split across pages) is kept.
    """
    if not text.endswith("]"):
        return text, None
    start = max(text.rfind(prefix) for prefix in _STATUS_PREFIXES)
    if start == -1:
        return text, None
    content = text[:start]
    if content.endswith(" "):
        content = content[:-1]
    return content, text[start + 1:-1]


def prompt_to_qname(prompt, zone):
    """Turn a prompt into a query name: each word (or dotted part) is a label.

    Raises ``ValueError`` with a readable message for prompts DNS can't carry.
    """
    try:
        prompt.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("the prompt must be plain ASCII (DNS names are)")
    labels = [label for word in prompt.split() for label in word.split(".") if label]
    if not labels:
        raise ValueError("--prompt must not be empty")
    for label in labels:
        if len(label) > 63:
            raise ValueError(f"word too long for a DNS label (max 63): {label[:20]}...")
    qname = ".".join(labels + [zone.strip(".")])
    if len(qname) > MAX_NAME_TEXT:
        raise ValueError(f"prompt too long: the name is {len(qname)} characters, "
                         f"DNS allows {MAX_NAME_TEXT}")
    return qname


def query_txt(qname, host, port, timeout, tries=3):
    """Send one TXT query (with EDNS) and return its text.

    Retries on timeout, since UDP packets can be lost.
    """
    family, _, _, _, addr = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
    for attempt in range(tries):
        qid = random.getrandbits(16)
        msg = bytearray(build_query(qname, qid=qid))
        msg[10:12] = b"\x00\x01"  # arcount = 1: the OPT record below
        msg += build_opt(buffer_size=MAX_UDP)
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.connect(addr)
            sock.send(bytes(msg))
            try:
                while True:
                    data = sock.recv(65535)
                    try:
                        header, _, answers, _, _ = parse_message(data)
                    except DNSError:
                        continue  # not a usable reply; keep waiting
                    if header["qr"] and header["id"] == qid:
                        break
            except socket.timeout:
                continue
            except ConnectionRefusedError:
                raise ClientError(f"nothing is listening on {host} port {port}")
        if header["tc"]:
            raise ClientError("the reply was truncated (it did not fit "
                              f"{CLASSIC_UDP}-{MAX_UDP} bytes)")
        if header["rcode"] != RCODE_NOERROR:
            raise ClientError(f"the server answered with rcode {header['rcode']} "
                              "(is --zone right?)")
        if not answers:
            raise ClientError("the server returned no answer")
        rdata, out, i = answers[0]["rdata"], [], 0
        while i < len(rdata):
            out.append(rdata[i + 1:i + 1 + rdata[i]])
            i += 1 + rdata[i]
        return b"".join(out).decode("utf-8")
    raise ClientError(f"no reply from {host} port {port} after {tries} tries")


def fetch_answer(qname, host, port, *, timeout=120.0, query_timeout=6.0,
                 max_pages=100):
    """Page through the answer for ``qname``.

    Returns ``(text, error)``; ``error`` is None for a complete answer.
    """
    deadline = time.monotonic() + timeout
    parts = []
    pages = 0
    busy_delay = 0.5
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "".join(parts), f"timed out after {timeout:g}s"
        try:
            reply = query_txt(qname, host, port, min(query_timeout, remaining))
        except ClientError as e:
            return "".join(parts), str(e)
        content, status = split_status(reply)
        if status is None:
            parts.append(content)  # e.g. the usage string at the zone apex
            return "".join(parts), None
        if status.startswith("thinking"):
            continue  # this page isn't full yet; the server already waited
        if status == "error: busy, try again":
            time.sleep(min(busy_delay, max(0.0, deadline - time.monotonic())))
            busy_delay = min(busy_delay * 2, 5.0)
            continue
        parts.append(content)
        if status == "done":
            return "".join(parts), None
        if status.startswith("error: "):
            return "".join(parts), status[len("error: "):]
        m = _MORE_RE.search(status)
        if not m:
            return "".join(parts), f"unexpected status: [{status}]"
        pages += 1
        if pages >= max_pages:
            return "".join(parts), f"stopped after --max-pages {max_pages}"
        qname = m.group(1)


def main(argv=None):
    p = argparse.ArgumentParser(description="Page through an ollama-over-dns answer.")
    p.add_argument("--prompt", required=True, help="the prompt (spaces are fine)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--zone", default="e.com")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="give up after this many seconds in total (default 120)")
    p.add_argument("--max-pages", type=int, default=100,
                   help="safety cap on pages fetched (default 100)")
    args = p.parse_args(argv)
    try:
        qname = prompt_to_qname(args.prompt, args.zone)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    answer, error = fetch_answer(qname, args.host, args.port, timeout=args.timeout,
                                 max_pages=args.max_pages)
    if answer:
        print(answer)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
