#!/usr/bin/env python3
"""Convenience client for ollama-over-dns.

The server has no real client -- ``dig`` is the client. This script is just a
thin wrapper that automates the manual paging loop: it fetches page 0, then
follows the ``[more: dig p<N+1>...]`` hint until ``[done]``, concatenating the
page text into the full answer. It shells out to ``dig`` for the transport.

Usage:
    python3 client.py --prompt "what does bruturva mean"
    python3 client.py --host 127.0.0.1 --port 5311 \
        --prompt "what does bruturva mean"
"""

import argparse
import re
import subprocess
import sys

# The status string is always the trailing bracketed token of the reply.
_STATUS_RE = re.compile(r"\[([^\]]*)\]\s*$")
# Inside a "[more: dig <qname> TXT]" hint, grab the next page's name.
_MORE_RE = re.compile(r"dig (\S+)\s+TXT")


def dig_txt(qname, host, port):
    """Return the concatenated TXT answer for ``qname`` (via `dig +short`)."""
    result = subprocess.run(
        ["dig", f"@{host}", "-p", str(port), qname, "TXT", "+short"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "dig failed")
    out = result.stdout.strip()
    if not out:
        return ""
    # dig +short prints TXT as space-separated quoted character-strings.
    return "".join(re.findall(r'"([^"]*)"', out))


def split_status(text):
    """Split a reply into ``(content, status_or_None)``.

    ``status`` is the bracketed token (e.g. ``done`` or the full ``[more: ...]``
    hint); ``content`` is everything before it.
    """
    m = _STATUS_RE.search(text)
    if not m:
        return text.rstrip(), None
    return text[: m.start()].rstrip(), m.group(1)


def main(argv=None):
    p = argparse.ArgumentParser(description="Page through an ollama-over-dns answer.")
    p.add_argument("--prompt", required=True, help="the prompt (spaces are fine)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5353)
    p.add_argument("--zone", default="e.com")
    p.add_argument("--max-pages", type=int, default=100,
                   help="safety cap on pages fetched (default 100)")
    args = p.parse_args(argv)
    if not args.prompt.strip():
        print("error: --prompt must not be empty", file=sys.stderr)
        return 2

    # Prompt words become DNS labels (spaces -> dots); hyphens are left as-is
    # (the server treats a hyphen inside a label as a space). The zone suffix is
    # appended verbatim.
    qname = f"{args.prompt.replace(' ', '.').strip('.')}.{args.zone}"
    parts = []
    try:
        for _ in range(args.max_pages):
            reply = dig_txt(qname, args.host, args.port)
            content, status = split_status(reply)
            parts.append(content)
            if status is None:
                break  # no status (e.g. usage string) -> nothing more to follow
            if status.startswith("done"):
                break
            m = _MORE_RE.search(status)
            if not m:
                break  # e.g. [error: ...] or [thinking...]
            qname = m.group(1)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    answer = "".join(parts)
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
