# Testing

Two suites, both stdlib, both run with:

```console
$ python -m unittest
```

(or `python -m unittest -v` for per-test output).

## `test_dns.py` — unit tests

One layer at a time, no network, no subprocesses:

- **Wire format** — `parse_message` / `build_*` round-trips, compressed names,
  EDNS0 OPT echo, and rejection of short/truncated packets.
- **Query → prompt** — `query_to_prompt` mapping, zone stripping, `pN` paging,
  out-of-zone `REFUSED`.
- **Server** — `Server.handle` for every case (usage string, empty NOERROR,
  error/busy replies), pagination math, repeat-query-doesn't-respawn,
  thinking-block stripping, multibyte-safe page cuts.
- **Backends** — `OllamaBackend` / `LlamaCppBackend` against stub HTTP servers
  (NDJSON / SSE), including the `<think>` strip and the wait/timeout →
  `[thinking…]` path.

## `test_e2e.py` — end-to-end tests

The whole stack together, not one layer at a time.

- **`TestSocketE2E`** — runs the real `Server` over a **real UDP socket**
  against a stub backend over **real HTTP**, sends a real DNS query on the wire,
  and asserts on the parsed response. Covers the AA flag, EDNS0 OPT echo, reply
  size < 1232 bytes, and paging (`[more]` → `[done]`). Deterministic; needs no
  external tools.
- **`TestDigE2E`** — drives a real `server.py` subprocess with the actual
  `dig` client, so the paging hint is exercised exactly as a user would see it.
  **Skipped automatically if `dig` is not installed** (e.g. on CI without
  `dnsutils`).

These spawn background threads and real sockets, so the module silences
`ResourceWarning` at import time — that's test hygiene, not a leak.

## Why the backends use stubs, not a live model

The unit and e2e tests never need a real LLM: the backends sit behind one
interface (`stream(prompt) -> iterator of chunks`), so a stub HTTP server that
emits NDJSON (ollama) or SSE (llama) is a faithful stand-in. To smoke-test
against a real model, run the server against your own backend — see the README
quickstart — the wire behaviour is identical.

## Running a single test

```console
$ python -m unittest test_e2e.TestSocketE2E.test_paging -v
```
