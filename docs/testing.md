# Testing

Two suites, both stdlib, both run with:

```console
$ python -m unittest
```

(or `python -m unittest -v` for per-test output).

## `test_dns.py` — unit tests

One layer at a time, no subprocesses (backend tests use stub HTTP servers on
localhost):

- **Wire format** (`RoundTripQuery`, `NameDecoding`, `TruncatedMessages`,
  `ResponseBuilding`) — `parse_message` / `build_*` round-trips, compressed
  names, trailing dots, and rejection of short/truncated packets.
- **Parser hardening** (`ParserHardening`, `QueryKinds`) — pointer loops,
  over-long names, reserved label types, unrepresentable label bytes, a seeded
  5000-packet fuzz test over `parse_message` + `handle`, dropped responses and
  `NOTIMP` for other opcodes.
- **Query → prompt** (`QueryToPrompt`) — `query_to_prompt` mapping, zone
  stripping, `pN` paging, out-of-zone names.
- **Server** (`ServerTests`) — `Server.handle` for every case (usage string,
  empty NOERROR at and below the apex, error/busy replies), pagination math,
  repeat-query-doesn't-respawn, multibyte-safe page cuts.
- **Thinking** (`SliceTests`, `StreamedThinking`) — `<think>` stripping on
  whole strings and on streams split one character per chunk, at every 2- and
  3-way split point, and through `Server.handle` mid-stream. Every
  intermediate text is checked, not just the final one.
- **Cache and paging** (`CacheAndPaging`) — failed generations not reused,
  full-page detection at multi-byte edges (with timing), partial failures,
  TTL and entry-cap eviction, concurrent first queries, slot release.
- **Reply size** (`ReplySize`) — a ~1900-case sweep of name lengths, page
  sizes, content and client buffers asserting every reply fits its limit; the
  worst case; the answer pointer; our own OPT; BADVERS; TC.
- **Backends** (`BackendTests`, `BackendConfig`) — exact request payloads for
  ollama and llama-server, real-shaped streams (`content: null`, usage chunks,
  `thinking`/`reasoning_content`), mid-stream and HTTP errors, and `--url` /
  `OLLAMA_HOST` forms.
- **Defaults** (`Defaults`) — the default bind and the usage string.
- **Client** (`ClientParsing`) — prompt → name, rejected prompts, and
  `split_status` (page whitespace, brackets in content and error reasons).

## `test_e2e.py` — end-to-end tests

The whole stack together, not one layer at a time.

- **`TestSocketE2E`** — runs the real `Server` over a **real UDP socket**
  against a stub backend over **real HTTP**, sends a real DNS query on the
  wire, and asserts on the parsed response: the AA flag, the OPT record, reply
  size < 1232 bytes, and paging (`[more]` → `[done]`).
- **`TestConcurrentServing`** — the real `serve()` loop: simultaneous slow
  queries answered within one `--wait`, a fast query not delayed by a waiting
  one, the in-flight cap, and a real `server.py` with concurrent prompts.
- **`TestClientE2E`** — `client.py` against a real server: the assembled
  answer must equal the stub's output byte for byte (quotes, backslashes, 2-
  to 4-byte characters, spaces at page edges) through `[thinking]` and
  `[more]`; the CLI with no `dig` on `PATH`; busy retries; errors, timeouts,
  truncation and wrong ids.
- **`TestHostilePacketsE2E`**, **`TestZoneApexE2E`**, **`TestReplySizeE2E`**,
  **`TestDefaultsE2E`** — a real `server.py` subprocess surviving a
  pointer-loop packet, answering NS/SOA at the apex, fitting a maximum-length
  name in 1232 bytes, and binding `127.0.0.1:5300` (and not the LAN address)
  by default.
- **`TestDigE2E`** and the `dig` tests in the classes above — drive a real
  `server.py` with the actual `dig` client, so replies are checked exactly as a
  user would see them. **Skipped automatically if `dig` is not installed** (CI
  installs `dnsutils`, so they run there).

These spawn background threads and real sockets, so the module silences
`ResourceWarning` at import time — that's test hygiene, not a leak.

## Why the backends use stubs, not a live model

The unit and e2e tests never need a real LLM: the backends sit behind one
interface (`stream(prompt) -> iterator of chunks`), so a stub HTTP server that
emits NDJSON (ollama) or SSE (llama) is a faithful stand-in. The stub streams
are copied from real ollama and llama-server output.

## Checking against a real model

Run the server against a real backend and page through an answer with the
client and with `dig`:

```console
# ollama (another machine works too: OLLAMA_HOST=192.168.1.215)
$ python3 server.py --backend ollama --model qwen2.5:3b --page-bytes 200

# or llama-server; --reasoning-format none leaves <think> in the content,
# which exercises the thinking strip
$ llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --port 8080 --reasoning-format none
$ python3 server.py --backend llamacpp --page-bytes 200

$ python3 client.py --prompt "explain how DNS caching works, with an example"
$ dig @127.0.0.1 -p 5300 what.does.bravurva.mean.e.com TXT
```

What to look for: the client prints a clean answer and exits 0; `--page-bytes
200` makes most answers span pages, so `[more]` is followed; no `<think>` text
or tags appear; quotes and punctuation come through intact.

## Running a single test

```console
$ python -m unittest test_e2e.TestSocketE2E.test_paging -v
```
