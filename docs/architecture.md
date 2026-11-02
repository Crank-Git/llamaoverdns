# Architecture

Internal design of `ollama-over-dns`. This is for reading the code, not for
running the server — see [`../README.md`](../README.md) for that.

The whole thing is one module, `server.py`, stdlib only. It reads top to bottom
in layers, each depending only on the ones above it:

```
dig  ──▶  DNS wire bytes  ──▶  parse  ──▶  query_to_prompt  ──▶  Server.handle
                                                                          │
        response bytes  ◀───  _txt_response  ◀───  fetch_page  ◀───  backend.stream(prompt)
                                                                          │
                                                    OllamaBackend / LlamaCppBackend  ──▶  HTTP
```

## 1. DNS wire format (parse + build)

`parse_message(data)` splits a raw packet into five parts:

```python
header, questions, answers, additional, opt = parse_message(data)
```

- `header` — a dict of the six 16-bit fields (`id`, `qdcount`, `ancount`, …)
  plus the decoded flags (`qr`, `opcode`, `tc`, …).
- `questions` — list of `{name, qtype, qclass}`.
- `answers` — raw answer records (not parsed further).
- `opt` — the EDNS0 `OPT` pseudo-record, or `None` if the query had none.

`build_query(...)` / `build_opt(...)` / `build_txt_rdata(...)` /
`assemble_response(...)` are the inverse. They are used by the server, the
tests, and `client.py`.

A DNS message is just length-prefixed fields, and a malformed or hostile
packet must never crash or stall the server — it raises `DNSError`, and the
loop logs and drops it. `decode_name` therefore:

- follows at most 127 compression pointers (a 255-byte name can't have more
  labels), so pointer loops fail fast;
- caps a name at 255 wire bytes and rejects the reserved label types
  (`0x40`/`0x80`) and labels that run past the message;
- refuses label bytes that would not echo back unchanged (`.`, control bytes,
  non-ASCII). A space inside a label is fine.

`encode_name` accepts a trailing dot and rejects empty labels and labels over
63 bytes. The parser tests include a seeded fuzz test over `parse_message` and
`handle`.

## 2. Query → prompt / page

`query_to_prompt(name, zone)` is the only place DNS names become text:

- The `--zone` suffix (`e.com`) is stripped.
- Names not under the zone return `None` → the server answers `REFUSED`.
- The remaining labels are joined on spaces (`-` in a label is a space), and
  lowercased (DNS case is unreliable).
- A leading `pN` label selects **page N** of the *same* generation.

It returns `(prompt, page)`. The zone itself (`e.com`) yields `("", 0)`, which
`Server.handle` recognizes as "return the usage string" for a TXT query.

## 3. Server.handle

One parsed request in, one response out (`None` means "drop it"):

| Situation | Response |
|---|---|
| a response (QR=1) | dropped |
| opcode other than QUERY | `NOTIMP` |
| EDNS version other than 0 | `BADVERS` (extended rcode 1) |
| name not under zone | `REFUSED` |
| any non-`TXT` type in the zone (apex included) | empty `NOERROR` |
| `TXT` for the zone itself | one-line usage string (built from zone, host, port) |
| `--max-gen` already running | TXT `[error: busy, try again]` |
| normal | TXT with the page text + status |

The response is a single TXT RR, TTL 0, `AA` set, split into ≤255-byte
character-strings:

- **Compression.** The answer's owner name is a pointer (`c0 0c`) to the
  question name, so the name is written once.
- **Own OPT.** When the query used EDNS, the reply carries the server's own
  `OPT` (1232-byte buffer, no options, DO clear) — never an echo of the
  client's, which would return its cookie and DO bit.
- **Size budget.** The limit is the client's EDNS buffer (at least 512, at
  most 1232), or 512 without EDNS. A reply over the limit is sent with no
  answer and `TC` set.
- **`max_page_bytes()`** computes the largest page for which the worst case —
  a 255-byte question name, a 253-character `[more]` hint, and the OPT —
  fits 1232 bytes: 664. `--page-bytes` must be 4–664, so an EDNS client with
  a 1232-byte buffer is never truncated. The 4-byte minimum is the widest
  UTF-8 character: a smaller page could never hold one.
- If the next page's name would exceed 253 characters, the hint is replaced
  by `[error: prompt too long to page; shorten it]`. Backend error reasons are
  cut to 200 characters.

## 4. Generation, caching, pagination

`Server` owns a `GenerationManager`. Each distinct prompt gets at most one
generation, cached for 10 minutes (64-entry cap, oldest finished entry evicted
first, in-flight work never evicted), with at most `--max-gen` running at once
(new prompts get `[error: busy, try again]`). A **failed** generation is never
reused: the next query for that prompt starts a fresh one.

The backend implements one interface: `stream(prompt) -> iterator of chunks`.
`_run` keeps the **raw** output and recomputes the served text from it after
every chunk with `strip_thinking(raw, final=False)`:

- closed `<think>…</think>` blocks are removed and an open block truncates the
  text. Working from the raw text matters: stripping already-stripped text
  loses the `<think>` tag and leaks the reasoning after it;
- while generating, a trailing partial tag (`<`, `<th`, …) is held back
  because the next chunk may complete it; when the stream ends it is kept;
- leading whitespace is dropped.

The served text therefore only ever grows, so a page that was already sent
stays valid.

`fetch_page` waits until the requested page is **full** — text exists past it,
or it is exactly `--page-bytes` — or generation finishes, or `--wait` runs
out, then appends a **status**:

| Status | Meaning |
|---|---|
| `[done]` | generation finished, this was the last page |
| `[more: dig p<N+1>… TXT]` | there is more; page forward |
| `[thinking… repeat the same query]` | page not full yet, generation ongoing |
| `[error: …]` | the backend failed; on the last page, after the text produced |

A page cut short to keep a multi-byte character whole is still "full" as soon
as the next page exists.

### Why paging chains through one generation

The `[more]` hint pages forward from the **core name** (the name with the zone
and the `pN` prefix stripped), not from the raw queried name. So `p1.foo.e.com`
→ `[more: dig p2.foo.e.com]`, not `p2.p1.foo.e.com`. Because every page of one
generation resolves to the same prompt, the second `dig` re-attaches to the
already-cached generation instead of starting a fresh one. This is the single
most common paging bug and it is what `test_page_forward_hint_uses_core_name`
guards against.

## 5. Backends

Two classes, one interface. `Server` never needs to know which is in use.

- **`OllamaBackend`** — POST to `{url}/api/chat` with `stream:true`, reads
  NDJSON lines. The system prompt is a `{"role": "system"}` message, the token
  cap is `options.num_predict`, and `think: false` keeps reasoning models from
  spending the budget on hidden thinking. Empty contents and the separate
  `message.thinking` field are skipped; an `{"error": …}` line raises.
  Requires `--model`.
- **`LlamaCppBackend`** — POST to `{url}/v1/chat/completions` with
  `stream:true`, parses SSE `data:` lines. It skips the role-only first chunk
  (`"content": null`), empty deltas, usage chunks with no `choices`, and
  `reasoning_content`; an error event raises. `model` is sent only when given,
  and `chat_template_kwargs.enable_thinking: false` asks Qwen3-style templates
  to skip reasoning.

Both turn HTTP errors into the backend's own message (`HTTP 404: model 'x'
not found`), report an unreachable backend as `cannot reach <url>: <reason>`,
and wait up to 120 s (a cold model load on a small board can take tens of
seconds). Anything a backend raises ends that generation with
`[error: <message>]`; it never takes the server down.

`make_backend(args)` wires CLI args to a backend. `normalize_url` accepts
`--url` and `OLLAMA_HOST` as `host`, `host:port`, `[v6]` or a full URL;
without a scheme it fills in `http://` and the default port (11434 or 8080),
as Ollama's own client does.

## 6. The server loop

`run_udp_server` binds the socket and calls `serve(sock, server)`. `serve`
reads packets and hands each one to its own thread, which parses it (logging
and dropping on `DNSError`), calls `handle`, and sends the reply. A query that
waits on a slow generation therefore never delays any other query. At most
`MAX_INFLIGHT` (64) packets are handled at once; beyond that they are dropped
and the client retries. `serve` returns when its `stop` event is set, which is
how the tests drive the real loop.

## 7. client.py

`client.py` reuses `build_query` / `build_opt` / `parse_message` to send its
own EDNS queries over UDP, so it doesn't need `dig`. `fetch_answer` repeats
the query on `[thinking]`, backs off on `busy`, follows `[more]` hints, and
stops on `[done]` or `[error]`, all within one overall timeout. `split_status`
removes exactly the one space the server puts before the status, so
whitespace at the end of a page survives and the assembled answer is
byte-for-byte what the model produced.
