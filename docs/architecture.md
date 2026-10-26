# Architecture

Internal design of `ollama-over-dns`. This is for reading the code, not for
running the server — see [`../README.md`](../README.md) for that.

The whole thing is one module, `server.py`, stdlib only. It reads top to bottom
in layers, each depending only on the ones above it:

```
dig  ──▶  DNS wire bytes  ──▶  parse  ──▶  query_to_prompt  ──▶  Server.handle
                                                                          │
        response bytes  ◀───  assemble_response  ◀───  fetch_page  ◀───  backend.stream(prompt)
                                                                          │
                                                    OllamaBackend / LlamaCppBackend  ──▶  HTTP
```

## 1. DNS wire format (parse + build)

`parse_message(data)` splits a raw packet into five parts:

```python
header, questions, answers, additional, opt = parse_message(data)
```

- `header` — a dict of the six 16-bit fields (`id`, `qdcount`, `ancount`, …).
- `questions` — list of `{name, qtype, qclass}`.
- `answers` — raw answer records (not parsed further; they're echoed back).
- `opt` — the EDNS0 `OPT` pseudo-record, or `None` if the query had none.

`build_query(...)` / `build_answer(...)` / `build_opt(...)` / `build_txt_rdata(...)`
are the inverse, and are used both by the server and by the tests. `build_query`
also backs the optional `client.py`.

Round-tripping through parse → build is covered by the tests
(`test_short_header_rejected`, `test_truncated_*_rejected`), because a DNS
message is just length-prefixed fields and a malformed/truncated packet must
never crash the server — it is logged and dropped.

## 2. Query → prompt / page

`query_to_prompt(name, zone)` is the only place DNS names become text:

- The `--zone` suffix (`e.com`) is stripped.
- Names not under the zone return `None` → the server answers `REFUSED`.
- The remaining labels are joined on spaces (`-` in a label is a space), and
  lowercased (DNS case is unreliable).
- A leading `pN` label selects **page N** of the *same* generation.

It returns `(prompt, page)`. The zone itself (`e.com`) yields `("", 0)`, which
`Server.handle` recognises as "return the usage string".

## 3. Server.handle

One parsed request in, one response out (`None` means "drop it"):

| Situation | Response |
|---|---|
| name not under zone | `REFUSED` |
| zone itself | one-line usage string |
| non-`TXT` in zone | empty `NOERROR` |
| generation error / busy | TXT with `[error: …]` |
| normal | TXT with the model's answer + status |

The response is a single TXT RR, split into ≤255-byte character-strings, TTL 0,
`AA` set, the query's OPT echoed back, total UDP < 1232 bytes.

## 4. Generation, caching, pagination

`Server` owns a `GenerationManager`. Each distinct prompt gets at most one
generation, cached for 10 minutes (64-entry cap, oldest-eviction), with at most
`--max-gen` running at once (new prompts get `[error: busy, try again]`).

The backend implements one interface: `stream(prompt) -> iterator of chunks`.
`fetch_page` pulls chunks off that iterator until the current page fills
(`--page-bytes`, default 600) or generation finishes, then returns the page's
text with a **status** appended:

| Status | Meaning |
|---|---|
| `[done]` | generation finished, this was the last page |
| `[more: dig p<N+1>… TXT]` | there is more; page forward |
| `[thinking… repeat the same query]` | page not full yet, generation ongoing |
| `[error: …]` | the backend failed |

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
  NDJSON lines. Default `{OLLAMA_HOST or http://localhost:11434}`. Requires
  `--model`.
- **`LlamaCppBackend`** — POST to `{url}/v1/chat/completions` with
  `stream:true`, parses SSE `data:` lines. Default `{url or http://localhost:8080}`; `--model` optional.

`make_backend(args)` wires CLI args to a backend. `--url` overrides the default
for whichever backend is selected.

## 6. The server loop

`run_udp_server` binds a UDP socket and loops: parse a packet (log + drop on
`DNSError`), call `handle`, send the reply. Generation runs in background
threads so a slow backend never blocks other queries.
