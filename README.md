# ollama-over-dns

A small, fun, intentionally dumb DNS server that answers `TXT` queries with
LLM output. There is **no client** — `dig` is the client.

```console
$ dig @127.0.0.1 -p 5353 what.does.bravurva.mean.e.com TXT
;; ANSWER SECTION:
what.does.bravurva.mean.e.com. 0 IN TXT "bravurva isn't a real word..." "[more: dig p1.what.does.bravurva.mean.e.com TXT]"
```

The labels **before the zone** are the prompt; the TXT reply is the model's
answer, paged.

## How a query maps to a prompt

Given `--zone e.com`, the server strips the `e.com` suffix and turns the
remaining labels into a prompt:

| Queried name | Prompt |
|---|---|
| `what.does.bravurva.mean.e.com` | `what does bruturva mean` |
| `why-is-the-sky-blue.e.com` | `why is the sky blue` (a `-` in a label is a space) |
| `p1.what.does.bravurva.mean.e.com` | `what does bruturva mean`, **page 1** |

- The name is lowercased (DNS case is unreliable).
- A leading `pN` label selects page `N` of the *same* generation (default 0).
  Paging chains through one cached generation: `p1.` → `p2.` → … → `[done]`.
- Names not under the zone get `REFUSED`. The zone itself (`e.com`) returns a
  one-line usage string. Non-`TXT` queries in the zone get an empty `NOERROR`.

## Quick start

### ollama

```bash
# 1. Make sure ollama is running and has a model.
ollama pull llama3.2

# 2. Start the server. OLLAMA_HOST is honored; default is http://localhost:11434.
python3 server.py --backend ollama --model llama3.2

# 3. Query it.
dig @127.0.0.1 -p 5353 what.does.bravurva.mean.e.com TXT
```

### llamacpp (`llama-server`)

```bash
# 1. Start llama-server with your model loaded (listens on :8080 by default).
llama-server -m path/to/model.gguf --port 8080

# 2. Start the server pointing at it. --model is optional (llama-server
#    serves a single loaded model and ignores it).
python3 server.py --backend llamacpp

# 3. Query it.
dig @127.0.0.1 -p 5353 what.does.bravurva.mean.e.com TXT
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `0.0.0.0` | UDP interface to bind |
| `--port` | `5353` | UDP port. Use `--port 53` when run as root (or with `cap_net_bind_service`) |
| `--zone` | `e.com` | Suffix stripped to form the prompt. Names not under it → `REFUSED` |
| `--backend` | `ollama` | `ollama` or `llamacpp` |
| `--url` | ollama: `$OLLAMA_HOST` or `http://localhost:11434` · llamacpp: `http://localhost:8080` | Backend HTTP URL |
| `--model` | *(required for ollama)* | Model name. Optional for llamacpp |
| `--max-tokens` | `256` | Max output tokens sent to the backend |
| `--wait` | `3.5` | Seconds to wait for the requested page (under dig's 5s timeout) |
| `--page-bytes` | `600` | UTF-8 bytes per page |
| `--max-gen` | `2` | Max concurrent generations |

## `dig` examples

1. Basic — the labels before `.e.com` are the prompt:

   ```bash
   dig @127.0.0.1 -p 5353 what.does.bravurva.mean.e.com TXT
   ```

2. Paging — a leading `p1.` fetches page 1 of the same generation. Follow the
   `[more: …]` hint until you see `[done]`:

   ```bash
   dig @127.0.0.1 -p 5353 p1.what.does.bravurva.mean.e.com TXT
   ```

3. Phrases and hyphens — a `-` inside a label is treated as a space:

   ```bash
   dig @127.0.0.1 -p 5353 why-is-the-sky-blue.e.com TXT
   ```

> `dig +short` prints each 255-byte chunk as a separate quoted string, so long
> answers look like `"chunk1" "chunk2"`. Plain `dig` (drop `+short`) is easier
> to read.

## Paging & status strings

Generation runs in a background thread, streaming into a per-prompt buffer
(keyed by a hash of the normalized prompt, **not** the page). The first query
starts it; repeat queries (including dig's automatic retries) don't restart it.
The server waits up to `--wait` seconds for the page to fill or generation to
finish, then replies with what exists. The reply is the page's text with a
status string appended:

| Status | Meaning |
|---|---|
| `[done]` | Last page, generation finished |
| `[more: dig p<N+1>.<name> TXT]` | Page is full; run that command for the next page |
| `[thinking... repeat the same query]` | Page isn't full yet; generation still running |
| `[error: <reason>]` | Backend unreachable or errored (never takes the server down) |

Pages are cut at character boundaries, so a multi-byte UTF-8 character is never
split across a page.

## Optional client

`dig` is the client, but following `[more: …]` by hand is tedious. `client.py`
automates the paging loop — it fetches page 0, follows the hint until `[done]`,
and prints the assembled answer:

```bash
python3 client.py --host 127.0.0.1 --port 5353 --prompt "what does bruturva mean"
```

Spaces in `--prompt` become labels (hyphens are preserved, since the server
treats a hyphen as a space).

## Notes & limitations

- **EDNS0.** The server parses the EDNS0 `OPT` record and echoes it back, and
  keeps total UDP responses under 1232 bytes. Very old non-EDNS clients are
  **not** supported (dig sends EDNS by default, so this is fine in practice).
- **Thinking models.** A fixed system prompt asks for plain text under 600
  characters, and `<think>…</think>` blocks are stripped from the output before
  it reaches DNS — even a block still open when generation is cut off.
- **Caching.** Finished generations are cached for 10 minutes (64-entry cap,
  oldest-eviction). At `--max-gen` concurrent generations, new prompts get
  `[error: busy, try again]`.
- **Page size vs. UDP.** With the default `--page-bytes 600` replies stay well
  under 1232 bytes; raising `--page-bytes` toward 1232 will push reply size up
  accordingly.

## Files

- `server.py` — DNS parse/build, query→prompt parsing, server loop, backends.
- `test_dns.py` — unit tests, one layer at a time (`python -m unittest`).
- `test_e2e.py` — end-to-end tests: real UDP socket + real `server.py`
  subprocess, against stub backends. Skips the `dig` leg if `dig` is absent.
- `client.py` — optional paging convenience wrapper (automates the `dig` loop).
- `docs/` — [`architecture.md`](docs/architecture.md) (internal design) and
  [`testing.md`](docs/testing.md).
- `.github/workflows/test.yml` — CI: runs the suite on Python 3.11–3.13.
