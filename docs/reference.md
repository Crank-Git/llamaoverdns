# llamaoverdns reference

Every server and client flag, the paging status strings, and the protocol limits.
For the internal design, read [architecture.md](architecture.md).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | UDP interface to bind. `0.0.0.0` exposes the model to every host that can reach this one |
| `--port` | `5300` | UDP port. Not 5353, which is mDNS. Use `--port 53` when run as root (or with `cap_net_bind_service`) |
| `--zone` | `e.com` | Suffix stripped to form the prompt. Names not under it → `REFUSED` |
| `--backend` | `ollama` | `ollama` or `llamacpp` |
| `--url` | ollama: `$OLLAMA_HOST` or `localhost:11434` · llamacpp: `localhost:8080` | Backend address: `host`, `host:port` or a full URL (`http://` and the default port are filled in) |
| `--model` | *(required for ollama)* | Model name. Optional for llamacpp |
| `--max-tokens` | `256` | Max output tokens (ollama `num_predict`, llama-server `max_tokens`) |
| `--wait` | `3.5` | Seconds a query waits for its page (under dig's 5s timeout) |
| `--page-bytes` | `600` | UTF-8 bytes per page, `4`–`664` (so every reply fits 1232 bytes) |
| `--max-gen` | `2` | Max concurrent generations |

## How a query maps to a prompt

Given `--zone e.com`, the server strips the `e.com` suffix and turns the
remaining labels into a prompt:

| Queried name | Prompt |
|---|---|
| `what.does.bravurva.mean.e.com` | `what does bravurva mean` |
| `why-is-the-sky-blue.e.com` | `why is the sky blue` (a `-` in a label is a space) |
| `p1.what.does.bravurva.mean.e.com` | `what does bravurva mean`, **page 1** |

- The name is lowercased (DNS case is unreliable).
- A leading `pN` label selects page `N` of the *same* generation (default 0).
  Paging chains through one cached generation: `p1.` → `p2.` → … → `[done]`.
- Names not under the zone get `REFUSED`. A `TXT` query for the zone itself
  (`e.com`) returns a one-line usage string. Any other type in the zone (the
  zone itself included) gets an empty `NOERROR`.

## Paging & status strings

Generation runs in a background thread, streaming into a per-prompt buffer
(keyed by a hash of the normalized prompt, **not** the page). The first query
starts it; repeat queries (including dig's automatic retries) don't restart it.
The server waits up to `--wait` seconds for the page to fill or generation to
finish, then replies with what exists. Each query is handled on its own
thread, so a query that is waiting never holds up anyone else's. The reply is
the page's text, a space, and a status string:

| Status | Meaning |
|---|---|
| `[done]` | Last page, generation finished |
| `[more: dig p<N+1>.<name> TXT]` | Page is full; run that command for the next page |
| `[thinking... repeat the same query]` | Page isn't full yet; generation still running |
| `[error: busy, try again]` | `--max-gen` generations are already running; retry shortly |
| `[error: <reason>]` | The backend failed. On the last page it follows whatever text was produced, so a cut-off answer never looks complete |
| `[error: prompt too long to page; shorten it]` | The next page's name would be longer than DNS allows |

Pages are cut at character boundaries, so a multi-byte UTF-8 character is never
split across a page.

## `dig` output

`dig +short` prints each 255-byte chunk as a separate quoted string, so long
answers look like `"chunk1" "chunk2"`, and it escapes quotes and non-ASCII
(`’` becomes `\226\128\153`). Plain `dig` (drop `+short`) is easier to read,
and `client.py` prints the answer as-is.

## Client flags

Each word of `--prompt` becomes a label (hyphens are preserved, since the
server treats a hyphen as a space). Flags: `--host` (`127.0.0.1`), `--port`
(`5300`), `--zone` (`e.com`), `--timeout` (`120` seconds in total) and
`--max-pages` (`100`). `client.py` exits 0 for a complete answer, 1 for an error or
timeout (printing any partial answer, with the reason on stderr), and 2 for a
prompt DNS can't carry (non-ASCII, a word over 63 characters, or a name over
253).

## Notes & limitations

- **EDNS0.** Replies carry the server's own `OPT` record (1232-byte buffer, no
  options) when the query had one, and never exceed what the client accepts:
  its EDNS buffer size, capped at 1232, or 512 without EDNS. A reply that
  doesn't fit is sent empty with the `TC` bit. There is no TCP listener, so
  non-EDNS clients only get answers short enough for 512 bytes (dig sends EDNS
  by default, so this is fine in practice). EDNS versions other than 0 get
  `BADVERS`.
- **Thinking models.** A fixed system prompt asks for plain text under 600
  characters, and `<think>…</think>` blocks are stripped from the streamed
  output before it reaches DNS: even when the tags arrive split across tokens,
  and even a block still open when generation is cut off.
- **Caching.** Finished generations are cached for 10 minutes (64-entry cap,
  oldest-eviction). Failed generations are not cached: the next query tries
  the backend again. At `--max-gen` concurrent generations, new prompts get
  `[error: busy, try again]`.
- **Through a resolver.** If the zone is delegated and queries arrive through
  a recursive resolver, a lower `--wait` (around 1) fits resolvers' shorter
  upstream timeouts; the resolver retries, and the retry reattaches to the
  running generation.
- **Exposure.** The server answers only standard queries (other opcodes get
  `NOTIMP`, responses are ignored) and handles at most 64 queries at once. It
  has no rate limiting or authentication; bind it to a public address only on
  purpose.
