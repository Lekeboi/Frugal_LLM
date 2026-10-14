# frugal-llm

How fast, and how expensive, is it to serve *your* model on *your* hardware?

`frugal-llm` points at any OpenAI-compatible endpoint — vLLM, TGI, SGLang,
llama.cpp, Ollama or a commercial API — and reports latency percentiles,
throughput and cost per million output tokens at each concurrency level you ask for.
It speaks only `POST /v1/chat/completions` with streaming, so there is nothing
engine-specific to install or configure.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Against a local Ollama:

```bash
ollama pull qwen2.5:0.5b
uvx frugal-llm bench --url http://localhost:11434/v1 --model qwen2.5:0.5b
```

A concurrency sweep with a cost model, against vLLM on a GPU you rent for £0.52/hour:

```bash
uvx frugal-llm bench --url http://localhost:8000/v1 --model mistral-7b-awq \
  -c 1,4,16,64 -n 200 --cost-per-hour 0.52 --label vllm-awq-L4
```

Results land in `./results/` as JSON (every individual sample) and Markdown (the
table below, ready to paste into an issue or a chat thread).

## Example output

> **TODO before release:** paste the real output of the Ollama quickstart run here.
> No numbers in this README are invented.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--url` | required | Base URL including `/v1`, e.g. `http://localhost:11434/v1` |
| `--model` | required | Model identifier as the server knows it |
| `--api-key` | none | Bearer token; also read from `FRUGAL_LLM_API_KEY`. Never written to results |
| `-c, --concurrency` | `1` | Comma-separated levels to sweep, e.g. `1,4,16,64`. Run one after another |
| `-n, --requests` | `50` | Measured requests per run. Use several times the concurrency |
| `--max-tokens` | `128` | Hard cap on output length |
| `--prompt-tokens` | `200` | Approximate length of generated prompts |
| `--prompts` | none | JSONL of `{"prompt": "..."}`, used in rotation instead of generated prompts |
| `--warmup` | `5` | Minimum warm-up requests per level; at least one per concurrent worker |
| `--repeats` | `3` | Independent runs per level. Single runs report noise as signal |
| `--cost-per-hour` | none | Hourly cost of the hardware; enables the cost column |
| `--currency` | `GBP` | Label only |
| `--timeout` | `120` | Seconds, applied per streamed chunk and as a total per-request deadline |
| `--out` | `./results` | Output directory |
| `--format` | `both` | `md`, `json` or `both` |
| `--label` | none | Free-text tag stored with the results and used in file names |

Re-aggregate a saved run without re-running it (for example after an upgrade changes
how reports are rendered):

```bash
uvx frugal-llm report results/20261014T120000Z-vllm-awq-L4.json
```

## What the numbers mean

| Metric | Definition |
|---|---|
| **TTFT** | Time to first token: request sent → first chunk with non-empty content (prefill). p50/p95/p99 |
| **ITL** | Inter-token latency: mean gap between content chunks within a request, excluding the first (decode). p50/p95 across requests |
| **E2E** | Request sent → final chunk. p50/p95/p99 in the JSON |
| **Throughput** | Output tokens of completed requests ÷ wall-clock measurement window |
| **Cost / 1M tokens** | `(cost_per_hour / 3600) / tokens_per_sec × 1,000,000` |
| **Err** | Failed or timed-out requests ÷ attempted. The report names each failure kind |

Latency is only ever reported as percentiles, never a mean: the tail is the point.
Each concurrency level is its own distribution; samples are never pooled across
levels. Within a level, latency samples from all repeats are pooled, while
throughput and cost show the median run with the min–max range across repeats.

## What this measures and what it doesn't

**It measures** speed and cost of generation as seen by a client: how long until the
first token, how fast tokens stream, how many tokens per second the endpoint sustains
at a given concurrency, and what that costs at the hourly rate you supply.

**It does not measure:**

- **Output quality or accuracy.** Not at all, in v0.1. A fast, cheap model that
  answers badly will look good here. Evaluate quality separately before deciding.
- **Your real traffic.** Load is closed-loop (N requests always in flight), with fixed
  output length and synthetic or supplied prompts. Bursty traffic, long contexts or
  variable output lengths will behave differently.
- **Network you don't benchmark through.** Numbers include the path between the
  client and the server. Run the client next to the server to measure the server.
- **Exact token counts when the server hides them.** Output tokens come from the
  server's `usage` report; if it doesn't send one, frugal-llm counts content chunks,
  which undercounts servers that pack several tokens per chunk. The report says
  which method was used.
- **Cost beyond the hourly rate.** Idle time, egress, storage and engineering time
  are not included.

## Why the numbers can be trusted

- **Every rule is tested against a mock server** that emits chunks at precisely
  controlled times, so the suite proves the tool measures the server and not itself:
  TTFT against a known prefill delay, genuinely concurrent requests (past 100),
  warm-up excluded, levels never pooled, percentile and cost arithmetic by hand.
- **Warm-up does double duty.** Warm-up requests warm the server *and* open every
  worker's connection, so measured requests don't pay TCP/TLS handshakes and the
  TTFT tail isn't handshake noise.
- **Every concurrent worker owns exactly one connection.** A shared connection pool
  either caps you at httpx's default 100 connections (silently serialising the load)
  or, sized up, burns client CPU matching requests to connections — at 150 concurrent
  requests that turned a 1 s server delay into a 6 s round.
- **The client checks itself.** It samples its own event-loop lag during every run
  and warns in the report when the client, not the server, may have shaped the
  latencies (for example on an overloaded laptop at high concurrency).
- **Prompts never share a prefix**, so server-side prefix caches can't flatter TTFT.
- **Failures fail loudly.** A preflight request catches a wrong URL, model or key
  before anything is measured; an unreachable endpoint aborts without writing
  partial results. Timeouts are per phase and named (`read_timeout`,
  `total_timeout`, ...), so a stalled server is distinguishable from a slow one.
- **HTTP/1.1 only.** HTTP/2 multiplexing would change what "concurrency" means.
  Recorded in every result file.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The tests need no GPU and no model: they run against `tests/mock_server.py`, which
you can also start by hand (`uv run python tests/mock_server.py`) and point the CLI at.

## License

Apache-2.0. See [LICENSE](LICENSE).
