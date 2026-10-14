# frugal-llm

Benchmark latency, throughput and cost of any OpenAI-compatible LLM endpoint.

> Work in progress: v0.1 is being built against [SPEC.md](SPEC.md). Quickstart and
> real output will be added once the tool runs end to end.

## Why the numbers can be trusted

- **Warm-up does double duty.** Warm-up requests warm the server *and* the client's
  connection pool. Without them, the first measured requests pay TCP and TLS handshake
  costs and the TTFT tail is handshake noise rather than server behaviour.
- **Every concurrent worker owns exactly one connection.** A single shared connection
  pool either caps you at httpx's default 100 connections (silently serialising the
  load) or, sized up, burns client CPU matching requests to connections — at 150
  concurrent requests that turned a 1 s server delay into a 6 s round. One connection
  per worker avoids both. The test suite sweeps past 100 and checks warm rounds stay
  close to the server's own delay.
- **The client must never block its own event loop.** A test proves that a blocked
  loop inflates measured TTFT by 3x without the server changing at all.
- **Timeouts are per phase, and failures say which phase.** The read timeout applies
  to each streamed chunk; a total deadline bounds the whole request (`total_timeout`).
  A `pool_timeout` means the client ran out of capacity; a `read_timeout` means the
  server stalled. They are reported apart.
- **HTTP/1.1 only.** HTTP/2 multiplexing puts N streams on one connection, which changes
  what "concurrency" means. v0.1 keeps it off and records that in every result file.
