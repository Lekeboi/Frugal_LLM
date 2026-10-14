# frugal-llm

Benchmark latency, throughput and cost of any OpenAI-compatible LLM endpoint.

> Work in progress: v0.1 is being built against [SPEC.md](SPEC.md). Quickstart and
> real output will be added once the tool runs end to end.

## Why the numbers can be trusted

- **Warm-up does double duty.** Warm-up requests warm the server *and* the client's
  connection pool. Without them, the first measured requests pay TCP and TLS handshake
  costs and the TTFT tail is handshake noise rather than server behaviour.
- **The connection pool is sized from the concurrency level.** httpx defaults to 100
  connections; above that, requests would queue in the client and the load generator
  would silently serialise. The test suite sweeps past 100 to keep it that way.
- **Timeouts are per phase, and failures say which phase.** The read timeout applies
  to each streamed chunk, not the whole response. A `pool_timeout` means the client ran
  out of capacity; a `read_timeout` means the server stalled. They are reported apart.
- **HTTP/1.1 only.** HTTP/2 multiplexing puts N streams on one connection, which changes
  what "concurrency" means. v0.1 keeps it off and records that in every result file.
