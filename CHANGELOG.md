# Changelog

## 0.2.0

Quality and cost measurement. The result format is **provisional** until the first full study
on real hardware completes: quality and cost have so far been exercised only against a 0.5B
model on development subsets (see README, "Results format").

### Added

- `frugal-llm quality`: scores a model on shipped, versioned task sets (all 1,319 GSM8K test
  questions; a fixed 1,000-question MMLU subset across all 57 subjects) with greedy decoding,
  fixed prompts and fixed answer-extraction rules. Saves each answer as it arrives;
  `--resume` continues an interrupted run. Retries HTTP 429/503 honouring `Retry-After`.
- `frugal-llm compare`: quality against an FP16 baseline. Paired accuracy difference with
  confidence intervals and an exact McNemar test (Holm-corrected across formats), answer
  agreement and output divergence, the FP16-vs-FP16 noise floor, and the smallest
  difference the comparison could detect. Warns if the baseline's server reports quantised
  weights.
- `frugal-llm cost`: break-even traffic and utilisation against user-supplied, sourced and
  dated API prices, from saved `bench` results; `--curve-json` for charting.
- Every result records the client machine, what the server reports about itself (its
  `/models` entry, `Server` header and build fingerprint, verbatim) and `--meta key=value`.
- `bench` saves each finished run immediately, so a crash costs only the run in progress;
  `report` renders an incomplete sweep with a banner.
- Method notes: `docs/quality-method.md`, `docs/cost-method.md`.

### Fixed

- Runs in which the client froze for over a second (e.g. the machine slept) are excluded
  from aggregates, including a freeze in the final moments of a run.
- Generated prompts are new for every run (random seed, recorded), so server prompt caches
  can no longer flatter repeat runs; server-reported cache hits are recorded and flagged.
- Results are saved before anything is printed, and terminal output never crashes on
  characters the console can't encode (Windows).

## 0.1.0

First release: `frugal-llm bench`, latency (TTFT, ITL, E2E percentiles), throughput and cost
per million output tokens against any OpenAI-compatible endpoint.
