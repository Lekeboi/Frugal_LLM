# Contributing to frugal-llm

Two kinds of contribution are welcome: **results** from your own hardware, and **changes to
the tool**. Results are the more valuable of the two: every comparable measurement on a GPU
nobody else has published makes the whole corpus more useful.

## Contributing results

A result is useful to others only if it can be compared with the study's. That needs the
same method, the same settings and enough recorded context for a stranger to check it.

### 1. Measure

Serve your model through any OpenAI-compatible server (tested with vLLM, llama.cpp and
Ollama; SGLang and TGI should work but are untested, so results from them are especially
welcome), then run the same sequence the study ran for each format
([`scripts/run_cell.sh`](scripts/run_cell.sh) is the exact recipe). For one format:

```bash
META=(--meta gpu="NVIDIA GeForce RTX 3060" --meta driver=580.65 \
      --meta engine=vllm --meta engine_version=0.30.0 \
      --meta weights=Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e9394b818fdfca67ab2a00ecc7dd641 \
      --meta format=awq --meta provider=runpod-secure \
      --meta rate_usd_per_hour=0.25 --meta rate_date=2026-10-16)

# Speed and cost: four concurrency levels, three repeats each
uvx frugal-llm bench --url http://localhost:8000/v1 --model <name> \
    -c 1,4,16,64 -n 50,100,200,400 --repeats 3 --max-tokens 128 \
    --cost-per-hour 0.25 --currency USD --label <gpu>-<format> "${META[@]}"

# Quality: both shipped task sets, at concurrency 16
for task in gsm8k mmlu; do
  uvx frugal-llm quality --url http://localhost:8000/v1 --model <name> \
      --task $task -c 16 --label <gpu>-<format> "${META[@]}"
done
```

The GPU, driver, rate and provider above are examples; record your own.

For a quality comparison, run the **unquantised (BF16 or FP16) weights twice** on the same
hardware (labels ending `-a` and `-b`): the second run is the noise floor that every other
format is judged against. Then check your own numbers with
`frugal-llm compare <bf16-a> <others...> --noise-floor <bf16-b>`.

**Record everything.** The `--meta` values above are the minimum: GPU, driver, engine and its
version, the weights with their exact revision, the format, and, if you rent, the provider,
hourly rate and the date you saw that rate. On hardware you own, either leave out
`--cost-per-hour` and the rate (speed and quality are still comparable), or enter what renting
the same GPU costs, with its source, and say so in your submission. Never enter 0: it would
report free tokens.

**Do not change the method.** Keep greedy decoding, `--max-tokens 128` for `bench` and the
default 1,024 for `quality`, and the shipped task sets. A result with a changed method is
still welcome, but it must say what changed and it will not be compared directly.

### 2. Check before submitting

- Every `bench` file has `"complete": true`, and every `quality` file answered every item
  (1,319 for GSM8K, 1,000 for MMLU). `frugal-llm report` and `compare` print error and
  truncation counts: zero is normal, anything else needs a sentence of explanation.
- Nothing private is in the files. Results record your endpoint URL, client machine and
  `--meta` values; API keys are never written.

### 3. Submit

Open a **Result submission** issue (the template asks for the details above) and attach the
JSON files as a `.zip`, or open a pull request adding them under
`community-results/<yyyy-mm-dd>-<gpu>-<model>/` with a short `README.md`. Results are
published as submitted, credited to you, and never edited; if something looks wrong, we
discuss it in the issue.

### Compatibility

Every file records `schema_version` (`0.1` for `bench`, `quality-0.1` for `quality`). Results
with the same schema version are comparable; a new version is documented in
[CHANGELOG.md](CHANGELOG.md) with what changed.

## Contributing to the tool

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The tests need no GPU and no model: they run against `tests/mock_server.py`. A change that
affects what is measured needs a test that would fail without it.

Three rules decide most reviews:

1. **Only the OpenAI-compatible HTTP API, never engine-specific code.** If a change needs
   `if engine == "vllm"`, the design is wrong. A server that behaves differently is handled
   gracefully or refused with a clear message, for every server alike.
2. **Measurement honesty.** Quantiles, never means; no pooling across concurrency levels;
   time to first token and inter-token latency kept apart; token counts taken from the
   response and the method recorded; warm-up outside the measurement window. Uncertainty is
   stated in the output, not left to the reader.
3. **Out of scope:** a web UI or dashboard, plugins or adapters, GPU provisioning inside the
   tool, distributed load generation, a results database or leaderboard, and anything beyond
   inference. These are reasonable ideas; they are not this tool.

Bugs: open a **Bug report** issue with the command you ran, what you expected, what happened
and the output of `frugal-llm --version`.
