# Results: what quantisation costs and saves on affordable GPUs

> **Final, 2026-10-16.** The method was fixed before the first measurement in
> [study-protocol.md](study-protocol.md); every change made after that is listed at the end
> of that document, with its reason. The raw data is in
> [`study/2026-10-qwen2.5-7b/`](https://github.com/Lekeboi/Frugal_LLM/tree/master/study/2026-10-qwen2.5-7b).

One model, **Qwen2.5-7B-Instruct**, served with **vLLM 0.30.0** in four formats: BF16 (the
baseline, as published), AWQ 4-bit, GPTQ 4-bit and GPTQ 8-bit, all from Qwen's own repositories
and pinned by revision. Each GPU tier ran every format on a single rented pod (RunPod Secure
Cloud), with the same pinned scripts. Every number on this page is generated from the saved
result files by [`analyse_study.py`](https://github.com/Lekeboi/Frugal_LLM/blob/master/scripts/analyse_study.py) and
[`compare_tiers.py`](https://github.com/Lekeboi/Frugal_LLM/blob/master/scripts/compare_tiers.py); nothing is typed in by hand.

## Summary

- **Within this study's resolution, no format lost accuracy.** The comparisons could detect
  differences of about 1–2.3 points; smaller ones are reported as "no difference detected",
  never as "no loss". On GSM8K (1,319 items) and an MMLU subset (1,000 items), no quantised
  format differed significantly from BF16 on any of the three GPUs that ran the quality tests
  (the A100 was a speed-only reference). Both 4-bit formats scored ~1 point lower on MMLU on
  all three GPUs: a consistent small deficit, but below our resolution of about 1–2 points,
  and one result rather than three, since the GPUs answered the same questions. GPTQ 8-bit
  was indistinguishable from BF16.
- **4-bit is 2.1–2.9× faster for a single user and 1.1–1.6× faster under load** than BF16 on
  the same GPU.
- **Hourly price did not rank cost per token.** At 64 concurrent requests the two consumer
  cards, an RTX 3090 (USD 0.50/h) and an RTX 4090 (USD 0.89/h), produced output tokens at about
  half the cost of an L4 (USD 0.59/h) or an A100 (USD 1.79/h): with AWQ, USD 0.089 and 0.080
  per 1M, against 0.169 and 0.159.
- **Self-hosting beats the only per-token API for this model only at high, steady traffic:**
  about 250,000 requests a day on an RTX 3090 and 440,000 on an RTX 4090 (21–30% of measured
  capacity, around the clock), and more on the other GPUs.
- **The API's quality matched the self-hosted baseline,** but its outputs diverged from it as
  much as a 4-bit format's do, and the provider does not say what precision it serves.

## What to do differently

For a 7B-class model served with vLLM, on tasks like these:

- **Pick a GPU by cost per token at your load, not by hourly price.** The L4, second-cheapest
  to rent, was the most expensive to run; the RTX 3090 and 4090 were the cheapest. Measure
  your own endpoint before choosing: `frugal-llm bench` takes one command, and the hourly rate
  turns its throughput into a cost.
- **Quantise, knowing the likely price.** 4-bit weights were up to 2.9× faster, with no
  significant accuracy loss but a possible one-point loss on knowledge questions. If that
  matters, GPTQ 8-bit was indistinguishable from BF16 and still 1.1–1.9× faster.
- **Below roughly a quarter of a million requests a day, paying per token is cheaper.** That is
  the break-even for a single RTX 3090 kept steadily busy, and higher for every other GPU
  measured. Self-host for privacy, control or latency, not to save money, unless your traffic
  is that high and that steady.
- **Do not assume an API will keep serving a given model, or serve it at full precision.**
  Of three providers listing this model, one served it per token; it did not say at what
  precision, and its answers diverged from full precision as much as a 4-bit model's.
- **Check what transfers.** Arithmetic and multiple-choice knowledge are narrow. If your
  workload is different (long context, code, open-ended writing), these quality results may
  not hold for it, and the speed results depend on your prompt and output lengths.

## What this is not

Not new: engine comparisons and quantisation evaluations are plentiful, from peer-reviewed
vLLM-against-TGI studies to vendor benchmarks. Nothing here is claimed to be first.

The closest prior work found is [Bench360](https://arxiv.org/abs/2511.16682) (arXiv:2511.16682;
Stuhlmann, Fadel Argerich and Fürst, 2025), which measures task quality and system metrics
across four inference engines on L4, A10 and A30 GPUs under a 24 GB memory budget. It varies
model size and precision together (for example Gemma-2 9B at FP16 against 27B at INT4), so it
does not isolate the effect of precision, and its cost analysis is energy-based. This study
holds the model fixed and varies only its precision, tests the quality difference with paired
statistics against a measured noise floor, and prices the result in dollars per token,
measured from outside through the OpenAI-compatible API, by a method fixed in advance and
from raw data anyone can check.

## 1. The harness

The study is fully rerunnable from this repository: the vLLM image is pinned
(`vllm/vllm-openai:v0.30.0`), the weights are pinned by revision, each pod fetched the cell
scripts from a fixed commit, and frugal-llm itself was installed at a fixed version (0.2.1).
Each result file records the GPU, driver, engine version, the server's own description of the
model, the hourly rate and its date. Raw results are committed with the release that follows
this study, so every table and chart can be regenerated, or checked, without rerunning
anything.

The GPU time for the whole study, including a shakedown run, cost about USD 4.40 (RunPod
billing), and the API quality check about USD 0.10.

## 2. Speed on affordable GPUs

| GPU | USD/hour | BF16 | AWQ 4-bit | GPTQ 4-bit | GPTQ 8-bit |
|---|---:|---:|---:|---:|---:|
| | | *output tokens/s, one request at a time* | | | |
| RTX 3090 | 0.50 | 49.9 | 138.0 | 138.8 | 92.1 |
| L4 | 0.59 | 17.6 | 51.5 | 51.8 | 31.3 |
| RTX 4090 | 0.89 | 62.4 | 162.0 | 162.8 | 104.2 |
| A100 80GB PCIe | 1.79 | 91.5 | 193.6 | — | — |
| | | *output tokens/s, 64 concurrent requests* | | | |
| RTX 3090 | 0.50 | 1,247 | 1,564 | 1,468 | 1,427 |
| L4 | 0.59 | 626 | 972 | 924 | 719 |
| RTX 4090 | 0.89 | 2,102 | 3,075 | 3,000 | 2,464 |
| A100 80GB PCIe | 1.79 | 2,772 | 3,134 | — | — |

Hourly rates are RunPod Secure Cloud on-demand prices on 15 October 2026, the day each pod
was rented. The A100 is a reference point and, as planned before any result, ran latency only
(BF16 and AWQ). Each figure is the median
of three repeats; the full tables, with time to first token and inter-token latency at 1, 4,
16 and 64 concurrent requests, are in each tier's summary.

![Output throughput by GPU and format](results/throughput-by-gpu.png)

Quantisation helps most where a GPU is short of memory bandwidth: a single user on the L4 gets
2.9× the speed from 4-bit weights. Under heavy load the gain shrinks as the GPU becomes
compute-bound: at 64 concurrent requests AWQ is only 1.13× BF16 on the A100 and 1.25× on the
RTX 3090, and an A100 running AWQ is barely faster than an RTX 4090 running it (3,134 against
3,075 tokens/s).

## 3. Cost

Cost per 1M output tokens is the hourly rate divided by measured output throughput: the GPU
kept fully busy at that load. At 64 concurrent requests:

| GPU | USD/hour | BF16 | AWQ 4-bit | GPTQ 4-bit | GPTQ 8-bit |
|---|---:|---:|---:|---:|---:|
| RTX 3090 | 0.50 | **0.111** | 0.089 | 0.095 | **0.097** |
| L4 | 0.59 | 0.262 | 0.169 | 0.177 | 0.228 |
| RTX 4090 | 0.89 | 0.118 | **0.080** | **0.082** | 0.100 |
| A100 80GB PCIe | 1.79 | 0.179 | 0.159 | — | — |

![Cost per 1M output tokens by GPU and format](results/cost-per-mtok.png)

Hourly price does not rank cost per token; throughput does. The RTX 4090 costs 51% more per
hour than the L4 but produces 3.2–3.4× the tokens. The RTX 3090, cheapest per hour, is the
cheapest per token for BF16 and GPTQ 8-bit; for 4-bit formats the RTX 4090 is 10–14% cheaper,
because 4-bit weights speed the RTX 3090 up less under load (1.25× against the 4090's 1.46×).

### Against an API

The comparison is against an API serving **the same model**, so that only the price differs.
On 2026-10-16 the only live per-token provider of Qwen2.5-7B-Instruct found was **Phala via
OpenRouter**, at USD 0.10 per 1M input and 0.20 per 1M output tokens. Together AI still lists
the model at USD 0.30/0.30 but no longer serves it per token (requests are refused unless an
hourly dedicated endpoint is rented), and DeepInfra lists it as deprecated. Per-token access
to a specific, slightly older model can disappear; a self-hosted copy cannot.

Requests are compared whole, at the shape measured (about 233 tokens in and 120 out). At 64
concurrent requests, self-hosting becomes cheaper than the API above:

| GPU | Break-even traffic, requests/day | Utilisation needed, around the clock |
|---|---:|---:|
| RTX 3090 | about 250,000 | 23% (4-bit) to 29% (BF16) |
| RTX 4090 | about 440,000 | 21% (4-bit) to 30% (BF16) |
| L4 | about 300,000 | 44% (4-bit) to 67% (BF16) |
| A100 80GB PCIe | about 890,000 | 41% (AWQ) to 46% (BF16) |

Break-even traffic barely depends on the format, because one instance is rented either way;
the format changes how much of that instance the traffic uses, and so how much headroom is
left before a second one is needed.

![Daily cost, RTX 4090 against the API](results/breakeven-rtx4090.png)

Not included: idle time, engineering time, storage and egress, which all favour the API. The
utilisation figure is the assumption everything rests on; the method is in
[cost-method.md](cost-method.md).

## 4. Quality against BF16

Greedy decoding (temperature 0, top_p 1, seed 0), at most 1,024 output tokens, 16 requests at
a time, scored by fixed rules ([quality-method.md](quality-method.md)). BF16 ran twice on each
GPU; the second run is the noise floor, showing how much answers change with nothing changed.
Differences are paired (same items), with 95% confidence intervals and Holm-corrected exact
McNemar tests.

**Resolution first.** With these question counts, the smallest difference a comparison could
detect ranged from about 1 point (GPTQ 8-bit, whose answers barely change) to 2.3 points
(4-bit formats on MMLU). A smaller difference is reported as "no difference detected", which
is not the same as "no loss".

| Format | GPU | GSM8K, points vs BF16 (95% CI) | MMLU, points vs BF16 (95% CI) |
|---|---|---:|---:|
| AWQ 4-bit | RTX 3090 | −0.8 (−2.1 to +0.6) | −0.9 (−2.4 to +0.6) |
| | RTX 4090 | −0.7 (−2.0 to +0.7) | −0.9 (−2.4 to +0.6) |
| | L4 | −0.8 (−2.2 to +0.5) | −0.9 (−2.4 to +0.6) |
| GPTQ 4-bit | RTX 3090 | +0.0 (−1.3 to +1.3) | −1.0 (−2.6 to +0.5) |
| | RTX 4090 | +0.2 (−1.2 to +1.5) | −1.2 (−2.8 to +0.4) |
| | L4 | −0.2 (−1.5 to +1.2) | −1.1 (−2.7 to +0.5) |
| GPTQ 8-bit | RTX 3090 | +0.0 (−0.9 to +0.9) | +0.1 (−0.6 to +0.8) |
| | RTX 4090 | +0.2 (−0.7 to +1.2) | +0.1 (−0.4 to +0.6) |
| | L4 | −0.2 (−1.1 to +0.8) | +0.0 (−0.8 to +0.8) |

BF16 itself scored 91.4–91.7% on GSM8K (Qwen reports 91.6%) and 73.8% on MMLU, on all three
GPUs.

**No difference was detected for any format on any GPU.** Every interval includes zero.

Both 4-bit formats scored ~1 point lower on MMLU on all three GPUs: a consistent small
deficit, but below our resolution of about 1–2 points. The consistency is one result, not
three: the GPUs answered the same questions with the same weights, and about 30 of the 33
MMLU questions AWQ lost were the same on all three. An exploratory test
(not in the pre-registered plan; [study-protocol.md](study-protocol.md), change 5) therefore
treats the item, not the GPU, as the unit, averaging each item's difference over the GPUs:

| Format, item-level across the three GPUs | GSM8K, points (95% CI) | MMLU, points (95% CI) |
|---|---:|---:|
| BF16 rerun (control) | −0.1 (−0.5 to +0.4) | −0.0 (−0.3 to +0.2) |
| AWQ 4-bit | −0.8 (−2.0 to +0.5) | −0.9 (−2.3 to +0.5) |
| GPTQ 4-bit | −0.0 (−1.2 to +1.2) | −1.1 (−2.7 to +0.4) |
| GPTQ 8-bit | +0.0 (−0.7 to +0.7) | +0.1 (−0.5 to +0.6) |
| Both 4-bit formats | −0.4 (−1.4 to +0.6) | −1.0 (−2.2 to +0.2), p = 0.10 |

None is significant (Holm-corrected p ≥ 0.81 across all eight tests; the uncorrected p for
both 4-bit formats on MMLU is 0.10). The honest reading: 4-bit may cost about one point on
MMLU, a loss of up to about two points cannot be ruled out, and a larger question set would
be needed to settle it. [`scripts/pooled_quality.py`](https://github.com/Lekeboi/Frugal_LLM/blob/master/scripts/pooled_quality.py)
reproduces the table.

A narrower item-level test of the two 4-bit formats on MMLU alone agrees: AWQ −0.90 points
(95% CI −2.33 to +0.50), GPTQ-Int4 −1.10 (−2.60 to +0.40); both intervals include zero
([study-protocol.md](study-protocol.md), change 7).

![Accuracy difference from BF16, RTX 4090](results/quality-rtx4090.png)

### Different is not worse

Quantised formats change answers far more than they change accuracy:

| | GSM8K, same final answer as BF16 | MMLU, same final answer as BF16 |
|---|---:|---:|
| BF16 rerun (noise floor) | 96.4–97.0% | 99.0–99.5% |
| GPTQ 8-bit | 95.5–95.7% | 98.3–99.3% |
| AWQ / GPTQ 4-bit | 91.6–92.5% | 91.1–92.3% |

(Ranges span the three GPUs.) 8-bit is close to the noise floor. 4-bit changes about one final
answer in twelve, and most of those changes cancel out: GPTQ 4-bit on GSM8K lost and gained
about 38 right answers each. Where they do not cancel, the gap is small: AWQ on GSM8K lost
about 46 and gained about 36, and both 4-bit formats on MMLU lost 32–39 and gained 23–27,
which is the one-point deficit above.

### The API

The same quality runs against Phala via OpenRouter scored 92.6% on GSM8K and 74.1% on MMLU:
+1.1 points (−0.2 to +2.5) and +0.3 (−0.7 to +1.3) against the RTX 4090 BF16 baseline, so no
difference was detected. Its outputs, however, matched the baseline's final answer on only
92.3% of GSM8K items and were word-for-word identical on 0.5%, which is the divergence of a
4-bit format, not of a BF16 rerun. The provider is serving something other than this exact
configuration (a different precision, engine or both) and does not report which. The cost
comparison above is therefore like for like in measured accuracy, not in serving setup.

## 5. Limitations

As stated in the protocol before any result: one model and one engine; quality covers
arithmetic reasoning and recalled knowledge only, on public sets that may be in training data;
one provider and one date for prices; closed-loop synthetic load. In addition, found during the
study:

- The API's precision is unreported, and only one per-token provider of this model existed.
- Recording which upstream served each API request (`served_by`) was added to frugal-llm
  after this study's API run; at the time, Phala was the only provider listed, and served a
  test request.
- Tier 1 ran on an RTX 4090 when the planned RTX 3090 could not be placed; the RTX 3090 was
  run afterwards and is reported alongside it, not instead of it.

## Reproducing

```bash
S=study/2026-10-qwen2.5-7b
uv run --with matplotlib python scripts/analyse_study.py $S/t1-rtx4090 \
    --tier t1-rtx4090 --out analysis/t1-rtx4090 \
    --api "name=phala-via-openrouter,in=0.10,out=0.20,source=https://openrouter.ai/api/v1/models/qwen/qwen-2.5-7b-instruct/endpoints,date=2026-10-16"
uv run --with matplotlib python scripts/compare_tiers.py \
    $S/t1-rtx3090 $S/t1-rtx4090 $S/t2-l4 $S/ref-a100 --out analysis/tiers
```

Each GPU cell is [`scripts/run_cell.sh`](https://github.com/Lekeboi/Frugal_LLM/blob/master/scripts/run_cell.sh), run on a pod by
[`scripts/pod_entry.sh`](https://github.com/Lekeboi/Frugal_LLM/blob/master/scripts/pod_entry.sh).
