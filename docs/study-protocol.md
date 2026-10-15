# Study protocol: what quantisation costs and saves on affordable GPUs

Written 2026-10-15, **before any study measurement**. Changes made after the first study
result must be listed at the end, with the reason, exactly as in the quality method note.

## 1. The question

What is the cheapest way to self-host a useful open-weight model, and what do you lose by going
cheap? Concretely, for one 7B instruction model served with vLLM:

- how fast it is and what it costs per request at each precision, on GPUs people can afford;
- how much its answers change, and whether its accuracy drops, when quantised;
- above what traffic self-hosting is cheaper than an API serving the same model.

## 2. Model and formats

**Qwen2.5-7B-Instruct** (Apache-2.0), chosen because its quantised builds are published by its
own authors and verifiably come from the same checkpoint. All weights are pinned by revision:

| Format | Repository @ revision | Size |
|---|---|---|
| BF16 (baseline, as published) | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | 15.2 GB |
| AWQ 4-bit | `Qwen/Qwen2.5-7B-Instruct-AWQ` @ `b25037543e9394b818fdfca67ab2a00ecc7dd641` | 5.6 GB |
| GPTQ 4-bit | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` @ `e9c932ac1893a49ae0fc497ad6e1e86e2e39af20` | 5.6 GB |
| GPTQ 8-bit | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8` @ `6711f2b6fc95545efe6018c469eca6832e28cefd` | 8.9 GB |

Provenance checked before the study: every quantised repo declares the BF16 repo as its base,
has an identical architecture configuration, and ships tokenizer files with identical content
(the GPTQ files differ only in formatting). This does not prove the weights came from exactly
this revision; the BF16 noise floor against each format's divergence would expose a different
model.

Candidates considered and rejected: Qwen3-8B and Qwen3.5-9B (reason by default, and disabling
it needs an engine-specific request option; Qwen3.5-9B also has no official quantised builds);
Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 (no quantised builds whose base can be
verified).

## 3. Hardware and software

| Tier | GPU | Why |
|---|---|---|
| 1 | NVIDIA RTX 3090, 24 GB | Consumer card many self-hosters own or rent cheaply |
| 2 | NVIDIA L4, 24 GB | The entry inference card of the major clouds |
| Reference | NVIDIA A100 80 GB | The class published benchmarks use; latency only, to anchor the results |

All tiers rented from RunPod **Secure Cloud** (RunPod's own data centres, for consistent
hardware), hourly rates recorded as paid with their date. Stock varies by the hour, so if a
planned GPU is unavailable when its tier is due, the next Secure Cloud option in this order is
used, and the substitution is reported with the results:

| Tier | Planned | Substitutes, in order |
|---|---|---|
| 1 | RTX 3090 | RTX 4090, then RTX A5000 |
| 2 | L4 | none (the tier waits for stock) |
| Reference | A100 PCIe 80 GB | A100 SXM 80 GB |

All formats of a tier run on **one pod**, so comparisons within a tier share identical
hardware, and each pod's host (GPU details, CPU, memory) is recorded with its results. The
pinned vLLM image is built for CUDA 13.0, so pods are created only on hosts supporting it.
Engine: **vLLM 0.30.0** (`vllm/vllm-openai:v0.30.0`), released 2026-09-22. Client: frugal-llm,
a pinned release, running on the same machine as the server so no network latency is measured.

**Each tier is completed in full before the next is started.** If the study stops early, the
completed tiers are reported, and incomplete ones are not.

## 4. What is run (one "cell" = one tier × one format)

Exactly `scripts/run_cell.sh`, identical on every tier:

1. Serve the pinned weights: `--max-model-len 4096 --gpu-memory-utilization 0.90`, defaults
   otherwise.
2. Latency: `frugal-llm bench -c 1,4,16,64 -n 50,100,200,400 --repeats 3 --max-tokens 128`.
3. Quality: `frugal-llm quality` on all 1,319 GSM8K and the 1,000-question MMLU set, greedy
   decoding, `max_tokens 1024`, **concurrency 16** for every run. BF16 is run **twice**; the
   second run is the noise floor.
4. Every result records the GPU, driver, vLLM version, weights revision, provider, rate and date.

Formats per tier: BF16, AWQ, GPTQ-Int4 and GPTQ-Int8 on tiers 1 and 2; BF16 and AWQ latency only
on the reference tier.

A **shakedown** precedes the study: one GPU-hour on tier 1 running every step at tiny size
(`SHAKEDOWN=1`), to catch load failures, timeouts and recording problems before the real runs.
Its results are not part of the study.

## 5. Analysis, fixed in advance

- **Latency and throughput:** as `frugal-llm bench` reports them: quantiles only, never pooled
  across concurrency levels; repeats shown as median and range.
- **Quality:** `frugal-llm compare` per tier and task, each quantised format against BF16 run
  *a*, with BF16 run *b* as the noise floor. Accuracy with Wilson intervals; paired difference
  with Newcombe intervals and exact McNemar tests, Holm-corrected across the formats of one
  tier and task; answer agreement and output divergence. Method: `docs/quality-method.md`.
- **Cost:** `frugal-llm cost` per tier, using the measured capacity and the rate actually
  paid, against APIs serving the same model, with prices read from the providers' own pages
  and dated on the day of analysis. Break-even traffic and utilisation, never a single ratio.
  Method: `docs/cost-method.md`.
- **The headline claim** for each format is the paired accuracy difference and its interval,
  stated with the comparison's resolution. A difference below resolution is reported as "no
  difference detected", never as "no loss".

## 6. Limitations, stated before any result

- One model (Qwen2.5-7B-Instruct), one engine (vLLM 0.30.0): results may not transfer to other
  models, sizes or engines.
- Quality covers arithmetic reasoning (GSM8K) and recalled knowledge (MMLU subset) only, not
  instruction following, long context, safety or writing. Both sets are public and may be in
  the model's training data; only the comparison against BF16 is meaningful.
- At these sample sizes, accuracy differences under roughly 3–4 points cannot be detected.
- Quality is measured at concurrency 16; all formats are compared under identical batching.
- Hardware from one provider at one point in time; prices change.
- Latency comes from closed-loop synthetic load; real traffic is burstier.
- Cost excludes idle time, engineering time, storage and egress, and rests on a stated
  utilisation assumption.
- No novelty is claimed: engine and quantisation comparisons exist. The contribution is the
  method, the affordable hardware, and quality measured alongside speed and cost.

## 7. Raw data

Every result file from the study (not the shakedown) is committed to this repository, so any
table or chart can be regenerated or checked with `report`, `compare` and `cost`.

## Changes after the first study result

1. **Run order** (2026-10-15, after tier 1). Tier 1 ran on the RTX 4090 substitute (the RTX
   3090 could not be placed on four attempts). The L4 for tier 2 then had no Secure Cloud stock,
   so the reference tier (latency only) was run while a script retried the L4. This changes
   the order of runs only, not any measurement; each tier still runs entirely on one pod.
2. **RTX 3090 tier added** (2026-10-15, after tier 2). With the other tiers complete, the
   originally planned RTX 3090 was retried once (Secure Cloud, no substitute, two-hour limit)
   and placed. It runs the same cells, pins and commit as tier 2, labelled `t1-rtx3090`. The
   RTX 4090 tier is kept, not replaced: both are reported, and no earlier result changes.
3. **Report wording and one chart fix** (2026-10-15). The server's `system_fingerprint` is
   now labelled "fingerprint", not "build": vLLM issues a new one at every launch, so it
   identifies a server process, not a software build. The break-even chart's self-hosted line
   could add an instance one sample early wherever traffic was an exact multiple of one
   instance's capacity (floating-point rounding; most visible as a spike at the final point);
   the instance count is now rounded before it is taken. Neither change alters any measurement
   or table value: regenerated summaries differ from the earlier ones only in the label.
4. **API comparator** (2026-10-16). Together AI still lists Qwen2.5-7B-Instruct-Turbo at
   USD 0.30 / 0.30 per 1M tokens but no longer serves it per token: requests are refused
   with `model_not_available` unless a dedicated (hourly) endpoint is rented. The only live
   per-token provider found was Phala via OpenRouter (USD 0.10 in / 0.20 out per 1M,
   2026-10-16; DeepInfra lists the model as deprecated). Phala via OpenRouter becomes the API
   comparison for cost; Together's listed price is reported only as a listed, unpurchasable
   price. The API's own quality was measured with the same task sets, settings (greedy,
   max_tokens 1024, concurrency 16) and statistics as the GPU tiers, in one run per task.
   OpenRouter does not report the upstream model's precision, and frugal-llm does not record
   which upstream served each request; Phala was the only provider listed, and served a
   test request.
