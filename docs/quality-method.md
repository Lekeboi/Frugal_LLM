# Quality measurement: method note

Status: settled, written before any quality code or results. Version 0.1.

This note fixes how frugal-llm will measure what a model loses when it is made cheaper
(quantised). It is written before any results exist, so the method can't be bent to fit them.
Anything that changes after a quality result has been produced must be listed under
"Changes since first result".

## 1. The question

**How much does a model's output change, and how much worse does it get on checkable tasks,
when it is served in a cheaper format instead of FP16?**

This is a relative question. It does not ask how good the model is. Every quality number is
a comparison against an FP16 baseline of the same model, measured with:

- the same prompts, in the same order;
- the same decoding settings;
- the same serving engine and version;
- the same hardware, in the same session, with the same harness.

The baseline is always measured, never taken from a model card or paper.

## 2. What is measured

Two methods, from the scope document's three. The third, an LLM judge on open-ended
prompts, is deferred: it is the weakest evidence and the first thing to drop.

### Method 1: checkable task sets (the backbone)

Tasks whose answers can be checked mechanically, so no human or judge model is involved.

| Set | Source | Licence | Items | Scored by |
|---|---|---|---|---|
| Arithmetic reasoning | GSM8K test split | MIT | all 1,319 | exact numeric match of the final answer |
| Multiple-choice knowledge | MMLU test split, fixed stratified subset | MIT | 1,000 (target; see §6) | exact match of the chosen letter A–D |

Why these two:
- Both are MIT-licensed, so the exact items can ship inside the Apache-2.0 package and anyone
  can rerun them. ARC was considered and rejected: its CC-BY-SA licence is awkward to ship.
- They stress different things: GSM8K needs multi-step reasoning and long generation, while
  MMLU tests recalled knowledge with a one-letter answer.
- Code tasks (HumanEval, MBPP) are **deferred**. Scoring them means executing model-written
  code, which needs a proper sandbox; doing that badly is a security risk on a user's machine.

What this does not measure, stated plainly: general capability, long-context behaviour,
instruction following, safety, multilingual quality and creative writing. A format that keeps
GSM8K and MMLU scores intact may still have degraded on any of these.

### Method 2: divergence from the FP16 baseline (the complement)

The same prompts with greedy decoding: how often, and how much, does the cheaper format's
output differ from FP16's? This needs no ground truth and answers "what do you lose" most
directly. Measured on the same items as Method 1 (no separate prompt set):

| Metric | Definition |
|---|---|
| Answer agreement | Share of items where the extracted final answer equals FP16's, right or wrong |
| Exact output match | Share of items where the whole output text is identical to FP16's |
| First divergence | Median word position where the output first differs from FP16's |
| Word edit distance | Word-level edit distance to FP16's output, divided by FP16's length |

All four work on text, split on whitespace. Token IDs are not used, because the
OpenAI-compatible API does not reliably return them and tokenisers differ between models.

**Different is not the same as worse.** A quantised model can diverge from FP16 and still be
correct, or diverge and be more correct. Divergence is reported next to accuracy and never
as a quality score by itself.

## 3. How a quality run works

Quality runs are a separate run type from latency runs, with their own command and result file.
Latency runs cap output length to keep throughput comparable; that cap would cut answers off and
corrupt the scores.

Request settings, identical for every model and format:

| Setting | Value | Why |
|---|---|---|
| `temperature` | 0 | Greedy decoding; sampling noise would swamp the quantisation effect |
| `top_p` | 1 | No truncation of the distribution |
| `seed` | 0 | Sent for servers that honour it; ignored harmlessly by those that don't |
| `max_tokens` | GSM8K 512, MMLU 32 | Long enough not to truncate (see the truncation rule below) |
| System prompt | none | The model's own chat template only |
| Shots | zero-shot | No examples in the prompt, so the prompt set is all there is |

Prompt templates, fixed and versioned with the task set:

- **GSM8K:** the question, then: *"Solve the problem step by step. End with a final line of the
  form `Answer: <number>`."*
- **MMLU:** the subject, question and four options labelled A–D, then:
  *"Answer with the letter of the correct option only."*

Answer extraction, fixed in advance:

- **GSM8K:** the number after the last `Answer:` (case-insensitive, and also with a
  fullwidth colon); if that's absent, the last number in the output. Commas, currency signs
  and trailing `.0` are removed and values compared as exact decimals. Which rule was used is
  recorded per item.
- **MMLU:** in this order: a letter A–D at the start of the output, alone or followed by
  punctuation (`C`, `C.`, `(C)`, `**C**`); else the letter in a phrase like "the answer is C";
  else the first standalone letter A–D. The order stops a reply such as "A good answer is C"
  being read as A. Which rule was used is recorded per item.
- An output with no extractable answer scores as wrong, and the count of such items is reported
  separately, per format. A format that stops producing parseable answers has degraded, and the
  reader should be able to see that this is how.

Truncation rule: if more than 1% of FP16 baseline outputs hit `max_tokens`, the limit is too low.
It is raised and **every** format is rerun at the new limit. The truncation rate is reported for
every format.

Everything is recorded per item: the full output text, the extracted answer, which extraction
rule applied, `finish_reason`, and token counts. The task set's file is identified by its
SHA-256 hash in every result, so a reader can confirm exactly which items were run.

## 4. Controls

**Noise floor.** Greedy decoding is not perfectly deterministic: GPU kernels and request
batching can change the numerics between runs. So the FP16 baseline is run **twice**, and the
divergence between the two FP16 runs is reported as the noise floor. Divergence of a cheaper
format is only meaningful above that floor. Both FP16 runs and every quantised run use the same
concurrency, because batching is one of the noise sources.

**Prompt caching.** Quality runs deliberately reuse identical prompts across formats, so server
prompt caches will hit. Caching should change speed, not output; any effect it does have on the
numbers shows up in the noise floor. Cache use is recorded per request, as in latency runs. No
timing from a quality run is ever reported as a latency result.

**Contamination.** GSM8K and MMLU are public and probably appear in the training data of
current models, which inflates absolute scores. The comparison is still fair, because FP16 and
the cheaper formats are the same weights and saw the same data, but absolute accuracies are not
evidence of general ability, and the write-up must say so.

## 5. Statistics

Comparisons are **paired**: both formats answer the same items, so they are compared item by
item, not as two separate accuracies.

- **Accuracy:** reported with a 95% Wilson score interval.
- **Difference against FP16:** the counts of items only FP16 got right (b) and only the cheaper
  format got right (c) are always shown. The difference (c − b)/n is reported with a 95%
  confidence interval (Newcombe's method for paired proportions), and its significance with an
  exact McNemar test.
- **Several formats:** when more than one format is compared against the same baseline, p-values
  are Holm-corrected, and every format tested is reported, not only the interesting ones.

**What the study can and cannot detect.** These sample sizes give 80% power at 5% significance:

| Accuracy drop to detect | If the formats disagree on 5% of items | 10% | 20% |
|---|---|---|---|
| 2 points | 979 items | 1,960 | 3,923 |
| 3 points | 434 | 870 | 1,742 |
| 5 points | — | 312 | 626 |
| 10 points | — | — | 155 |

With GSM8K's 1,319 items, a drop of about 3 points is detectable if the formats disagree on
10% of items or fewer. **Drops of 1–2 points cannot be told apart from noise at these sizes.**
The report will say "no difference detected; differences under about 3 points are below this
study's resolution", never "no quality loss".

## 6. Decisions

Settled on 2026-10-15, before any quality code or results:

1. **MMLU subset: 1,000 items.** It keeps a run affordable on a laptop and matches GSM8K's
   resolution. The subset is stratified (a fixed number of items per subject), chosen once with
   a fixed seed, and shipped as a file; the full 14,042 items are not used.
2. **Development runs use the first 100 items** of a task set, labelled as a development subset
   in the result file and never published as a result. A full GSM8K run on a laptop CPU takes
   hours, because every answer generates about 250 tokens at about 15 tokens/s.
3. **The study model** is chosen in the measurement-matrix workstream (B), not here. The method
   is model-agnostic; development uses Qwen2.5-0.5B-Instruct, which has an FP16 GGUF file and
   eight quantised formats for llama.cpp, so the whole FP16-versus-quantised pipeline can be
   built and tested locally at no cost.

## 7. How this fits the tool

The design rule still holds: only the OpenAI-compatible API, no engine-specific code. A quality
run is a chat-completions request without streaming timing, and the baseline is just another run
against a different endpoint or model name. The intended interface, to be confirmed when built:

```bash
frugal-llm quality --url URL --model MODEL --task gsm8k --label fp16-run1
frugal-llm compare results/fp16-run1.json results/q4_k_m.json --noise-floor results/fp16-run2.json
```

`compare` works only on saved results, so the analysis can be redone without re-running any
model, which matters once runs cost money.

## Changes since first result

None yet.
