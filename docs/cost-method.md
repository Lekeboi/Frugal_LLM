# Cost comparison: method note

Version 0.1. How `frugal-llm cost` compares self-hosting with a commercial API, and why it
answers with a break-even point rather than a ratio.

## The trap this avoids

A rented GPU costs the same per hour whether it serves a million tokens or none. So any
single figure like "self-hosting is 5× cheaper" is really a statement about an assumed load,
usually a flatteringly high one. The honest answer to "is self-hosting worth it for me?" is:
**above what sustained traffic does it become cheaper?** That is what this tool reports.

## What is compared

**Whole requests, of the shape that was measured.** A benchmark measures, at each
concurrency level, how many requests per second the hardware completes (its capacity) and
how many tokens go in and come out per request, as counted by the server.

- **API cost of one request** = prompt tokens × input price + output tokens × output price.
- **Self-hosted cost of one request at 100% use** = hourly rate ÷ 3,600 ÷ capacity.

Self-hosting gets no separate input and output price, because none exists to measure: one
GPU-hour processes prompts and generates replies together, and splitting that hour between
them would be an invented allocation. Comparing whole requests prices the input side fully,
on both sides, without inventing anything. Token counts are means, not quantiles: cost is
additive, so the average request is what is billed.

If the server did not report prompt-token counts, no API comparison is made for that
level. Pricing only the output would make self-hosting look cheaper than it is.

## Break-even

One instance costs the hourly rate H around the clock; the API costs r × 3,600 × A per hour
at traffic r requests per second, where A is its price per request. They are equal at

    r* = H / (3,600 × A)

reported as requests per day and as **break-even utilisation** r* ÷ capacity: the share of
the instance's measured capacity that must be used, around the clock, for it to cost no more
than the API. Above 100%, the hardware costs more per request than the API even when fully
busy, and the report says so.

`--curve-json` writes the data behind a break-even chart: daily cost of each option from zero
traffic to three instances' capacity. Self-hosting is a step function, because carrying more
traffic means adding whole instances.

## Inputs, always sourced and dated

- **Hourly rate:** `--hourly-rate`, with a required `--rate-source` (provider and machine)
  and `--rate-date`. Spot and on-demand prices move.
- **API prices:** `--api 'name=…,in=…,out=…,source=…,date=…'`, per million tokens, entered
  by the user. They are never built into the tool, where they would be stale within weeks.
- All prices are taken as entered, in one currency; nothing is converted.

## Stated in every report

- Utilisation is the assumption every figure rests on. Costs at utilisation u are 1/u times
  the 100% figures.
- Capacity is what was measured at each concurrency level. Higher concurrency might raise it,
  at a cost in latency, which is shown next to it.
- Not included: idle time between workloads, engineering time, storage, egress, and the
  quality difference between the self-hosted model and the API's model. That last one is
  what `frugal-llm quality` exists to measure, and a cost comparison between models of
  different quality is only half an answer.
- Stalled runs (the client froze, e.g. the machine slept) are left out, as in latency reports.
