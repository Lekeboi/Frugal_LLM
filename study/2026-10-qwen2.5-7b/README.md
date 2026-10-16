# Raw data: Qwen2.5-7B-Instruct on affordable GPUs (October 2026)

Every file the study produced, unedited. The write-up is [docs/results.md](../../docs/results.md);
the method, fixed before the first measurement, is
[docs/study-protocol.md](../../docs/study-protocol.md).

| Folder | What ran | Where |
|---|---|---|
| `t1-rtx3090/` | BF16 (twice), AWQ, GPTQ-Int4, GPTQ-Int8: speed and quality | RTX 3090, RunPod Secure, EU-CZ-1, USD 0.50/h |
| `t1-rtx4090/` | the same | RTX 4090, RunPod Secure, USD 0.89/h |
| `t2-l4/` | the same | L4, RunPod Secure, EU-RO-1, USD 0.59/h |
| `ref-a100/` | BF16 and AWQ, speed only (reference) | A100 80GB PCIe, RunPod Secure, USD 1.79/h |
| `api-openrouter-phala/` | quality only | Phala via OpenRouter, USD 0.10 / 0.20 per 1M tokens |

In each GPU folder:

- `*-<tier>-<format>.json`: `bench` results, every request of every repeat at concurrency
  1, 4, 16 and 64, with the server's own description of itself and the environment.
- `*-quality-<task>-<tier>-<format>.json`: `quality` results, every item's full output, the
  extracted answer and the rule that extracted it. `bf16-a` is the baseline, `bf16-b` its
  rerun (the noise floor).
- `*.md`: the report `bench` printed at the time.
- `host-info.txt`: `nvidia-smi -q` and the host's CPU and memory, recorded before the run.
- `cell-<format>.log` and `vllm-<format>.log`: the run's own log and the server's.

Check any number with the tool itself, without rerunning anything:

```bash
uvx frugal-llm report t2-l4/*Z-t2-l4-awq.json
uvx frugal-llm compare t2-l4/*-quality-gsm8k-t2-l4-bf16-a.json t2-l4/*-quality-gsm8k-t2-l4-awq.json \
    --noise-floor t2-l4/*-quality-gsm8k-t2-l4-bf16-b.json
```

These files were written by frugal-llm 0.2.1 and say `"schema_status": "provisional"`; the
format is identical to the stable schema 0.1 declared in 0.3.0.
