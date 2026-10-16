# frugal-llm

How fast, and how expensive, is it to serve *your* model on *your* hardware, and what do you
lose by serving a cheaper version of it?

frugal-llm is an open-source command-line tool that points at any OpenAI-compatible endpoint
(vLLM, SGLang, TGI, llama.cpp, Ollama or a commercial API) and measures latency, throughput,
cost per token, and the quality lost to quantisation, with stated uncertainty.

```bash
uvx frugal-llm bench --url http://localhost:8000/v1 --model <name>
```

## The first study

**[Results: what quantisation costs and saves on affordable GPUs](results.md)**: Qwen2.5-7B-Instruct
in four formats on an RTX 3090, RTX 4090, L4 and A100, with quality measured against BF16 and
cost compared with the only API still serving the model per token.

- No quantised format showed a detectable accuracy loss on GSM8K or MMLU.
- 4-bit weights were 2.1–2.9× faster for a single user.
- The consumer cards produced tokens at about half the cost of an L4 or A100.

## Method

Written before the first measurement, with every later change logged:

- [Study protocol](study-protocol.md): the question, model, hardware and analysis
- [Quality method](quality-method.md): task sets, decoding, scoring and statistics
- [Cost method](cost-method.md): break-even against per-token API prices

## Links

- Source and README: [github.com/Lekeboi/Frugal_LLM](https://github.com/Lekeboi/Frugal_LLM)
- Package: [pypi.org/project/frugal-llm](https://pypi.org/project/frugal-llm/)
- Raw data: [study/2026-10-qwen2.5-7b](https://github.com/Lekeboi/Frugal_LLM/tree/master/study/2026-10-qwen2.5-7b)
- Contributing your own results: [CONTRIBUTING.md](https://github.com/Lekeboi/Frugal_LLM/blob/master/CONTRIBUTING.md)

Apache-2.0.
