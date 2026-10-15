"""Quality runs: score a model on shipped task sets with greedy decoding.

The method is fixed in docs/quality-method.md; this module implements it and nothing
else. Quality runs are separate from latency runs: no timing here is ever reported as
a latency result, and output length is set for complete answers, not comparability.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import hashlib
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path

import httpx

from frugal_llm.bench import BenchConfig, classify_error, make_client
from frugal_llm.report import SCHEMA_STATUS

SCHEMA_VERSION = "quality-0.1"
TASK_VERSION = "v1"
TASKS = ("gsm8k", "mmlu")
DEFAULT_MAX_TOKENS = {"gsm8k": 1024, "mmlu": 1024}
# Greedy decoding, identical for every model and format (method note, section 3).
SAMPLING = {"temperature": 0, "top_p": 1, "seed": 0}
RETRYABLE = frozenset({429, 503})
MAX_BACKOFF_S = 60.0


class QualityAborted(Exception):
    """The endpoint cannot be scored; nothing meaningful can be written."""


# --- task sets ----------------------------------------------------------------------


@dataclass(frozen=True)
class TaskItem:
    id: str
    prompt: str
    answer: str


def render_gsm8k(record: dict) -> str:
    return (
        f"{record['question']}\n\n"
        "Solve the problem step by step. End with a final line of the form `Answer: <number>`."
    )


def render_mmlu(record: dict) -> str:
    options = "\n".join(
        f"{letter}. {text}" for letter, text in zip("ABCD", record["choices"], strict=True)
    )
    subject = record["subject"].replace("_", " ")
    return (
        f"The following is a multiple-choice question about {subject}.\n\n"
        f"{record['question']}\n\n{options}\n\n"
        "Answer with the letter of the correct option only."
    )


RENDER = {"gsm8k": render_gsm8k, "mmlu": render_mmlu}


def load_task(task: str) -> tuple[list[TaskItem], str]:
    """The shipped items, in their fixed order, and the SHA-256 of the file they came from.

    The hash is checked against the manifest: an edited task file is a different study.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")
    folder = resources.files("frugal_llm") / "tasks"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tasks"][task]
    data = (folder / entry["file"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"task file {entry['file']} does not match its manifest hash")
    records = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    render = RENDER[task]
    return [TaskItem(r["id"], render(r), r["answer"]) for r in records], digest


# --- answer extraction (fixed in advance; method note, section 3) --------------------

_NUMBER = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"
# Also accepts the fullwidth colon (U+FF1A), which Qwen models sometimes write.
_ANSWER_LINE = re.compile(r"answer\s*[:：]\s*\**\s*(" + _NUMBER + ")", re.IGNORECASE)  # noqa: RUF001
_ANY_NUMBER = re.compile(_NUMBER)


def normalise_number(text: str) -> Decimal | None:
    cleaned = text.replace(",", "").replace("$", "").replace(" ", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


# Models that reason state their result at the end, often as LaTeX \boxed{...}.
_BOXED_NUMBER = re.compile(r"\\boxed\{\s*(" + _NUMBER + r")\s*\}")


def extract_gsm8k(output: str) -> tuple[str | None, str]:
    """Number after the last 'Answer:'; else the last \\boxed{N}; else the last number."""
    raw, rule = None, "none"
    for name, found in (
        ("answer_line", _ANSWER_LINE.findall(output)),
        ("boxed", _BOXED_NUMBER.findall(output)),
        ("last_number", _ANY_NUMBER.findall(output)),
    ):
        if found:
            raw, rule = found[-1], name
            break
    value = normalise_number(raw) if raw is not None else None
    if value is None:
        return None, "none"
    return format(value.normalize(), "f"), rule


# A leading letter counts only alone or followed by punctuation: "C", "C.", "(C)", "**C**".
_LEADING_LETTER = re.compile(r"^\W*([ABCD])(?:\s*$|[.):*\]])")
_BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?([ABCD])(?![A-Za-z])")
# "the correct answer is: \[ \boxed{D} \]", "The correct option is C. True, False."
_ANSWER_PHRASE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:)?\s*[:\s\\\[\(\*$]*(?:\\boxed\{|\\text\{)?\s*"
    r"([ABCD])(?![A-Za-z])",
    re.IGNORECASE,
)
_STANDALONE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
# In longer replies a bare letter is usually part of the working, not the answer.
SHORT_REPLY_WORDS = 20


def extract_mmlu(output: str) -> tuple[str | None, str]:
    """In order: a leading letter; the last \\boxed{X}; the last 'answer/option is X';
    and only for short replies, the first standalone A-D. Otherwise no answer: a guess
    from a letter inside long reasoning would score noise."""
    text = output.strip()
    if match := _LEADING_LETTER.search(text):
        return match.group(1), "leading_letter"
    for rule, pattern in (("boxed", _BOXED_LETTER), ("answer_phrase", _ANSWER_PHRASE)):
        if found := pattern.findall(text):
            return found[-1].upper(), rule
    if len(text.split()) <= SHORT_REPLY_WORDS and (match := _STANDALONE.search(text)):
        return match.group(1), "short_reply_letter"
    return None, "none"


EXTRACT = {"gsm8k": extract_gsm8k, "mmlu": extract_mmlu}


def is_correct(task: str, extracted: str | None, answer: str) -> bool:
    if extracted is None:
        return False
    if task == "gsm8k":
        return normalise_number(extracted) == normalise_number(answer)
    return extracted == answer


# --- one item -------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemResult:
    id: str
    output: str | None
    extracted: str | None
    rule: str
    correct: bool
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int
    error: str | None = None


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """Wait the server asks for (seconds or an HTTP date), else exponential backoff."""
    header = response.headers.get("retry-after")
    if header:
        with contextlib.suppress(ValueError):
            return min(float(header), MAX_BACKOFF_S)
        with contextlib.suppress(TypeError, ValueError):
            when = email.utils.parsedate_to_datetime(header)
            return min(max((when - datetime.now(UTC)).total_seconds(), 0.0), MAX_BACKOFF_S)
    return min(2.0**attempt, MAX_BACKOFF_S)


async def ask(
    client: httpx.AsyncClient,
    *,
    task: str,
    item: TaskItem,
    model: str,
    max_tokens: int,
    max_retries: int,
    sleep: Callable[[float], object] = asyncio.sleep,
) -> ItemResult:
    """One non-streaming request. Rate limits (429) and 503s are retried politely, because
    a quality score is only valid if every item is answered; anything else is recorded."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": item.prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        **SAMPLING,
    }
    for attempt in itertools.count(1):
        try:
            response = await client.post("/chat/completions", json=body)
            if response.status_code in RETRYABLE and attempt <= max_retries:
                await sleep(_retry_after(response, attempt))
                continue
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            error = (
                classify_error(exc) if isinstance(exc, httpx.HTTPError) else "malformed_response"
            )
            return ItemResult(item.id, None, None, "none", False, None, None, None, attempt, error)
        choice = (data.get("choices") or [{}])[0]
        output = (choice.get("message") or {}).get("content") or ""
        if choice.get("finish_reason") == "length":
            # Runaway or cut-off output is never graded from its unfinished text.
            extracted, rule = None, "truncated"
        else:
            extracted, rule = EXTRACT[task](output)
        usage = data.get("usage") or {}
        return ItemResult(
            id=item.id,
            output=output,
            extracted=extracted,
            rule=rule,
            correct=is_correct(task, extracted, item.answer),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            attempts=attempt,
        )
    raise AssertionError("unreachable")


# --- a whole run ----------------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    url: str
    model: str
    api_key: str | None
    task: str
    limit: int | None
    concurrency: int
    max_tokens: int
    max_retries: int
    timeout: float
    label: str | None


def _client_config(cfg: QualityConfig) -> BenchConfig:
    # Reuse the latency harness's client (one connection per worker, per-phase timeouts).
    return BenchConfig(
        url=cfg.url, model=cfg.model, api_key=cfg.api_key, concurrency=[cfg.concurrency],
        requests=1, max_tokens=cfg.max_tokens, prompt_tokens=0, prompts_path=None, warmup=0,
        repeats=1, cost_per_hour=None, currency="GBP", timeout=cfg.timeout, out=Path("."),
        format="json", label=cfg.label,
    )  # fmt: skip


def run_header(
    cfg: QualityConfig,
    task_sha256: str,
    n_items: int,
    started_at: datetime,
    server: dict | None = None,
    meta: dict | None = None,
) -> dict:
    """Everything that defines the run. A resumed run must match it exactly."""
    from frugal_llm.report import environment_info

    return {
        "schema_version": SCHEMA_VERSION,
        "schema_status": SCHEMA_STATUS,
        "kind": "quality",
        "label": cfg.label,
        "started_at": started_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "url": cfg.url,
            "model": cfg.model,
            "task": cfg.task,
            "task_version": TASK_VERSION,
            "task_sha256": task_sha256,
            "items": n_items,
            "development_subset": cfg.limit is not None,
            "concurrency": cfg.concurrency,
            "max_tokens": cfg.max_tokens,
            **SAMPLING,
        },
        "environment": environment_info(server, meta),
    }


def development_subset(items: list[TaskItem], n: int) -> list[TaskItem]:
    """n items spread evenly through the task file, so every subject is represented.
    (The first n would be the first few subjects alphabetically.)"""
    if n >= len(items):
        return items
    return [items[i * len(items) // n] for i in range(n)]


@dataclass
class PreparedRun:
    items: list[TaskItem]
    header: dict
    done: dict[str, dict]
    partial_path: Path


def prepare_run(
    cfg: QualityConfig,
    partial_path: Path,
    *,
    resume: bool = False,
    server: dict | None = None,
    meta: dict | None = None,
) -> PreparedRun:
    """Load the task, and either start a new partial file or pick up an interrupted one.

    Items that failed with an error are not carried over, so a resume retries them.
    """
    items, digest = load_task(cfg.task)
    if cfg.limit is not None:
        items = development_subset(items, cfg.limit)
    header = run_header(cfg, digest, len(items), datetime.now(UTC), server, meta)
    done: dict[str, dict] = {}
    if resume:
        lines = partial_path.read_text(encoding="utf-8").splitlines()
        previous = json.loads(lines[0])
        if previous["config"] != header["config"]:
            raise QualityAborted("the partial file was made with different settings; start afresh")
        header = previous  # keep the original start time and environment
        done = {r["id"]: r for r in map(json.loads, lines[1:]) if not r.get("error")}
    else:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(header) + "\n", encoding="utf-8", newline="\n")
    return PreparedRun(items, header, done, partial_path)


async def run_quality(
    cfg: QualityConfig,
    run: PreparedRun,
    *,
    on_item: Callable[[ItemResult], None] | None = None,
) -> dict:
    """Score every remaining item, appending each result to the partial file as it
    completes, so an interrupted run costs at most the items in flight."""
    done = dict(run.done)
    todo = [item for item in run.items if item.id not in done]
    async with contextlib.AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(make_client(_client_config(cfg)))
            for _ in range(cfg.concurrency)
        ]
        if todo and not done:
            first = await ask(
                clients[0], task=cfg.task, item=todo[0], model=cfg.model,
                max_tokens=cfg.max_tokens, max_retries=cfg.max_retries,
            )  # fmt: skip
            if first.error:
                raise QualityAborted(f"the first request failed ({first.error})")
            _append(run.partial_path, first, on_item)
            done[first.id] = asdict(first)
            todo = todo[1:]

        queue = iter(todo)

        async def worker(client: httpx.AsyncClient) -> None:
            for item in queue:
                result = await ask(
                    client, task=cfg.task, item=item, model=cfg.model,
                    max_tokens=cfg.max_tokens, max_retries=cfg.max_retries,
                )  # fmt: skip
                _append(run.partial_path, result, on_item)
                done[item.id] = asdict(result)

        async with asyncio.TaskGroup() as tg:
            for client in clients:
                tg.create_task(worker(client))

    return {**run.header, "items": [done[item.id] for item in run.items]}


def _append(path: Path, result: ItemResult, on_item: Callable | None) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    if on_item:
        on_item(result)


def summarise_run(doc: dict) -> dict:
    items = doc["items"]
    scored = [i for i in items if not i["error"]]
    return {
        "items": len(items),
        "scored": len(scored),
        "errors": len(items) - len(scored),
        "correct": sum(i["correct"] for i in scored),
        "no_answer": sum(i["extracted"] is None for i in scored),
        "truncated": sum(i["finish_reason"] == "length" for i in scored),
        "retried": sum(i["attempts"] > 1 for i in items),
    }


def write_result(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
