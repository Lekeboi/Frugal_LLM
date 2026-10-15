"""Compare saved quality runs against an FP16 baseline. Pure functions over result docs.

Works only from saved results, so the analysis can be redone without re-running any
model. Method: docs/quality-method.md, sections 2 (divergence) and 5 (statistics).
"""

from __future__ import annotations

import dataclasses
import statistics
from dataclasses import dataclass

from frugal_llm.environment import summary_line
from frugal_llm.stats import (
    holm,
    mcnemar_exact,
    minimum_detectable_difference,
    paired_difference_ci,
    wilson,
)

# Settings that must match for two runs to be comparable at all.
MUST_MATCH = ("task", "task_version", "task_sha256", "items", "development_subset",
              "max_tokens", "temperature", "top_p", "seed")  # fmt: skip


# Weight formats a server may report for an unquantised baseline.
FULL_PRECISION = frozenset({"F32", "F16", "BF16", "FP32", "FP16"})


class NotComparable(ValueError):
    pass


def reported_weights(doc: dict) -> str | None:
    """The weight format the server itself reported for this run, if it reported one."""
    server = (doc.get("environment") or {}).get("server") or {}
    meta = (server.get("model_entry") or {}).get("meta")
    return meta.get("ftype") if isinstance(meta, dict) else None


def label_of(doc: dict) -> str:
    return doc.get("label") or doc["config"]["model"]


def check_comparable(baseline: dict, other: dict) -> list[str]:
    """Raise if the runs measured different things; return softer warnings otherwise."""
    for key in MUST_MATCH:
        if baseline["config"][key] != other["config"][key]:
            raise NotComparable(
                f"{label_of(other)!r} differs from the baseline in {key}: "
                f"{other['config'][key]!r} vs {baseline['config'][key]!r}"
            )
    if baseline["config"]["concurrency"] != other["config"]["concurrency"]:
        return [
            f"{label_of(other)} ran at concurrency {other['config']['concurrency']}, the "
            f"baseline at {baseline['config']['concurrency']}; batching can change greedy "
            "outputs, so divergence may be overstated."
        ]
    return []


def word_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over whitespace-separated words."""
    x, y = a.split(), b.split()
    # Shared prefix and suffix cost nothing; trimming them keeps this fast.
    while x and y and x[0] == y[0]:
        x, y = x[1:], y[1:]
    while x and y and x[-1] == y[-1]:
        x, y = x[:-1], y[:-1]
    previous = list(range(len(y) + 1))
    for i, wx in enumerate(x, 1):
        current = [i]
        for j, wy in enumerate(y, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (wx != wy)))
        previous = current
    return previous[-1]


def first_divergence(a: str, b: str) -> int | None:
    """Index of the first differing word, or None if the outputs are identical."""
    x, y = a.split(), b.split()
    for i, (wx, wy) in enumerate(zip(x, y, strict=False)):
        if wx != wy:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


@dataclass(frozen=True)
class Comparison:
    label: str
    n: int
    excluded: int
    both_right: int
    baseline_only: int  # b
    candidate_only: int  # c
    both_wrong: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    baseline_accuracy: float
    difference: float
    difference_ci: tuple[float, float]
    p_value: float
    answer_agreement: float
    exact_match: float
    first_divergence_median: float | None
    edit_distance_median: float
    no_answer: int
    truncated: int
    detectable: float | None
    p_adjusted: float | None = None


def compare_pair(baseline: dict, candidate: dict) -> Comparison:
    base = {i["id"]: i for i in baseline["items"] if not i["error"]}
    cand = {i["id"]: i for i in candidate["items"] if not i["error"]}
    ids = [i["id"] for i in baseline["items"] if i["id"] in base and i["id"] in cand]
    n = len(ids)
    if n == 0:
        raise NotComparable(f"{label_of(candidate)!r} shares no scored items with the baseline")
    pairs = [(base[i], cand[i]) for i in ids]
    a = sum(bi["correct"] and ci["correct"] for bi, ci in pairs)
    b = sum(bi["correct"] and not ci["correct"] for bi, ci in pairs)
    c = sum(ci["correct"] and not bi["correct"] for bi, ci in pairs)
    d = n - a - b - c
    divergences = [first_divergence(bi["output"], ci["output"]) for bi, ci in pairs]
    edit = [
        word_edit_distance(bi["output"], ci["output"]) / max(len(bi["output"].split()), 1)
        for bi, ci in pairs
    ]
    differing = [x for x in divergences if x is not None]
    return Comparison(
        label=label_of(candidate),
        n=n,
        excluded=len(baseline["items"]) - n,
        both_right=a,
        baseline_only=b,
        candidate_only=c,
        both_wrong=d,
        accuracy=(a + c) / n,
        accuracy_ci=wilson(a + c, n),
        baseline_accuracy=(a + b) / n,
        difference=(c - b) / n,
        difference_ci=paired_difference_ci(a, b, c, d),
        p_value=mcnemar_exact(b, c),
        answer_agreement=sum(bi["extracted"] == ci["extracted"] for bi, ci in pairs) / n,
        exact_match=sum(bi["output"] == ci["output"] for bi, ci in pairs) / n,
        first_divergence_median=statistics.median(differing) if differing else None,
        edit_distance_median=statistics.median(edit),
        no_answer=sum(ci["extracted"] is None for _, ci in pairs),
        truncated=sum(ci["finish_reason"] == "length" for _, ci in pairs),
        detectable=minimum_detectable_difference(n, (b + c) / n),
    )


def compare_runs(
    baseline: dict, candidates: list[dict], noise_floor: dict | None = None
) -> tuple[list[Comparison], Comparison | None, list[str]]:
    """Every candidate against the baseline, Holm-corrected as one family of claims.

    The noise floor (a second baseline run) is a control, not a claim, so it is reported
    but left out of the multiple-comparison family.
    """
    warnings = []
    weights = reported_weights(baseline)
    if weights and weights.upper() not in FULL_PRECISION:
        warnings.append(
            f"the baseline's server reported its weights as {weights}, not full precision. "
            "Every difference below is measured against a quantised model."
        )
    for other in [*candidates, *([noise_floor] if noise_floor else [])]:
        warnings += check_comparable(baseline, other)
    results = [compare_pair(baseline, c) for c in candidates]
    adjusted = holm([r.p_value for r in results])
    results = [_with_adjusted(r, p) for r, p in zip(results, adjusted, strict=True)]
    floor = compare_pair(baseline, noise_floor) if noise_floor else None
    return results, floor, warnings


def _with_adjusted(r: Comparison, p: float) -> Comparison:
    return dataclasses.replace(r, p_adjusted=p)


# --- rendering ----------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _pts(x: float) -> str:
    return f"{x * 100:+.1f}"


def verdict(r: Comparison) -> str:
    lo, hi = r.difference_ci
    ci = f"95% CI {_pts(lo)} to {_pts(hi)} points"
    if r.p_adjusted is not None and r.p_adjusted < 0.05:
        direction = "lower" if r.difference < 0 else "higher"
        return (
            f"**{r.label}**: accuracy {abs(r.difference) * 100:.1f} points {direction} than the "
            f"baseline ({ci}; p = {r.p_adjusted:.3g}, Holm-corrected)."
        )
    if r.detectable is None:
        resolution = "the two runs never disagreed on correctness, so no resolution can be stated"
    else:
        resolution = (
            f"differences under about {r.detectable * 100:.1f} points are below this "
            "comparison's resolution"
        )
    return f"**{r.label}**: no difference detected ({ci}); {resolution}."


def render_markdown(
    baseline: dict,
    results: list[Comparison],
    floor: Comparison | None,
    warnings: list[str],
    extra_docs: list[dict] | None = None,
) -> str:
    """``extra_docs``: the candidate and noise-floor runs, for their environment lines."""
    extra_docs = extra_docs or []
    cfg = baseline["config"]
    rows = [r for r in [floor, *results] if r is not None]
    lines = [
        f"## frugal-llm quality comparison — {cfg['task']} {cfg['task_version']} "
        f"({cfg['items']} items)",
        "",
    ]
    if cfg["development_subset"]:
        lines += [
            "> **Development subset.** These items are a slice of the task set used while "
            "building and testing. Not a publishable result.",
            "",
        ]
    lines += [
        f"Baseline: **{label_of(baseline)}** ({cfg['model']} @ {cfg['url']}). Greedy decoding "
        f"(temperature {cfg['temperature']}, top_p {cfg['top_p']}, seed {cfg['seed']}), "
        f"max_tokens {cfg['max_tokens']}, concurrency {cfg['concurrency']}.",
        "",
        "| Run | Accuracy (95% CI) | vs baseline, points (95% CI) | Only baseline right / only this right | p (Holm) |",  # noqa: E501
        "|---|---:|---:|---:|---:|",
    ]
    base_ok = rows[0].baseline_accuracy if rows else 0.0
    n = rows[0].n if rows else 0
    lo, hi = wilson(round(base_ok * n), n) if n else (0.0, 1.0)
    lines.append(
        f"| {label_of(baseline)} (baseline) | {_pct(base_ok)} ({_pct(lo)} to {_pct(hi)}) | | | |"
    )
    for r in rows:
        name = f"{r.label} (noise floor: baseline rerun)" if r is floor else r.label
        dlo, dhi = r.difference_ci
        p = "control" if r is floor else f"{r.p_adjusted:.3g}"
        lines.append(
            f"| {name} | {_pct(r.accuracy)} ({_pct(r.accuracy_ci[0])} to {_pct(r.accuracy_ci[1])}) "
            f"| {_pts(r.difference)} ({_pts(dlo)} to {_pts(dhi)}) "
            f"| {r.baseline_only} / {r.candidate_only} | {p} |"
        )
    lines += [
        "",
        "Divergence from the baseline's outputs (different is not the same as worse):",
        "",
        "| Run | Same final answer | Identical output | Median first divergence (words) | Median word edit distance | No answer | Truncated |",  # noqa: E501
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        name = f"{r.label} (noise floor)" if r is floor else r.label
        first = "—" if r.first_divergence_median is None else f"{r.first_divergence_median:.0f}"
        lines.append(
            f"| {name} | {_pct(r.answer_agreement)} | {_pct(r.exact_match)} | {first} "
            f"| {r.edit_distance_median:.2f} | {r.no_answer} | {r.truncated} |"
        )
    runs = [baseline, *extra_docs]
    environments = [(label_of(d), summary_line(d.get("environment") or {})) for d in runs]
    if any(env for _, env in environments):
        lines += ["", "Environment, as each server reported it:", ""]
        lines += [f"- {name}: {env or 'nothing reported'}" for name, env in environments]
    lines += ["", *(f"- {verdict(r)}" for r in results)]
    if floor:
        lines.append(
            f"- Noise floor: two baseline runs agreed on {_pct(floor.answer_agreement)} of "
            f"final answers and produced identical output for {_pct(floor.exact_match)} of items. "
            "Divergence above this is attributable to the format, not to run-to-run noise."
        )
    else:
        lines.append(
            "- No noise floor supplied: without a second baseline run, divergence cannot be "
            "separated from run-to-run nondeterminism."
        )
    excluded = max((r.excluded for r in rows), default=0)
    if excluded:
        lines.append(f"- {excluded} items were excluded because a run returned an error for them.")
    lines += [f"- WARNING: {w}" for w in warnings]
    lines += [
        "- Scored by fixed rules in docs/quality-method.md. GSM8K and MMLU measure arithmetic "
        "reasoning and recalled knowledge only, not general capability; both are public and "
        "may be in training data, so absolute accuracy is not evidence of general ability.",
        "",
    ]
    return "\n".join(lines)
