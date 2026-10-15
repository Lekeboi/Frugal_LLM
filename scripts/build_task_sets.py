"""Build the versioned quality task sets shipped in src/frugal_llm/tasks/.

Run once per task-set version; the output files are committed and shipped with the
package, so every user runs exactly the same items. Rerunning this script must
reproduce byte-identical files (checked against the hashes in manifest.json).

    uv run --with pyarrow python scripts/build_task_sets.py

(pyarrow reads MMLU's parquet file; it is needed only here, not by the tool.)

Method: docs/quality-method.md, sections 2 and 6.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "src" / "frugal_llm" / "tasks"
VERSION = "v1"

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
MMLU_URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
MMLU_TOTAL = 14042
MMLU_SUBSET = 1000
SEED = f"frugal-llm-mmlu-{VERSION}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    path.write_bytes(data)
    return sha256(data)


def gsm8k(client: httpx.Client) -> tuple[list[dict], str]:
    raw = client.get(GSM8K_URL).raise_for_status().content
    records = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        item = json.loads(line)
        match = re.search(r"####\s*(.+?)\s*$", item["answer"])
        if not match:
            raise ValueError(f"GSM8K test item {i}: no '#### answer' line")
        answer = match.group(1).replace(",", "")
        float(answer)  # every GSM8K answer is a plain number; fail loudly otherwise
        records.append(
            {"id": f"gsm8k-test-{i:04d}", "question": item["question"], "answer": answer}
        )
    return records, sha256(raw)


def mmlu_rows(client: httpx.Client) -> tuple[list[dict], str]:
    import io

    import pyarrow.parquet as pq

    raw = client.get(MMLU_URL).raise_for_status().content
    rows = pq.read_table(io.BytesIO(raw)).to_pylist()
    if len(rows) != MMLU_TOTAL:
        raise ValueError(f"expected {MMLU_TOTAL} MMLU test rows, got {len(rows)}")
    return rows, sha256(raw)


def mmlu_subset(rows: list[dict]) -> list[dict]:
    """Stratified: every subject gets floor or ceil of an equal share, chosen by seed."""
    by_subject: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_subject.setdefault(row["subject"], []).append(i)
    subjects = sorted(by_subject)
    base, extra = divmod(MMLU_SUBSET, len(subjects))
    rng = random.Random(SEED)
    gets_extra = set(rng.sample(subjects, extra))
    chosen = []
    for subject in subjects:
        quota = base + (subject in gets_extra)
        chosen += sorted(rng.sample(by_subject[subject], quota))
    return [
        {
            "id": f"mmlu-test-{i:05d}",
            "subject": rows[i]["subject"],
            "question": rows[i]["question"],
            "choices": rows[i]["choices"],
            "answer": "ABCD"[rows[i]["answer"]],
        }
        for i in chosen
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        gsm_records, gsm_source = gsm8k(client)
        rows, mmlu_source = mmlu_rows(client)
    mmlu_records = mmlu_subset(rows)

    manifest = {
        "version": VERSION,
        "tasks": {
            "gsm8k": {
                "file": f"gsm8k-{VERSION}.jsonl",
                "items": len(gsm_records),
                "sha256": write_jsonl(OUT / f"gsm8k-{VERSION}.jsonl", gsm_records),
                "source": GSM8K_URL,
                "source_sha256": gsm_source,
                "license": "MIT, Copyright (c) 2021 OpenAI",
            },
            "mmlu": {
                "file": f"mmlu-{VERSION}.jsonl",
                "items": len(mmlu_records),
                "sha256": write_jsonl(OUT / f"mmlu-{VERSION}.jsonl", mmlu_records),
                "source": MMLU_URL,
                "source_sha256": mmlu_source,
                "subset": f"{MMLU_SUBSET} of {MMLU_TOTAL}, stratified by subject, seed {SEED!r}",
                "license": "MIT, Copyright (c) 2020 Dan Hendrycks",
            },
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                       newline="\n")  # fmt: skip
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
