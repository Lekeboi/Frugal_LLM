"""Create one RunPod pod from a priority list of GPU types, retrying until one can be placed.

Scarce GPUs show as "in stock" yet are often taken in the seconds between a stock read and a
create. This tries each candidate in order, every --interval seconds, until one is created or
--timeout passes, then stops: at most ONE pod is ever created.

    python scripts/runpod_create_retry.py body.json --gpu "NVIDIA L4" --rate 0.59 --timeout 10800

body.json is the create-pod body (RunPod API v2, POST /v2/pods) without `gpu.id`; each
--gpu fills it in turn, and --rate sets the matching RATE env var recorded with the results.
The API key comes from RUNPOD_API_KEY and is never printed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2/pods"


def create(body: dict, key: str) -> tuple[int, dict | str]:
    request = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:300]
    except OSError as exc:
        return 0, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("body", help="create-pod body as JSON (without gpu.id)")
    parser.add_argument(
        "--gpu", action="append", required=True, help="GPU type id, in priority order"
    )
    parser.add_argument(
        "--rate", action="append", required=True, help="USD/hour for each --gpu, same order"
    )
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=10800)
    args = parser.parse_args()
    if len(args.gpu) != len(args.rate):
        parser.error("give one --rate per --gpu")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("RUNPOD_API_KEY is not set", flush=True)
        return 2
    with open(args.body, encoding="utf-8") as f:
        template = json.load(f)

    deadline = time.monotonic() + args.timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        for gpu, rate in zip(args.gpu, args.rate, strict=True):
            body = copy.deepcopy(template)
            body["gpu"]["id"] = gpu
            body["env"]["RATE"] = rate
            status, result = create(body, key)
            stamp = time.strftime("%H:%M:%S", time.gmtime())
            if 200 <= status < 300 and isinstance(result, dict) and result.get("id"):
                print(f"[{stamp}] CREATED pod {result['id']} on {gpu} at ${rate}/h "
                      f"(attempt {attempt})", flush=True)  # fmt: skip
                return 0
            capacity = isinstance(result, str) and "no longer any instances" in result
            outcome = "unavailable" if capacity else f"HTTP {status}: {result}"
            print(f"[{stamp}] attempt {attempt}: {gpu} {outcome}", flush=True)
            if not capacity:  # anything but a stock miss is a real error: stop, don't spin
                return 1
        time.sleep(args.interval)
    print("TIMEOUT: no pod created", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
