"""Stop the pod this runs on, using RunPod's pod-scoped key, so an unattended pod never keeps
a GPU billing. Stopping releases the GPU and keeps the pod's disk (and the results on it).

Uses only the standard library: the pod image is not ours to install into. Exits 0 when the
stop was accepted, 1 otherwise; the outcome is printed either way.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _post(url: str, key: str, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # RunPod's edge rejects urllib's default User-Agent (Cloudflare error 1010).
            "User-Agent": "frugal-llm-selfstop/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode(errors="replace")[:300]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:300]
    except OSError as exc:
        return 0, str(exc)


def main() -> int:
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        print("self-stop: RUNPOD_POD_ID or RUNPOD_API_KEY not set; cannot stop this pod")
        return 1
    status, text = _post(f"https://rest.runpod.io/v1/pods/{pod}/stop", key)
    print(f"self-stop: REST stop -> HTTP {status} {text}")
    if 200 <= status < 300:
        return 0
    # Fallback: the GraphQL API, which pod-scoped keys have historically been able to use.
    query = f'mutation {{ podStop(input: {{podId: "{pod}"}}) {{ id desiredStatus }} }}'
    mutation = {"query": query}
    status, text = _post("https://api.runpod.io/graphql", key, mutation)
    print(f"self-stop: GraphQL podStop -> HTTP {status} {text}")
    return 0 if 200 <= status < 300 and '"errors"' not in text else 1


if __name__ == "__main__":
    sys.exit(main())
