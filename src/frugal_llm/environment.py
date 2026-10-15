"""Capture everything about where a result came from (scope Workstream D).

An unrecorded variable is a result nobody can reproduce. Three sources:

- the client machine, read locally;
- the server, through the OpenAI-compatible API only: the ``/models`` entry for the
  model, the ``Server`` header and the ``system_fingerprint`` a completion returns.
  These are recorded verbatim, never parsed per engine: whatever llama.cpp, vLLM or
  Ollama choose to report is kept as they reported it;
- what the API cannot reveal (GPU, driver, provider, quantisation method, the date of
  the hourly rate), supplied by the user as ``--meta key=value``.
"""

from __future__ import annotations

import os
import platform
import sys

import httpx

from frugal_llm import __version__


def client_info() -> dict:
    return {
        "frugal_llm_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "executable_is_frozen": getattr(sys, "frozen", False),
    }


def parse_meta(pairs: list[str] | None) -> dict[str, str]:
    """``["gpu=RTX 3060", "provider=runpod"]`` -> ``{"gpu": "RTX 3060", ...}``."""
    meta: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--meta expects key=value, got {pair!r}")
        meta[key.strip()] = value.strip()
    return meta


async def server_info(client: httpx.AsyncClient, model: str) -> dict:
    """What the server says about itself, verbatim. Never fails a run: an endpoint that
    lists no models or returns no fingerprint is recorded as such."""
    info: dict = {"server_header": None, "model_entry": None, "models_listed": None,
                  "system_fingerprint": None}  # fmt: skip
    try:
        response = await client.get("/models")
        info["server_header"] = response.headers.get("server")
        if response.status_code == 200:
            body = response.json()
            # OpenAI wraps the list as {"data": [...]}; some providers (Together) return it bare.
            data = body.get("data") if isinstance(body, dict) else body
            data = [m for m in data or [] if isinstance(m, dict)] if isinstance(data, list) else []
            info["models_listed"] = [m.get("id") for m in data]
            info["model_entry"] = next((m for m in data if m.get("id") == model), None)
        else:
            info["models_error"] = f"http_{response.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        info["models_error"] = type(exc).__name__
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return info  # nothing is listening; the preflight will say so plainly
    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": model,
                "max_tokens": 1,
                "stream": False,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        info["server_header"] = info["server_header"] or response.headers.get("server")
        if response.status_code == 200:
            info["system_fingerprint"] = response.json().get("system_fingerprint")
    except (httpx.HTTPError, ValueError):
        pass
    return info


def summary_line(environment: dict) -> str | None:
    """One readable line for reports: server, fingerprint, precision as reported, user meta."""
    server = environment.get("server") or {}
    entry = server.get("model_entry") or {}
    meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
    parts = []
    if server.get("server_header"):
        parts.append(f"server {server['server_header']}")
    if server.get("system_fingerprint"):
        parts.append(f"fingerprint {server['system_fingerprint']}")
    if meta.get("ftype"):
        parts.append(f"weights {meta['ftype']} (as reported)")
    parts += [f"{k}={v}" for k, v in (environment.get("meta") or {}).items()]
    return ", ".join(parts) or None
