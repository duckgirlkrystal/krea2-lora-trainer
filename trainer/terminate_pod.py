#!/usr/bin/env python3
"""
Rationale (verified live, Aug 2026): this exists because the two shutdown
mechanisms that were supposed to be free both turned out not to work.

  1. RunPod's server-side `terminateAfter` did NOT fire. A pod created with
     a deadline 10 minutes out was still RUNNING 12 minutes past it.
  2. `runpodctl` is preinstalled on the pod but is NOT authenticated:
     "Runpod config file not found ... API key not found". There is also no
     `runpodctl pod delete` subcommand - the correct form is
     `runpodctl remove pod <id>`.
  3. Neither RUNPOD_POD_ID nor RUNPOD_API_KEY is present in the pod's
     environment, in any shell, interactive or not.
  4. urllib's default User-Agent ("Python-urllib/3.x") is rejected by
     RunPod's edge with a 403 BEFORE the key is ever checked. Same key, same
     request, same machine, only the UA changed: 403 vs 200. This silently
     killed both HTTP routes on the first live run and left runpodctl as the
     only thing standing. Hence USER_AGENT below - do not remove it.

So the pod cannot identify itself or authenticate itself. Both the id and
the key have to be handed to it by the client. Without this, the only thing
standing between the user and an open-ended GPU bill is their laptop staying
awake - which is not acceptable for people who cannot read a stack trace.

What it does:
  Terminates this pod, tries every available route, and verifies the pod is
  actually gone rather than trusting an HTTP 200.

Usage:
  python3 terminate_pod.py <pod_id> <api_key>

Maintenance: stdlib only (urllib), because it must work before, during and
after any pip failure. Do not add dependencies.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_V2 = "https://api.runpod.io/v2"
GONE_STATUSES = {"TERMINATED", "EXITED"}
# Load-bearing. See point 4 in the module docstring: the default urllib UA
# gets a 403 from RunPod's edge before authentication happens.
USER_AGENT = "curl/8.0"


def log(message: str) -> None:
    print(f"TERMINATE: {message}", flush=True)


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def _post_graphql(api_key: str, query: str, timeout: int = 30) -> dict:
    payload = json.dumps({"query": query}).encode()
    request = urllib.request.Request(
        GRAPHQL_URL, data=payload, method="POST", headers=_headers(api_key),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def is_gone(pod_id: str, api_key: str) -> bool:
    """Ask RunPod whether the pod is actually terminated.

    Never infer success from an HTTP 200: RunPod returns application errors
    inside a 200 body.
    """
    query = (
        'query { pod(input: {podId: "%s"}) { id desiredStatus } }' % pod_id
    )
    try:
        body = _post_graphql(api_key, query)
    except Exception as exc:
        log(f"could not check status ({exc})")
        return False

    if body.get("errors"):
        # A pod that no longer exists is a perfectly good outcome.
        message = str(body["errors"][0].get("message", "")).lower()
        if "not found" in message or "does not exist" in message:
            return True
        return False

    pod = (body.get("data") or {}).get("pod")
    if pod is None:
        return True
    return str(pod.get("desiredStatus", "")).upper() in GONE_STATUSES


def try_graphql(pod_id: str, api_key: str) -> None:
    query = 'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id
    try:
        body = _post_graphql(api_key, query)
        if body.get("errors"):
            log(f"graphql said: {body['errors'][0].get('message')}")
        else:
            log("graphql podTerminate accepted")
    except Exception as exc:
        log(f"graphql route failed ({exc})")


def try_rest(pod_id: str, api_key: str) -> None:
    request = urllib.request.Request(
        f"{REST_V2}/pods/{pod_id}", method="DELETE", headers=_headers(api_key),
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            log(f"rest delete returned {response.status}")
    except urllib.error.HTTPError as exc:
        log(f"rest delete returned {exc.code}")
    except Exception as exc:
        log(f"rest route failed ({exc})")


def try_runpodctl(pod_id: str, api_key: str) -> None:
    """Last resort. runpodctl ships unauthenticated, so configure it first.

    Verified: `runpodctl pod delete` does not exist; the correct form is
    `runpodctl remove pod <id>`.
    """
    try:
        subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            capture_output=True, timeout=60,
        )
        result = subprocess.run(
            ["runpodctl", "remove", "pod", pod_id],
            capture_output=True, timeout=120, text=True,
        )
        log(f"runpodctl exited {result.returncode}: "
            f"{(result.stdout + result.stderr).strip()[:200]}")
    except FileNotFoundError:
        log("runpodctl is not installed")
    except Exception as exc:
        log(f"runpodctl route failed ({exc})")


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        log("no pod id supplied; cannot terminate")
        return 1
    pod_id = sys.argv[1].strip()
    api_key = sys.argv[2].strip() if len(sys.argv) > 2 else ""

    log(f"shutting down pod {pod_id}")
    if not api_key:
        log("no API key supplied; cannot authenticate to RunPod")
        return 1

    if is_gone(pod_id, api_key):
        log("pod is already gone")
        return 0

    for attempt in range(1, 5):
        try_graphql(pod_id, api_key)
        time.sleep(5)
        if is_gone(pod_id, api_key):
            log("confirmed terminated")
            return 0

        try_rest(pod_id, api_key)
        time.sleep(5)
        if is_gone(pod_id, api_key):
            log("confirmed terminated")
            return 0

        if attempt >= 2:
            try_runpodctl(pod_id, api_key)
            time.sleep(5)
            if is_gone(pod_id, api_key):
                log("confirmed terminated")
                return 0

        log(f"still running after attempt {attempt}, retrying")
        time.sleep(15)

    log("EVERY ROUTE FAILED - this pod is still billing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
