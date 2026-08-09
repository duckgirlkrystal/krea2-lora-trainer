#!/usr/bin/env python3
"""
Rationale (verified): This script owns the entire pod lifecycle - create,
train, collect, destroy - rather than attaching to a pod someone else brought
up. The user has no ops tooling, no pre-provisioned infrastructure, and no way
to clean up by hand if something is left half-created, so the thing that
starts a pod has to be the thing that guarantees it dies.

It depends only on `requests` and `paramiko`, for the reasons in _shared.py.

What it does:
  1. Loads settings, or runs the first-time wizard
  2. Prepares the photos into a clean dataset
  3. Generates an in-memory SSH keypair and injects the public half at pod
     creation via PUBLIC_KEY, so nothing is written to the user's RunPod
     account and nothing lands in C:\\Users\\...\\.ssh
  4. Picks an available GPU, shows the cost, and asks before spending money
  5. Creates the pod, arms a pod-side watchdog, and sets terminateAfter
  6. Uploads the dataset, config, captioner and training script over SFTP
  7. Monitors, reconnecting through dropped SSH sessions
  8. Verifies the first checkpoint is not a dead all-zero LoRA (issue #925)
  9. Downloads the LoRA files and preview images
 10. Terminates the pod and confirms it is gone

Usage:
  Double-click "1 - TRAIN IN THE CLOUD.bat" in the parent folder.
  python train_runpod.py                 (same thing, from a terminal)
  python train_runpod.py --shutdown-all  (kill every pod on the account)
  python train_runpod.py --setup-only    (validate on the pod, do not train)

Billing safety, in the order it actually protects you (measured, Aug 2026):

  1. The pod-side watchdog. An independent detached process that terminates
     the pod after a hard deadline no matter what happens to training or to
     the client. Verified to survive its parent SSH session ending.
  2. The pod-side terminator, run when training finishes or dies.
  3. This client's `finally`, which deletes the pod and confirms it is gone.
  4. "3 - EMERGENCY - SHUT DOWN CLOUD.bat", which kills everything.
  5. terminateAfter. DO NOT RELY ON THIS. A pod created with a deadline ten
     minutes out was still RUNNING twelve minutes past it. It is set because
     it costs nothing, but it is the weakest layer, not the backstop.

Maintenance: Pod creation uses the legacy GraphQL podFindAndDeployOnDemand
mutation on purpose - REST v2's CreatePodRequest has no gpuTypeIdList. If
RunPod retires GraphQL, creation must move to v2; the billing story does not
depend on GraphQL-only fields, so that migration is safe.
"""
from __future__ import annotations

# =============================================================================
#  SETTINGS - you can fill these in if you would rather not answer questions.
#  Leaving them blank is fine: the program will just ask you once and
#  remember. Keep the quotes.
# =============================================================================

MY_RUNPOD_API_KEY = ""
MY_HUGGINGFACE_TOKEN = ""
MY_CHARACTER_NAME = ""

# =============================================================================
#  Nothing below here needs changing.
# =============================================================================

import argparse
import io
import os
import posixpath
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import _shared as sh
from _shared import UserFacingError
import _configs

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_V2 = "https://api.runpod.io/v2"

REMOTE_WORK = "/workspace/krea2-training"
REMOTE_DATASET = f"{REMOTE_WORK}/dataset"
REMOTE_OUTPUT = f"{REMOTE_WORK}/output"
REMOTE_CONFIG = f"{REMOTE_WORK}/config.yaml"
# Cloud runs train with sampling off and render previews afterwards instead,
# in a fresh process. The pod-side wrapper runs this config only once training
# has exited 0, and swallows every failure: the LoRA is the deliverable.
REMOTE_PREVIEW_CONFIG = f"{REMOTE_WORK}/preview_config.yaml"
REMOTE_SCRIPT = f"{REMOTE_WORK}/setup_and_train.sh"
REMOTE_CAPTIONER = f"{REMOTE_WORK}/caption_dataset.py"
REMOTE_TERMINATOR = f"{REMOTE_WORK}/terminate_pod.py"
REMOTE_LOG = f"{REMOTE_WORK}/training.log"
REMOTE_DONE = f"{REMOTE_WORK}/TRAINING_DONE"
REMOTE_CLIENT_DONE = f"{REMOTE_WORK}/CLIENT_DONE"

# The private half of the throwaway pod key. Persisted so an interrupted run
# can be resumed rather than abandoned with the pod still billing.
POD_KEY_FILE = sh.WORK_DIR / "pod_key"

# ai-toolkit keeps thumbnails in a hidden samples/.thumbs/ directory, named
# "<original>.jpg.jpg". Without this, every preview image is collected twice -
# once real, once as a thumbnail wearing a doubled suffix - and the user gets
# sixteen files for eight pictures.
NO_HIDDEN_DIRS = r"-not -path '*/.*/*'"

# Every pod this trainer starts is named with this prefix, which is how the
# emergency shutdown tells its own pods from anything else on the account.
POD_NAME_PREFIX = "krea2-"

POD_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
CONTAINER_DISK_GB = 50
VOLUME_GB = 100

# Two groups of candidates. Every card within a group runs the same training
# config, so whichever one RunPod's scheduler hands us, the config is valid.
# Group A is the reference configuration; group B is the cheaper fallback
# used only when no 80GB card is free.
GPU_GROUPS: list[dict[str, Any]] = [
    {
        "name": "80GB",
        "tier": "big",
        "seconds_per_step": 1.8,
        "gpus": [
            "NVIDIA A100 80GB PCIe",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA H100 PCIe",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA H200",
        ],
        "fallback_price": 1.49,
    },
    {
        "name": "48GB",
        "tier": "32gb",
        "seconds_per_step": 3.2,
        "gpus": [
            "NVIDIA L40S",
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            "NVIDIA L40",
        ],
        "fallback_price": 0.99,
    },
]

SETUP_HOURS = 1.0          # env install plus the ~35GB model download
MIN_TERMINATE_HOURS = 4.0  # never set the billing deadline tighter than this


# ---------------------------------------------------------------------------
# RunPod API
# ---------------------------------------------------------------------------

class RunPod:
    def __init__(self, api_key: str) -> None:
        import requests

        self.requests = requests
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        """Run a GraphQL call.

        RunPod returns application errors inside an HTTP 200 body, so
        checking response.ok is not enough and never has been.
        """
        payload: dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables
        try:
            response = self.requests.post(
                GRAPHQL_URL, headers=self.headers, json=payload, timeout=60
            )
        except Exception as exc:
            raise UserFacingError(f"Could not reach RunPod: {exc}")

        if response.status_code in (401, 403):
            raise UserFacingError(
                "RunPod rejected your API key. It may have been deleted or it\n"
                "  may not have write permission. Delete my_settings.json and\n"
                "  run this again to enter a new one."
            )
        try:
            body = response.json()
        except Exception:
            raise UserFacingError(
                f"RunPod sent back something unreadable (code {response.status_code})."
            )
        if body.get("errors"):
            message = body["errors"][0].get("message", "unknown error")
            raise RunPodApiError(message)
        return body.get("data") or {}

    def rest(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self.requests.request(
                method, f"{REST_V2}{path}", headers=self.headers, timeout=60, **kwargs
            )
        except Exception:
            return None
        if response.status_code >= 400:
            return None
        if not response.content:
            return {}
        try:
            return response.json()
        except Exception:
            return {}

    # -- catalog ------------------------------------------------------------

    def gpu_catalog(self) -> dict[str, dict]:
        """Return {gpu_id: {price, availability}} or {} if unavailable."""
        data = self.rest(
            "GET", "/catalog/gpus",
            params={"include": "AVAILABILITY", "cloud": "SECURE", "count": 1},
        )
        if not data:
            return {}
        entries = data.get("gpus") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return {}
        catalog: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            gpu_id = entry.get("id") or entry.get("displayName") or entry.get("name")
            if not gpu_id:
                continue
            price = entry.get("price") or {}
            catalog[gpu_id] = {
                "price": price.get("secure") or price.get("community"),
                "availability": (entry.get("availability") or "UNKNOWN").upper(),
            }
        return catalog

    # -- pods ---------------------------------------------------------------

    def create_pod(
        self, *, name: str, gpu_ids: list[str], public_key: str,
        env: dict[str, str], terminate_after: str,
    ) -> dict:
        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id name imageName costPerHr machineId
          }
        }
        """
        env_list = [{"key": k, "value": v} for k, v in env.items()]
        # Both key names are set deliberately. RunPod's docs are inconsistent
        # about which one a user-supplied key should use, and start.sh appends
        # rather than overwrites, so a duplicate is harmless.
        env_list.append({"key": "PUBLIC_KEY", "value": public_key})
        env_list.append({"key": "SSH_PUBLIC_KEY", "value": public_key})

        variables = {"input": {
            "name": name,
            "imageName": POD_IMAGE,
            "cloudType": "SECURE",
            "gpuTypeId": gpu_ids[0],
            "gpuTypeIdList": gpu_ids,
            "gpuCount": 1,
            "containerDiskInGb": CONTAINER_DISK_GB,
            "volumeInGb": VOLUME_GB,
            "volumeMountPath": "/workspace",
            "minVcpuCount": 8,
            "minMemoryInGb": 32,
            "ports": "22/tcp",
            "startSsh": True,
            "supportPublicIp": True,
            "terminateAfter": terminate_after,
            "env": env_list,
        }}
        data = self.graphql(mutation, variables)
        pod = data.get("podFindAndDeployOnDemand")
        if not pod:
            raise RunPodApiError("RunPod did not return a pod.")
        return pod

    def get_pod(self, pod_id: str) -> dict | None:
        query = """
        query getPod($input: PodFilter!) {
          pod(input: $input) {
            id name desiredStatus costPerHr
            runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
          }
        }
        """
        try:
            data = self.graphql(query, {"input": {"podId": pod_id}})
        except RunPodApiError:
            return None
        return data.get("pod")

    def list_pods(self) -> list[dict]:
        query = "query { myself { pods { id name desiredStatus costPerHr } } }"
        data = self.graphql(query)
        myself = data.get("myself") or {}
        return myself.get("pods") or []

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod, trying both API surfaces.

        This is the safety-critical path, so it deliberately does not give up
        after one failure.
        """
        if self.rest("DELETE", f"/pods/{pod_id}") is not None:
            return True
        try:
            self.graphql(
                "mutation terminate($input: PodTerminateInput!) { podTerminate(input: $input) }",
                {"input": {"podId": pod_id}},
            )
            return True
        except Exception:
            pass
        try:
            self.graphql(
                'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id
            )
            return True
        except Exception:
            return False


class RunPodApiError(Exception):
    """An error message that came from RunPod itself."""


_NO_CAPACITY = re.compile(
    r"no longer any instances available|not enough|no instances", re.IGNORECASE
)
_NO_DISK = re.compile(r"disk space", re.IGNORECASE)


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

def generate_keypair() -> tuple[Any, str]:
    """Create a throwaway SSH keypair that only ever exists in memory.

    paramiko has no Ed25519Key.generate(), so the key is built with
    `cryptography` (which paramiko already depends on) and handed to paramiko
    through an in-memory buffer. Nothing is written to ~/.ssh, so there are no
    Windows file-permission problems and nothing is left on the account.

    The PEM is returned as well because it MUST be saved next to the pod id.
    Verified the hard way on 2026-08-04: when the client process died, the
    only key that could reach the pod died with it, leaving a paid A100 that
    could not be reconnected to, tailed, or collected from - just billed.
    """
    import paramiko
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    ).decode()
    public_openssh = private.public_key().public_bytes(
        serialization.Encoding.OpenSSH,
        serialization.PublicFormat.OpenSSH,
    ).decode()

    key = paramiko.Ed25519Key.from_private_key(io.StringIO(private_pem))
    return key, f"{public_openssh} krystals-lora-trainer", private_pem


def load_saved_key(private_pem: str) -> Any:
    import paramiko

    return paramiko.Ed25519Key.from_private_key(io.StringIO(private_pem))


class Ssh:
    """A reconnecting SSH/SFTP session to the pod."""

    def __init__(self, host: str, port: int, key: Any, username: str = "root") -> None:
        self.host = host
        self.port = port
        self.key = key
        self.username = username
        self.client = None
        # Set when the pod reports it can shut itself down. Until then this
        # client is the only thing standing between the user and an open-ended
        # GPU bill, so the finally in run() must terminate unconditionally.
        self.self_terminate_armed = False

    def connect(self, timeout_seconds: int = 240) -> None:
        import paramiko

        deadline = time.time() + timeout_seconds
        last_error = ""
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            client = paramiko.SSHClient()
            # The pod generates fresh host keys on first boot, so there is
            # nothing to have trusted in advance.
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    pkey=self.key,
                    timeout=20,
                    banner_timeout=30,
                    auth_timeout=30,
                    allow_agent=False,
                    look_for_keys=False,
                )
                self.client = client
                return
            except Exception as exc:
                last_error = str(exc)
                try:
                    client.close()
                except Exception:
                    pass
                time.sleep(min(10, 2 + attempt))
        raise UserFacingError(
            "Could not connect to the rented computer.\n\n"
            f"  Technical detail: {last_error}\n\n"
            "  The pod was created but is not accepting connections. Run\n"
            "  \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\" to make sure you are\n"
            "  not being charged, then try again."
        )

    def ensure(self) -> None:
        if self.client is not None:
            transport = self.client.get_transport()
            if transport is not None and transport.is_active():
                return
        self.close()
        self.connect(timeout_seconds=120)

    def run(self, command: str, timeout: int = 300) -> tuple[str, str, int]:
        self.ensure()
        assert self.client is not None
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        stdin.close()
        try:
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
        except Exception:
            # A command that leaves a background process holding the channel
            # will never send EOF, so the read blocks until the socket
            # timeout. Close the channel and report failure rather than
            # letting a stuck command wedge the whole run.
            try:
                stdout.channel.close()
            except Exception:
                pass
            return "", "command timed out", 124
        return out, err, code

    def run_detached(self, command: str, log_path: str) -> None:
        """Start a long-lived background command and return immediately.

        setsid plus full fd redirection is required: anything inheriting the
        SSH channel's descriptors keeps exec_command() blocked.
        """
        wrapped = (
            f"setsid nohup bash -c {_shell_quote(command)} "
            f"> {log_path} 2>&1 < /dev/null & echo DETACHED"
        )
        self.run(wrapped, timeout=60)

    def run_streaming(self, command: str, timeout: int = 7200) -> int:
        """Run a command, printing pod output as it arrives."""
        self.ensure()
        assert self.client is not None
        stdin, stdout, stderr = self.client.exec_command(
            command, timeout=timeout, get_pty=True
        )
        stdin.close()
        for raw in iter(stdout.readline, ""):
            line = _strip_ansi(raw.rstrip("\r\n"))
            if "SELF_TERMINATE_ARMED=1" in line:
                self.self_terminate_armed = True
            if line.strip():
                sh.say("    " + line)
        return stdout.channel.recv_exit_status()

    def sftp(self):
        self.ensure()
        assert self.client is not None
        return self.client.open_sftp()

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None


_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------------------
# Cost and planning
# ---------------------------------------------------------------------------

def choose_gpu_group(runpod: RunPod, only: str = "") -> tuple[dict, list[str], float]:
    """Pick a GPU group, preferring the reference 80GB configuration.

    Availability from the catalog is a hint, not a reservation, so the
    returned list is still an ordered fallback for the scheduler to work
    through.

    `only` restricts the search to one group by name. Nothing a friend runs
    ever sets it; it exists so a maintainer can rehearse a run on the cheaper
    tier without waiting for the 80GB cards to sell out.
    """
    catalog = runpod.gpu_catalog()
    if not catalog:
        sh.step("Could not read the GPU price list; using standard settings.")

    groups = [g for g in GPU_GROUPS if not only or g["name"] == only]
    for group in groups:
        available = []
        prices = []
        for gpu_id in group["gpus"]:
            info = catalog.get(gpu_id)
            if info is None:
                # Unknown to the catalog (or no catalog at all): still worth
                # asking the scheduler for.
                available.append(gpu_id)
                continue
            if info["availability"] == "NONE":
                continue
            available.append(gpu_id)
            if info.get("price"):
                prices.append(info["price"])
        if available:
            # Quote the dearest card in the group, not the cheapest. The
            # scheduler chooses from `available` and the user has no way to
            # insist on the bargain, so a minimum is a best case dressed up
            # as a forecast. Measured Aug 2026: a run quoted at $0.44/hr from
            # the catalog minimum was billed by RunPod at $0.99/hr. For a
            # tool whose promise is "no surprise bills", quoting half the
            # real number is the one number it must not get wrong.
            price = max(prices) if prices else group["fallback_price"]
            return group, available, price

    raise UserFacingError(
        "RunPod currently has no suitable graphics cards free.\n\n"
        "  This happens at busy times. Wait 15 minutes and try again, or\n"
        "  train on your own PC instead with \"2 - TRAIN ON MY PC.bat\".\n\n"
        "  Nothing was created and you have not been charged."
    )


def estimate_hours(group: dict, steps: int) -> float:
    return SETUP_HOURS + (steps * group["seconds_per_step"]) / 3600.0


# ---------------------------------------------------------------------------
# Upload / download
# ---------------------------------------------------------------------------

def upload_everything(
    ssh: Ssh, dataset_dir: Path, config_text: str, preview_config_text: str = ""
) -> None:
    ssh.run(f"mkdir -p {REMOTE_DATASET} {REMOTE_OUTPUT}")

    existing: dict[str, int] = {}
    out, _, code = ssh.run(
        f"find {REMOTE_DATASET} -maxdepth 1 -type f -printf '%s %f\\n' 2>/dev/null"
    )
    if code == 0:
        for line in out.strip().splitlines():
            size, _, name = line.partition(" ")
            if name and size.isdigit():
                existing[name] = int(size)

    files = sorted(p for p in dataset_dir.iterdir() if p.is_file())
    sftp = ssh.sftp()
    try:
        uploaded = skipped = 0
        for index, path in enumerate(files, start=1):
            if existing.get(path.name) == path.stat().st_size:
                skipped += 1
                continue
            sftp.put(str(path), posixpath.join(REMOTE_DATASET, path.name))
            uploaded += 1
            if uploaded == 1 or uploaded % 10 == 0:
                sh.step(f"  sent {index} of {len(files)} files")
        if skipped:
            sh.step(f"  {skipped} file(s) were already there")

        with sftp.open(REMOTE_CONFIG, "w") as handle:
            handle.write(config_text)

        if preview_config_text:
            with sftp.open(REMOTE_PREVIEW_CONFIG, "w") as handle:
                handle.write(preview_config_text)

        for local_name, remote_path in (
            ("setup_and_train.sh", REMOTE_SCRIPT),
            ("caption_dataset.py", REMOTE_CAPTIONER),
            ("terminate_pod.py", REMOTE_TERMINATOR),
        ):
            local_path = sh.TRAINER_DIR / local_name
            if not local_path.exists():
                raise UserFacingError(f"A program file is missing: {local_name}")
            sftp.put(str(local_path), remote_path)
    finally:
        try:
            sftp.close()
        except Exception:
            pass

    # Windows checkouts carry CRLF, which breaks a bash shebang line.
    ssh.run(f"sed -i 's/\\r$//' {REMOTE_SCRIPT} && chmod +x {REMOTE_SCRIPT}")
    ssh.run(f"sed -i 's/\\r$//' {REMOTE_CAPTIONER}")
    ssh.run(f"sed -i 's/\\r$//' {REMOTE_TERMINATOR}")
    sh.ok(f"{len(files)} files uploaded.")


def remote_checkpoints(ssh: Ssh) -> list[str]:
    out, _, code = ssh.run(
        f"find {REMOTE_OUTPUT} -name '*.safetensors' 2>/dev/null | sort"
    )
    if code != 0:
        return []
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def download_file(ssh: Ssh, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp = ssh.sftp()
    try:
        sftp.get(remote_path, str(local_path))
        return local_path.exists() and local_path.stat().st_size > 0
    except Exception as exc:
        sh.warn(f"Could not download {posixpath.basename(remote_path)}: {exc}")
        return False
    finally:
        try:
            sftp.close()
        except Exception:
            pass


class IncrementalCollector:
    """Brings checkpoints and preview images home as soon as they appear.

    Collecting everything only after training exits means twenty minutes of
    nothing followed by four files at once, which is the wrong shape: the
    reason for keeping intermediate checkpoints is to compare them, and
    people want to look as they go rather than wait for a batch.

    A checkpoint is several hundred MB and is written progressively, so a
    file is only fetched once its size has stopped changing between two
    polls. That is what stops a half-written safetensors being pulled down
    and mistaken for a finished one.
    """

    IMAGE_NAMES = ("*.jpg", "*.png", "*.jpeg", "*.webp")

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.tracker = sh.SettleTracker()

    def _find_command(self) -> str:
        images = " -o ".join(f"-name '{pattern}'" for pattern in self.IMAGE_NAMES)
        return (
            rf"find {REMOTE_OUTPUT} {NO_HIDDEN_DIRS} \( -name '*.safetensors' "
            rf"-o \( -path '*samples*' \( {images} \) \) \) "
            rf"-printf '%s\t%p\n' 2>/dev/null | sort -k2"
        )

    def _destination(self, remote_path: str) -> Path:
        name = posixpath.basename(remote_path)
        if remote_path.endswith(".safetensors"):
            return self.output_dir / name
        return self.output_dir / "preview_images" / name

    def poll(self, ssh: Ssh) -> int:
        """Fetch anything new and settled. Returns how many files landed."""
        try:
            out, _, code = ssh.run(self._find_command(), timeout=90)
        except Exception:
            return 0
        if code != 0:
            return 0

        landed = 0
        for line in out.strip().splitlines():
            if "\t" not in line:
                continue
            raw_size, _, remote_path = line.partition("\t")
            remote_path = remote_path.strip()
            if not remote_path or self.tracker.is_done(remote_path):
                continue
            try:
                size = int(raw_size.strip())
            except ValueError:
                continue
            if not self.tracker.is_ready(remote_path, size):
                continue

            local_path = self._destination(remote_path)
            if local_path.exists() and local_path.stat().st_size == size:
                self.tracker.mark_done(remote_path)
                continue

            if download_file(ssh, remote_path, local_path):
                self.tracker.mark_done(remote_path)
                landed += 1
                if remote_path.endswith(".safetensors"):
                    sh.ok(f"  Saved {local_path.name} ({sh.human_size(size)})")
                    sh.say(f"     -> {self.output_dir}")
                else:
                    sh.ok(f"  Saved preview image {local_path.name}")
        return landed


def download_results(ssh: Ssh, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for remote_path in remote_checkpoints(ssh):
        local_path = output_dir / posixpath.basename(remote_path)
        if local_path.exists():
            out, _, _ = ssh.run(f"stat -c %s {remote_path}")
            if out.strip().isdigit() and int(out.strip()) == local_path.stat().st_size:
                saved.append(local_path)
                continue
        sh.step(f"  downloading {local_path.name} ...")
        if download_file(ssh, remote_path, local_path):
            saved.append(local_path)

    out, _, _ = ssh.run(
        rf"find {REMOTE_OUTPUT} {NO_HIDDEN_DIRS} -path '*samples*' "
        rf"\( -name '*.jpg' -o -name '*.png' "
        rf"-o -name '*.jpeg' -o -name '*.webp' \) 2>/dev/null | sort"
    )
    previews = [line.strip() for line in out.strip().splitlines() if line.strip()]
    if previews:
        preview_dir = output_dir / "preview_images"
        sh.step(f"  downloading {len(previews)} preview image(s) ...")
        for remote_path in previews:
            download_file(ssh, remote_path, preview_dir / posixpath.basename(remote_path))

    out, _, _ = ssh.run(f"tail -c 400000 {REMOTE_LOG} 2>/dev/null")
    if out.strip():
        (output_dir / "training_log.txt").write_text(out, encoding="utf-8")

    return saved


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def monitor(ssh: Ssh, steps: int, output_dir: Path) -> str:
    """Watch the run until it finishes. Returns 'done', 'dead' or 'failed'."""
    printer = sh.ProgressPrinter(interval=60)
    collector = IncrementalCollector(output_dir)
    checked_first_checkpoint = False
    downloading_announced = False
    started = time.time()
    consecutive_errors = 0

    sh.say()
    sh.say("  Training is running on the rented computer.")
    sh.say("  You can leave this window open and go and do something else.")
    sh.say("  Closing it will NOT stop the training, but it will stop the")
    sh.say("  automatic shutdown, so please leave it running if you can.")
    sh.say()
    sh.say("  Snapshots and preview pictures are copied into the 'output'")
    sh.say("  folder as soon as each one is ready, so you can look at them")
    sh.say("  while the rest carries on.")
    sh.say()

    while True:
        time.sleep(20)
        try:
            out, _, _ = ssh.run(f"tail -c 40000 {REMOTE_LOG} 2>/dev/null", timeout=90)
            done, _, done_code = ssh.run(f"test -f {REMOTE_DONE} && echo yes", timeout=60)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors == 1:
                sh.step("Connection hiccup, reconnecting...")
            if consecutive_errors > 30:
                raise UserFacingError(
                    f"Lost contact with the rented computer ({exc}).\n\n"
                    "  Run \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\" to make\n"
                    "  sure you are not still being charged."
                )
            try:
                ssh.close()
                ssh.connect(timeout_seconds=120)
            except Exception:
                pass
            continue

        text = _strip_ansi(out)
        progress = sh.parse_progress(text, steps)

        if progress["step"] is None:
            if not downloading_announced and time.time() - started > 60:
                downloading_announced = True
                sh.say("  Setting up and downloading the AI model (about 35 GB).")
                sh.say("  This part takes 30-45 minutes. Nothing is wrong.")
            if "CAPTION:" in text and "CAPTION: wrote" not in text:
                printer.maybe_print({"step": None}, force=False)
        else:
            printer.maybe_print(progress)

        if not checked_first_checkpoint:
            checkpoints = remote_checkpoints(ssh)
            if checkpoints:
                checked_first_checkpoint = True
                if not check_first_checkpoint(ssh, checkpoints[0], output_dir):
                    return "dead"

        # Only start pulling files down once the dead-LoRA gate has passed,
        # so a run that is going to be thrown away does not spend bandwidth
        # (or the user's attention) on worthless checkpoints.
        if checked_first_checkpoint:
            collector.poll(ssh)

        if "yes" in done.strip() and done_code == 0:
            break

        if "TRAINING_EXIT_CODE=" in text:
            break

    out, _, _ = ssh.run(f"tail -c 40000 {REMOTE_LOG} 2>/dev/null", timeout=90)
    text = _strip_ansi(out)
    match = re.search(r"TRAINING_EXIT_CODE=(\d+)", text)
    if match and match.group(1) != "0":
        code = match.group(1)
        if code in ("124", "137"):
            sh.warn("Training hit its time limit and was stopped.")
            sh.step("Any checkpoints saved so far will still be downloaded.")
            return "done"
        sh.warn(f"Training ended with an error (code {code}).")
        tail = "\n".join(text.strip().splitlines()[-20:])
        sh.say()
        sh.say("  Last few lines from the training log:")
        for line in tail.splitlines():
            sh.say("    " + line)
        sh.say()
        return "failed"
    return "done"


def check_first_checkpoint(ssh: Ssh, remote_path: str, output_dir: Path) -> bool:
    """Download the first saved LoRA and confirm it is not empty.

    ai-toolkit issue #925 produces Krea 2 LoRAs whose lora_B tensors are all
    exactly zero while the loss curve looks perfectly healthy. Catching it at
    the first save costs one download; not catching it costs the whole run.
    """
    sh.say()
    sh.step("First snapshot saved. Checking it actually learned something...")
    temp_path = output_dir / ("_check_" + posixpath.basename(remote_path))
    if not download_file(ssh, remote_path, temp_path):
        sh.warn("Could not check the snapshot. Carrying on anyway.")
        return True

    try:
        alive, message = sh.verify_checkpoint_alive(temp_path)
    except Exception as exc:
        sh.warn(f"Could not check the snapshot ({exc}). Carrying on anyway.")
        return True
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass

    if alive:
        sh.ok(f"Looks good. {message}")
        sh.say()
        return True

    sh.say()
    sh.problem(f"The snapshot is empty: {message}")
    sh.say(sh.DEAD_LORA_EXPLANATION)
    return False


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def forget_pod() -> None:
    """Drop the resume breadcrumbs once the pod is confirmed gone."""
    for path in (sh.ROOT / "LAST_POD_ID.txt", POD_KEY_FILE):
        try:
            path.unlink()
        except Exception:
            pass


def is_our_pod(pod: dict, recorded_id: str = "") -> bool:
    """Did this trainer start that pod?

    Recognised by name as well as by the recorded id, because the id is kept
    in a local file that has been seen to go missing. Without the name check,
    a lost note would make the trainer's own pod look like a stranger's and
    the emergency button would stop to ask permission - at the exact moment
    the user needs it to just work.
    """
    if recorded_id and pod.get("id") == recorded_id:
        return True
    return str(pod.get("name") or "").startswith(POD_NAME_PREFIX)


def read_recorded_pod_id() -> str:
    """The id of the pod this trainer started, if we still have a note of it."""
    try:
        lines = (sh.ROOT / "LAST_POD_ID.txt").read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    return lines[0].strip() if lines else ""


def live_previous_pod(runpod: RunPod) -> str:
    """Id of a still-running pod from an earlier run, or "" if there is none.

    The client dying does not stop the pod: training carries on and the pod
    waits for someone to come and collect. Without this, the next launch rents
    a SECOND computer while the first is still billing and still holding the
    only copy of the results.

    Guarded by is_our_pod() rather than trusting the recorded id alone -
    RunPod recycles ids, and a stale note once pointed the emergency shutdown
    at a stranger's pod.
    """
    if not POD_KEY_FILE.exists():
        return ""
    pod_id = read_recorded_pod_id()
    if not pod_id:
        return ""
    try:
        pod = runpod.get_pod(pod_id)
    except Exception:
        return ""
    if not pod or not is_our_pod(pod, pod_id):
        return ""
    if (pod.get("desiredStatus") or "").upper() in ("TERMINATED", "EXITED", "FAILED"):
        return ""
    return pod_id


def find_running_pod(runpod: RunPod) -> str:
    """Ask RunPod which pod to reconnect to when the local note is missing."""
    try:
        pods = runpod.list_pods()
    except Exception:
        pods = []
    live = [
        pod for pod in pods
        if (pod.get("desiredStatus") or "").upper() not in ("TERMINATED", "EXITED")
    ]

    if not live:
        raise UserFacingError(
            "There is no interrupted run to pick up.\n\n"
            "  Nothing is running on your RunPod account, so there is\n"
            "  nothing to reconnect to and nothing is costing you money.\n\n"
            "  Start a normal run when you are ready."
        )
    if len(live) > 1:
        listed = "\n".join(f"       {pod.get('id')}" for pod in live)
        raise UserFacingError(
            "More than one rented computer is running, so I cannot tell\n"
            "  which one belongs to your interrupted run:\n\n"
            f"{listed}\n\n"
            "  To stop paying for all of them, run\n"
            "  \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\"."
        )

    pod_id = str(live[0].get("id") or "")
    sh.step("No note of which computer to use; found one running and will use it.")
    return pod_id


def record_pod_id(pod_id: str, private_pem: str = "") -> None:
    """Write the pod id and its SSH key to disk the instant they exist.

    If everything else fails, these two files are how the user (or we) can
    find an orphaned pod later - and, thanks to the key, actually get back
    into it with --resume instead of writing off the run.
    """
    try:
        if private_pem:
            sh.WORK_DIR.mkdir(parents=True, exist_ok=True)
            POD_KEY_FILE.write_text(private_pem, encoding="utf-8")
            try:
                os.chmod(POD_KEY_FILE, 0o600)
            except Exception:
                pass
    except Exception:
        pass
    # Not silent on failure. This note is what stands between the user and a
    # rented computer they cannot find, so if it cannot be written they need
    # to hear the id now, while it is still on screen.
    id_file = sh.ROOT / "LAST_POD_ID.txt"
    try:
        id_file.write_text(
            f"{pod_id}\n"
            f"created: {datetime.now().isoformat(timespec='seconds')}\n"
            f"If this run was interrupted, you can pick it back up with\n"
            f"  python trainer/train_runpod.py --resume\n"
            f"If you are being charged unexpectedly, delete this pod at\n"
            f"{sh.RUNPOD_PODS_URL}\n",
            encoding="utf-8",
        )
        if not id_file.exists():
            raise OSError("the file was not there afterwards")
    except Exception as exc:
        sh.warn(f"Could not save a note of the computer's id ({exc}).")
        sh.warn(f"Write this down in case you need it: {pod_id}")


def shut_down(runpod: RunPod, pod_id: str, ssh: Ssh | None) -> None:
    """Terminate the pod and make sure it is really gone."""
    if not pod_id:
        return

    # CLIENT_DONE is deliberately NOT written here. It means "I have the
    # results, stop waiting and shut down", and this function runs on every
    # exit path including the ones where nothing was downloaded at all. Written
    # unconditionally it was a lie that collapsed the pod's own grace window -
    # the last chance to salvage a run - at the exact moment it was needed.
    # collect_results() writes it, once a download has actually succeeded.
    if ssh is not None:
        ssh.close()

    sh.step("Shutting down the rented computer...")
    for attempt in range(4):
        try:
            runpod.terminate_pod(pod_id)
        except Exception:
            pass
        time.sleep(4)
        try:
            pod = runpod.get_pod(pod_id)
        except Exception:
            pod = None
        if pod is None or (pod.get("desiredStatus") or "").upper() in (
            "TERMINATED", "EXITED",
        ):
            sh.ok("Shut down. You are no longer being charged for it.")
            forget_pod()
            return
        if attempt < 3:
            time.sleep(5)

    sh.say()
    sh.problem("COULD NOT CONFIRM THE RENTED COMPUTER SHUT DOWN.")
    sh.say()
    sh.say("  Please check this by hand right now so you are not charged:")
    sh.say(f"    {sh.RUNPOD_PODS_URL}")
    sh.say(f"  Delete the pod with id: {pod_id}")
    sh.say()


def shutdown_all(api_key: str, assume_yes: bool = False) -> int:
    runpod = RunPod(api_key)
    pods = runpod.list_pods()
    live = [
        pod for pod in pods
        if (pod.get("desiredStatus") or "").upper() not in ("TERMINATED", "EXITED")
    ]

    if not live:
        sh.ok("Nothing is running. You are not being charged for any computers.")
        forget_pod()
        return 0

    # Anything this trainer started is ours to stop, no questions asked. Pods
    # it did not start are somebody's work: this button killed an unrelated
    # $0.99/hr pod in Aug 2026 and RunPod deletes rather than stops, so there
    # was nothing to undo. A friend with a RunPod account used only for this
    # will never see the prompt, because everything running will be ours.
    recorded = read_recorded_pod_id()
    ours = [pod for pod in live if is_our_pod(pod, recorded)]
    others = [pod for pod in live if not is_our_pod(pod, recorded)]

    if others:
        sh.say()
        if ours:
            sh.warn("These are running but were NOT started by this trainer:")
        else:
            sh.warn("These are running, but I have no note of starting any"
                    " of them:")
        for pod in others:
            name = pod.get("name") or pod.get("id")
            cost = pod.get("costPerHr")
            sh.say(f"       {name} ({pod.get('id')})"
                   + (f" - ${cost}/hour" if cost else ""))
        sh.say()
        # If the question cannot be asked (no console, piped input, a
        # double-clicked window that lost its stdin) treat it as "no" rather
        # than letting the exception escape. Our own pod still gets stopped;
        # aborting here would leave everything running, which is the one
        # outcome this button exists to prevent.
        def _asked_and_agreed() -> bool:
            try:
                return sh.confirm("Shut these down too?")
            except Exception:
                sh.say()
                sh.warn("Could not ask, so leaving those alone.")
                return False

        if assume_yes or _asked_and_agreed():
            ours = live
        else:
            sh.say()
            if not ours:
                sh.ok("Nothing was stopped.")
                sh.say("  If you are being charged unexpectedly, stop them"
                       " by hand at")
                sh.say(f"    {sh.RUNPOD_PODS_URL}")
                return 0
            sh.step("Leaving those alone.")

    live = ours
    sh.warn(f"Shutting down {len(live)} computer(s).")
    sh.say()
    failures = []
    for pod in live:
        pod_id = pod.get("id", "")
        name = pod.get("name") or pod_id
        cost = pod.get("costPerHr")
        label = f"{name} ({pod_id})"
        if cost:
            label += f" - ${cost}/hour"
        sh.step(f"Stopping {label}")
        ok_flag = False
        for _ in range(3):
            try:
                if runpod.terminate_pod(pod_id):
                    ok_flag = True
                    break
            except Exception:
                pass
            time.sleep(3)
        if ok_flag:
            sh.ok(f"  stopped {pod_id}")
        else:
            failures.append(pod_id)
            sh.problem(f"  could NOT stop {pod_id}")

    sh.say()
    if failures:
        sh.problem("Some computers could not be stopped automatically.")
        sh.say()
        sh.say("  PLEASE DO THIS BY HAND RIGHT NOW:")
        sh.say(f"    1. Open {sh.RUNPOD_PODS_URL}")
        sh.say("    2. Delete these:")
        for pod_id in failures:
            sh.say(f"       {pod_id}")
        sh.say()
        return 1

    sh.ok("Everything is shut down. You are no longer being charged.")
    forget_pod()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    sh.header("Krystal's LoRA Trainer - Cloud")

    settings = sh.load_or_ask(
        need_runpod=True,
        character_name=MY_CHARACTER_NAME,
        huggingface_token=MY_HUGGINGFACE_TOKEN,
        runpod_api_key=MY_RUNPOD_API_KEY,
    )
    steps = args.steps or settings.training_steps or 2000
    slug = sh.make_slug(settings.character_name)
    run_name = f"{slug}_krea2"

    sh.header("Step 1 of 5 - your photos")
    sh.prepare_photos(sh.PHOTOS_DIR, sh.DATASET_DIR)
    # Deliberately NO caption writing here. Verified live on 2026-08-04: the
    # pod skips captioning when it already sees one .txt per image, so writing
    # fallback captions before upload silently disabled the Qwen3-VL captioner
    # on every cloud run, training on "<trigger>, a photograph of <trigger>".
    # setup_and_train.sh applies the identical fallback AFTER the captioner,
    # so nothing is lost by leaving the dataset caption-free at this point.

    runpod = RunPod(settings.runpod_api_key)

    # Recovery has to be the thing that already happened, not a flag. The
    # --resume path existed but no .bat ever passed it, so a user whose window
    # closed mid-run had a working recovery route and no way to reach it.
    # Same button, no arguments: if the earlier computer is still alive, go
    # back to it rather than renting a second one alongside it.
    if live_previous_pod(runpod):
        sh.say()
        sh.ok("An earlier run is still going on a rented computer.")
        sh.step("Reconnecting to it instead of renting another one.")
        return resume(args)

    sh.header("Step 2 of 5 - renting a computer")
    group, gpu_ids, price = choose_gpu_group(runpod, getattr(args, "gpu_group", ""))
    hours = estimate_hours(group, steps)
    cost = hours * price
    tier = _configs.TIER_BY_KEY[group["tier"]]

    sh.say(f"  Graphics card:   {gpu_ids[0]} or similar")
    sh.say(f"  Price:           up to ${price:.2f} per hour")
    sh.say(f"  Expected time:   about {sh.human_duration(hours * 3600)}")
    sh.say(f"  Expected cost:   up to about ${cost:.2f}")
    sh.say()
    sh.say("  The computer shuts down automatically when it is finished.")
    sh.say("  A hard deadline is also set on RunPod's side as a backstop.")
    sh.say()

    if not args.yes and not sh.confirm("Start training?"):
        sh.say()
        sh.ok("Cancelled. Nothing was created and you have not been charged.")
        return 0

    # Sampling is switched OFF for every cloud run regardless of tier, and
    # previews are rendered afterwards from the finished LoRA instead.
    #
    # Rationale (Aug 2026, from a real user's failed run): ai-toolkit calls
    # self.sample() unguarded inside its training loop, so ANY error while
    # writing a preview image aborts the whole job. An OSError [Errno 5] at
    # PIL's fp.close() killed a 2000-step run at step 250. Cloud runs are
    # unattended, billed by the hour, and keep the only copy of their results
    # on a pod that deletes itself, so a nice-to-have image is not worth the
    # risk of losing the run. Local runs keep their tier default: the user is
    # sitting there, nothing is being billed, and a crash costs only time.
    config_text = _configs.build_training_config(
        run_name=run_name,
        trigger_word=settings.trigger_word,
        dataset_dir=REMOTE_DATASET,
        output_dir=REMOTE_OUTPUT,
        steps=steps,
        tier=tier,
        sample_during_training=False,
    )
    preview_config_text = _configs.build_preview_config(
        run_name=run_name,
        trigger_word=settings.trigger_word,
        dataset_dir=REMOTE_DATASET,
        output_dir=REMOTE_OUTPUT,
        lora_path=f"{REMOTE_OUTPUT}/{run_name}/{run_name}.safetensors",
        tier=tier,
    )

    key, public_key, private_pem = generate_keypair()
    deadline = datetime.now(timezone.utc) + timedelta(
        hours=max(MIN_TERMINATE_HOURS, hours * 2.5)
    )
    terminate_after = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")

    sh.say()
    sh.step("Asking RunPod for a computer...")
    pod = None
    last_error = ""
    for attempt in range(4):
        try:
            pod = runpod.create_pod(
                name=f"{POD_NAME_PREFIX}{slug}",
                gpu_ids=gpu_ids,
                public_key=public_key,
                env={
                    # Keep the 35GB of model downloads on the big volume
                    # rather than the small container disk.
                    "HF_HOME": "/workspace/hf-cache",
                    "HF_TOKEN": settings.huggingface_token,
                },
                terminate_after=terminate_after,
            )
            break
        except RunPodApiError as exc:
            last_error = str(exc)
            if _NO_DISK.search(last_error):
                raise UserFacingError(
                    "RunPod could not find a computer with enough disk space.\n\n"
                    "  This usually clears within a few minutes. Please try again."
                )
            if _NO_CAPACITY.search(last_error):
                sh.step(f"  none free right now, retrying ({attempt + 1} of 4)...")
                time.sleep(30)
                continue
            raise UserFacingError(f"RunPod could not start a computer:\n\n  {last_error}")

    if pod is None:
        raise UserFacingError(
            "RunPod has no free graphics cards at the moment.\n\n"
            f"  Their message: {last_error}\n\n"
            "  Wait 15 minutes and try again. Nothing was created and you\n"
            "  have not been charged."
        )

    pod_id = pod["id"]
    record_pod_id(pod_id, private_pem)
    sh.ok(f"Computer rented (id {pod_id}).")

    # The estimate above is a forecast from a price list; this is the rate
    # RunPod is actually charging. Show it whenever the two differ enough to
    # matter, so nobody finds out from their statement instead.
    try:
        actual_hourly = float(pod.get("costPerHr") or 0)
    except (TypeError, ValueError):
        actual_hourly = 0.0
    if actual_hourly > 0:
        sh.step(f"Actual price: ${actual_hourly:.2f} per hour "
                f"(about ${actual_hourly * hours:.2f} for the whole run)")
        if actual_hourly > price * 1.1:
            sh.warn("That is dearer than the estimate above. If it is more"
                    " than you want to spend,")
            sh.warn("close this window and run"
                    " \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\" now.")

    sh.step(f"Automatic shutdown deadline: {terminate_after}")

    ssh: Ssh | None = None
    finished = False
    try:
        sh.step("Waiting for it to start up...")
        host, port = wait_for_ssh_endpoint(runpod, pod_id)
        sh.ok("It is up. Connecting...")

        ssh = Ssh(host, port, key)
        ssh.connect(timeout_seconds=300)
        sh.ok("Connected.")

        sh.header("Step 3 of 5 - sending your photos")
        upload_everything(ssh, sh.DATASET_DIR, config_text, preview_config_text)

        sh.header("Step 4 of 5 - training")
        sh.say("  Installing software and downloading the AI model.")
        sh.say("  This part takes 30-45 minutes and prints a lot of text.")
        sh.say()

        # Every one of these is passed explicitly rather than relying on the
        # pod's environment. Verified live: variables set at pod creation do
        # not reach a non-interactive `ssh host command` shell, so POD_ID in
        # particular has to come from us or self-termination cannot work.
        env_prefix = (
            f'HF_TOKEN="{settings.huggingface_token}" '
            f'RUN_NAME="{run_name}" '
            f'TRIGGER_WORD="{settings.trigger_word}" '
            f'HF_HOME="/workspace/hf-cache" '
            f'POD_ID="{pod_id}" '
            f'RUNPOD_API_KEY="{settings.runpod_api_key}" '
            f'SELF_TERMINATE=1 '
            f'MAX_TRAIN_HOURS={max(6, int(hours * 3) + 1)} '
            f'GRACE_HOURS=2 '
        )
        if args.setup_only:
            env_prefix += "SETUP_ONLY=1 "

        code = ssh.run_streaming(f"{env_prefix}bash {REMOTE_SCRIPT}", timeout=5400)
        if code != 0:
            # Setup failed, so the pod is not in a state where its own shutdown
            # can be trusted - the wrapper that runs the terminator may never
            # have started even if the watchdog did arm. Take the killer role
            # back rather than leave a broken pod billing until the 9h ceiling.
            ssh.self_terminate_armed = False
            raise UserFacingError(
                "Setting up the rented computer failed. The messages above\n"
                "  explain why. Nothing further was started, and the computer\n"
                "  is about to be shut down so you stop being charged."
            )

        if args.setup_only:
            sh.ok("Setup-only check passed. Nothing was trained.")
            return 0

        outcome = collect_results(ssh, steps, slug, settings.trigger_word)
        finished = True
        return outcome

    finally:
        # Terminating here used to be unconditional. That made this client the
        # sole custodian of a run whose only copy lives on a pod that deletes
        # itself: a Ctrl+C, a crash, or a closed console at step 1500 destroyed
        # every checkpoint. It stays unconditional in the one window where it
        # has to be - before the pod reports it can shut itself down - because
        # RunPod's own terminateAfter was measured NOT to fire, and a pod with
        # no killer at all is an open-ended bill.
        if finished or ssh is None or not ssh.self_terminate_armed:
            shut_down(runpod, pod_id, ssh)
        else:
            sh.say()
            sh.warn("Stopping before your results were saved to this PC.")
            sh.say()
            sh.say("  The rented computer is still working and will shut")
            sh.say("  itself down on its own. Nothing is lost yet.")
            sh.say()
            sh.say("  To pick it up again, double-click:")
            sh.say("      1 - TRAIN IN THE CLOUD.bat")
            sh.say("  It will reconnect to the same computer automatically.")
            sh.say()
            sh.say("  If you would rather stop and not be charged any more,")
            sh.say("  double-click \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\".")
            sh.say()
            if ssh is not None:
                ssh.close()


def collect_results(ssh: Ssh, steps: int, slug: str, trigger_word: str) -> int:
    """Watch training to the end, then bring the LoRA and previews home.

    Shared by a normal run and by --resume, so a resumed session behaves
    identically to one that was never interrupted.
    """
    result = monitor(ssh, steps, sh.OUTPUT_DIR)

    sh.header("Step 5 of 5 - collecting your results")
    saved = download_results(ssh, sh.OUTPUT_DIR)

    # Only now is it true. Releases the pod from its grace wait so it shuts
    # itself down within seconds instead of idling on the clock.
    if saved:
        try:
            ssh.run(f"touch {REMOTE_CLIENT_DONE}", timeout=30)
        except Exception:
            pass

    if result == "dead":
        sh.warn("The LoRA came out empty, so it is not worth keeping.")
        return 1
    if result == "failed" and not saved:
        return 1

    sh.summarise_results(sh.OUTPUT_DIR, slug)
    sh.say(f"  Type the word '{trigger_word}' in a prompt to use it.")
    sh.say()
    return 0


def resume(args: argparse.Namespace) -> int:
    """Reconnect to the pod from an interrupted run and finish the job.

    Exists because a client death used to be unrecoverable: the SSH key was
    generated in memory, so when the process went away the paid pod became
    unreachable. Now the key is on disk beside the pod id.
    """
    sh.header("Krystal's LoRA Trainer - resuming")

    settings = sh.load_or_ask(
        need_runpod=True,
        character_name=MY_CHARACTER_NAME,
        huggingface_token=MY_HUGGINGFACE_TOKEN,
        runpod_api_key=MY_RUNPOD_API_KEY,
    )
    steps = args.steps or settings.training_steps or 2000
    slug = sh.make_slug(settings.character_name)

    id_file = sh.ROOT / "LAST_POD_ID.txt"
    if not POD_KEY_FILE.exists():
        raise UserFacingError(
            "There is no interrupted run to pick up.\n\n"
            "  The key needed to get back into a rented computer is not\n"
            "  here, so there is nothing to reconnect to.\n\n"
            "  If you think one is still running and costing you money,\n"
            "  run \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\".\n\n"
            "  Otherwise just start a normal run."
        )

    key = load_saved_key(POD_KEY_FILE.read_text(encoding="utf-8"))
    runpod = RunPod(settings.runpod_api_key)

    # The key alone is enough to get back in, so a missing id is recoverable:
    # ask RunPod what is running instead of giving up. This matters because
    # the id file is the one piece of state written locally, and a run was
    # observed live (Aug 2026) where the key landed and the id did not,
    # which would otherwise have stranded a paid pod with no way back in.
    pod_id = ""
    if id_file.exists():
        lines = id_file.read_text(encoding="utf-8").splitlines()
        pod_id = lines[0].strip() if lines else ""
    if not pod_id:
        pod_id = find_running_pod(runpod)
    pod = runpod.get_pod(pod_id)
    status = ((pod or {}).get("desiredStatus") or "").upper()
    if not pod or status in ("EXITED", "TERMINATED", "FAILED"):
        try:
            id_file.unlink()
        except Exception:
            pass
        raise UserFacingError(
            f"That computer (id {pod_id}) is already shut down.\n\n"
            "  There is nothing to reconnect to and you are not being\n"
            "  charged for it. Start a fresh run when you are ready."
        )

    sh.ok(f"Found your computer still running (id {pod_id}).")

    ssh: Ssh | None = None
    finished = False
    try:
        host, port = wait_for_ssh_endpoint(runpod, pod_id)
        ssh = Ssh(host, port, key)
        ssh.connect(timeout_seconds=300)
        sh.ok("Reconnected. Picking up where it left off.")

        # Whether this client may stand down as the pod's killer is a fact to
        # check, not to assume: a pod whose setup died before the watchdog was
        # armed is still RUNNING and still reachable, and standing down there
        # would leave it billing with nothing able to stop it. Require both the
        # terminator script and a live wrapper to run it.
        _, _, term_code = ssh.run(
            f"test -f {REMOTE_WORK}/terminate_pod.sh", timeout=30
        )
        _, _, wrap_code = ssh.run("pgrep -f run_training.sh > /dev/null", timeout=30)
        ssh.self_terminate_armed = term_code == 0 and wrap_code == 0

        outcome = collect_results(ssh, steps, slug, settings.trigger_word)
        finished = True
        return outcome
    finally:
        if finished or ssh is None or not ssh.self_terminate_armed:
            shut_down(runpod, pod_id, ssh)
        else:
            sh.say()
            sh.warn("Stopping before your results were saved to this PC.")
            sh.say("  The rented computer is still working and will shut")
            sh.say("  itself down on its own. Double-click")
            sh.say("  \"1 - TRAIN IN THE CLOUD.bat\" to pick it up again.")
            sh.say()
            ssh.close()


def wait_for_ssh_endpoint(runpod: RunPod, pod_id: str, timeout: int = 600) -> tuple[str, int]:
    """Poll until the pod is running and exposes port 22 over direct TCP.

    Direct TCP is mandatory: RunPod's ssh.runpod.io proxy does not support
    SFTP, and we have a dataset to upload and a LoRA to bring back.
    """
    deadline = time.time() + timeout
    announced = False
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        if pod:
            status = (pod.get("desiredStatus") or "").upper()
            runtime = pod.get("runtime") or {}
            for port in runtime.get("ports") or []:
                if port.get("privatePort") == 22 and port.get("ip") and port.get("publicPort"):
                    if str(port.get("type", "tcp")).lower() != "tcp":
                        continue
                    return str(port["ip"]), int(port["publicPort"])
            if status in ("EXITED", "TERMINATED", "FAILED"):
                raise UserFacingError(
                    f"The rented computer stopped before it was ready (status {status}).\n\n"
                    "  This is usually a temporary RunPod problem. Please try again."
                )
            if not announced and status == "RUNNING":
                announced = True
                sh.step("  started, waiting for remote access to come up...")
        time.sleep(10)

    raise UserFacingError(
        "The rented computer never became reachable.\n\n"
        "  Run \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\" to make sure you are\n"
        "  not being charged, then try again."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a Krea 2 character LoRA on a rented cloud GPU"
    )
    parser.add_argument("--shutdown-all", action="store_true",
                        help="Terminate every pod on the account and exit")
    parser.add_argument("--setup-only", action="store_true",
                        help="Validate everything on the pod but do not train")
    parser.add_argument("--steps", type=int, default=0, help="Override training steps")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation")
    parser.add_argument("--resume", action="store_true",
                        help="Reconnect to an interrupted run and collect its results")
    parser.add_argument("--gpu-group", default="",
                        choices=[""] + [g["name"] for g in GPU_GROUPS],
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.shutdown_all:
            sh.header("Emergency shutdown")
            api_key = MY_RUNPOD_API_KEY
            if not api_key:
                saved = sh._load_settings_file()
                api_key = saved.get("runpod_api_key", "")
            if not api_key:
                api_key = sh.ask_secret("Paste your RunPod API key")
            if not api_key:
                raise UserFacingError(
                    "Without your RunPod key I cannot shut anything down.\n\n"
                    "  Please do it by hand instead:\n"
                    f"    {sh.RUNPOD_PODS_URL}"
                )
            return shutdown_all(api_key, assume_yes=args.yes)
        if args.resume:
            return resume(args)
        return run(args)

    except UserFacingError as exc:
        sh.say()
        sh.say("=" * 66)
        sh.problem(str(exc))
        sh.say("=" * 66)
        sh.say()
        return 1
    except KeyboardInterrupt:
        sh.say()
        sh.warn("Stopped by you.")
        sh.say()
        sh.say("  IMPORTANT: if a computer was already rented, it may still be")
        sh.say("  running. Double-click \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\"")
        sh.say("  to be certain you are not being charged.")
        sh.say()
        return 130
    except Exception as exc:
        sh.say()
        sh.say("=" * 66)
        sh.problem(f"Something unexpected went wrong: {exc}")
        sh.say("=" * 66)
        sh.say()
        sh.say("  Please send a screenshot of this window.")
        sh.say()
        sh.say("  Then double-click \"3 - EMERGENCY - SHUT DOWN CLOUD.bat\" to")
        sh.say("  make sure you are not being charged for anything.")
        sh.say()
        return 1


if __name__ == "__main__":
    sys.exit(main())
