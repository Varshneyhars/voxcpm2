"""Example client for calling the deployed VoxCPM2 RunPod endpoint.

Usage:
    export RUNPOD_API_KEY=...
    export RUNPOD_ENDPOINT_ID=...
    python client_example.py
"""
import base64
import os
import sys
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def run_sync(payload: dict, timeout: int = 300) -> dict:
    r = requests.post(f"{BASE}/runsync", json=payload, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run_async(payload: dict, poll_interval: float = 2.0) -> dict:
    r = requests.post(f"{BASE}/run", json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    job_id = r.json()["id"]
    while True:
        s = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS, timeout=30).json()
        if s["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return s
        time.sleep(poll_interval)


def save_output(result: dict, path: str) -> None:
    out = result.get("output") or {}
    if "error" in out:
        print(f"[error] {out['error']}", file=sys.stderr)
        sys.exit(1)
    audio_b64 = out["audio_base64"]
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    print(f"saved: {path} ({out['duration_seconds']:.2f}s @ {out['sample_rate']}Hz)")


if __name__ == "__main__":
    payload = {
        "input": {
            "text": "Hello from VoxCPM2 running on RunPod serverless!",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }
    }
    result = run_sync(payload)
    save_output(result, "output.wav")

    payload_design = {
        "input": {
            "text": "(A young woman, gentle and sweet voice)Welcome to VoxCPM2.",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }
    }
    save_output(run_sync(payload_design), "voice_design.wav")
