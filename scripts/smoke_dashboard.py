"""Check real launcher startup and shutdown from an isolated tracked source archive."""

from __future__ import annotations

import io
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=PROJECT_ROOT)
    with tempfile.TemporaryDirectory(prefix="jobcopilot-launcher-smoke-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(root, filter="data")
        home = root / "isolated-home"
        home.mkdir()
        env = {
            "PATH": os.defpath,
            "HOME": str(home),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        health_url = f"http://127.0.0.1:{port}/_stcore/health"
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with (root / "launcher.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable, "run_dashboard.py", "--server.headless=true",
                    "--server.address=127.0.0.1", f"--server.port={port}",
                    "--browser.gatherUsageStats=false",
                ],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Launcher exited early: {process.returncode}")
                    try:
                        with client.open(health_url, timeout=1) as response:
                            if response.status == 200 and response.read().strip() == b"ok":
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Dashboard health endpoint did not become ready")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise RuntimeError("Launcher did not stop after SIGINT") from None
            if process.returncode != 0:
                raise RuntimeError(f"Launcher shutdown exit code: {process.returncode}")
            try:
                with client.open(health_url, timeout=1):
                    raise RuntimeError("Dashboard health endpoint remained open after shutdown")
            except (urllib.error.URLError, TimeoutError):
                pass
        print("Isolated supported launcher HTTP startup and shutdown passed.")


if __name__ == "__main__":
    main()
