#!/usr/bin/env python
"""One-command launcher: workers + gateway + web UI.

    python run.py                 # 2 tensor-parallel workers, opens http://localhost:8000
    python run.py --tp 4 --port 9000
    python run.py --model-dir /path/to/Llama --tokenizer /path/to/Llama
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def check_dependencies() -> None:
    mods = {"torch": "torch", "grpc": "grpcio", "grpc_tools": "grpcio-tools", "google.protobuf": "protobuf",
            "safetensors": "safetensors", "fastapi": "fastapi", "uvicorn": "uvicorn", "websockets": "websockets"}
    missing = []
    for m, pkg in mods.items():
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(pkg)
    if missing:
        sys.exit("Missing packages: " + ", ".join(missing) + "\nInstall them with:\n  pip install -r requirements.txt")


def ensure_proto() -> None:
    """Regenerate gRPC stubs if they are missing or incompatible with the installed protobuf."""
    try:
        importlib.import_module("grid.proto.inference_pb2_grpc")
        return
    except Exception:
        print("[run] (re)generating gRPC stubs for your protobuf version ...")
    subprocess.check_call([sys.executable, "-m", "grpc_tools.protoc", "-I.", "--python_out=.", "--grpc_python_out=.",
                           "grid/proto/inference.proto"], cwd=ROOT)
    for m in [k for k in sys.modules if k.startswith("grid.proto")]:
        del sys.modules[m]
    importlib.import_module("grid.proto.inference_pb2_grpc")


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        return s.connect_ex(("127.0.0.1", port)) != 0


def find_ports(start: int, count: int) -> int:
    base = start
    while base < 65000:
        if all(port_free(base + i) for i in range(count)):
            return base
        base += count
    sys.exit("no free ports found")


def stop_all(procs) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 8
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel size (number of worker processes)")
    ap.add_argument("--port", type=int, default=8000, help="web UI / API port")
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "models", "tiny-llama"))
    ap.add_argument("--tokenizer", default="byte")
    ap.add_argument("--num-blocks", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-seqs", type=int, default=16)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    def _term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    check_dependencies()
    ensure_proto()
    if not port_free(a.port):
        sys.exit(f"Port {a.port} is already in use. Choose another with:  python run.py --port {a.port + 1}")

    if not os.path.exists(os.path.join(a.model_dir, "config.json")):
        print(f"[run] creating tiny demo model in {a.model_dir}")
        from grid.model.config import ModelConfig
        from grid.weights.make_tiny_model import make
        make(a.model_dir, ModelConfig())
    from grid.model.config import ModelConfig as _MC
    _MC.from_dir(a.model_dir).validate_tp(a.tp)

    grpc_base = find_ports(50051, a.tp)
    dist_port = find_ports(29500, 1)
    addrs = [f"127.0.0.1:{grpc_base + r}" for r in range(a.tp)]
    env = {**os.environ, "PYTHONPATH": ROOT, "GRID_MODEL_DIR": a.model_dir, "GRID_TOKENIZER": a.tokenizer,
           "GRID_WORKERS": ",".join(addrs), "WORLD_SIZE": str(a.tp), "MASTER_ADDR": "127.0.0.1",
           "MASTER_PORT": str(dist_port), "GRID_NUM_BLOCKS": str(a.num_blocks), "GRID_BLOCK_SIZE": str(a.block_size),
           "GRID_MAX_BATCH_SEQS": str(a.max_batch_seqs), "PYTHONUNBUFFERED": "1"}
    procs: list[subprocess.Popen] = []
    try:
        print(f"[run] starting {a.tp} worker(s) + gateway ...")
        for r in range(a.tp):
            procs.append(subprocess.Popen([sys.executable, "-m", "grid.worker.server", "--rank", str(r),
                                           "--grpc-port", str(grpc_base + r)], env=env, cwd=ROOT))
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "grid.gateway.app:get_app", "--factory",
                                       "--host", "0.0.0.0", "--port", str(a.port), "--log-level", "warning"],
                                      env=env, cwd=ROOT))
        url = f"http://localhost:{a.port}"
        deadline = time.time() + 240
        while time.time() < deadline:
            dead = [p for p in procs if p.poll() is not None]
            if dead:
                print("[run] a process exited unexpectedly (see log above)")
                return
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as resp:
                    if json.load(resp).get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("[run] cluster did not become ready in time")
            return
        print(f"\n[run] READY  ->  {url}\n[run] press Ctrl+C to stop\n")
        if not a.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("[run] a process exited - shutting down")
    except KeyboardInterrupt:
        print("\n[run] stopping ...")
    finally:
        stop_all(procs)


if __name__ == "__main__":
    main()
