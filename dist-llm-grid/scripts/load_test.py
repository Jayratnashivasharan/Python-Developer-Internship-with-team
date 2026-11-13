"""Concurrent load test over the WebSocket API.

    python scripts/load_test.py --url ws://localhost:8000/ws/generate --concurrency 32 --requests 128

Reports client-side TTFT, end-to-end latency, throughput, then pulls /metrics from the
gateway for server-side inter-node latency (gRPC overhead and all-reduce time).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request

import websockets


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))] if s else float("nan")


async def one(url, prompt, max_new, results, sem):
    async with sem:
        t0 = time.perf_counter()
        ttft, n, err = None, 0, None
        try:
            async with websockets.connect(url, max_size=2**22) as ws:
                await ws.send(json.dumps({"prompt": prompt, "max_new_tokens": max_new, "temperature": 0.8, "top_k": 40}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg["type"] == "token":
                        n += 1
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                    elif msg["type"] == "done":
                        break
                    else:
                        err = msg.get("message")
                        break
        except Exception as e:
            err = repr(e)
        results.append({"ttft": ttft, "total": (time.perf_counter() - t0) * 1000, "tokens": n, "err": err})


async def main(a):
    results: list[dict] = []
    sem = asyncio.Semaphore(a.concurrency)
    prompt = "Explain distributed inference. " * a.prompt_repeat
    t0 = time.perf_counter()
    await asyncio.gather(*(one(a.url, prompt, a.max_new_tokens, results, sem) for _ in range(a.requests)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if not r["err"]]
    tokens = sum(r["tokens"] for r in ok)
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    tot = [r["total"] for r in ok]
    report = {
        "concurrency": a.concurrency, "requests": a.requests, "succeeded": len(ok), "failed": len(results) - len(ok),
        "wall_s": round(wall, 2), "throughput_tok_s": round(tokens / wall, 1), "req_per_s": round(len(ok) / wall, 2),
        "ttft_ms": {"p50": round(pct(ttft, 50), 1), "p95": round(pct(ttft, 95), 1), "p99": round(pct(ttft, 99), 1)},
        "latency_ms": {"p50": round(pct(tot, 50), 1), "p95": round(pct(tot, 95), 1), "p99": round(pct(tot, 99), 1)},
        "errors": sorted({r["err"] for r in results if r["err"]})[:5],
    }
    try:
        base = a.url.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]
        report["server_metrics"] = json.load(urllib.request.urlopen(base + "/metrics", timeout=5))
    except Exception as e:
        report["server_metrics"] = f"unavailable: {e!r}"
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws/generate")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt-repeat", type=int, default=2)
    ap.add_argument("--out", default="")
    asyncio.run(main(ap.parse_args()))
