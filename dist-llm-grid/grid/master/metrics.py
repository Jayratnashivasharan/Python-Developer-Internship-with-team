from __future__ import annotations

import time
from collections import deque


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p / 100 * len(s)))], 3)


class Metrics:
    """Rolling-window metrics exposed at /metrics (TTFT, network overhead, throughput)."""

    def __init__(self, window: int = 2000):
        self.started = time.time()
        self.requests_total = 0
        self.requests_finished = 0
        self.requests_failed = 0
        self.tokens_generated = 0
        self.steps = 0
        self.ttft_ms = deque(maxlen=window)
        self.step_ms = deque(maxlen=window)        # master-observed wall time per step
        self.compute_ms = deque(maxlen=window)     # slowest worker's forward time
        self.comm_ms = deque(maxlen=window)        # slowest worker's collective time
        self.net_ms = deque(maxlen=window)         # step wall time - slowest compute = gRPC + scheduling overhead
        self.batch_size = deque(maxlen=window)

    def record_step(self, wall_ms: float, replies, batch: int) -> None:
        self.steps += 1
        comp = max(r.compute_ms for r in replies)
        self.step_ms.append(wall_ms)
        self.compute_ms.append(comp)
        self.comm_ms.append(max(r.comm_ms for r in replies))
        self.net_ms.append(max(0.0, wall_ms - comp))
        self.batch_size.append(batch)

    def snapshot(self, running: int, waiting: int, kv_usage: float) -> dict:
        up = max(1e-9, time.time() - self.started)

        def block(d):
            return {"avg": round(sum(d) / len(d), 3) if d else None, "p50": _pct(d, 50), "p95": _pct(d, 95), "p99": _pct(d, 99)}

        return {
            "uptime_s": round(up, 1),
            "requests": {"total": self.requests_total, "finished": self.requests_finished,
                         "failed": self.requests_failed, "running": running, "waiting": waiting},
            "tokens_generated": self.tokens_generated,
            "tokens_per_s": round(self.tokens_generated / up, 2),
            "steps": self.steps,
            "kv_cache_usage": round(kv_usage, 4),
            "ttft_ms": block(self.ttft_ms),
            "step_ms": block(self.step_ms),
            "worker_compute_ms": block(self.compute_ms),
            "worker_allreduce_ms": block(self.comm_ms),
            "inter_node_overhead_ms": block(self.net_ms),
            "avg_batch_size": round(sum(self.batch_size) / len(self.batch_size), 2) if self.batch_size else None,
        }
