import threading
import time


class Stats:
    """Thread-safe counters shared by pipeline stages and the dashboard."""
    FIELDS = ("reads_total", "reads_seeded", "reads_filtered", "reads_skipped",
              "reads_aligned", "reads_lowscore", "gpu_cpu_mismatch", "batches",
              "bases", "bytes_done", "bytes_total")

    def __init__(self, backend: str = "?"):
        self._lock = threading.Lock()
        self.backend = backend
        self.device = ""
        self.t0 = time.perf_counter()
        self.t_end = None
        self.queue_depth = 0
        self.kernel_seconds = 0.0
        self.variant_events = 0
        for f in self.FIELDS:
            setattr(self, f, 0)
        self.bytes_total = 1

    def add(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    @property
    def elapsed(self):
        return (self.t_end or time.perf_counter()) - self.t0

    @property
    def reads_per_sec(self):
        return self.reads_total / self.elapsed if self.elapsed > 0 else 0.0

    def finish(self):
        self.t_end = time.perf_counter()

    def as_dict(self):
        d = {f: getattr(self, f) for f in self.FIELDS if f not in ("bytes_done", "bytes_total")}
        d.update(backend=self.backend, device=self.device, elapsed_s=round(self.elapsed, 3),
                 reads_per_sec=round(self.reads_per_sec, 1),
                 kernel_seconds=round(self.kernel_seconds, 3))
        return d
