"""Live terminal dashboard (Rich): throughput, progress, pipeline health."""
from __future__ import annotations
import time
from collections import deque

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


class Dashboard:
    def __init__(self, stats, enabled: bool = True, refresh: float = 4.0):
        self.stats, self.enabled, self.refresh = stats, enabled, refresh
        self._live = None
        self._hist = deque(maxlen=12)       # (time, reads) for instantaneous rate

    def _rate(self):
        now = time.perf_counter()
        self._hist.append((now, self.stats.reads_total))
        (t0, r0), (t1, r1) = self._hist[0], self._hist[-1]
        return (r1 - r0) / (t1 - t0) if t1 > t0 else 0.0

    def render(self):
        s = self.stats
        pct = 100.0 * s.bytes_done / max(s.bytes_total, 1)
        rate = self._rate()
        gpu_share = 100.0 * s.kernel_seconds / s.elapsed if s.elapsed else 0.0
        t = Table.grid(padding=(0, 2))
        t.add_column(style="cyan"); t.add_column(justify="right"); t.add_column(style="cyan"); t.add_column(justify="right")
        t.add_row("Backend", f"{s.backend} {s.device}".strip(), "Elapsed", f"{s.elapsed:7.1f} s")
        t.add_row("Reads processed", f"{s.reads_total:,}", "Throughput (now)", f"{rate:,.0f} reads/s")
        t.add_row("Seeded -> GPU", f"{s.reads_seeded:,}", "Throughput (avg)", f"{s.reads_per_sec:,.0f} reads/s")
        t.add_row("Filtered (no seed)", f"{s.reads_filtered:,}", "Kernel busy", f"{gpu_share:5.1f} %")
        t.add_row("Aligned (score ok)", f"{s.reads_aligned:,}", "Prep queue depth", f"{s.queue_depth}")
        t.add_row("Low score", f"{s.reads_lowscore:,}", "Variant events", f"{s.variant_events:,}")
        t.add_row("Batches", f"{s.batches:,}", "GPU/CPU mismatches", f"{s.gpu_cpu_mismatch}")
        bar = ProgressBar(total=100, completed=pct, width=60)
        return Panel(Group(t, Text(""), bar, Text(f"{pct:5.1f}% of input file")),
                     title="[bold green]gpualign[/] - alignment progress", border_style="green")

    def __enter__(self):
        if self.enabled:
            self._live = Live(self.render(), refresh_per_second=self.refresh, auto_refresh=False)
            self._live.__enter__()
            import threading
            self._stop = threading.Event()

            def loop():
                while not self._stop.wait(1.0 / self.refresh):
                    self._live.update(self.render(), refresh=True)
            self._th = threading.Thread(target=loop, daemon=True)
            self._th.start()
        return self

    def __exit__(self, *exc):
        if self._live:
            self._stop.set(); self._th.join()
            self._live.update(self.render(), refresh=True)
            self._live.__exit__(*exc)
