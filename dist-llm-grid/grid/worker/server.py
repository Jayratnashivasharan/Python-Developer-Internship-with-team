"""Worker process: one tensor-parallel rank exposed through gRPC.

    RANK=0 WORLD_SIZE=2 python -m grid.worker.server --grpc-port 50051

Start-up order: join torch.distributed -> load this rank's weight shard -> serve gRPC.
The master retries ``Health`` until every worker reports ``ready``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
import torch.distributed as dist

from grid.config import Settings
from grid.model.parallel import TPContext
from grid.proto import inference_pb2 as pb
from grid.proto import inference_pb2_grpc as pb_grpc
from grid.worker.runner import ModelRunner

log = logging.getLogger("grid.worker")


def resolve_rank() -> int:
    """RANK env var, or the StatefulSet ordinal from the hostname (grid-worker-1 -> 1)."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    m = re.search(r"-(\d+)$", socket.gethostname())
    if not m:
        raise RuntimeError("set RANK, or run with a hostname that ends in -<ordinal>")
    return int(m.group(1))


def init_distributed(settings: Settings, rank: int, world: int) -> tuple[TPContext, torch.device, str]:
    """NCCL when GPUs are present (the Python binding to NCCL is torch.distributed), else Gloo."""
    use_cuda = torch.cuda.is_available()
    backend = settings.dist_backend
    if backend == "auto":
        backend = "nccl" if use_cuda else "gloo"
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if use_cuda and backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{settings.master_addr}:{settings.master_port}",
            rank=rank,
            world_size=world,
            device_id=device if device.type == "cuda" else None,
        )
    tp = TPContext(rank=rank, world_size=world, group=None, sync_for_timing=device.type == "cuda")
    return tp, device, backend


class WorkerService(pb_grpc.WorkerServicer):
    def __init__(self, runner: ModelRunner, backend: str):
        self.runner, self.backend = runner, backend
        # ONE thread => collectives are issued in the same order on every rank (no deadlocks).
        self.exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tp-step")

    async def Step(self, request: pb.StepRequest, context) -> pb.StepReply:  # noqa: N802
        loop = asyncio.get_running_loop()
        try:
            nxt, compute_ms, comm_ms = await loop.run_in_executor(self.exec, self.runner.step, request)
        except Exception as e:  # surface model errors to the master instead of hanging it
            log.exception("step failed")
            await context.abort(grpc.StatusCode.INTERNAL, repr(e))
        return pb.StepReply(step_id=request.step_id, rank=self.runner.tp.rank, next_tokens=nxt,
                            compute_ms=compute_ms, comm_ms=comm_ms)

    async def Health(self, request, context) -> pb.HealthReply:  # noqa: N802
        r = self.runner
        return pb.HealthReply(ready=True, rank=r.tp.rank, world_size=r.tp.world_size, backend=self.backend,
                              device=str(r.device), free_kv_blocks=r.s.num_blocks)


async def serve(args) -> None:
    settings = Settings()
    rank = args.rank if args.rank is not None else resolve_rank()
    world = int(os.environ.get("WORLD_SIZE", settings.world_size))
    threads = int(os.environ.get("GRID_TORCH_THREADS", max(1, (os.cpu_count() or 1) // world)))
    torch.set_num_threads(threads)

    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [rank {rank}] %(name)s: %(message)s")
    log.info("joining group (rank %d / world %d) via %s:%d", rank, world, settings.master_addr, settings.master_port)
    tp, device, backend = init_distributed(settings, rank, world)
    runner = ModelRunner(settings, tp, device)

    server = grpc.aio.server(options=[
        ("grpc.max_receive_message_length", 64 * 2**20),
        ("grpc.max_send_message_length", 64 * 2**20),
        ("grpc.keepalive_time_ms", 20_000),
        ("grpc.http2.max_pings_without_data", 0),
    ])
    pb_grpc.add_WorkerServicer_to_server(WorkerService(runner, backend), server)
    port = args.grpc_port or settings.grpc_port
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    log.info("ready: gRPC on :%d, backend=%s, device=%s, threads=%d", port, backend, device, threads)
    await server.wait_for_termination()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--grpc-port", type=int, default=None)
    asyncio.run(serve(ap.parse_args()))


if __name__ == "__main__":
    main()
