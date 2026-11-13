from __future__ import annotations

import asyncio
import logging

import grpc

from grid.proto import inference_pb2 as pb
from grid.proto import inference_pb2_grpc as pb_grpc

log = logging.getLogger("grid.master")

_OPTS = [
    ("grpc.max_receive_message_length", 64 * 2**20),
    ("grpc.max_send_message_length", 64 * 2**20),
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.tcp_user_timeout_ms", 10_000),
]


class WorkerClient:
    def __init__(self, address: str, expected_rank: int):
        self.address, self.expected_rank = address, expected_rank
        self.channel = grpc.aio.insecure_channel(address, options=_OPTS)
        self.stub = pb_grpc.WorkerStub(self.channel)

    async def wait_ready(self, timeout_s: float) -> pb.HealthReply:
        loop, deadline, delay = asyncio.get_running_loop(), None, 0.5
        deadline = loop.time() + timeout_s
        while True:
            try:
                rep = await self.stub.Health(pb.HealthRequest(), timeout=5)
                if rep.ready:
                    if rep.rank != self.expected_rank:
                        raise RuntimeError(f"{self.address} reports rank {rep.rank}, expected {self.expected_rank}; "
                                           "GRID_WORKERS must be listed in rank order")
                    return rep
            except grpc.aio.AioRpcError:
                pass
            if loop.time() > deadline:
                raise TimeoutError(f"worker {self.address} not ready after {timeout_s}s")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 3.0)

    async def step(self, req: pb.StepRequest, timeout_s: float) -> pb.StepReply:
        return await self.stub.Step(req, timeout=timeout_s)

    async def close(self) -> None:
        await self.channel.close()
