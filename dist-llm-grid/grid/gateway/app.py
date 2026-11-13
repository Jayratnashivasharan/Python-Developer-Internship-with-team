"""Client-facing API gateway (FastAPI).

    GET  /health            liveness + cluster readiness
    GET  /metrics           TTFT, step latency, inter-node overhead, KV usage, throughput
    POST /v1/generate       one-shot completion (JSON)
    POST /v1/generate/stream  Server-Sent Events token stream
    WS   /ws/generate       real-time token streaming over a WebSocket

WebSocket protocol (JSON text frames)
    client -> {"prompt": "...", "max_new_tokens": 64, "temperature": 0.0, "top_k": 0}
    server -> {"type": "token", "token_id": 12, "text": "a"}   (repeated)
    server -> {"type": "done", "finish_reason": "stop|length", "ttft_ms": 12.3, "tokens": 17}
    server -> {"type": "error", "message": "..."}
One socket may carry many sequential requests; closing it aborts the in-flight one.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from grid.config import Settings
from grid.master.engine import Engine, QueueFull, SamplingParams
from grid.model.config import ModelConfig
from grid.tokenizer import StreamDecoder, load_tokenizer

log = logging.getLogger("grid.gateway")
STATIC = Path(__file__).parent / "static"


class GenerateIn(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(64, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=5.0)
    top_k: int = Field(0, ge=0)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [gateway] %(name)s: %(message)s")
        mcfg = ModelConfig.from_dir(settings.model_dir)
        tok = load_tokenizer(settings.tokenizer)
        engine = Engine(settings, eos_id=tok.eos_id, vocab_size=mcfg.vocab_size, max_len=mcfg.max_position_embeddings)
        await engine.start()
        app.state.engine, app.state.tok, app.state.mcfg = engine, tok, mcfg
        yield
        await engine.stop()

    app = FastAPI(title="Distributed LLM Grid", version="1.0.0", lifespan=lifespan)

    def _prep(body: GenerateIn):
        eng: Engine = app.state.engine
        tok = app.state.tok
        params = SamplingParams(body.max_new_tokens, body.temperature, body.top_k)
        try:
            return eng, tok, eng.submit(tok.encode(body.prompt), params)
        except QueueFull as e:
            raise HTTPException(503, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/info")
    async def info():
        eng: Engine = app.state.engine
        c = app.state.mcfg
        h, i, L, V = c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.vocab_size
        kv_dim = c.num_key_value_heads * c.head_dim
        params = V * h * 2 + L * (2 * h * h + 2 * h * kv_dim + 3 * h * i + 2 * h) + h
        return {
            "model": os.path.basename(os.path.normpath(settings.model_dir)),
            "demo_model": os.path.basename(os.path.normpath(settings.model_dir)) == "tiny-llama",
            "parameters": params, "layers": L, "hidden_size": h, "heads": c.num_attention_heads,
            "kv_heads": c.num_key_value_heads, "max_context": c.max_position_embeddings,
            "tensor_parallel_size": settings.world_size, "workers": eng.worker_info,
            "kv_block_size": settings.block_size, "kv_num_blocks": settings.num_blocks,
            "kv_capacity_tokens": settings.block_size * settings.num_blocks,
            "max_batch_seqs": settings.max_batch_seqs, "ready": eng.ready,
        }

    @app.get("/health")
    async def health():
        eng: Engine = app.state.engine
        return {"status": "ok" if eng.ready else "starting", "workers": settings.workers,
                "tensor_parallel_size": settings.world_size}

    @app.get("/metrics")
    async def metrics():
        return app.state.engine.stats()

    @app.post("/v1/generate")
    async def generate(body: GenerateIn):
        eng, tok, req = _prep(body)
        dec, text = StreamDecoder(tok), []
        try:
            async for t in eng.stream(req):
                text.append(dec.push(t))
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        return {"text": "".join(text), "tokens": req.generated, "finish_reason": req.finish_reason,
                "ttft_ms": req.ttft_ms, "total_ms": (time.perf_counter() - req.arrival) * 1000}

    @app.post("/v1/generate/stream")
    async def generate_stream(body: GenerateIn):
        eng, tok, req = _prep(body)

        async def gen():
            dec = StreamDecoder(tok)
            try:
                async for t in eng.stream(req):
                    yield f"data: {json.dumps({'token_id': t, 'text': dec.push(t)})}\n\n"
                yield f"data: {json.dumps({'done': True, 'finish_reason': req.finish_reason, 'ttft_ms': req.ttft_ms})}\n\n"
            except RuntimeError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.websocket("/ws/generate")
    async def ws_generate(ws: WebSocket):
        await ws.accept()
        eng: Engine = app.state.engine
        tok = app.state.tok
        req = None
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    body = GenerateIn(**json.loads(raw))
                    req = eng.submit(tok.encode(body.prompt),
                                     SamplingParams(body.max_new_tokens, body.temperature, body.top_k))
                except QueueFull as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
                except Exception as e:  # bad JSON / validation
                    await ws.send_json({"type": "error", "message": f"bad request: {e}"})
                    continue
                dec, n = StreamDecoder(tok), 0
                try:
                    async for t in eng.stream(req):
                        n += 1
                        await ws.send_json({"type": "token", "token_id": t, "text": dec.push(t)})
                    await ws.send_json({"type": "done", "finish_reason": req.finish_reason,
                                        "ttft_ms": req.ttft_ms, "tokens": n})
                except RuntimeError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
        except WebSocketDisconnect:
            if req is not None:
                eng.abort(req)   # free KV blocks of the abandoned request immediately

    return app


def get_app():
    return create_app()
