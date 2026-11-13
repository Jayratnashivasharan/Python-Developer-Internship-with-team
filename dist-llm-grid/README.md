# Distributed Memory Grid for Real-Time LLM Inference

A from-scratch distributed inference engine in **PyTorch + gRPC + asyncio** that shards LLM weights across
several modest nodes (tensor parallelism), keeps GPU/CPU memory tidy with a **paged KV cache**, and serves
many users at once with **continuous batching**, streamed token-by-token over **WebSockets**.

| Requirement | Where it lives |
|---|---|
| Master–worker topology, asyncio + gRPC | `grid/master/`, `grid/worker/`, `grid/proto/inference.proto` |
| safetensors + memory-mapped, shard-only loading | `grid/weights/loader.py` |
| Tensor parallelism (heads + FFN) with `torch.distributed` | `grid/model/parallel.py`, `grid/model/llama.py` |
| Async KV-cache manager (no fragmentation) | `grid/kvcache/paged.py` |
| Continuous batching (iteration-level) | `grid/master/engine.py` |
| NCCL (auto) / Gloo backend | `grid/worker/server.py::init_distributed` |
| FastAPI + WebSocket streaming gateway | `grid/gateway/app.py` |
| Docker / Swarm / Kubernetes with node affinity | `deploy/` |
| Load testing (TTFT, inter-node latency) | `scripts/load_test.py`, `/metrics` |
| Web application (chat, live dashboard, load test) | `grid/gateway/static/index.html` |

## Quick start (CPU only, no GPU needed)
```bash
pip install -r requirements.txt
python run.py                      # starts 2 workers + gateway, opens http://localhost:8000
```
That's it. The first start creates a tiny demo model automatically. Press **Ctrl+C** to stop everything.

Options: `python run.py --tp 4 --port 9000 --no-browser`, or a real model:
`python run.py --model-dir /path/to/Llama --tokenizer /path/to/Llama` (see `docs/DEPLOYMENT.md`).

### The web application (http://localhost:8000)
| Tab | What it does |
|---|---|
| **Playground** | Chat with the cluster; tokens stream live over WebSocket; temperature / top-k / length controls; Stop button |
| **Dashboard** | Live tokens/s, running/waiting requests, TTFT, step time, all-reduce time, gRPC overhead, KV-cache usage and charts |
| **Load test** | Fires N concurrent requests from the browser and reports TTFT p50/p95/p99, throughput, latency |
| **Cluster & API** | Model details, worker ranks/backends, API reference |

The UI is responsive (phone, tablet, desktop), has light/dark themes and needs no internet access.
The demo model has random weights, so its text is gibberish — it proves the *system*. Use a real
HuggingFace Llama checkpoint for meaningful text.

### Other commands
```bash
python scripts/ws_client.py "Hello"                       # terminal streaming client
python scripts/load_test.py --concurrency 32 --requests 128   # terminal load test
python -m pytest tests -q                                  # 8 tests
```

## API
* `WS /ws/generate` — send `{"prompt": "...", "max_new_tokens": 64, "temperature": 0.7, "top_k": 40}`, receive
  `{"type":"token","text":...}` frames, then `{"type":"done",...}`.
* `POST /v1/generate` (JSON), `POST /v1/generate/stream` (SSE), `GET /health`, `GET /metrics`.

## Project layout
```
grid/
  config.py            env-based settings shared by all processes
  proto/               gRPC contract + generated stubs (make proto)
  model/               Llama architecture, tensor-parallel layers
  weights/             safetensors shard loader, tiny-model generator
  kvcache/             paged block allocator
  worker/              gRPC worker + model runner (one TP rank)
  master/              engine (continuous batching), worker client, metrics
  gateway/             FastAPI app (HTTP, SSE, WebSocket) + static/index.html web UI
run.py                 one-command launcher (workers + gateway + UI)
scripts/               load_test, ws_client
tests/                 TP equivalence, paged cache, continuous batching
deploy/                docker, swarm, k8s
docs/                  architecture, deployment, evaluation report
```
Read `docs/ARCHITECTURE.md` first; it explains the design and the honest limits.
