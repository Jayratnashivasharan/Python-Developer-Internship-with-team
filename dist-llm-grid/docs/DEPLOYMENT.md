# Deployment

## Local (no Docker)
```bash
pip install -r requirements.txt
python run.py                                  # web UI at http://localhost:8000
python scripts/ws_client.py "Hello"            # streaming demo
python scripts/load_test.py --concurrency 32 --requests 128
```
Real model: `python run.py --tp 2 --model-dir /models/Llama-3.2-1B --tokenizer /models/Llama-3.2-1B`
(needs `pip install transformers`; `num_attention_heads`, `num_key_value_heads`, `intermediate_size` must divide by `--tp`).

## Multiple machines (manual)
On every node: same repo, same model dir, same env:
```bash
export GRID_MODEL_DIR=/models/llama GRID_WORKERS=nodeA:50051,nodeB:50051 WORLD_SIZE=2 MASTER_ADDR=nodeA MASTER_PORT=29500
RANK=0 python -m grid.worker.server   # on nodeA
RANK=1 python -m grid.worker.server   # on nodeB
uvicorn grid.gateway.app:get_app --factory --host 0.0.0.0 --port 8000   # on any node
```
`GRID_WORKERS` must be listed **in rank order**; the master verifies this at start-up.
Open ports: 50051 (gRPC), 29500 (rendezvous), plus the ephemeral range used by Gloo/NCCL between workers.

## Docker Compose / Swarm / Kubernetes
| Target | File | Node affinity |
|---|---|---|
| Compose | `deploy/docker/docker-compose.yml` | – |
| Swarm | `deploy/swarm/stack.yml` | placement constraints on node labels `role`, `rank` |
| Kubernetes | `deploy/k8s/grid.yaml` | `nodeAffinity` (label `grid/role=worker`) + `podAntiAffinity` (1 rank/node) |

Build: `docker build -f deploy/docker/Dockerfile -t grid:cpu .` (GPU: see header of the Dockerfile).
Rank in Kubernetes is derived from the StatefulSet ordinal (`grid-worker-1` → rank 1).

## Configuration reference (env vars, `grid/config.py`)
| Var | Default | Meaning |
|---|---|---|
| `GRID_MODEL_DIR` | models/tiny-llama | HF-style dir with `config.json` + `*.safetensors` |
| `GRID_TOKENIZER` | byte | `byte` or HF tokenizer path/name |
| `GRID_WORKERS` | localhost:50051,localhost:50052 | worker gRPC addresses in rank order |
| `GRID_DIST_BACKEND` | auto | `nccl` / `gloo` / auto |
| `GRID_DTYPE` | float32 | float32, float16, bfloat16 |
| `GRID_NUM_BLOCKS`, `GRID_BLOCK_SIZE` | 512, 16 | KV pool geometry (**same on all nodes**) |
| `GRID_MAX_BATCH_SEQS` | 16 | max concurrent sequences per iteration |
| `GRID_MAX_PREFILL_TOKENS` | 512 | prompt-token budget admitted per iteration |
| `GRID_MAX_QUEUE` | 256 | waiting requests before HTTP 503 |
| `GRID_STEP_TIMEOUT` | 60 | seconds before a step is declared failed |
