# Architecture

```
                        WebSocket / SSE / HTTP
   clients  ───────────────────────────────────►  Gateway (FastAPI)            grid/gateway/app.py
                                                        │ submit()/stream()
                                                  ┌─────▼──────────────────┐
                                                  │ Engine (master)        │   grid/master/engine.py
                                                  │  • continuous batching │
                                                  │  • KV BlockAllocator   │   grid/kvcache/paged.py
                                                  │  • metrics             │
                                                  └─────┬──────────────────┘
                              gRPC Step(batch)  (control plane, protobuf)
                     ┌──────────────┼───────────────┐
              ┌──────▼─────┐ ┌──────▼─────┐  ...  ┌─▼──────────┐
              │ Worker 0   │ │ Worker 1   │       │ Worker N-1 │              grid/worker/server.py
              │ shard 0/N  │ │ shard 1/N  │       │ shard N-1  │              grid/worker/runner.py
              │ KV tensors │ │ KV tensors │       │ KV tensors │
              └──────▲─────┘ └──────▲─────┘       └─▲──────────┘
                     └──── torch.distributed all-reduce (NCCL on GPU / Gloo on CPU) ────┘
                                    (data plane, activations)
```

## Two planes
* **Control plane (gRPC)** – the master tells *every* worker the same ragged batch (token ids, positions,
  block tables). Messages are a few hundred bytes, so gRPC overhead is negligible and measured
  (`inter_node_overhead_ms` in `/metrics`).
* **Data plane (torch.distributed)** – activations are exchanged by collectives. This is the Python binding
  to NCCL (`backend="nccl"` is auto-selected when CUDA is present; Gloo otherwise).
  Note: “microsecond” latency is achievable only on RDMA/NVLink fabrics; on commodity Ethernet expect
  ~0.1–1 ms per all-reduce, which is why the design needs only **2 all-reduces per layer**.

## Tensor parallelism (grid/model/parallel.py, llama.py)
| Weight | Split | Comm |
|---|---|---|
| q/k/v_proj (attention heads) | rows → each rank owns `H/N` heads and `Hkv/N` KV heads | none |
| o_proj | columns | all-reduce |
| gate/up_proj (FFN) | rows | none |
| down_proj | columns | all-reduce |
| embeddings, norms, lm_head | replicated | none |

Constraint: `heads`, `kv_heads` and `intermediate_size` divisible by N (validated at start-up).
Vocab-parallel embedding / lm_head is the natural next optimisation for very large vocabularies.

## Weight loading (grid/weights/loader.py)
`safe_open` memory-maps the checkpoint; `get_slice` reads only this rank's rows/columns, so no node ever
loads the full model. Parameter names match HuggingFace Llama (single file or sharded with an index).

## Paged KV cache (grid/kvcache/paged.py)
Pool of fixed-size blocks; each sequence owns a block list. Same-sized blocks ⇒ no external fragmentation.
The master owns the allocator (guarded by an `asyncio.Lock`), workers only store tensors and receive block
tables each step ⇒ nothing to synchronise. Admission reserves `prompt + max_new_tokens` blocks, which
guarantees a running sequence can never run out of memory mid-generation (no preemption needed).

## Continuous batching (grid/master/engine.py)
Each iteration: retire finished/cancelled → admit waiting requests (FCFS; limited by `max_batch_seqs`,
`max_prefill_tokens`, free blocks) → one forward pass where running sequences decode 1 token while new
ones prefill in the *same* batch → stream tokens. Verified by `tests/test_engine.py` and by
`test_ragged_batch_matches_individual_runs` (mixed batches are numerically identical to separate runs).

## Failure behaviour
Step timeout/worker error → current batch is failed with an error event, blocks freed, engine continues.
A dead worker breaks the collective group, so the supervisor (Swarm/K8s `restart`) must restart the whole
worker set; elastic re-sharding is out of scope.

## Known simplifications (honest limits)
* Attention loops over sequences in Python (clean and correct; a fused varlen/paged kernel is the production upgrade).
* Single master (scheduler is a single point of failure by design — simple and consistent).
* Sampling supports greedy / temperature / top-k.
