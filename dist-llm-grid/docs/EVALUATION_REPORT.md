# Final Evaluation Report — Distributed Memory Grid for Real-Time LLM Inference
Prepared for: Zaalima Development Pvt. Ltd.  |  Version: v1.0

## 1. Mid-project review (end of Week 2)
| Criterion | Status | Evidence |
|---|---|---|
| Network topology & master–worker communication operational | ✅ | `grid/proto/inference.proto`, `grid/master/worker_client.py`, `grid/worker/server.py`; `/health` shows all ranks |
| Tensor parallelism distributes weights without data loss | ✅ | `tests/test_tensor_parallel.py`: logits for TP=2 and TP=4 equal single-device (atol 1e-4); shard-only loading test |

## 2. Final review (end of Week 4)
| Criterion | Status | Evidence |
|---|---|---|
| Engine generates tokens across multiple nodes | ✅ | Local run: 2 worker processes (Gloo/CPU), 72/72 requests OK; multi-host via `deploy/` |
| Continuous batching & KV-cache optimisation verified | ✅ | `tests/test_engine.py`, `test_ragged_batch_matches_individual_runs`, `test_paged_cache.py`; avg batch size 10.51 under load; KV usage returned to 0% (no leaks) |
| Architecture report & deployment scripts | ✅ | `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `deploy/*` |

## 3. Load-test results (reference run)
**Environment:** 1 shared CPU core in a sandbox, TP=2 (Gloo), random 4-layer demo model, 24 concurrent WebSocket clients,
72 requests × 32 new tokens. Absolute numbers are *not* representative of GPU/multi-host hardware — re-run
`scripts/load_test.py` on the target cluster and paste the results here. Raw data: `docs/load_test_tp2_cpu.json`.

| Metric | Value |
|---|---|
| Success | 72/72 |
| Throughput | 605.5 tok/s (20.37 req/s) |
| TTFT p50 / p95 / p99 (client) | 292.2 / 762.6 / 915.8 ms |
| End-to-end latency p50 / p95 | 977.3 / 1516.2 ms |
| Step time (server) avg / p95 | 19.073 / 33.32 ms |
| Inter-node overhead (gRPC + scheduling) avg / p95 | 2.074 / 3.76 ms |
| All-reduce time per step avg | 10.028 ms |

TTFT under 24-way concurrency is dominated by queueing for batch slots (`max_batch_seqs=16`), not by network cost.

## 4. Limitations & future work
Fused paged-attention kernel, vocab-parallel lm_head, pipeline parallelism for >N-heads limits, elastic recovery, prefix caching, authentication on the gateway.
