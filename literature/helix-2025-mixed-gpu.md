# Helix: Serving LLMs over Heterogeneous GPUs via Max-Flow
## Literature Note — LLVM Dublin 2026

**Paper:** "Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow"
**Venue:** ASPLOS 2025 (30th ACM ASPLOS, Rotterdam, March 2025)
**Authors:** CMU Parallel Data Laboratory (Thesys-lab)
**arXiv:** https://arxiv.org/abs/2406.01566
**Code:** https://github.com/Thesys-lab/Helix-ASPLOS25
**Relevance Score:** 7/10

---

## Finding

Helix achieves up to **3.3x throughput improvement** and **66% / 24% latency reduction** (prompt / decode phases) over prior approaches by jointly optimizing model placement and request scheduling in heterogeneous GPU clusters of 24–42 nodes.

---

## Core Technical Contribution

### Problem formulation

Prior serving systems (e.g., plain vLLM, Alpa) separate model placement from request scheduling and assume homogeneous GPU clusters. In a heterogeneous cluster, the two problems are deeply entangled: a placement decision that ignores scheduling creates bottlenecks; a scheduler that ignores placement cannot balance load.

Helix reframes the entire cluster as a **directed weighted graph**:
- **Nodes** = GPU instances, with capacity proportional to memory and compute throughput
- **Edges** = network links between nodes, with capacity set by measured bandwidth
- **Objective** = maximize total request throughput = max-flow through the graph

### MILP-based model placement (ILPLayout)

Given the graph, Helix solves a **Mixed Integer Linear Programming (MILP)** problem to determine:
1. Which model layers reside on which GPU nodes
2. How pipeline parallelism is organized across heterogeneous GPU types

Key insight: LLM layers are homogeneous (transformer blocks are structurally identical), so the number of MILP variables and constraints scales **linearly** in the number of nodes and edges — not quadratically — making the problem tractable even for 42-node clusters. Gurobi is used as the MILP solver.

Three heuristic baselines (round-robin, greedy, memory-proportional) are compared; the MILP layout consistently dominates.

### Per-request pipeline scheduler

Standard serving systems assign a fixed set of pipeline replicas and route requests round-robin. Helix introduces **per-request pipelines**: each incoming request gets its own independently selected path through the GPU graph at dispatch time, based on current load and capacity.

The scheduler:
1. At request arrival, enumerates available paths through the placed model layers
2. Selects the path with the highest instantaneous residual capacity (max-flow residual)
3. Dispatches prefill and decode tokens along that path
4. Releases capacity when the request completes

This dynamic per-request dispatch is the runtime analog of our libkdl dispatch: instead of routing requests, libkdl routes kernel variants — but the capacity-aware selection principle is the same.

### Profiling and heterogeneity capture

Helix profiles each GPU type offline to populate edge/node capacities:
- **Compute capacity:** token throughput (tokens/sec) per GPU type for prefill and decode stages
- **Memory capacity:** KV cache memory budget per GPU
- **Network capacity:** inter-node bandwidth (GB/s), measured at deployment

No online per-kernel profiling occurs. Capacities are static inputs to the MILP. This is a design choice that trades dynamic adaptivity for tractable planning.

---

## Architecture Overview

```
Cluster profiling → Capacity graph (nodes=GPUs, edges=links)
                         ↓
            MILP solver (Gurobi) → Model placement plan
                         ↓
        Runtime scheduler: per-request pipeline selection
                         ↓
      vLLM 0.4.0post1 for kernel execution per GPU
      ZeroMQ for inter-node messaging
      Unified page pool (KV cache) atop vLLM
```

Implementation: 1.5k LoC Python + 1.7k LoC C++.

---

## Performance Results

| Metric | Improvement vs baselines |
|--------|--------------------------|
| Throughput | Up to 3.3x |
| Prompt latency | Up to 66% reduction |
| Decode latency | Up to 24% reduction |
| Cluster sizes tested | 24–42 GPU nodes |

Baselines tested: Petals (decentralized serving), AlpaServe (static pipeline), and round-robin vLLM variants.

---

## Relevance to libkdl

### Direct connections

1. **Heterogeneous dispatch model:** Helix routes *requests* to GPU pipelines based on real-time capacity; libkdl routes *kernel variants* to GPU hardware based on detected capability. The abstraction level differs (request scheduling vs kernel dispatch) but the problem structure is analogous: one artifact (model / kernel binary), multiple possible execution targets, need for efficient dynamic selection.

2. **Capacity-aware selection:** Helix's residual-capacity scheduler is essentially a lightweight runtime oracle. Our libkdl dispatch table lookup is the kernel-level equivalent — matching capability profile to pre-compiled kernel variant.

3. **Proof that static dispatch is insufficient:** Helix's 3.3x gain over round-robin (which is effectively static scheduling) directly quantifies the cost of ignoring hardware heterogeneity at dispatch time. This is a strong citation for motivating dynamic dispatch in libkdl.

4. **vLLM integration:** Helix layers on top of vLLM. Our work could similarly complement vLLM's kernel dispatch layer (Triton-generated kernels) with cross-vendor runtime selection.

### Key difference from libkdl

Helix operates at the **cluster/serving** level — it decides which machine runs which layers. libkdl operates at the **process/kernel** level — it decides which compiled kernel variant runs on the detected GPU. Helix assumes each GPU already has the correct kernel for its type; libkdl is the mechanism that would make that assumption hold without requiring separate binary builds.

---

## Risks / Gaps

- Helix's MILP planning is **offline / static** — it does not react to GPU failures or workload distribution shifts mid-deployment. The paper acknowledges this as future work.
- Profiling is per-GPU-type, not per-kernel. If a GPU model (e.g., RTX 3090 vs A100) runs the same kernel at wildly different latencies, the capacity estimate may be coarse.
- The 3.3x gain is cluster-level throughput, not per-kernel speedup. Citing this figure in a kernel-dispatch context requires careful framing.
- Code is open-source but tied to vLLM 0.4.0 and Gurobi (commercial solver).

---

## Key Citations from This Paper

- Helix itself: Thesys-lab, ASPLOS 2025, ACM DL: https://dl.acm.org/doi/10.1145/3669940.3707215
- vLLM (used as kernel executor): Kwon et al., SOSP 2023
- AlpaServe (baseline): Li et al., OSDI 2022
- Petals (baseline): Borzunov et al., 2022

---

## Notes for Poster

- Use the 3.3x figure to quantify "cost of ignoring heterogeneity" in the motivation section
- Contrast: Helix solves placement+scheduling at cluster level; libkdl solves kernel selection at process level — complementary, not competing
- Cite in "Related Work: Runtime-Heterogeneous Serving Systems"
- The per-request pipeline scheduler is a direct inspiration for libkdl's per-dispatch capability query
