# Post-Dublin 6-Month Roadmap — Runtime Variant Selection for LLVM GPU Offloading

**Author:** Akash (IIT Patna, CERN GSoC alumnus, vLLM contributor)
**Created:** 2026-04-10
**Scope:** April 2026 (post-EuroLLVM Dublin) through October 2026 (LLVM US Dev Meeting)
**Prerequisites:** EuroLLVM poster presented, metadata RFC posted to Discourse, prototype benchmarks published

---

## Executive Summary

The EuroLLVM Dublin poster established three things: (1) a 5-key metadata vocabulary for OffloadBinary, (2) the first published flame graph of the LLVM GPU dispatch stack, and (3) a design sketch for `#gpu.runtime_select`. The 6-month roadmap converts these from poster contributions into upstream patches, a full paper, and a GSoC-viable project. The critical path is: RFC acceptance (May) -> first patches landed (June-July) -> full paper submitted (August) -> LLVM US Dev Meeting talk (October).

---

## 1. Conference Follow-Ups

### 1.1 Full Paper Submission Targets

| Venue | Deadline (est.) | Format | Fit | Priority |
|-------|-----------------|--------|-----|----------|
| **CGO 2027** | ~September 2026 | 12-page full paper | **Best fit.** CGO published Proteus (Georgakoudis 2025), SYCL-MLIR (2024), MLIR itself (Lattner 2021). Compiler+runtime dispatch is core CGO territory. | **PRIMARY** |
| **CC 2027** (ACM SIGPLAN Compiler Construction) | ~November 2026 | 12-page full paper | Good fit for the MLIR attribute design and IR-level dispatch mechanism. CC values novel IR extensions. | SECONDARY |
| **LLVM Dev Meeting US 2026** (October, San Jose) | ~July 2026 talk proposal | 25-min technical talk | **Must submit.** Present the implemented `#gpu.runtime_select` attribute with upstream patch status. Talk proposals are lightweight (abstract + bio). | **REQUIRED** |
| **IWOCL/SYCLcon 2027** | ~January 2027 | Short paper / talk | Relevant for the cross-vendor dispatch angle. AdaptiveCpp SSCP (Alpay 2023, 2025) was published here. Good for SYCL community visibility. | OPTIONAL |
| **SC 2026 Workshop (P3HPC or LLVM-HPC)** | ~August 2026 | 6-8 page workshop paper | The performance portability angle (P3 metric, ALPAKA/Kokkos comparison) fits P3HPC. LLVM-HPC workshop is direct community overlap. | STRETCH |

**Paper evolution from poster:**

The poster presents three contributions: metadata vocabulary (T07), dispatch flame graph (T19), and `#gpu.runtime_select` design sketch (T01). The full paper must add:

1. **Implementation evidence** -- working `RuntimeSelectAttr.cpp` with upstream-quality tests (the poster acknowledges "zero lines of MLIR C++ exist")
2. **Multi-hardware evaluation** -- at minimum GTX 1650 (Turing) + one AMD GPU (borrow/cloud). The poster explicitly states "AMD path mocked only"
3. **End-to-end workload** -- dispatch a real ML kernel (GEMM or attention) across heterogeneous targets, not just null kernels
4. **Comparison with IREE HAL dispatch** -- the poster references IREE issues #50/#12230/#15334 but provides no head-to-head measurement
5. **Formal cost model analysis** -- the poster calls the selection logic a "weighted heuristic"; the paper needs to evaluate against an oracle or autotuned baseline

### 1.2 Talk Proposal for LLVM US Dev Meeting 2026

**Proposed title:** "From First-Compatible to Best-Compatible: Runtime Variant Selection for `gpu.binary`"

**Abstract draft (200 words):**

> MLIR's gpu-module-to-binary pass can compile a single gpu.module to NVIDIA, AMD, and Intel targets simultaneously. The OffloadBinary container carries all variants. But at runtime, LLVM's offload stack picks the first compatible image and stops -- PR #186088's parseOffloadBinary loop has no ranking mechanism and zero timing instrumentation.
>
> We present three contributions. First, a standard metadata vocabulary (5 keys: min_sm, min_gfx, requires_features, variant_priority, variant_tag) that extends OffloadBinary's string-map to enable runtime selection beyond triple/arch matching. Second, the first published per-layer latency decomposition of the LLVM GPU dispatch stack: hot-path dispatch floor 4.26 us, module load 10.1 us warm / 54.6 us cold, variant selection 4 ns amortized (measured on GTX 1650, CUDA 13.1). Third, `#gpu.runtime_select` -- a new MLIR attribute implementing OffloadingLLVMTranslationAttrInterface that emits a dispatch table and vendor-detection stub at compile time with zero steady-state overhead. The design is inspired by CPU Function Multi-Versioning IFunc resolvers.
>
> We discuss the upstream patch status, community feedback from EuroLLVM Dublin, and the path from metadata RFC to landed attribute.

**Submission timeline:** Talk proposals typically due ~July for October meeting. Monitor llvm.org/devmtg/ for the 2026 US call-for-talks.

---

## 2. Upstream Patch Timeline — 12 Patches over 6 Months

Ordered by ascending review risk. Each patch is independently reviewable. The dependency graph is linear for the first 3 patches, then fans out.

### Phase 1: Foundation (May-June 2026) — Lowest Risk

| # | Patch | Component | LOC | Risk | Depends On | Target Week |
|---|-------|-----------|-----|------|------------|-------------|
| 1 | Header constants for metadata keys | `llvm/include/llvm/Object/OffloadBinary.h` | ~30 | **Trivial** — additive constants, zero functional change | RFC acceptance on Discourse | May W1 |
| 2 | Documentation for OffloadBinary metadata | `llvm/docs/OffloadBinary.rst` (new) | ~100 | **Low** — docs-only, no code change | Patch 1 | May W2 |
| 3 | `isMetadataCompatible()` Tier 1 extension | `offload/plugins-nextgen/common/PluginInterface.cpp` | ~40 | **Low** — opt-in check, backward-compatible, no behavior change for binaries without new keys | Patch 1 | May W3-W4 |

**Gating milestone:** If Patch 1 gets LGTM, the vocabulary is accepted and all subsequent patches build on standard keys.

### Phase 2: Backend Writers (June-July 2026) — Medium Risk, Parallelizable

| # | Patch | Component | LOC | Risk | Depends On | Target Week |
|---|-------|-----------|-----|------|------------|-------------|
| 4 | AMDGPU writer: emit `min_gfx` from target-id | `llvm/lib/Target/AMDGPU/` (OffloadBinary emission path) | ~60 | **Medium** — touches AMDGPU backend; needs Saiyedul Islam review | Patch 1 | June W1-W2 |
| 5 | NVPTX writer: emit `min_sm` from chip attribute | clang-linker-wrapper pipeline / NVPTX backend | ~60 | **Medium** — must verify correct integration point (ClangOffloadWrapper.cpp may be superseded) | Patch 1 | June W1-W2 |
| 6 | `llvm-offload-binary --annotate` flag | `llvm/tools/llvm-offload-binary/` | ~80 | **Low** — tooling addition, no runtime behavior change | Patch 1 | June W3 |
| 7 | `variant_priority` / `variant_tag` propagation from `gpu.object` to OffloadBinary | `mlir/lib/Target/LLVMIR/Dialect/GPU/` | ~50 | **Medium** — MLIR-side metadata propagation | Patch 1 | June W3-W4 |

**Patches 4-7 are independent of each other.** Submit all four in parallel after Patch 1 lands.

### Phase 3: The Attribute (July-September 2026) — Higher Risk

| # | Patch | Component | LOC | Risk | Depends On | Target Week |
|---|-------|-----------|-----|------|------------|-------------|
| 8 | `#gpu.runtime_select` TableGen definition | `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td` | ~30 | **Medium** — new attribute in GPU dialect; needs Fabian Mora alignment on RFC #88170 | RFC #88170 direction | July W1 |
| 9 | `RuntimeSelectAttr.cpp` (embedBinary + launchKernel) | `mlir/lib/Target/LLVMIR/Dialect/GPU/` | ~400 | **High** — core implementation; the most reviewed patch in the series | Patch 8 | July W2-August W2 |
| 10 | `GPURuntimeSelectWrappers.cpp` | `mlir/lib/ExecutionEngine/` | ~200 | **Medium** — runtime helper library; dlopen-based vendor detection | Patch 9 | August W1-W2 |
| 11 | `--gpu-mark-runtime-select` pass | `mlir/lib/Dialect/GPU/Transforms/` | ~50 | **Low** — convenience pass; walks modules with 2+ targets | Patch 8 | August W1 |
| 12 | Integration tests (2 .mlir files + lit config) | `mlir/test/Dialect/GPU/` | ~100 | **Low** — tests; most reviewers prefer these alongside the implementation | Patch 9 | August W2-W3 |

**Total: ~1,200 LOC across 12 patches.**

### Contingency: PR #186088 Merge Scenarios

| Scenario | Impact on Roadmap |
|----------|-------------------|
| **#186088 merges before May** | Reframe: "first-compatible-wins is now mainline DEFAULT; ranked selection is the obvious next step." Patch 3 extends the merged `isMetadataCompatible()`. Strongest position. |
| **#186088 remains stalled** | Patch 3 targets the existing `isMetadataCompatible()` from PR #185663. Still viable; the consumer hook exists even without generalized OffloadBinary support. |
| **#186088 is closed/rejected** | Reframe: the metadata vocabulary is independently useful for `llvm-offload-binary` tooling and documentation. `#gpu.runtime_select` operates at the MLIR layer, above liboffload, and does not depend on #186088. |

### Contingency: RFC #88170 Outcomes

| Scenario | Impact on Patch 8-9 |
|----------|---------------------|
| **RFC #88170 lands before July** | `#gpu.runtime_select` fills the policy slot the RFC explicitly creates. Best case. |
| **RFC #88170 changes `gpu.binary` structure** | TableGen definition (Patch 8) changes; `embedBinary`/`launchKernel` logic unchanged. Moderate rework. |
| **RFC #88170 stalls indefinitely** | `#gpu.runtime_select` lands independently as a standalone `OffloadingLLVMTranslationAttrInterface` implementation. The interface extension point does not depend on the RFC. |

---

## 3. GSoC 2027 Proposal Angle

### 3.1 Project Framing

**Title:** "Runtime Variant Selection for LLVM GPU Offloading: From Metadata to Dispatch"

**Scope:** Implement and upstream the `#gpu.runtime_select` attribute, extending LLVM's GPU offload stack from first-compatible to best-compatible binary selection.

**Why it works as a GSoC project:**

1. **Bounded scope:** 12 patches, ~1,200 LOC total, 6-month timeline fits a GSoC medium/large project
2. **Clear deliverables:** Each patch is independently testable and mergeable
3. **Existing design:** The refined-design-v1.md provides complete pseudocode for every component
4. **Existing prototype:** libkdl (5,100 LOC) validates the runtime mechanics on real hardware
5. **Mentor availability:** Joseph Huber (liboffload maintainer) and Joel Denny (KernelInfo pass author) are the natural mentors -- both are already CC'd on the RFC
6. **Student profile:** Requires familiarity with MLIR TableGen, LLVM IR emission, and GPU runtime APIs. An advanced undergrad or early-stage grad student with LLVM experience could complete this.

### 3.2 Proposed Milestones (12-week GSoC timeline)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Metadata RFC posted, header constants patch submitted | Patches 1-2 reviewed |
| 3-4 | `isMetadataCompatible()` extension + one backend writer | Patches 3 + (4 or 5) landed |
| 5-6 | Second backend writer + tooling patch | Patches (4 or 5) + 6 landed |
| 7-8 | `#gpu.runtime_select` TableGen + initial `embedBinary()` | Patch 8 reviewed, Patch 9 WIP |
| 9-10 | Complete `RuntimeSelectAttr.cpp` + runtime helpers | Patches 9-10 reviewed |
| 11-12 | Integration tests + `--gpu-mark-runtime-select` pass + final review | Patches 11-12 landed, project complete |

### 3.3 Mentor Candidates

| Person | Affiliation | Why |
|--------|-------------|-----|
| **Joseph Huber** (`@jhuber6`) | AMD, liboffload/OffloadBinary maintainer | Owns the runtime consumer code. Most patches touch his domain. Primary mentor. |
| **Joel Denny** (`@jdenny-ornl`) | ORNL, KernelInfo pass author | Owns the metadata production pipeline (D123878). Co-mentor for backend writer patches. |
| **Fabian Mora** (`@fabianmcg`) | RFC #88170 author, GPU dialect cleanup | Alignment on `gpu.binary` policy slot. Advisory role. |

### 3.4 Application Strategy

- GSoC 2027 applications typically open January 2027
- By January 2027, the metadata vocabulary patches (1-6) should already be landed, providing concrete upstream evidence
- The GSoC project scope would be Patches 7-12 (the MLIR attribute and runtime components)
- Alternatively, if all 12 patches are landed by the proposer before GSoC opens, the GSoC project could extend to **Phase 2 features**: learned cost model, resource-usage keys (Tier 2: sgpr_count, vgpr_count), or multi-device dispatch

---

## 4. Collaboration Targets

### 4.1 National Labs

| Person / Group | Lab | Connection Point | Outreach Action |
|----------------|-----|------------------|-----------------|
| **Joel Denny** | ORNL | KernelInfo pass (D123878) -- metadata pipeline from compiler to OffloadBinary | Post in RFC thread; mention KernelInfo-to-OffloadBinary writer as a joint effort. Denny's pass produces the data; our vocabulary standardizes where it goes. |
| **Giorgis Georgakoudis** | LLNL | Proteus JIT (CGO 2025) -- runtime specialization composable with AOT variant selection | Email after Dublin. Proteus JIT + `#gpu.runtime_select` AOT is a natural composition: JIT specializes within a variant tier, AOT selects the tier. Joint paper opportunity for CGO 2027. |
| **Sunita Chandrasekaran / IRIS team** | ORNL/U Delaware | IRIS task-based runtime (IEEE TPDS 2024) -- runtime dispatch across CUDA/HIP/L0/OpenCL | IRIS wraps vendor APIs but has no cost-model selection. Our metadata vocabulary could provide the ranking signal IRIS lacks. Workshop paper at SC 2026 (LLVM-HPC). |
| **HEP-CCE / CMS Experiment** | Fermilab / CERN | ALPAKA production deployment with 80 build configurations | The poster's "so what?" story. Contact via CERN GSoC alumni network or HEP-CCE mailing list. A letter of support or co-authorship on the CGO paper would strengthen the use-case argument. |

### 4.2 Industry (Vendor Compiler Teams)

| Person / Group | Company | Connection Point | Outreach Action |
|----------------|---------|------------------|-----------------|
| **Joseph Huber** (`@jhuber6`) | AMD | liboffload maintainer, PR #186088 reviewer | Already engaged. Continue RFC thread interaction. Push for Patch 3 review. |
| **Alex Duran** (`@adurang`) | Intel/BSC | PR #186088 author (OffloadBinary generalization) | Post in PR #186088 thread: "We have complementary metadata vocabulary work. Should we coordinate?" Duran's PR establishes the multi-plugin OffloadBinary path; our metadata enriches what flows through it. |
| **Saiyedul Islam** | AMD(?) | AMDGPU offload, CC on RFC | Directly relevant for Patch 4 (AMDGPU writer for `min_gfx`). Request review when patch is ready. |
| **sarnex** (Samuel Antao?) | Intel | L0 plugin maintainer, commented on #186088 | Relevant for Intel/SPIR-V OffloadBinary path. The metadata vocabulary benefits L0 plugin's `isMetadataCompatible()`. |
| **Yury Plyakhin** | (OffloadBinary format) | PR #169425 (format v2) | Format version awareness. CC on metadata RFC to ensure vocabulary does not conflict with format v2 changes. |

### 4.3 Academic Collaborators

| Person / Group | Institution | Connection Point |
|----------------|-------------|------------------|
| **Aksel Alpay** | Heidelberg | AdaptiveCpp SSCP (IWOCL 2023, 2025). Single-pass JIT is complementary to AOT variant selection. |
| **Pablo Martinez** | (rocMLIR, EuroLLVM talk) | MLIR compilation for AMD GPUs. `#gpu.runtime_select` would let rocMLIR-compiled kernels be selected alongside NVPTX kernels at runtime. |
| **Pennycook et al.** | Intel | P3 (Performance Portability) metric. Our dispatch mechanism changes the P3 calculation -- runtime selection improves P3 by avoiding worst-case fallback. Evaluation methodology collaboration. |

---

## 5. Research Extensions

### 5.1 Immediate Extensions (6-12 months, paper-ready)

| Extension | Description | Effort | Paper Venue |
|-----------|-------------|--------|-------------|
| **Learned cost model** | Replace the "weighted heuristic with vendor-specific constants" (kdl.c:1013-1088) with an ML-trained recommender. Train on kernel metadata (register count, shared mem, workgroup size) + device features -> predict best variant. The cuBLAS precedent (93% optimal) suggests ML-based selection significantly outperforms heuristics. | 3-6 months | CGO 2027 (as evaluation section) or standalone ML-for-compilers paper |
| **Multi-device pipeline dispatch** | Extend from single-kernel dispatch to graph-level dispatch: given a computation graph and N devices, assign subgraphs to devices based on cost model. This is the IREE #15334 problem (multi-versioning epic) solved at a different layer. | 6-12 months | SC 2027 workshop or ASPLOS 2028 |
| **Proteus + runtime_select composition** | Proteus (LLNL) does JIT specialization at runtime; `#gpu.runtime_select` does AOT variant selection. Compose: AOT selects the best pre-compiled variant for the detected hardware, Proteus further specializes it at runtime for the specific input shape. Measure the combined P3 improvement. | 3-6 months, requires LLNL collaboration | CGO 2027 (joint with Georgakoudis) |
| **Resource-usage keys (Tier 2)** | The deferred keys: `sgpr_count`, `vgpr_count`, `registers_per_thread`, `shared_mem_bytes`. These require KernelInfo-to-OffloadBinary writer integration in each backend. Enables occupancy-aware variant selection. | 2-3 months per backend | Follow-up RFC on Discourse + upstream patches |
| **SPIR-V fallback tier** | Add SPIR-V as a universal fallback in `#gpu.runtime_select`. When no native binary matches, JIT-compile a SPIR-V variant via the appropriate vendor's SPIR-V consumer (Vulkan, L0, clvk). Measure the P3 cost of the SPIR-V fallback (literature suggests 50-80% of native). | 3-6 months | IWOCL/SYCLcon 2027 |

### 5.2 Longer-Term Research Directions (12-24 months)

| Direction | Description | Connection |
|-----------|-------------|------------|
| **Feedback-directed variant selection** | Use runtime profiling data (kernel execution time, occupancy achieved) to adjust variant selection decisions online. Similar to PGO for dispatch: if the cost model was wrong, switch variants on the next invocation. | Connects to PR #190708 (PGO for AMDGPU offload) |
| **Cross-node dispatch in distributed settings** | Extend from single-node multi-GPU to multi-node heterogeneous clusters. The metadata vocabulary works unchanged; the dispatch mechanism needs a network-aware cost model. | Connects to HEP-CCE distributed use case, Helix 2025 (mixed GPU serving) |
| **Formal verification of dispatch correctness** | Prove that the variant selection mechanism always selects a functionally correct variant (i.e., the selected binary's requirements are satisfied by the device's capabilities). This is non-trivial for `requires_features` with vendor-specific token semantics. | Connects to AMD formal memory model work (EuroLLVM talk by van Houtryve) |
| **torch.compile integration** | Implement a torch.compile backend that uses MLIR multi-target compilation + `#gpu.runtime_select` dispatch. This is the "connection to PyTorch/TF ecosystem" that reviewer 91B demanded. A torch.compile backend that transparently dispatches across NVIDIA/AMD without user intervention would be a significant practical contribution. | Connects to torch-mlir-bridge.md analysis, vLLM experience |

### 5.3 Experiment Wishlist (Requires Hardware Access)

| Experiment | Hardware Needed | Purpose |
|------------|-----------------|---------|
| AMD validation | MI250X or MI300X (cloud or lab access) | Remove "AMD path mocked only" disclaimer. Measure actual HIP dispatch latency for head-to-head comparison with CUDA. |
| Intel validation | Ponte Vecchio or Arc (Aurora access or Intel DevCloud) | Validate Level Zero dispatch path. Exercise the full tri-vendor story (NVVM + ROCDL + XeVM). |
| Multi-GPU dispatch | System with 2+ different GPU models | Test actual runtime variant selection on a heterogeneous system (e.g., NVIDIA + AMD in same host, or two NVIDIA generations). |
| Scale test | 8-GPU node | Measure dispatch table scaling: does variant selection overhead increase with device count? Expected: O(N*M) where N = devices, M = variants. |
| End-to-end ML workload | Any multi-GPU system | Dispatch a real GEMM/attention kernel across heterogeneous targets. Measure throughput vs. single-target baseline. |

---

## 6. Community Building

### 6.1 Discourse Engagement (Continuous)

| Action | Timeline | Category |
|--------|----------|----------|
| **Post metadata vocabulary RFC** | April W3 (immediately after Dublin) | Runtimes / Offloading |
| **Post flame graph results** | May W1 (after RFC discussion starts) | Runtimes / Offloading |
| **Post `#gpu.runtime_select` RFC** | June W1 (after metadata RFC gets initial feedback) | MLIR / GPU Dialect |
| **Respond to every review comment within 48 hours** | Continuous | -- |
| **Post monthly status updates in RFC threads** | Monthly | -- |
| **Engage in existing threads:** RFC #88170 (GPU cleanup), #74302 (llvm-project/offload), #75611 (offload roadmap) | Continuous | -- |

### 6.2 LLVM Mailing List / Discourse Presence

| Thread | Action | Purpose |
|--------|--------|---------|
| RFC #88170 (Cleaning the GPU Dialect) | Post: "We propose `#gpu.runtime_select` as the first dispatch-policy implementation for the policy slot this RFC creates." | Establish alignment with Fabian Mora's cleanup direction |
| RFC #75611 (offload roadmap) | Post: "Multi-image handling is listed as future work. We have a concrete metadata vocabulary and measurement data." | Get on the official roadmap |
| RFC #74302 (Introducing llvm-project/offload) | Monitor for architectural changes that affect the plugin interface | Stay aware of liboffload evolution |
| GPU/Offloading Workshop call-for-talks (when announced) | Submit talk proposal | Community visibility |

### 6.3 Workshop Proposals

| Workshop | Venue | Topic | Timeline |
|----------|-------|-------|----------|
| **LLVM-HPC Workshop** | SC 2026 (November) | "Runtime Dispatch for Heterogeneous GPU Clusters: Metadata, Measurement, and Mechanism" | Paper deadline ~August 2026 |
| **P3HPC Workshop** | SC 2026 (November) | "How Runtime Variant Selection Changes the Performance Portability Equation" | Paper deadline ~August 2026 |
| **GPU/Offloading Workshop** | LLVM US Dev Meeting 2026 (October) | Tutorial or lightning talk on `#gpu.runtime_select` implementation | Proposal deadline ~July 2026 |

### 6.4 Open-Source Artifacts

| Artifact | Location | Timeline |
|----------|----------|----------|
| **libkdl prototype** (cleaned, documented) | GitHub public repo | May 2026 -- publish alongside RFC |
| **Benchmark reproduction scripts** | Same repo, `benchmarks/` directory | May 2026 |
| **Flame graph SVGs** | Same repo, `results/` directory | May 2026 |
| **RFC patch drafts** on Phabricator/GitHub PRs | llvm/llvm-project | Per patch timeline above |
| **runtime_select_poc** (OffloadBinary PoC) | Same repo | Already exists, clean up and document |

### 6.5 Blog / Technical Writing

| Post | Platform | Timeline | Purpose |
|------|----------|----------|---------|
| "The GPU Dispatch Stack Nobody Measured" | LLVM blog (if accepted) or personal blog | May 2026 | Publicize flame graph results; drive traffic to RFC |
| "From `ld.so` to GPU: Runtime Variant Selection in LLVM" | Personal blog / Medium | June 2026 | Accessible explanation for non-LLVM-experts; useful for GSoC recruitment |
| "EuroLLVM Dublin Trip Report" | LLVM Discourse (Trip Reports category) | April W4 | Community engagement; document conversations with Huber, Mora, Denny |

---

## 7. Month-by-Month Timeline

### April 2026 (Post-Dublin)

- [x] Present poster at EuroLLVM Dublin (April 15)
- [ ] Have 1:1 conversations with Huber, Mora, Denny at the conference
- [ ] Post trip report on Discourse
- [ ] Post metadata vocabulary RFC on Discourse (rfc-READY-TO-POST.md is ready)
- [ ] Check PR #186088 merge status and update framing

### May 2026

- [ ] Respond to RFC feedback; iterate on key definitions based on community input
- [ ] Submit Patch 1 (header constants) and Patch 2 (documentation)
- [ ] Submit Patch 3 (isMetadataCompatible extension)
- [ ] Publish libkdl prototype and benchmarks on GitHub
- [ ] Post flame graph results on Discourse
- [ ] Submit talk proposal for LLVM US Dev Meeting 2026

### June 2026

- [ ] Submit Patches 4-7 in parallel (AMDGPU writer, NVPTX writer, tooling, MLIR propagation)
- [ ] Begin `RuntimeSelectAttr.cpp` implementation
- [ ] Contact Georgakoudis (LLNL) re: Proteus composition paper
- [ ] Contact HEP-CCE re: use-case validation / co-authorship
- [ ] Blog post: "The GPU Dispatch Stack Nobody Measured"

### July 2026

- [ ] Submit Patch 8 (TableGen definition)
- [ ] Submit Patch 9 (RuntimeSelectAttr.cpp) -- the core implementation
- [ ] LLVM US Dev Meeting talk proposal deadline (verify exact date)
- [ ] Begin CGO 2027 paper draft (sections 1-3: intro, background, related work)
- [ ] Seek AMD hardware access for real HIP validation

### August 2026

- [ ] Submit Patches 10-12 (runtime helpers, convenience pass, integration tests)
- [ ] Complete CGO 2027 paper draft (sections 4-6: design, evaluation, discussion)
- [ ] SC 2026 workshop paper deadlines (LLVM-HPC, P3HPC)
- [ ] Internal review of CGO paper with collaborators

### September 2026

- [ ] **CGO 2027 submission deadline** (verify exact date)
- [ ] All 12 upstream patches submitted; target 6+ landed
- [ ] Prepare LLVM US Dev Meeting talk slides
- [ ] CC 2027 paper deadline consideration (if CGO rejected, pivot here)

### October 2026

- [ ] **LLVM US Dev Meeting** (San Jose) -- present talk
- [ ] Engage GSoC 2027 planning (identify potential students, draft project idea)
- [ ] Post-mortem: which patches landed? What community feedback requires design changes?
- [ ] Begin GSoC 2027 project idea page draft for llvm.org/gsoc

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Metadata RFC rejected: community prefers different key names or structure | Medium | High -- blocks all downstream patches | Lead with the *mechanism* (standardized keys in string-map) not the *specific names*. Be willing to rename keys based on feedback. The RFC is a starting point, not a decree. |
| `#gpu.runtime_select` deemed out-of-scope for MLIR GPU dialect | Low-Medium | High -- blocks Patches 8-12 | Fallback: implement as an out-of-tree attribute first. The `OffloadingLLVMTranslationAttrInterface` is explicitly designed for extension. Out-of-tree attributes are loadable via plugins. |
| PR #186088 merges with a different metadata approach | Low | Medium -- requires vocabulary alignment | Monitor the PR. If Duran/Huber define metadata keys in #186088, adopt their naming. Our RFC should be posted first to establish priority. |
| No AMD hardware access for 6 months | Medium | Medium -- weakens paper evaluation section | Use AMD cloud instances (Azure NC series with MI300X, or AMD Instinct Cloud). Budget ~$50-100 for cloud GPU hours. Alternatively, contact AMD DevRel for hardware loan. |
| CGO 2027 paper rejected | Medium | Medium -- delays publication but not upstream work | Pivot to CC 2027 (November deadline) or ASPLOS 2028. The upstream patches are valuable regardless of paper acceptance. |
| Joseph Huber becomes unresponsive / leaves project | Low | High -- blocks review of most patches | Build relationships with multiple reviewers: Denny, Duran, Plyakhin. Never depend on a single reviewer. |
| GSoC 2027 not accepted as LLVM project | Low | Low -- GSoC is a nice-to-have, not a dependency | The upstream patches proceed regardless. GSoC would accelerate Phase 2 features (learned cost model, resource-usage keys). |

---

## 9. Success Metrics

### 6-Month Checkpoint (October 2026)

| Metric | Target | Stretch |
|--------|--------|---------|
| Upstream patches submitted | 12 | All 12 reviewed |
| Upstream patches landed | 6+ (Phases 1-2) | All 12 |
| Discourse RFC threads | 2 (metadata + attribute) | 3 (+ flame graph tooling) |
| Conference submissions | 2 (LLVM US talk + CGO paper) | 3 (+ SC workshop) |
| External collaborators engaged | 3 (Huber, Denny, 1 lab) | 5+ |
| Blog posts published | 1 | 3 |
| GitHub stars on prototype repo | 10 | 50 |
| AMD hardware validation | Cloud hours acquired | Physical hardware access |

### 12-Month Checkpoint (April 2027)

| Metric | Target | Stretch |
|--------|--------|---------|
| All 12 patches landed upstream | Yes | Phase 2 patches also started |
| CGO 2027 paper accepted | Submitted | Accepted |
| GSoC 2027 project accepted | Project idea posted | Student matched |
| Community adoption | 1 downstream user (torch-mlir, IREE, or chipStar) | 3+ downstream users |
| Discourse presence | Regular contributor to GPU/Offload threads | Recognized community member |

---

## Appendix A: Key File References

| File | What It Contains |
|------|------------------|
| `proposals/proposal-v2.md` | Complete poster proposal with 3 contributions, risk register, and reviewer responses |
| `proposals/refined-design-v1.md` | Full technical design: TableGen, embedBinary pseudocode, measurement methodology, RFC structure |
| `proposals/rfc-READY-TO-POST.md` | Discourse-ready RFC text for metadata vocabulary |
| `proposals/rfc-metadata-vocabulary.md` | Detailed RFC with composition examples and upgrade path |
| `proposals/elevator-pitch.md` | 15/30/60-second scripts and per-person talking points |
| `research/pr-status-check.md` | Status of PRs #186088, #185663, and related upstream work as of April 9, 2026 |
| `research/benchmark-results.md` | Raw benchmark data from GTX 1650 (bench_dispatch, bench_layers) |
| `research/real-offloadbinary-results.md` | PoC results with real CUBINs packed into OffloadBinary format |
| `/home/akash/PROJECTS/LLVM/research/paper-outline.md` | Full paper structure with section-by-section literature mapping and citation priority |

## Appendix B: Upstream PR/RFC Dependency Map

```
                    RFC: Metadata Vocabulary
                    (Discourse, April W3)
                            |
                   +--------+--------+
                   |                 |
              Patch 1            (community
           (header consts)       feedback)
                   |
          +--------+--------+--------+
          |        |        |        |
       Patch 3  Patch 4  Patch 5  Patch 6
       (isMeta) (AMDGPU) (NVPTX)  (tooling)
          |        |
          |     Patch 7
          |   (MLIR propagation)
          |        |
          +--------+
                |
         RFC: #gpu.runtime_select
         (Discourse, June W1)
                |
           Patch 8
         (TableGen def)
                |
       +--------+--------+
       |        |        |
    Patch 9  Patch 10  Patch 11
    (impl)   (runtime) (pass)
       |
    Patch 12
    (tests)
```

---

*Roadmap produced: 2026-04-10*
*Based on: proposal-v2.md, refined-design-v1.md, pr-status-check.md, paper-outline.md, elevator-pitch.md, benchmark-results.md, real-offloadbinary-results.md*
*All timelines are estimates; actual dates depend on community review cadence and hardware access*
