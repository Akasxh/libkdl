# Paper Reviews — Submission #91

## Review #91A — Weak Accept (3/5)
**Expertise:** 3/5 (Knowledgeable)

**Summary:** Although LLVM and MLIR enable portable code generation for machine learning, there is still a major runtime limitation: current systems cannot dynamically inspect available hardware and automatically route kernels across mixed NVIDIA, AMD, and CPU setups without recompiling for each target. It proposes to investigate architectures for vendor-agnostic runtime dispatch, drawing on existing ecosystems such as SYCL as well as lighter-weight, header-only designs to close this gap.

**Feedback:** Runtime support for heterogeneous ML workloads is an important and relatively under-explored problem, and the motivation you outline is compelling. However, from the current description it is not clear what concrete mechanism, design, or system the authors actually propose beyond surveying existing approaches and highlighting this gap. But I am fine having this talk as Lightning one.

**Key Takeaway:** Need a **concrete mechanism/design** — not just a survey.

---

## Review #91B — Weak Reject (2/5)
**Expertise:** 3/5 (Knowledgeable)

**Feedback:**
- In ML, kernels are well known at compile time along with tensor sizes → static graph of kernels
- Systems have dedicated accelerators for specific math ops → correct accelerator chosen at compile time
- Parallelism between multiple instances of an accelerator is still efficient — show how this fits
- ML uses PyTorch/TensorFlow — explain how runtimes collaborate with compiled kernel graphs from those frameworks

**Key Takeaway:** Address the **"ML kernels are static"** argument. Show value beyond compile-time dispatch. Connect to PyTorch/TF ecosystem.

---

## Review #91C — Weak Reject (2/5)
**Expertise:** 2/5 (Some familiarity)

**Feedback:**
- Abstract doesn't clearly explain if this is a survey or a proposal
- References very specific stacks that not everyone knows
- Recommend better motivation, introduction, and problem statement

**Key Takeaway:** **Clarify scope** — survey vs. proposal. Better intro for non-experts.

---

## Review #91D — Accept (4/5)
**Expertise:** 3/5 (Knowledgeable)

**Feedback:**
- Idea is very interesting and an important problem
- IREE references for context:
  - https://github.com/iree-org/iree/issues/15334
  - https://github.com/iree-org/iree/issues/12230
  - https://github.com/iree-org/iree/issues/50 (issue since IREE project start)
- IREE has a SPIR-V backend that generates vendor-agnostic code given specified hardware features — not fully vendor-locked
- Why SYCL specifically? The idea is more general — multi-versioned kernels specialized at JIT time by querying hardware features
- SPIR-V and even HIP/CUDA can do this — what makes SYCL special?

**Key Takeaway:** **Broaden beyond SYCL.** Frame as multi-versioned kernel JIT dispatch. Address IREE SPIR-V capabilities honestly. Explore what makes each approach unique.

---

## Synthesis of Reviews → Action Items

1. **MUST have a concrete contribution** — design/prototype, not just survey (91A, 91C)
2. **Address "ML is static" objection** — show dynamic scenarios: mixed hardware, model serving, edge deployment (91B)
3. **Connect to PyTorch/TF ecosystem** — show where runtime dispatch fits in real ML pipelines (91B)
4. **Broaden beyond SYCL** — compare SPIR-V, HIP, multi-versioned kernels, ALPAKA (91D)
5. **Better framing** — clear problem statement accessible to non-specialists (91C)
6. **Acknowledge IREE SPIR-V** correctly — it CAN generate vendor-agnostic code (91D)
