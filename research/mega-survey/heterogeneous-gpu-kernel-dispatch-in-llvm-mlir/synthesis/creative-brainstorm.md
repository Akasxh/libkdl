# See agent output for full content - writing summary version

## Top 20 Poster Ideas for EuroLLVM Dublin 2026 (Non-libkdl)

### TOP PICKS (Highest feasibility × resonance):

**#15. offload-ld — dlopen for GPU Kernels** (Novelty: 10/10)
- `offload_dlopen()` / `offload_dlsym()` / `offload_dlclose()` for GPU kernels
- Addresses Issue #75356 (open 2.5 years)
- ~500 LOC C, upstream to offload/tools/

**#5. rankImage() callback for liboffload** (Novelty: 8/10)
- Replace "first compatible wins" in PR #186088 with scored selection
- The literal follow-up PR the community asked for
- Joseph Huber will stop at this poster

**#1. gpu.select_variant — Runtime variant selection MLIR op** (Novelty: 9/10)
- Replace compile-time #gpu.select_object with runtime callback
- Fills 3-year gap in GPU/Offloading workshop topics

**#12. Cross-vendor dispatch overhead measurement** (Novelty: 8/10)
- First published comparison: cuLaunchKernel vs ol* API vs CPU fallback
- Novel data nobody else has, runs on GTX 1650

### COMBO STRATEGIES:

**Combo A: "The Missing Runtime Half"** (Ideas 5+8+12)
rankImage() + capability annotations + dispatch overhead measurement

**Combo B: "offload-ld: dlopen for GPUs"** (Ideas 15+18+6)
offload-ld + llvm-offload-nm + olEnumerateSymbols

**Combo C: "Static Analysis to Runtime Dispatch"** (Ideas 4+10+8)
KernelInfo bridge + FLOP counting + capability metadata

See full agent output for all 20 ideas with details.
