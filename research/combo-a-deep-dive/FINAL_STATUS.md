# FINAL STATUS — EuroLLVM Dublin 2026

## Converged + Updated: April 10 landscape refresh complete.

### Score: 3.675/5 → estimated 3.9/5 after updates
- Novelty: 3.5→3.75 | Soundness: 4.0 | Significance: 3.0→3.25 | Presentation: 4.0 | Reproducibility: 4.0
- Improved through multiple review rounds
- **April 10 updates:** 4 new papers cited, HetGPU in comparison table, 4 new Q&A cards, Discourse-ready RFC

### To reach 4.5 (post-Dublin targets):
1. Compile RuntimeSelectAttr.cpp against MLIR trunk → +1.0 (needs mlir-opt)
2. AMD GPU measurement → +0.5 (needs ROCm hardware)

### What's ready for April 15:
- `poster/poster-combo-a.html` — A0, print-ready, now with HetGPU comparison + bibliography
- `proposals/extended-abstract-v3.tex` — 4 pages, now citing HetGPU + KernelEvolve + Universal GPU ISA + AdaptiveCpp
- `proposals/handout.pdf` — 1 page takeaway
- `proposals/rfc-FINAL.md` — substance
- `proposals/rfc-discourse-ready.md` — paste-ready for discourse.llvm.org
- `proposals/elevator-pitch.md` — practice this
- `proposals/qa-cards-final.md` — 29 cards (added HetGPU, KernelEvolve, AdaptiveCpp, Universal ISA)
- `proposals/visitor-personas.md` — read before the session
- `literature/` — 4 new papers analyzed (hetgpu, kernelevolve, universal-gpu-isa, adaptivecpp)
- `literature/competitive-landscape.md` — updated with all 4 new entries + references

### RuntimeSelectAttr Validation
- `experiments/prototype/src/RuntimeSelectAttr-validation.md` — API correctness report

### Action items:
1. `xdg-open poster/poster-combo-a.html` — verify poster looks correct
2. Recompile: `cd research/combo-a-deep-dive/proposals && pdflatex extended-abstract-v3.tex`
3. Post `rfc-discourse-ready.md` to https://discourse.llvm.org/c/runtimes/
4. Print poster A0 + 50 handout copies
5. Practice 60-second pitch + new Q&A cards (especially Q7-Q10)
6. Travel to Dublin
