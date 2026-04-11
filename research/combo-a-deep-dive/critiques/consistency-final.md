# Cross-File Consistency Check — Final

**3 CRITICAL issues** (all in OLD poster/index.html — new poster-combo-a.html addresses):
- C1: Old poster says "roofline rank" (new poster uses "weighted heuristic")
- C2: Old poster has MI300X (new poster uses GTX 1650 only)
- C3: Selection overhead varies: 2/3/4/6 ns — MUST normalize to one number

**Authoritative number for selection overhead: 4 ns**
Source: runtime_select_poc with real CUBINs (most recent, most realistic test)
- 2 ns: synthetic entries, 100K iterations (initial PoC)
- 3 ns: synthetic entries, different run (verification-poc)
- 4 ns: real CUBINs from directory (real-cubin-test-results)
- 6 ns: real OffloadBinary file (real-offloadbinary-results)

Use 4-6 ns range or "< 10 ns" conservatively. Best: cite 4 ns with real CUBINs, note 6 ns with OffloadBinary parsing overhead.

**All PR numbers consistent** (#186088, #185663, #148286) across proposal/paper/RFC.
**TaxBreak correctly "average"** (not median) everywhere.
**5 metadata keys consistent** across proposal/paper/RFC/handout.
