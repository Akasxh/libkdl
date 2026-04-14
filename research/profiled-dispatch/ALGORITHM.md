# Algorithm Specification: Profiled Adaptive Dispatch

**Authors:** Akash (IIT Patna)
**Context:** Extension to the LLVM Dublin 2026 poster on libkdl
**Date:** 2026-04-14

---

## 1. Overview

The profiled dispatch algorithm extends libkdl's kernel selection from a
single-shot cost model decision to a three-phase adaptive strategy:

1. **Cold start** -- Use roofline cost model (existing behavior)
2. **Exploration** -- Profile variant candidates via GPU event timing
3. **Exploitation** -- Cache and dispatch the empirically best variant

The algorithm operates per-context, where a context is uniquely identified
by the tuple (kernel_name, shape_hash, device_id).

---

## 2. Data Structures

### 2.1 Profile Entry

```c
typedef struct {
    uint32_t variant_index;
    uint32_t sample_count;       /* number of profiled invocations */
    double   mean_time_us;       /* running mean of execution time */
    double   m2;                 /* running M2 for Welford's online variance */
    double   min_time_us;        /* minimum observed time */
    double   max_time_us;        /* maximum observed time */
} kdl_profile_entry;
```

### 2.2 Context Profile

```c
typedef enum {
    KDL_PHASE_COLD,              /* no profiling data; use roofline */
    KDL_PHASE_EXPLORE,           /* actively profiling variants */
    KDL_PHASE_EXPLOIT,           /* converged; dispatching cached best */
} kdl_dispatch_phase;

typedef struct {
    /* Cache key */
    uint64_t context_hash;       /* hash(kernel_name, shape_hash, device_id) */

    /* Phase tracking */
    kdl_dispatch_phase phase;
    uint32_t total_invocations;  /* total calls in this context */
    uint32_t explore_index;      /* which variant we are currently profiling */

    /* Variant profiles */
    uint32_t num_variants;       /* N: number of viable variants */
    kdl_profile_entry profiles[KDL_MAX_CANDIDATES];

    /* Roofline ranking (prior) */
    uint32_t roofline_order[KDL_MAX_CANDIDATES]; /* variant indices sorted by roofline cost */

    /* Converged result */
    uint32_t best_variant;       /* index of empirically best variant */
    double   best_mean_us;       /* mean time of best variant */

    /* Re-validation */
    uint32_t invocations_since_revalidation;
} kdl_context_profile;
```

### 2.3 Context Hash Function

The cache key uniquely identifies a dispatch scenario:

```c
static uint64_t kdl_context_hash(const char *kernel_name,
                                  uint64_t shape_hash,
                                  int device_id) {
    uint64_t h = 14695981039346656037ULL; /* FNV-1a offset basis */
    /* Hash kernel name */
    for (const char *p = kernel_name; *p; p++) {
        h ^= (uint64_t)*p;
        h *= 1099511628211ULL; /* FNV-1a prime */
    }
    /* Mix in shape hash */
    h ^= shape_hash;
    h *= 1099511628211ULL;
    /* Mix in device ID */
    h ^= (uint64_t)device_id;
    h *= 1099511628211ULL;
    return h;
}
```

The shape_hash is computed by the caller from the kernel's dynamic dimensions.
For a GEMM with dimensions (M, N, K):

```c
uint64_t shape_hash = ((uint64_t)M << 40) ^ ((uint64_t)N << 20) ^ (uint64_t)K;
```

---

## 3. Core Algorithm

### 3.1 Pseudocode

```
FUNCTION profiled_dispatch(ctx, bundle, kernel_name, shape_hash, device_id, args):
    key = context_hash(kernel_name, shape_hash, device_id)
    profile = lookup_or_create(ctx.profile_cache, key)

    SWITCH profile.phase:

    CASE COLD:
        /* First invocation of this context. Use roofline to select
           and simultaneously begin profiling. */
        variants = contract_match(bundle, kernel_name, device_id)
        profile.num_variants = |variants|
        roofline_rank(variants, device_id, profile.roofline_order)

        IF profile.num_variants == 1:
            /* Only one viable variant -- skip exploration entirely */
            profile.phase = EXPLOIT
            profile.best_variant = variants[0]
            RETURN launch(variants[0], args)

        /* Launch the roofline-best variant with timing */
        best_roofline = profile.roofline_order[0]
        time_us = launch_timed(variants[best_roofline], args)
        update_profile(profile.profiles[best_roofline], time_us)

        profile.explore_index = 1  /* next variant to try */
        profile.phase = EXPLORE
        RETURN

    CASE EXPLORE:
        profile.total_invocations += 1

        /* Epsilon-greedy with termination: explore with probability epsilon,
           exploit with probability 1 - epsilon */
        epsilon = compute_epsilon(profile)

        IF random() < epsilon AND profile.explore_index < profile.num_variants:
            /* Explore: profile the next unfinished variant */
            idx = profile.roofline_order[profile.explore_index]
            time_us = launch_timed(variants[idx], args)
            update_profile(profile.profiles[idx], time_us)

            IF profile.profiles[idx].sample_count >= N_WARMUP:
                profile.explore_index += 1

            /* Check convergence */
            IF should_converge(profile):
                profile.best_variant = select_best(profile)
                profile.best_mean_us = profile.profiles[profile.best_variant].mean_time_us
                profile.phase = EXPLOIT

            RETURN

        ELSE:
            /* Exploit: dispatch current empirical best */
            current_best = select_best(profile)
            time_us = launch_timed(variants[current_best], args)
            update_profile(profile.profiles[current_best], time_us)
            RETURN

    CASE EXPLOIT:
        profile.total_invocations += 1
        profile.invocations_since_revalidation += 1

        /* Periodic re-validation check */
        IF profile.invocations_since_revalidation >= REVALIDATION_INTERVAL:
            trigger_revalidation(profile)

        /* Fast path: dispatch cached best, no timing */
        RETURN launch(variants[profile.best_variant], args)
```

### 3.2 Online Variance (Welford's Algorithm)

We maintain running statistics without storing individual samples:

```c
static void update_profile(kdl_profile_entry *entry, double time_us) {
    entry->sample_count++;
    uint32_t n = entry->sample_count;

    if (n == 1) {
        entry->mean_time_us = time_us;
        entry->m2 = 0.0;
        entry->min_time_us = time_us;
        entry->max_time_us = time_us;
        return;
    }

    double delta = time_us - entry->mean_time_us;
    entry->mean_time_us += delta / n;
    double delta2 = time_us - entry->mean_time_us;
    entry->m2 += delta * delta2;

    if (time_us < entry->min_time_us) entry->min_time_us = time_us;
    if (time_us > entry->max_time_us) entry->max_time_us = time_us;
}

static double profile_variance(const kdl_profile_entry *entry) {
    if (entry->sample_count < 2) return INFINITY;
    return entry->m2 / (entry->sample_count - 1);
}

static double profile_stddev(const kdl_profile_entry *entry) {
    return sqrt(profile_variance(entry));
}
```

### 3.3 Convergence Criterion

Exploration terminates when one of these conditions holds:

```c
#define N_WARMUP 5               /* minimum samples per variant */
#define CONFIDENCE_THRESHOLD 2.0 /* number of stddevs separating best from rest */
#define MAX_EXPLORE_INVOCATIONS 100  /* hard cap on exploration phase */

static int should_converge(const kdl_context_profile *profile) {
    /* Condition 1: All variants have been profiled N_WARMUP times */
    int all_profiled = 1;
    for (uint32_t i = 0; i < profile->num_variants; i++) {
        if (profile->profiles[i].sample_count < N_WARMUP) {
            all_profiled = 0;
            break;
        }
    }

    if (!all_profiled)
        return 0;

    /* Condition 2: The best variant is statistically separated from
       all others by CONFIDENCE_THRESHOLD standard deviations */
    uint32_t best = select_best(profile);
    double best_mean = profile->profiles[best].mean_time_us;
    double best_stderr = profile_stddev(&profile->profiles[best])
                         / sqrt(profile->profiles[best].sample_count);

    for (uint32_t i = 0; i < profile->num_variants; i++) {
        if (i == best) continue;
        double other_mean = profile->profiles[i].mean_time_us;
        double other_stderr = profile_stddev(&profile->profiles[i])
                              / sqrt(profile->profiles[i].sample_count);

        /* Combined standard error for the difference of means */
        double diff_stderr = sqrt(best_stderr * best_stderr
                                  + other_stderr * other_stderr);

        /* Is the gap statistically significant? */
        double gap = other_mean - best_mean;
        if (gap < CONFIDENCE_THRESHOLD * diff_stderr) {
            /* Cannot distinguish best from variant i with confidence */
            return 0;
        }
    }

    return 1;  /* All non-best variants are statistically worse */
}

/* Fallback: also converge after MAX_EXPLORE_INVOCATIONS regardless */
```

### 3.4 Best Variant Selection

```c
static uint32_t select_best(const kdl_context_profile *profile) {
    uint32_t best = 0;
    double best_time = INFINITY;

    for (uint32_t i = 0; i < profile->num_variants; i++) {
        if (profile->profiles[i].sample_count == 0) continue;
        if (profile->profiles[i].mean_time_us < best_time) {
            best_time = profile->profiles[i].mean_time_us;
            best = i;
        }
    }

    return best;
}
```

---

## 4. GPU Event-Based Timing

### 4.1 CUDA Path

```c
typedef CUevent cuda_event_t;
/* Obtained via dlsym from libcuda.so.1 */
typedef CUresult (*cuEventCreate_fn)(CUevent *, unsigned int);
typedef CUresult (*cuEventRecord_fn)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_fn)(CUevent);
typedef CUresult (*cuEventElapsedTime_fn)(float *, CUevent, CUevent);
typedef CUresult (*cuEventDestroy_fn)(CUevent);

static double launch_timed_cuda(kdl_kernel_t kernel, void **args) {
    CUevent start, stop;
    ctx->cuEventCreate(&start, 0 /* CU_EVENT_DEFAULT */);
    ctx->cuEventCreate(&stop, 0);

    ctx->cuEventRecord(start, kernel->stream);
    ctx->cuLaunchKernel(kernel->function,
                        kernel->grid_x, kernel->grid_y, kernel->grid_z,
                        kernel->block_x, kernel->block_y, kernel->block_z,
                        kernel->shared_mem, kernel->stream,
                        args, NULL);
    ctx->cuEventRecord(stop, kernel->stream);
    ctx->cuEventSynchronize(stop);

    float elapsed_ms;
    ctx->cuEventElapsedTime(&elapsed_ms, start, stop);

    ctx->cuEventDestroy(start);
    ctx->cuEventDestroy(stop);

    return (double)elapsed_ms * 1000.0;  /* return microseconds */
}
```

### 4.2 HIP Path

```c
/* Obtained via dlsym from libamdhip64.so */
typedef hipError_t (*hipEventCreate_fn)(hipEvent_t *);
typedef hipError_t (*hipEventRecord_fn)(hipEvent_t, hipStream_t);
typedef hipError_t (*hipEventSynchronize_fn)(hipEvent_t);
typedef hipError_t (*hipEventElapsedTime_fn)(float *, hipEvent_t, hipEvent_t);
typedef hipError_t (*hipEventDestroy_fn)(hipEvent_t);

static double launch_timed_hip(kdl_kernel_t kernel, void **args) {
    hipEvent_t start, stop;
    ctx->hipEventCreate(&start);
    ctx->hipEventCreate(&stop);

    ctx->hipEventRecord(start, kernel->stream);
    ctx->hipModuleLaunchKernel(kernel->function,
                                kernel->grid_x, kernel->grid_y, kernel->grid_z,
                                kernel->block_x, kernel->block_y, kernel->block_z,
                                kernel->shared_mem, kernel->stream,
                                args, NULL);
    ctx->hipEventRecord(stop, kernel->stream);
    ctx->hipEventSynchronize(stop);

    float elapsed_ms;
    ctx->hipEventElapsedTime(&elapsed_ms, start, stop);

    ctx->hipEventDestroy(start);
    ctx->hipEventDestroy(stop);

    return (double)elapsed_ms * 1000.0;  /* return microseconds */
}
```

### 4.3 Timing Considerations

**Why GPU events, not wall-clock time:**
- `clock_gettime(CLOCK_MONOTONIC)` measures wall-clock time including CPU-GPU
  synchronization overhead, kernel launch latency, and any queueing delays.
- GPU events measure *only* the time the kernel spends executing on the GPU,
  which is what we want to compare across variants.
- GPU event resolution is ~0.5us on NVIDIA (from cuEventElapsedTime documentation)
  and similar on AMD. This is sufficient to resolve 2-3% differences on
  millisecond-scale kernels.

**Warmup protocol:**
- The first invocation of a kernel on a context may suffer cold-start effects:
  instruction cache miss, TLB miss, memory page faults.
- We discard the first sample (sample_count starts at 0 but the first timing
  is recorded). The convergence criterion requires N_WARMUP = 5 samples, so
  the first (potentially noisy) sample is outvoted.
- Alternatively, we could explicitly run a warmup invocation without recording
  the time, but this wastes a kernel execution. Given the robustness of
  Welford's running mean to outliers in small samples, the simpler approach
  (record everything, let statistics handle it) is preferred.

---

## 5. Epsilon Schedule

The exploration probability epsilon controls the explore/exploit tradeoff:

```c
#define EPSILON_INITIAL  1.0    /* always explore at start */
#define EPSILON_MIN      0.0    /* fully exploit after convergence */
#define EPSILON_DECAY    0.85   /* multiplicative decay per exploration step */

static double compute_epsilon(const kdl_context_profile *profile) {
    if (profile->phase == KDL_PHASE_EXPLOIT)
        return 0.0;

    /* During exploration: decay epsilon as variants are profiled */
    uint32_t profiled_count = 0;
    for (uint32_t i = 0; i < profile->num_variants; i++) {
        if (profile->profiles[i].sample_count >= N_WARMUP)
            profiled_count++;
    }

    /* Start at 1.0, decay as more variants are characterized */
    double epsilon = EPSILON_INITIAL;
    for (uint32_t i = 0; i < profiled_count; i++) {
        epsilon *= EPSILON_DECAY;
    }

    return fmax(epsilon, EPSILON_MIN);
}
```

In practice, the epsilon schedule is dominated by the convergence criterion:
once all variants are profiled and statistical separation is achieved, the
phase transitions to EXPLOIT and epsilon drops to zero permanently.

---

## 6. Re-Validation Protocol

After convergence, the dispatched variant may become suboptimal if hardware
conditions change. The re-validation protocol detects this:

```c
#define REVALIDATION_INTERVAL 10000  /* invocations between checks */
#define REVALIDATION_SAMPLES  3      /* samples for re-check */
#define DRIFT_THRESHOLD       0.15   /* 15% slowdown triggers re-explore */

static void trigger_revalidation(kdl_context_profile *profile) {
    profile->invocations_since_revalidation = 0;

    /* Re-profile the current best */
    /* (done inline during the next REVALIDATION_SAMPLES invocations) */

    /* If the current best is now >DRIFT_THRESHOLD slower than its
       historical mean, reset to exploration phase */

    double current_mean = /* mean of last REVALIDATION_SAMPLES timings */;
    if (current_mean > profile->best_mean_us * (1.0 + DRIFT_THRESHOLD)) {
        /* Performance has drifted. Reset exploration. */
        profile->phase = KDL_PHASE_EXPLORE;
        profile->explore_index = 0;
        /* Reset all profile entries */
        for (uint32_t i = 0; i < profile->num_variants; i++) {
            profile->profiles[i].sample_count = 0;
            profile->profiles[i].mean_time_us = 0.0;
            profile->profiles[i].m2 = 0.0;
        }
    }
}
```

---

## 7. Integration with Existing libkdl

### 7.1 API Extension

The profiled dispatch integrates with the existing `kdl_select_kernel` API
through an opt-in flag:

```c
/* New flag for kdl_select_kernel */
#define KDL_SELECT_PROFILED  (1 << 0)  /* enable profiled adaptive dispatch */
#define KDL_SELECT_ROOFLINE  (0)       /* default: roofline-only (existing) */

/* Extended selection API */
kdl_status kdl_select_kernel_ex(kdl_ctx ctx, kdl_bundle_t bundle,
                                 const char *kernel_name,
                                 int device_index,
                                 uint64_t shape_hash,
                                 uint32_t flags,
                                 kdl_kernel_t *out);

/* Query profiling state */
kdl_status kdl_get_profile_info(kdl_ctx ctx,
                                 const char *kernel_name,
                                 uint64_t shape_hash,
                                 int device_index,
                                 kdl_dispatch_phase *phase,
                                 uint32_t *best_variant,
                                 double *best_time_us);
```

### 7.2 Backward Compatibility

The existing `kdl_select_kernel` function remains unchanged. It uses the
roofline cost model as before. The new `kdl_select_kernel_ex` function
adds the profiling capability while being fully backward-compatible:

```c
kdl_status kdl_select_kernel(kdl_ctx ctx, kdl_bundle_t bundle,
                              const char *kernel_name,
                              int device_index, kdl_kernel_t *out) {
    /* Existing behavior: roofline-only, no shape awareness */
    return kdl_select_kernel_ex(ctx, bundle, kernel_name, device_index,
                                 0 /* shape_hash=0 */, KDL_SELECT_ROOFLINE, out);
}
```

### 7.3 Memory Overhead

Per-context profile storage:

```
sizeof(kdl_context_profile) = 8 (hash)
                             + 4 (phase) + 4 (total_invocations) + 4 (explore_index)
                             + 4 (num_variants)
                             + 64 * sizeof(kdl_profile_entry)  /* 64 * 48 = 3072 */
                             + 64 * 4  /* roofline_order = 256 */
                             + 4 (best_variant) + 8 (best_mean_us)
                             + 4 (invocations_since_revalidation)
                             ~ 3.4 KB per context
```

For a typical workload with 15 contexts: 15 * 3.4 KB ~ 51 KB total.
This is negligible compared to kernel binary sizes (typically 10-100 KB each).

---

## 8. Complexity Summary

| Operation | Time | Space | When |
|-----------|------|-------|------|
| Context hash | O(len(kernel_name)) | O(1) | Every dispatch |
| Profile cache lookup | O(1) amortized (hash table) | O(C * 3.4KB) | Every dispatch |
| Roofline ranking | O(N log N) | O(N) | First invocation per context |
| Profiled launch (explore) | O(1) + kernel time | O(1) | During exploration |
| Convergence check | O(N) | O(1) | After each exploration step |
| Exploit dispatch | O(1) | O(1) | Steady state |
| Re-validation | O(1) | O(1) | Every 10000 invocations |

Where N = number of viable variants (< 10), C = number of distinct contexts.
