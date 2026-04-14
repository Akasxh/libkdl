/*
 * bench_profiled.c -- Demonstrate adaptive explore/exploit variant dispatch
 *
 * Creates 3 simulated kernel variants with different execution times
 * (implemented via nanosleep), runs the profiled dispatch system for
 * 100 iterations, and shows convergence to the fastest variant.
 *
 * Output columns:
 *   iter  chosen_variant  measured_ns  phase  cumulative_regret_ns
 *
 * After (KDL_PD_WARMUP_SAMPLES * n_variants) explore iterations, the
 * system should exclusively pick the fastest variant.
 *
 * Compile:
 *   gcc -O2 -Wall -std=c11 -I../src -o bench_profiled bench_profiled.c \
 *       ../src/profiled_dispatch.c -lm
 *
 * Run:
 *   ./bench_profiled
 *
 * Part of mlir-hetero-dispatch (LLVM Developers' Meeting, Dublin 2026).
 */

#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kdl.h"

/* ------------------------------------------------------------------ */
/*  Timing helpers                                                     */
/* ------------------------------------------------------------------ */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/*  Simulated kernel variants                                          */
/*                                                                     */
/*  Each variant sleeps for a different duration to simulate kernels   */
/*  with different performance characteristics:                        */
/*    variant 0: ~500 us  (slow)                                       */
/*    variant 1: ~100 us  (fastest)                                    */
/*    variant 2: ~300 us  (medium)                                     */
/*                                                                     */
/*  Small jitter is added via the timing measurement itself.           */
/* ------------------------------------------------------------------ */

#define N_VARIANTS 3

static const long variant_sleep_us[N_VARIANTS] = { 500, 100, 300 };
static const char *variant_names[N_VARIANTS] = { "sm_50_slow", "sm_80_fast", "sm_70_med" };

static double simulate_kernel(int variant_idx) {
    struct timespec req;
    req.tv_sec = 0;
    req.tv_nsec = variant_sleep_us[variant_idx] * 1000L;

    uint64_t t0 = now_ns();
    nanosleep(&req, NULL);
    uint64_t t1 = now_ns();

    return (double)(t1 - t0);
}

/* ------------------------------------------------------------------ */
/*  Main benchmark loop                                                */
/* ------------------------------------------------------------------ */

#define N_ITERATIONS 100
#define KERNEL_NAME  "matmul_tiled"
#define SHAPE_HASH   0xDEAD1234ULL
#define DEVICE_ID    0

int main(void) {
    kdl_pd_state_t pd = NULL;
    kdl_status rc;

    rc = kdl_pd_create(&pd);
    if (rc != KDL_SUCCESS) {
        fprintf(stderr, "kdl_pd_create failed: %d\n", rc);
        return 1;
    }

    printf("=== Profiled Dispatch Benchmark ===\n");
    printf("Variants:\n");
    for (int v = 0; v < N_VARIANTS; v++) {
        printf("  [%d] %-15s  simulated_time=%ld us\n",
               v, variant_names[v], variant_sleep_us[v]);
    }
    printf("\nWarmup samples per variant: %d\n", KDL_PD_WARMUP_SAMPLES);
    printf("Explore budget: %d iterations (worst case)\n",
           KDL_PD_WARMUP_SAMPLES * N_VARIANTS);
    printf("\n");

    /* CSV header */
    printf("%-6s  %-8s  %-14s  %-10s  %-20s  %s\n",
           "iter", "variant", "measured_ns", "phase",
           "cumul_regret_ns", "variant_name");
    printf("------  --------  --------------  ----------  "
           "--------------------  ---------------\n");

    double cumulative_regret = 0.0;
    double optimal_time_ns = (double)(variant_sleep_us[1] * 1000L);  /* variant 1 is fastest */

    int converged_at = -1;

    for (int i = 0; i < N_ITERATIONS; i++) {
        int chosen = -1;
        rc = kdl_pd_select(pd, KERNEL_NAME, SHAPE_HASH, DEVICE_ID, &chosen);
        if (rc != KDL_SUCCESS) {
            fprintf(stderr, "kdl_pd_select failed: %d\n", rc);
            break;
        }

        const char *phase;

        if (chosen == -1) {
            /* Phase 1: cold start -- pick variant 0 (metadata fallback) */
            chosen = 0;
            phase = "cold";
        } else {
            /* Check if we're still exploring */
            kdl_pd_profile_entry prof;
            rc = kdl_pd_get_profile(pd, KERNEL_NAME, SHAPE_HASH, DEVICE_ID, &prof);
            if (rc == KDL_SUCCESS) {
                int all_warm = 1;
                for (int v = 0; v < N_VARIANTS; v++) {
                    if (prof.variant_samples[v] < KDL_PD_WARMUP_SAMPLES) {
                        all_warm = 0;
                        break;
                    }
                }
                phase = all_warm ? "exploit" : "explore";
            } else {
                phase = "explore";
            }
        }

        /* Execute the chosen variant (simulated) */
        double elapsed = simulate_kernel(chosen);

        /* Record the measurement */
        rc = kdl_pd_record(pd, KERNEL_NAME, SHAPE_HASH, DEVICE_ID,
                           chosen, N_VARIANTS, elapsed);
        if (rc != KDL_SUCCESS) {
            fprintf(stderr, "kdl_pd_record failed: %d\n", rc);
            break;
        }

        /* Regret = time spent above optimal */
        double regret = elapsed - optimal_time_ns;
        if (regret < 0.0) regret = 0.0;
        cumulative_regret += regret;

        /* Track convergence point */
        if (converged_at < 0 && chosen == 1 &&
            strcmp(phase, "exploit") == 0) {
            converged_at = i;
        }

        printf("%-6d  %-8d  %-14.0f  %-10s  %-20.0f  %s\n",
               i, chosen, elapsed, phase, cumulative_regret,
               variant_names[chosen]);
    }

    /* Final summary */
    printf("\n=== Summary ===\n");

    kdl_pd_profile_entry final_prof;
    rc = kdl_pd_get_profile(pd, KERNEL_NAME, SHAPE_HASH, DEVICE_ID, &final_prof);
    if (rc == KDL_SUCCESS) {
        printf("Total dispatches:  %lu\n", (unsigned long)final_prof.dispatch_count);
        printf("Explore dispatches: %lu\n", (unsigned long)final_prof.explore_count);
        printf("Best variant:      [%d] %s\n",
               final_prof.best_variant_idx,
               (final_prof.best_variant_idx >= 0 &&
                final_prof.best_variant_idx < N_VARIANTS)
                   ? variant_names[final_prof.best_variant_idx]
                   : "?");
        printf("Best median time:  %.0f ns (%.1f us)\n",
               final_prof.best_time_ns,
               final_prof.best_time_ns / 1000.0);

        printf("\nPer-variant statistics:\n");
        printf("  %-4s  %-15s  %-10s  %-14s\n",
               "idx", "name", "samples", "median_ns");
        for (int v = 0; v < N_VARIANTS; v++) {
            printf("  %-4d  %-15s  %-10d  %-14.0f\n",
                   v, variant_names[v],
                   final_prof.variant_samples[v],
                   final_prof.variant_times[v]);
        }
    }

    if (converged_at >= 0) {
        printf("\nConverged to optimal at iteration: %d\n", converged_at);
    }
    printf("Cumulative regret:  %.0f ns (%.1f us)\n",
           cumulative_regret, cumulative_regret / 1000.0);

    kdl_pd_destroy(pd);
    return 0;
}
