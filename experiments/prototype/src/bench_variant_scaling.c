#define _POSIX_C_SOURCE 199309L

/*
 * bench_variant_scaling.c -- Variant-count scaling benchmark
 *
 * Measures how selection overhead scales with the number of variants
 * in the dispatch table. Creates synthetic dispatch tables with
 * N = {1, 2, 3, 5, 10, 20, 50, 100} entries and measures selection
 * time over 100,000 iterations for each.
 *
 * No real cubins or GPU drivers required -- purely measures the
 * select_best_entry() scan cost.
 *
 * Build:  cc -O2 -Wall -Wextra -std=c11 -o bench_variant_scaling \
 *             bench_variant_scaling.c -ldl
 *
 * Expected: linear scaling (O(N) scan) with very small constants
 * due to branch prediction on the sequential scan.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define VENDOR_NVIDIA 1
#define MAX_ENTRIES   128
#define ITERATIONS    100000

/* ------------------------------------------------------------------ */
/*  RuntimeSelectEntry -- mirrors runtime_select_poc.c                 */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t vendor_id;
    const void *blob_ptr;
    uint64_t blob_size;
    uint32_t min_sm;
    uint32_t variant_priority;
    char     variant_tag[32];
} RuntimeSelectEntry;

/* ------------------------------------------------------------------ */
/*  Timing helpers                                                     */
/* ------------------------------------------------------------------ */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/*  Selection logic -- identical to runtime_select_poc.c               */
/*  Strategy: rank_by_priority                                         */
/*    1. Filter: vendor_id must match                                  */
/*    2. Filter: min_sm <= device SM                                   */
/*    3. Rank: highest variant_priority wins                           */
/*    4. Tiebreak: highest min_sm (most specialized)                   */
/* ------------------------------------------------------------------ */

static int select_best_entry(const RuntimeSelectEntry *table, int num_entries,
                             uint32_t device_vendor, uint32_t device_sm) {
    int best_idx = -1;
    uint32_t best_priority = 0;
    uint32_t best_sm = 0;

    for (int i = 0; i < num_entries; i++) {
        const RuntimeSelectEntry *e = &table[i];

        if (e->vendor_id != device_vendor) continue;
        if (e->min_sm > device_sm) continue;

        if (e->variant_priority > best_priority ||
            (e->variant_priority == best_priority && e->min_sm > best_sm)) {
            best_idx = i;
            best_priority = e->variant_priority;
            best_sm = e->min_sm;
        }
    }

    return best_idx;
}

/* ------------------------------------------------------------------ */
/*  Build synthetic dispatch table with N entries                      */
/*                                                                    */
/*  Layout: one sm_75 entry (compatible with device sm_75), the rest  */
/*  are incompatible (sm_80, sm_86, sm_89, sm_90, ...).               */
/*  The matching entry is placed at a random position to avoid        */
/*  early-exit bias.                                                   */
/* ------------------------------------------------------------------ */

static const uint32_t INCOMPAT_SMS[] = {
    80, 86, 89, 90, 95, 100, 110, 120, 130, 140,
    150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    250, 260, 270, 280, 290, 300, 310, 320, 330, 340,
    350, 360, 370, 380, 390, 400, 410, 420, 430, 440,
    450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
    550, 560, 570, 580, 590, 600, 610, 620, 630, 640,
    650, 660, 670, 680, 690, 700, 710, 720, 730, 740,
    750, 760, 770, 780, 790, 800, 810, 820, 830, 840,
    850, 860, 870, 880, 890, 900, 910, 920, 930, 940,
    950, 960, 970, 980, 990, 1000, 1010, 1020, 1030
};
#define NUM_INCOMPAT_SMS (sizeof(INCOMPAT_SMS) / sizeof(INCOMPAT_SMS[0]))

static void build_table(RuntimeSelectEntry *table, int n,
                        int match_position) {
    int incompat_idx = 0;

    for (int i = 0; i < n; i++) {
        table[i].vendor_id = VENDOR_NVIDIA;
        table[i].blob_ptr = NULL;
        table[i].blob_size = 0;

        if (i == match_position) {
            /* Compatible entry: sm_75 matches device sm_75 */
            table[i].min_sm = 75;
            table[i].variant_priority = 5;
            snprintf(table[i].variant_tag, sizeof(table[i].variant_tag),
                     "sm_75");
        } else {
            /* Incompatible: min_sm > device sm_75 */
            table[i].min_sm = INCOMPAT_SMS[incompat_idx % NUM_INCOMPAT_SMS];
            table[i].variant_priority = 5;
            snprintf(table[i].variant_tag, sizeof(table[i].variant_tag),
                     "sm_%u", table[i].min_sm);
            incompat_idx++;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Comparison function for qsort (used for median calculation)        */
/* ------------------------------------------------------------------ */

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Run benchmark for a given table size                               */
/* ------------------------------------------------------------------ */

typedef struct {
    int n;
    double mean_ns;
    double median_ns;
    double p99_ns;
    double per_entry_ns;
    int selected_idx;
} BenchResult;

static BenchResult run_bench(int n) {
    RuntimeSelectEntry table[MAX_ENTRIES];
    BenchResult result;

    memset(&result, 0, sizeof(result));
    result.n = n;

    /* Place matching entry at end to force full scan (worst case) */
    int match_pos = n - 1;
    build_table(table, n, match_pos);

    /* Verify selection works */
    int idx = select_best_entry(table, n, VENDOR_NVIDIA, 75);
    if (idx != match_pos) {
        fprintf(stderr, "ERROR: expected idx=%d, got %d for n=%d\n",
                match_pos, idx, n);
        return result;
    }
    result.selected_idx = idx;

    /* Warmup: 10,000 iterations to stabilize branch predictor + caches */
    volatile int sink = 0;
    for (int i = 0; i < 10000; i++) {
        sink = select_best_entry(table, n, VENDOR_NVIDIA, 75);
    }
    (void)sink;

    /* Collect per-iteration timings in batches of 1000 */
    uint64_t *samples = malloc(ITERATIONS * sizeof(uint64_t));
    if (!samples) {
        fprintf(stderr, "ERROR: malloc failed for samples\n");
        return result;
    }

    for (int i = 0; i < ITERATIONS; i++) {
        uint64_t t0 = now_ns();
        sink = select_best_entry(table, n, VENDOR_NVIDIA, 75);
        uint64_t t1 = now_ns();
        samples[i] = t1 - t0;
    }
    (void)sink;

    /* Compute mean */
    uint64_t sum = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        sum += samples[i];
    }
    result.mean_ns = (double)sum / ITERATIONS;

    /* Compute median and p99 */
    qsort(samples, ITERATIONS, sizeof(uint64_t), cmp_u64);
    result.median_ns = (double)samples[ITERATIONS / 2];
    result.p99_ns = (double)samples[(int)(ITERATIONS * 0.99)];

    /* Per-entry cost */
    result.per_entry_ns = result.mean_ns / n;

    free(samples);
    return result;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
    static const int SIZES[] = {1, 2, 3, 5, 10, 20, 50, 100};
    static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    BenchResult results[8];

    printf("=== Variant-Count Scaling Benchmark ===\n");
    printf("Measures select_best_entry() cost vs dispatch table size\n");
    printf("Device: sm_75 (simulated), Match position: last entry\n");
    printf("Iterations: %d per size, Warmup: 10,000\n\n", ITERATIONS);

    /* Run benchmarks */
    for (int i = 0; i < NUM_SIZES; i++) {
        results[i] = run_bench(SIZES[i]);
        printf("N=%3d | mean=%7.1f ns | median=%7.1f ns | "
               "p99=%7.1f ns | per_entry=%5.2f ns | "
               "selected=[%d]\n",
               results[i].n,
               results[i].mean_ns,
               results[i].median_ns,
               results[i].p99_ns,
               results[i].per_entry_ns,
               results[i].selected_idx);
    }

    /* Summary table (markdown-friendly) */
    printf("\n--- Markdown Table ---\n");
    printf("| N | mean_ns | median_ns | p99_ns | per_entry_ns |\n");
    printf("|---|---------|-----------|--------|-------------|\n");
    for (int i = 0; i < NUM_SIZES; i++) {
        printf("| %d | %.1f | %.1f | %.1f | %.2f |\n",
               results[i].n,
               results[i].mean_ns,
               results[i].median_ns,
               results[i].p99_ns,
               results[i].per_entry_ns);
    }

    /* Scaling analysis */
    printf("\n--- Scaling Analysis ---\n");
    double base_ns = results[0].mean_ns;
    for (int i = 0; i < NUM_SIZES; i++) {
        double ratio = results[i].mean_ns / base_ns;
        int expected_ratio = SIZES[i];
        printf("N=%3d: %.1fx actual vs %dx expected (linear)\n",
               SIZES[i], ratio, expected_ratio);
    }

    printf("\n--- Conclusion ---\n");
    double ns_100 = results[NUM_SIZES - 1].mean_ns;
    if (ns_100 < 1000.0) {
        printf("Even at N=100 variants, selection takes <1 us (%.1f ns).\n",
               ns_100);
        printf("Selection overhead is negligible for any realistic fat "
               "binary.\n");
    } else if (ns_100 < 10000.0) {
        printf("At N=100, selection takes ~%.1f us. Still negligible vs "
               "kernel launch overhead (~5-20 us).\n", ns_100 / 1000.0);
    } else {
        printf("At N=100, selection takes ~%.1f us. Consider optimizing "
               "for large variant counts.\n", ns_100 / 1000.0);
    }

    return 0;
}
