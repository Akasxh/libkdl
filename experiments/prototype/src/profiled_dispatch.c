/*
 * profiled_dispatch.c -- Adaptive explore/exploit kernel variant selection
 *
 * Implements an online profiling strategy for KDL:
 *   Phase 1 (cold):    No measurements yet -- caller falls back to metadata.
 *   Phase 2 (explore): Each compatible variant is sampled KDL_PD_WARMUP_SAMPLES times.
 *   Phase 3 (exploit): The variant with the lowest median time is returned.
 *
 * The profile state is a linear-probe hash map keyed on
 *   hash(kernel_name) ^ shape_hash ^ device_id
 *
 * Part of mlir-hetero-dispatch (LLVM Developers' Meeting, Dublin 2026).
 */

#define _POSIX_C_SOURCE 199309L

#include "kdl.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Internal hash map entry                                            */
/* ------------------------------------------------------------------ */

#define PD_MAX_RAW_SAMPLES 64  /* raw samples kept per variant for median */

typedef struct {
    int      in_use;
    uint64_t key;  /* hash(kernel_name) ^ shape_hash ^ device_id */

    kdl_pd_profile_entry entry;

    /* Raw sample ring buffers for median computation */
    double   raw_samples[KDL_PD_MAX_VARIANTS][PD_MAX_RAW_SAMPLES];
    int      raw_count[KDL_PD_MAX_VARIANTS];   /* total samples seen */
    int      raw_write[KDL_PD_MAX_VARIANTS];   /* next write position (ring) */
} pd_map_slot;

struct kdl_pd_state {
    pd_map_slot slots[KDL_PD_MAP_SIZE];
};

/* ------------------------------------------------------------------ */
/*  Hash helpers                                                       */
/* ------------------------------------------------------------------ */

static uint64_t fnv1a_str(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    for (; *s; s++) {
        h ^= (uint64_t)(unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t make_key(const char *kernel_name, uint64_t shape_hash,
                          int device_id) {
    return fnv1a_str(kernel_name) ^ shape_hash ^ (uint64_t)device_id;
}

/* ------------------------------------------------------------------ */
/*  Median computation (insertion sort on small arrays)                */
/* ------------------------------------------------------------------ */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double compute_median(const double *samples, int count) {
    if (count <= 0) return 0.0;

    /* Copy to scratch, sort, return median */
    double scratch[PD_MAX_RAW_SAMPLES];
    int n = (count < PD_MAX_RAW_SAMPLES) ? count : PD_MAX_RAW_SAMPLES;
    memcpy(scratch, samples, (size_t)n * sizeof(double));
    qsort(scratch, (size_t)n, sizeof(double), cmp_double);

    if (n % 2 == 1) return scratch[n / 2];
    return (scratch[n / 2 - 1] + scratch[n / 2]) * 0.5;
}

/* ------------------------------------------------------------------ */
/*  Map lookup (linear probe)                                          */
/* ------------------------------------------------------------------ */

static pd_map_slot *map_find(struct kdl_pd_state *state, uint64_t key) {
    int start = (int)(key % KDL_PD_MAP_SIZE);
    for (int i = 0; i < KDL_PD_MAP_SIZE; i++) {
        int idx = (start + i) % KDL_PD_MAP_SIZE;
        pd_map_slot *slot = &state->slots[idx];
        if (!slot->in_use) return NULL;
        if (slot->key == key) return slot;
    }
    return NULL;
}

static pd_map_slot *map_insert(struct kdl_pd_state *state, uint64_t key) {
    int start = (int)(key % KDL_PD_MAP_SIZE);
    for (int i = 0; i < KDL_PD_MAP_SIZE; i++) {
        int idx = (start + i) % KDL_PD_MAP_SIZE;
        pd_map_slot *slot = &state->slots[idx];
        if (!slot->in_use) {
            memset(slot, 0, sizeof(*slot));
            slot->in_use = 1;
            slot->key = key;
            slot->entry.best_variant_idx = -1;
            slot->entry.best_time_ns = 0.0;
            return slot;
        }
        if (slot->key == key) return slot;
    }
    return NULL;  /* map full */
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

kdl_status kdl_pd_create(kdl_pd_state_t *out) {
    if (!out) return KDL_ERROR_INVALID_ARGUMENT;

    struct kdl_pd_state *state = calloc(1, sizeof(*state));
    if (!state) return KDL_ERROR_LOAD_FAILED;

    *out = state;
    return KDL_SUCCESS;
}

void kdl_pd_destroy(kdl_pd_state_t state) {
    free(state);
}

kdl_status kdl_pd_record(kdl_pd_state_t state, const char *kernel_name,
                          uint64_t shape_hash, int device_id,
                          int variant_idx, int n_variants, double elapsed_ns) {
    if (!state || !kernel_name) return KDL_ERROR_INVALID_ARGUMENT;
    if (variant_idx < 0 || variant_idx >= KDL_PD_MAX_VARIANTS)
        return KDL_ERROR_INVALID_ARGUMENT;
    if (n_variants <= 0 || n_variants > KDL_PD_MAX_VARIANTS)
        return KDL_ERROR_INVALID_ARGUMENT;

    uint64_t key = make_key(kernel_name, shape_hash, device_id);
    pd_map_slot *slot = map_insert(state, key);
    if (!slot) return KDL_ERROR_POOL_EXHAUSTED;

    kdl_pd_profile_entry *e = &slot->entry;
    e->n_variants = n_variants;
    e->dispatch_count++;

    /* Store raw sample in ring buffer */
    int wi = slot->raw_write[variant_idx] % PD_MAX_RAW_SAMPLES;
    slot->raw_samples[variant_idx][wi] = elapsed_ns;
    slot->raw_write[variant_idx] = wi + 1;
    slot->raw_count[variant_idx]++;
    e->variant_samples[variant_idx] = slot->raw_count[variant_idx];

    /* Recompute median for this variant */
    int n = slot->raw_count[variant_idx];
    if (n > PD_MAX_RAW_SAMPLES) n = PD_MAX_RAW_SAMPLES;
    e->variant_times[variant_idx] = compute_median(
        slot->raw_samples[variant_idx], n);

    /* Check if still in explore phase */
    int all_warmed = 1;
    for (int v = 0; v < n_variants; v++) {
        if (e->variant_samples[v] < KDL_PD_WARMUP_SAMPLES) {
            all_warmed = 0;
            break;
        }
    }

    if (!all_warmed) {
        e->explore_count = e->dispatch_count;
    }

    /* Update best variant (only meaningful after warmup) */
    if (all_warmed) {
        int best = 0;
        double best_t = e->variant_times[0];
        for (int v = 1; v < n_variants; v++) {
            if (e->variant_times[v] < best_t) {
                best_t = e->variant_times[v];
                best = v;
            }
        }
        e->best_variant_idx = best;
        e->best_time_ns = best_t;
    }

    return KDL_SUCCESS;
}

kdl_status kdl_pd_select(kdl_pd_state_t state, const char *kernel_name,
                          uint64_t shape_hash, int device_id,
                          int *out_variant) {
    if (!state || !kernel_name || !out_variant)
        return KDL_ERROR_INVALID_ARGUMENT;

    uint64_t key = make_key(kernel_name, shape_hash, device_id);
    pd_map_slot *slot = map_find(state, key);

    /* Phase 1: cold start -- no profile data at all */
    if (!slot) {
        *out_variant = -1;
        return KDL_SUCCESS;
    }

    kdl_pd_profile_entry *e = &slot->entry;

    /* Phase 2: explore -- find a variant that needs more samples */
    for (int v = 0; v < e->n_variants; v++) {
        if (e->variant_samples[v] < KDL_PD_WARMUP_SAMPLES) {
            *out_variant = v;
            return KDL_SUCCESS;
        }
    }

    /* Phase 3: exploit -- return the best known variant */
    *out_variant = e->best_variant_idx;
    return KDL_SUCCESS;
}

kdl_status kdl_pd_get_profile(kdl_pd_state_t state, const char *kernel_name,
                               uint64_t shape_hash, int device_id,
                               kdl_pd_profile_entry *out) {
    if (!state || !kernel_name || !out)
        return KDL_ERROR_INVALID_ARGUMENT;

    uint64_t key = make_key(kernel_name, shape_hash, device_id);
    pd_map_slot *slot = map_find(state, key);
    if (!slot) return KDL_ERROR_NO_MATCHING_VARIANT;

    *out = slot->entry;
    return KDL_SUCCESS;
}
