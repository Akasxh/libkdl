/*
 * GPURuntimeSelectWrappers.c -- Runtime helper library for #gpu.runtime_select
 *
 * Implements the vendor detection and dispatch-table ranking functions
 * that RuntimeSelectAttr::embedBinary() links against.  These are the
 * host-side runtime helpers called from the LLVM IR emitted by
 * RuntimeSelectAttr (Section 1.6 of refined-design-v1.md).
 *
 * The functions mirror the dlopen-probe pattern from kdl.c and
 * runtime_select_poc.c but are structured as a position-independent
 * shared library with a stable C ABI.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -std=c11 -shared -fPIC \
 *      -o libGPURuntimeSelectWrappers.so GPURuntimeSelectWrappers.c \
 *      -ldl -lpthread
 *
 * Link pattern (matches mlir/lib/ExecutionEngine/ convention):
 *   ExecutionEngine sharedLibPaths += "libGPURuntimeSelectWrappers.so"
 */

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Vendor enum (matches RuntimeSelectEntry.vendor_id in emitted IR)  */
/* ------------------------------------------------------------------ */

#define VENDOR_CPU    0
#define VENDOR_NVIDIA 1
#define VENDOR_AMD    2
#define VENDOR_INTEL  3

/* ------------------------------------------------------------------ */
/*  CUDA driver API attribute IDs                                      */
/* ------------------------------------------------------------------ */

#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

/* ------------------------------------------------------------------ */
/*  Dispatch table entry layout                                        */
/*  Must match the %RuntimeSelectEntry struct emitted by embedBinary() */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t vendor_id;
    uint32_t min_sm;
    uint32_t variant_priority;
    uint32_t _pad;
    const void *blob_ptr;
    uint64_t blob_size;
} RuntimeSelectEntry;

/* ------------------------------------------------------------------ */
/*  Selection strategies                                               */
/* ------------------------------------------------------------------ */

#define STRATEGY_FIRST_COMPATIBLE   0
#define STRATEGY_RANK_BY_PRIORITY   1
#define STRATEGY_RANK_BY_SM_CLOSEST 2

/* ------------------------------------------------------------------ */
/*  Cached detection state (thread-safe via pthread_once)              */
/* ------------------------------------------------------------------ */

static pthread_once_t g_detect_once = PTHREAD_ONCE_INIT;
static uint32_t       g_vendor      = VENDOR_CPU;
static uint32_t       g_device_sm   = 0;

/* Cached dlopen handles -- kept open for the process lifetime so that
   subsequent driver API calls (cuModuleLoad etc.) work. */
static void *g_cuda_handle = NULL;

/* ------------------------------------------------------------------ */
/*  Internal: probe NVIDIA (libcuda.so.1)                              */
/* ------------------------------------------------------------------ */

static int probe_nvidia(void) {
    void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h)
        return 0;

    /* Resolve cuInit and verify driver is functional */
    int (*fn_cuInit)(unsigned int) =
        (int (*)(unsigned int))dlsym(h, "cuInit");
    if (!fn_cuInit || fn_cuInit(0) != 0) {
        dlclose(h);
        return 0;
    }

    /* Resolve device enumeration */
    int (*fn_cuDeviceGetCount)(int *) =
        (int (*)(int *))dlsym(h, "cuDeviceGetCount");
    int (*fn_cuDeviceGet)(int *, int) =
        (int (*)(int *, int))dlsym(h, "cuDeviceGet");
    int (*fn_cuDeviceGetAttribute)(int *, int, int) =
        (int (*)(int *, int, int))dlsym(h, "cuDeviceGetAttribute");

    if (!fn_cuDeviceGetCount || !fn_cuDeviceGet ||
        !fn_cuDeviceGetAttribute) {
        dlclose(h);
        return 0;
    }

    int count = 0;
    if (fn_cuDeviceGetCount(&count) != 0 || count == 0) {
        dlclose(h);
        return 0;
    }

    int dev = 0;
    if (fn_cuDeviceGet(&dev, 0) != 0) {
        dlclose(h);
        return 0;
    }

    /* Query SM version: major*10 + minor (e.g., 75 for GTX 1650) */
    int major = 0, minor = 0;
    fn_cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    fn_cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    g_cuda_handle = h;
    g_device_sm = (uint32_t)(major * 10 + minor);
    g_vendor = VENDOR_NVIDIA;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Internal: probe AMD (libamdhip64.so)                               */
/* ------------------------------------------------------------------ */

static int probe_amd(void) {
    void *h = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
    if (!h)
        return 0;

    int (*fn_hipInit)(unsigned int) =
        (int (*)(unsigned int))dlsym(h, "hipInit");
    if (!fn_hipInit || fn_hipInit(0) != 0) {
        dlclose(h);
        return 0;
    }

    /* AMD detected; SM version stays 0 (not applicable). */
    g_vendor = VENDOR_AMD;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Internal: probe Intel Level Zero (libze_loader.so.1)               */
/* ------------------------------------------------------------------ */

static int probe_intel(void) {
    void *h = dlopen("libze_loader.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h)
        return 0;

    /* zeInit takes a ze_init_flags_t (uint32_t). 0 = all device types. */
    int (*fn_zeInit)(uint32_t) =
        (int (*)(uint32_t))dlsym(h, "zeInit");
    if (!fn_zeInit || fn_zeInit(0) != 0) {
        dlclose(h);
        return 0;
    }

    g_vendor = VENDOR_INTEL;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Internal: one-shot detection (called via pthread_once)             */
/* ------------------------------------------------------------------ */

static void detect_vendor_once(void) {
    /* Probe order: NVIDIA -> AMD -> Intel -> CPU fallback.
       Matches the priority in runtime_select_poc.c and kdl.c. */
    if (probe_nvidia()) return;
    if (probe_amd())    return;
    if (probe_intel())  return;
    /* g_vendor remains VENDOR_CPU */
}

/* ================================================================== */
/*  PUBLIC API -- extern "C" symbols called from emitted LLVM IR       */
/* ================================================================== */

/*
 * __gpu_runtime_select_detect_vendor
 *
 * Returns the vendor enum for the first functional GPU runtime found.
 * Thread-safe: detection runs exactly once per process via pthread_once.
 *
 * Return values:
 *   0 = CPU only (no GPU runtime found)
 *   1 = NVIDIA (libcuda.so.1 + cuInit succeeded)
 *   2 = AMD    (libamdhip64.so + hipInit succeeded)
 *   3 = Intel  (libze_loader.so.1 + zeInit succeeded)
 */
uint32_t __gpu_runtime_select_detect_vendor(void) {
    pthread_once(&g_detect_once, detect_vendor_once);
    return g_vendor;
}

/*
 * __gpu_runtime_select_get_device_sm
 *
 * Returns the NVIDIA SM version as major*10+minor (e.g., 75 for sm_75).
 * Returns 0 for non-NVIDIA vendors or if detection has not run.
 * Implicitly triggers detection if it hasn't happened yet.
 */
uint32_t __gpu_runtime_select_get_device_sm(void) {
    pthread_once(&g_detect_once, detect_vendor_once);
    return g_device_sm;
}

/*
 * __gpu_runtime_select_rank
 *
 * Iterates a dispatch table of RuntimeSelectEntry structs and returns
 * the index of the best entry for the given vendor/device combination.
 *
 * Parameters:
 *   table      - pointer to array of RuntimeSelectEntry
 *   num_entries - number of entries in the table
 *   vendor     - detected vendor enum (from detect_vendor)
 *   device_sm  - device SM version (from get_device_sm, 0 if non-NVIDIA)
 *   strategy   - selection strategy:
 *                  0 = first_compatible (matches PR #186088 behavior)
 *                  1 = rank_by_priority (highest variant_priority wins)
 *                  2 = rank_by_sm_closest (closest SM match wins)
 *
 * Returns:
 *   Index of the best entry (>= 0), or -1 if no compatible entry found.
 */
int32_t __gpu_runtime_select_rank(
        const RuntimeSelectEntry *table,
        uint32_t num_entries,
        uint32_t vendor,
        uint32_t device_sm,
        uint32_t strategy) {

    if (!table || num_entries == 0)
        return -1;

    /* ---- Strategy 0: first compatible ---- */
    if (strategy == STRATEGY_FIRST_COMPATIBLE) {
        for (uint32_t i = 0; i < num_entries; i++) {
            if (table[i].vendor_id != vendor)
                continue;
            if (vendor == VENDOR_NVIDIA && table[i].min_sm > device_sm)
                continue;
            return (int32_t)i;
        }
        return -1;
    }

    /* ---- Strategy 1: rank by variant_priority ---- */
    if (strategy == STRATEGY_RANK_BY_PRIORITY) {
        int32_t  best_idx      = -1;
        uint32_t best_priority = 0;
        uint32_t best_sm       = 0;

        for (uint32_t i = 0; i < num_entries; i++) {
            if (table[i].vendor_id != vendor)
                continue;
            if (vendor == VENDOR_NVIDIA && table[i].min_sm > device_sm)
                continue;

            /* Highest priority wins; tiebreak: highest min_sm */
            if (table[i].variant_priority > best_priority ||
                (table[i].variant_priority == best_priority &&
                 table[i].min_sm > best_sm)) {
                best_idx      = (int32_t)i;
                best_priority = table[i].variant_priority;
                best_sm       = table[i].min_sm;
            }
        }
        return best_idx;
    }

    /* ---- Strategy 2: rank by SM closeness ---- */
    if (strategy == STRATEGY_RANK_BY_SM_CLOSEST) {
        int32_t  best_idx  = -1;
        uint32_t best_dist = UINT32_MAX;

        for (uint32_t i = 0; i < num_entries; i++) {
            if (table[i].vendor_id != vendor)
                continue;
            if (vendor == VENDOR_NVIDIA && table[i].min_sm > device_sm)
                continue;

            /* Distance = device_sm - entry.min_sm (lower is better,
               i.e., prefer the most specialized compatible binary). */
            uint32_t dist = device_sm - table[i].min_sm;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx  = (int32_t)i;
            }
        }
        return best_idx;
    }

    /* Unknown strategy: fall back to first_compatible */
    for (uint32_t i = 0; i < num_entries; i++) {
        if (table[i].vendor_id != vendor)
            continue;
        if (vendor == VENDOR_NVIDIA && table[i].min_sm > device_sm)
            continue;
        return (int32_t)i;
    }
    return -1;
}
