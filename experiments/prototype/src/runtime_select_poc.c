#define _POSIX_C_SOURCE 199309L

/*
 * runtime_select_poc.c -- Proof-of-concept for #gpu.runtime_select
 *
 * Demonstrates what RuntimeSelectAttr::embedBinary() would emit as
 * LLVM IR, implemented in C for the EuroLLVM Dublin poster demo.
 *
 * The mechanism:
 *   1. Vendor detection via dlopen (probes libcuda.so.1 / libamdhip64.so)
 *   2. Dispatch table construction with {vendor_id, blob_ptr, size, sm_ver}
 *   3. Selection logic: rank entries by device SM compatibility
 *   4. Module loading via cuModuleLoadData on the selected variant
 *   5. Kernel launch via cuLaunchKernel
 *   6. Timing of each phase with clock_gettime(CLOCK_MONOTONIC)
 *
 * Build:  make runtime_select_poc
 *    or:  gcc -O2 -Wall -std=c11 -o runtime_select_poc runtime_select_poc.c -ldl
 *
 * Usage:  ./runtime_select_poc [path/to/dir_with_cubins]
 *         ./runtime_select_poc [path/to/file.offloadbin]
 *         If no argument given, uses embedded synthetic test data.
 *
 * Requires: NVIDIA GPU with CUDA driver installed (libcuda.so.1).
 *           No compile-time CUDA dependency -- everything via dlopen.
 */

#include <dlfcn.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  OffloadBinary format constants                                     */
/*  Source: llvm/include/llvm/Object/OffloadBinary.h                  */
/* ------------------------------------------------------------------ */

#define OFFLOAD_BINARY_MAGIC   UINT64_C(0x10FF10AD)
#define OFFLOAD_BINARY_VERSION UINT64_C(1)

typedef struct __attribute__((packed)) {
    uint64_t magic;
    uint64_t version;
    uint64_t size;
    uint64_t entry_offset;
    uint64_t entry_count;
    uint64_t padding;
} ObFileHeader;

typedef struct __attribute__((packed)) {
    uint64_t the_size;
    uint64_t image_offset;
    uint64_t image_size;
    uint64_t string_offset;
    uint64_t string_size;
} ObEntryHeader;

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define VENDOR_NONE   0
#define VENDOR_NVIDIA 1
#define VENDOR_AMD    2
#define VENDOR_SPIRV  3

#define MAX_ENTRIES   16
#define MAX_PATH_LEN  512

/* CUDA driver API types (opaque pointers) */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;

/* CUDA device attribute IDs */
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

/* ------------------------------------------------------------------ */
/*  RuntimeSelectEntry -- the dispatch table struct                    */
/*  Mirrors %RuntimeSelectEntry in the LLVM IR emission               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t vendor_id;
    const void *blob_ptr;     /* pointer to cubin/hsaco data */
    uint64_t blob_size;
    uint32_t min_sm;          /* from OffloadBinary metadata key "min_sm" */
    uint32_t variant_priority; /* from metadata key "variant_priority" */
    char     variant_tag[32]; /* human-readable label */
} RuntimeSelectEntry;

/* ------------------------------------------------------------------ */
/*  Dispatch table -- populated at startup, read at kernel launch     */
/* ------------------------------------------------------------------ */

static RuntimeSelectEntry g_dispatch_table[MAX_ENTRIES];
static int                g_num_entries = 0;
static int                g_selected_idx = -1;
static CUmodule           g_module_ptr = NULL;

/* ------------------------------------------------------------------ */
/*  CUDA driver API function pointers (populated via dlopen)          */
/* ------------------------------------------------------------------ */

static void *g_libcuda = NULL;

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGetCount)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetAttribute)(int *, int, CUdevice);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned,
                                    unsigned, CUstream,
                                    void **, void **);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuStreamSynchronize)(CUstream);

/* ------------------------------------------------------------------ */
/*  Timing helpers                                                     */
/* ------------------------------------------------------------------ */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/*  Phase 1: Vendor detection via dlopen                              */
/*  Mirrors __gpu_runtime_select_detect_vendor() in emitted LLVM IR   */
/* ------------------------------------------------------------------ */

static uint32_t detect_vendor(uint32_t *out_sm_major, uint32_t *out_sm_minor,
                              char *dev_name, int name_len) {
    *out_sm_major = 0;
    *out_sm_minor = 0;
    if (dev_name && name_len > 0) dev_name[0] = '\0';

    g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!g_libcuda) {
        fprintf(stderr, "[runtime_select] libcuda.so.1 not found\n");
        return VENDOR_NONE;
    }

    #define LOAD(name) *(void **)(&p_##name) = dlsym(g_libcuda, #name)
    LOAD(cuInit);
    LOAD(cuDeviceGetCount);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetAttribute);
    LOAD(cuDeviceGetName);
    LOAD(cuCtxCreate);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleGetFunction);
    LOAD(cuLaunchKernel);
    LOAD(cuModuleUnload);
    LOAD(cuStreamSynchronize);
    #undef LOAD

    if (!p_cuInit || p_cuInit(0) != 0) {
        fprintf(stderr, "[runtime_select] cuInit failed\n");
        return VENDOR_NONE;
    }

    int count = 0;
    if (p_cuDeviceGetCount(&count) != 0 || count == 0) {
        fprintf(stderr, "[runtime_select] no CUDA devices\n");
        return VENDOR_NONE;
    }

    CUdevice dev;
    if (p_cuDeviceGet(&dev, 0) != 0) return VENDOR_NONE;

    int major = 0, minor = 0;
    if (!p_cuDeviceGetAttribute) return VENDOR_NONE;
    p_cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    p_cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    *out_sm_major = (uint32_t)major;
    *out_sm_minor = (uint32_t)minor;

    if (p_cuDeviceGetName && dev_name && name_len > 0)
        p_cuDeviceGetName(dev_name, name_len, dev);

    /* Create a context for module loading */
    if (p_cuCtxCreate) {
        CUcontext ctx_handle;
        if (p_cuCtxCreate(&ctx_handle, 0, dev) != 0) {
            fprintf(stderr, "[runtime_select] cuCtxCreate failed\n");
            return VENDOR_NONE;
        }
    }

    return VENDOR_NVIDIA;
}

/* ------------------------------------------------------------------ */
/*  Phase 2: Build dispatch table from directory of cubins             */
/*  Mirrors the N global constants emitted by embedBinary()           */
/* ------------------------------------------------------------------ */

static void *load_file(const char *path, uint64_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (uint64_t)sz;
    return buf;
}

/* Parse SM version from filename like "kernel_sm75.cubin" -> 75 */
static uint32_t parse_sm_from_filename(const char *name) {
    const char *p = strstr(name, "sm");
    if (!p) return 0;
    p += 2;
    if (*p == '_') p++;
    return (uint32_t)atoi(p);
}

static int build_dispatch_table_from_dir(const char *dir_path) {
    DIR *d = opendir(dir_path);
    if (!d) {
        fprintf(stderr, "[runtime_select] cannot open directory: %s\n", dir_path);
        return -1;
    }

    struct dirent *ent;
    g_num_entries = 0;
    while ((ent = readdir(d)) != NULL && g_num_entries < MAX_ENTRIES) {
        const char *name = ent->d_name;
        size_t len = strlen(name);
        if (len < 6) continue;

        /* Accept .cubin files */
        if (strcmp(name + len - 6, ".cubin") != 0) continue;

        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s/%s", dir_path, name);

        uint64_t sz = 0;
        void *blob = load_file(path, &sz);
        if (!blob) continue;

        RuntimeSelectEntry *e = &g_dispatch_table[g_num_entries];
        e->vendor_id = VENDOR_NVIDIA;
        e->blob_ptr = blob;
        e->blob_size = sz;
        e->min_sm = parse_sm_from_filename(name);
        e->variant_priority = (e->min_sm >= 90) ? 10 : 5; /* higher SM = higher priority */
        snprintf(e->variant_tag, sizeof(e->variant_tag), "sm_%u", e->min_sm);

        printf("  [%d] loaded %s (%lu bytes, min_sm=%u, priority=%u)\n",
               g_num_entries, name, (unsigned long)sz, e->min_sm, e->variant_priority);
        g_num_entries++;
    }
    closedir(d);
    return g_num_entries;
}

/* ------------------------------------------------------------------ */
/*  Parse arch string "sm_75" -> uint32_t 75                         */
/* ------------------------------------------------------------------ */

static uint32_t parse_sm_from_arch_str(const char *arch) {
    /* arch is e.g. "sm_75" or "sm_86" */
    const char *p = arch;
    if (strncmp(p, "sm_", 3) == 0) p += 3;
    else if (strncmp(p, "sm", 2) == 0) p += 2;
    return (uint32_t)atoi(p);
}

/* ------------------------------------------------------------------ */
/*  Build dispatch table from LLVM OffloadBinary file                 */
/*  Mirrors what embedBinary() emits as @kernels_blob_{idx} globals   */
/* ------------------------------------------------------------------ */

static int build_dispatch_table_from_offloadbin(const char *path) {
    /* Read entire file */
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[runtime_select] cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz <= (long)sizeof(ObFileHeader)) {
        fprintf(stderr, "[runtime_select] offloadbin too small\n");
        fclose(f); return -1;
    }
    uint8_t *buf = malloc((size_t)fsz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        fprintf(stderr, "[runtime_select] read error\n");
        fclose(f); free(buf); return -1;
    }
    fclose(f);

    /* Validate header */
    ObFileHeader *hdr = (ObFileHeader *)buf;
    if (hdr->magic != OFFLOAD_BINARY_MAGIC) {
        fprintf(stderr, "[runtime_select] bad OffloadBinary magic: 0x%lX\n",
                (unsigned long)hdr->magic);
        free(buf); return -1;
    }
    if (hdr->version != OFFLOAD_BINARY_VERSION) {
        fprintf(stderr, "[runtime_select] unsupported OffloadBinary version: %lu\n",
                (unsigned long)hdr->version);
        free(buf); return -1;
    }

    printf("  OffloadBinary: magic=0x%lX version=%lu entries=%lu\n",
           (unsigned long)hdr->magic,
           (unsigned long)hdr->version,
           (unsigned long)hdr->entry_count);

    /* Walk entries, build dispatch table */
    uint64_t entry_off = hdr->entry_offset;
    g_num_entries = 0;

    for (uint64_t i = 0; i < hdr->entry_count && g_num_entries < MAX_ENTRIES; i++) {
        if (entry_off + sizeof(ObEntryHeader) > (uint64_t)fsz) {
            fprintf(stderr, "  entry %lu out of bounds\n", (unsigned long)i);
            break;
        }

        ObEntryHeader *ehdr = (ObEntryHeader *)(buf + entry_off);

        /* Parse string table for arch and kind */
        const char *strtab = (const char *)(buf + entry_off + ehdr->string_offset);
        uint64_t strtab_end = ehdr->string_size;
        char arch_str[32] = {0};
        char kind_str[32] = {0};
        char triple_str[64] = {0};
        uint64_t pos = 0;
        while (pos < strtab_end) {
            const char *key = strtab + pos;
            size_t klen = strlen(key);
            pos += klen + 1;
            if (pos >= strtab_end) break;
            const char *val = strtab + pos;
            size_t vlen = strlen(val);
            pos += vlen + 1;
            if (strcmp(key, "arch")   == 0) strncpy(arch_str,   val, sizeof(arch_str)   - 1);
            if (strcmp(key, "kind")   == 0) strncpy(kind_str,   val, sizeof(kind_str)   - 1);
            if (strcmp(key, "triple") == 0) strncpy(triple_str, val, sizeof(triple_str) - 1);
        }

        /* Only accept CUDA entries for NVIDIA dispatch */
        if (strcmp(kind_str, "cuda") != 0) {
            entry_off += ehdr->the_size;
            continue;
        }

        /* Copy image out of mmap'd buffer (we own buf, but entries share it) */
        uint64_t img_sz = ehdr->image_size;
        uint8_t *img_copy = malloc((size_t)img_sz);
        if (!img_copy) { entry_off += ehdr->the_size; continue; }
        memcpy(img_copy, buf + entry_off + ehdr->image_offset, (size_t)img_sz);

        uint32_t min_sm = parse_sm_from_arch_str(arch_str);

        RuntimeSelectEntry *e = &g_dispatch_table[g_num_entries];
        e->vendor_id = VENDOR_NVIDIA;
        e->blob_ptr  = img_copy;
        e->blob_size = img_sz;
        e->min_sm    = min_sm;
        e->variant_priority = (min_sm >= 90) ? 10 : 5;
        snprintf(e->variant_tag, sizeof(e->variant_tag), "%s", arch_str);

        printf("  [%d] %s triple=%s kind=%s (%lu bytes, min_sm=%u, priority=%u)\n",
               g_num_entries, arch_str, triple_str, kind_str,
               (unsigned long)img_sz, min_sm, e->variant_priority);

        g_num_entries++;
        entry_off += ehdr->the_size;
    }

    free(buf);  /* entries hold copies of image data */
    return g_num_entries;
}

/* Build a synthetic table for demo when no cubin directory is given */
static void build_synthetic_table(void) {
    /*
     * In the real LLVM IR emission, each #gpu.object blob becomes a
     * @kernels_blob_{idx} global constant. Here we simulate 3 entries
     * with NULL blobs -- the selection logic is the interesting part.
     */
    static const char tag_sm50[] = "sm_50";
    static const char tag_sm75[] = "sm_75";
    static const char tag_sm90[] = "sm_90";

    g_dispatch_table[0] = (RuntimeSelectEntry){
        .vendor_id = VENDOR_NVIDIA, .blob_ptr = NULL, .blob_size = 0,
        .min_sm = 50, .variant_priority = 1, .variant_tag = {0}
    };
    memcpy(g_dispatch_table[0].variant_tag, tag_sm50, sizeof(tag_sm50));

    g_dispatch_table[1] = (RuntimeSelectEntry){
        .vendor_id = VENDOR_NVIDIA, .blob_ptr = NULL, .blob_size = 0,
        .min_sm = 75, .variant_priority = 5, .variant_tag = {0}
    };
    memcpy(g_dispatch_table[1].variant_tag, tag_sm75, sizeof(tag_sm75));

    g_dispatch_table[2] = (RuntimeSelectEntry){
        .vendor_id = VENDOR_NVIDIA, .blob_ptr = NULL, .blob_size = 0,
        .min_sm = 90, .variant_priority = 10, .variant_tag = {0}
    };
    memcpy(g_dispatch_table[2].variant_tag, tag_sm90, sizeof(tag_sm90));

    g_num_entries = 3;
}

/* ------------------------------------------------------------------ */
/*  Phase 3: Selection logic                                          */
/*  Mirrors __gpu_runtime_select_rank() in emitted LLVM IR            */
/*                                                                    */
/*  Strategy: rank_by_priority                                        */
/*    1. Filter: entry.vendor_id must match detected vendor           */
/*    2. Filter: entry.min_sm <= device SM version (compatibility)    */
/*    3. Rank: highest variant_priority wins                          */
/*    4. Tiebreak: highest min_sm (most specialized)                  */
/* ------------------------------------------------------------------ */

static int select_best_entry(uint32_t device_vendor,
                             uint32_t device_sm) {
    int best_idx = -1;
    uint32_t best_priority = 0;
    uint32_t best_sm = 0;

    for (int i = 0; i < g_num_entries; i++) {
        RuntimeSelectEntry *e = &g_dispatch_table[i];

        /* Filter: vendor must match */
        if (e->vendor_id != device_vendor) continue;

        /* Filter: device must meet minimum SM requirement */
        if (e->min_sm > device_sm) continue;

        /* Rank: highest priority wins, tiebreak on SM */
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
/*  Phase 4+5: Module load and kernel launch                          */
/*  Mirrors the global_ctors stub + launchKernel() emission           */
/* ------------------------------------------------------------------ */

static int load_and_launch(int entry_idx) {
    RuntimeSelectEntry *e = &g_dispatch_table[entry_idx];

    if (!e->blob_ptr || e->blob_size == 0) {
        printf("  (synthetic entry -- skipping actual GPU load/launch)\n");
        return 0;
    }

    if (!p_cuModuleLoadData) {
        fprintf(stderr, "[runtime_select] cuModuleLoadData not available\n");
        return -1;
    }

    /* Phase 4: cuModuleLoadData -- mirrors global_ctors module load */
    uint64_t t0 = now_ns();
    CUresult res = p_cuModuleLoadData(&g_module_ptr, e->blob_ptr);
    uint64_t t1 = now_ns();

    if (res != 0) {
        fprintf(stderr, "[runtime_select] cuModuleLoadData failed: %d\n", res);
        return -1;
    }
    printf("  module_load_ns: %lu\n", (unsigned long)(t1 - t0));

    /* Get kernel function handle */
    CUfunction func = NULL;
    uint64_t t2 = now_ns();
    res = p_cuModuleGetFunction(&func, g_module_ptr, "_null_kernel");
    uint64_t t3 = now_ns();

    if (res != 0) {
        /* Try alternate name -- cubin might use different mangling */
        res = p_cuModuleGetFunction(&func, g_module_ptr, "null_kernel");
        t3 = now_ns();
    }
    if (res != 0) {
        fprintf(stderr, "[runtime_select] cuModuleGetFunction failed: %d\n", res);
        fprintf(stderr, "  (expected kernel named '_null_kernel' or 'null_kernel')\n");
        p_cuModuleUnload(g_module_ptr);
        g_module_ptr = NULL;
        return -1;
    }
    printf("  get_function_ns: %lu\n", (unsigned long)(t3 - t2));

    /* Phase 5: cuLaunchKernel -- 1 thread, 0 shared mem, null stream */
    uint64_t t4 = now_ns();
    res = p_cuLaunchKernel(func,
                           1, 1, 1,    /* grid: 1x1x1 */
                           1, 1, 1,    /* block: 1x1x1 */
                           0,          /* shared mem bytes */
                           NULL,       /* stream (default) */
                           NULL,       /* kernel params */
                           NULL);      /* extra */
    uint64_t t5 = now_ns();

    if (res != 0) {
        fprintf(stderr, "[runtime_select] cuLaunchKernel failed: %d\n", res);
        p_cuModuleUnload(g_module_ptr);
        g_module_ptr = NULL;
        return -1;
    }

    /* Synchronize to ensure launch completed */
    if (p_cuStreamSynchronize) p_cuStreamSynchronize(NULL);
    uint64_t t6 = now_ns();

    printf("  launch_ns: %lu\n", (unsigned long)(t5 - t4));
    printf("  sync_ns:   %lu\n", (unsigned long)(t6 - t5));

    /* Cleanup */
    p_cuModuleUnload(g_module_ptr);
    g_module_ptr = NULL;

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Main: orchestrates the full runtime_select pipeline               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    const char *cubin_dir = (argc > 1) ? argv[1] : NULL;

    printf("=== #gpu.runtime_select Proof-of-Concept ===\n");
    printf("Demonstrates RuntimeSelectAttr::embedBinary() mechanism in C\n\n");

    /* --- Phase 1: Vendor detection --- */
    printf("[Phase 1] Vendor detection (dlopen probe)\n");
    uint32_t sm_major = 0, sm_minor = 0;
    char dev_name[256] = {0};

    uint64_t t_detect_start = now_ns();
    uint32_t vendor = detect_vendor(&sm_major, &sm_minor, dev_name, sizeof(dev_name));
    uint64_t t_detect_end = now_ns();

    uint32_t device_sm = sm_major * 10 + sm_minor;
    printf("  vendor:      %s (id=%u)\n",
           vendor == VENDOR_NVIDIA ? "NVIDIA" :
           vendor == VENDOR_AMD    ? "AMD" : "NONE", vendor);
    printf("  device:      %s\n", dev_name[0] ? dev_name : "(unknown)");
    printf("  sm_version:  %u (sm_%u%u)\n", device_sm, sm_major, sm_minor);
    printf("  detect_ns:   %lu\n\n", (unsigned long)(t_detect_end - t_detect_start));

    /* --- Phase 2: Build dispatch table --- */
    printf("[Phase 2] Dispatch table construction\n");
    uint64_t t_table_start = now_ns();
    if (cubin_dir) {
        /* Detect if argument is an OffloadBinary file (.offloadbin extension) */
        size_t arglen = strlen(cubin_dir);
        int is_offloadbin = (arglen > 11 &&
                             strcmp(cubin_dir + arglen - 11, ".offloadbin") == 0);
        if (is_offloadbin) {
            printf("  source: OffloadBinary file %s\n", cubin_dir);
            if (build_dispatch_table_from_offloadbin(cubin_dir) < 0) {
                fprintf(stderr, "Failed to parse OffloadBinary\n");
                return 1;
            }
        } else {
            printf("  source: directory %s\n", cubin_dir);
            if (build_dispatch_table_from_dir(cubin_dir) < 0) {
                fprintf(stderr, "Failed to build dispatch table\n");
                return 1;
            }
        }
    } else {
        printf("  source: synthetic entries (no cubin directory given)\n");
        build_synthetic_table();
    }
    uint64_t t_table_end = now_ns();
    printf("  entries:     %d\n", g_num_entries);
    printf("  table_ns:    %lu\n\n", (unsigned long)(t_table_end - t_table_start));

    /* --- Phase 3: Selection --- */
    printf("[Phase 3] Variant selection (strategy=rank_by_priority)\n");
    uint64_t t_select_start = now_ns();
    g_selected_idx = select_best_entry(vendor, device_sm);
    uint64_t t_select_end = now_ns();

    if (g_selected_idx < 0) {
        fprintf(stderr, "  NO COMPATIBLE ENTRY FOUND\n");
        fprintf(stderr, "  (device sm_%u, vendor=%u, %d entries checked)\n",
                device_sm, vendor, g_num_entries);
        return 1;
    }

    RuntimeSelectEntry *selected = &g_dispatch_table[g_selected_idx];
    printf("  selected:    [%d] %s (min_sm=%u, priority=%u)\n",
           g_selected_idx, selected->variant_tag,
           selected->min_sm, selected->variant_priority);
    printf("  select_ns:   %lu\n\n", (unsigned long)(t_select_end - t_select_start));

    /* --- Phase 4+5: Load and launch --- */
    printf("[Phase 4+5] Module load + kernel launch\n");
    int rc = load_and_launch(g_selected_idx);

    /* --- Summary --- */
    printf("\n=== Timing Summary ===\n");
    uint64_t detect_ns = t_detect_end - t_detect_start;
    uint64_t table_ns  = t_table_end - t_table_start;
    uint64_t select_ns = t_select_end - t_select_start;
    printf("  detect_ns:   %lu\n", (unsigned long)detect_ns);
    printf("  table_ns:    %lu\n", (unsigned long)table_ns);
    printf("  select_ns:   %lu\n", (unsigned long)select_ns);
    printf("  total_overhead_ns: %lu  (detect + table + select)\n",
           (unsigned long)(detect_ns + table_ns + select_ns));
    printf("\n");

    /* --- Selection benchmark: measure isolated select_best_entry cost --- */
    printf("=== Selection Microbenchmark (100,000 iterations) ===\n");
    uint64_t bench_start = now_ns();
    volatile int sink = 0;
    for (int i = 0; i < 100000; i++) {
        sink = select_best_entry(vendor, device_sm);
    }
    uint64_t bench_end = now_ns();
    uint64_t per_select = (bench_end - bench_start) / 100000;
    printf("  per_select_ns: %lu\n", (unsigned long)per_select);
    printf("  (this is the runtime cost added by #gpu.runtime_select\n");
    printf("   vs. #gpu.select_object's zero-cost compile-time selection)\n");
    (void)sink;

    /* Cleanup loaded blobs */
    for (int i = 0; i < g_num_entries; i++) {
        if (g_dispatch_table[i].blob_ptr && cubin_dir)
            free((void *)g_dispatch_table[i].blob_ptr);
    }

    if (g_libcuda) dlclose(g_libcuda);

    return rc;
}
