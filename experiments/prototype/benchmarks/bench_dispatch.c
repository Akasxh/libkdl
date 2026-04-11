/*
 * bench_dispatch.c -- Dispatch overhead microbenchmark
 *
 * Measures latency of each kdl API call in isolation:
 *   kdl_init, kdl_load_bundle, kdl_select_kernel (cold/cached), kdl_launch
 *
 * If libcuda.so.1 is present, also compares against a direct cuLaunchKernel
 * baseline to quantify the overhead introduced by kdl dispatch.
 *
 * Timing: clock_gettime(CLOCK_MONOTONIC), 1000 iterations per phase.
 * Output: CSV to stdout (import into plot_results.py).
 *
 * Compile:
 *   gcc -O2 -Wall -I../src -L.. -lkdl -ldl -lm -o bench_dispatch bench_dispatch.c
 *
 * Run (CPU-only, no GPU required):
 *   LD_LIBRARY_PATH=.. ./bench_dispatch
 */

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kdl.h"

/* ------------------------------------------------------------------ */
/* Timing helpers                                                       */
/* ------------------------------------------------------------------ */

#define N_ITER 1000

static inline double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Sort doubles in-place (insertion sort — fine for N_ITER=1000). */
static void sort_doubles(double *arr, int n)
{
    for (int i = 1; i < n; i++) {
        double key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

typedef struct {
    double mean_ns;
    double median_ns;
    double p99_ns;
    double min_ns;
    double max_ns;
} bench_stats;

static bench_stats compute_stats(double *samples, int n)
{
    bench_stats s;
    double sum = 0.0;
    s.min_ns = samples[0];
    s.max_ns = samples[0];
    for (int i = 0; i < n; i++) {
        sum += samples[i];
        if (samples[i] < s.min_ns) s.min_ns = samples[i];
        if (samples[i] > s.max_ns) s.max_ns = samples[i];
    }
    s.mean_ns = sum / n;

    /* work on a sorted copy for percentiles */
    double *sorted = malloc(n * sizeof(double));
    if (!sorted) { s.median_ns = s.p99_ns = 0; return s; }
    memcpy(sorted, samples, n * sizeof(double));
    sort_doubles(sorted, n);
    s.median_ns = sorted[n / 2];
    s.p99_ns    = sorted[(int)(n * 0.99)];
    free(sorted);
    return s;
}

/* ------------------------------------------------------------------ */
/* Demo MTB bundle (embedded, so the benchmark is self-contained)      */
/* ------------------------------------------------------------------ */

/*
 * We build a minimal valid MTB in memory and write it to a temp file so that
 * kdl_load_bundle can be exercised without requiring a real compiled kernel.
 *
 * Layout mirrors Section 4.1 of ARCHITECTURE.md:
 *   [Header 32 bytes][KernelTable 12 bytes][VariantTable 40 bytes]
 *   [StringTable][BinaryData (1-byte dummy)]
 */

#define MTB_MAGIC "KDL_MTB\0"

#pragma pack(push, 1)
typedef struct {
    char     magic[8];       /* "KDL_MTB\0" */
    uint32_t version;        /* 1 */
    uint32_t num_kernels;
    uint32_t num_variants;
    uint32_t string_table_offset;
    uint32_t binary_data_offset;
    uint32_t reserved;
} mtb_header;

typedef struct {
    uint32_t name_offset;
    uint32_t first_variant_idx;
    uint32_t num_variants;
} mtb_kernel_entry;

typedef struct {
    uint32_t target_kind;         /* KDL_TARGET_X86_64 = 3 */
    uint32_t target_chip_offset;
    uint32_t contract_offset;
    uint32_t priority;
    uint64_t binary_offset;
    uint64_t binary_size;
    uint32_t entry_point_offset;
    uint32_t reserved;
} mtb_variant_entry;
#pragma pack(pop)

/*
 * String table (offsets noted):
 *   0  : "matmul\0"         (7)
 *   7  : "x86-64-v3\0"      (10)
 *   17 : "{\"target\":\"x86\"}\0" (17)
 *   34 : "matmul_entry\0"   (13)
 *   47 : total
 */
static const char STRTAB[] =
    "matmul\0"
    "x86-64-v3\0"
    "{\"target\":\"x86\"}\0"
    "matmul_entry\0";

#define STRTAB_LEN sizeof(STRTAB)

/* Write the demo MTB to a temp file; return 0 on success. */
static int write_demo_bundle(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t header_sz  = sizeof(mtb_header);
    uint32_t ktable_sz  = sizeof(mtb_kernel_entry);    /* 1 kernel */
    uint32_t vtable_sz  = sizeof(mtb_variant_entry);   /* 1 variant */
    uint32_t strtab_off = header_sz + ktable_sz + vtable_sz;
    uint32_t bindata_off = strtab_off + (uint32_t)STRTAB_LEN;

    mtb_header hdr;
    memcpy(hdr.magic, MTB_MAGIC, 8);
    hdr.version             = 1;
    hdr.num_kernels         = 1;
    hdr.num_variants        = 1;
    hdr.string_table_offset = strtab_off;
    hdr.binary_data_offset  = bindata_off;
    hdr.reserved            = 0;

    mtb_kernel_entry ke;
    ke.name_offset       = 0;   /* "matmul" */
    ke.first_variant_idx = 0;
    ke.num_variants      = 1;

    mtb_variant_entry ve;
    ve.target_kind        = 3;  /* X86_64 */
    ve.target_chip_offset = 7;  /* "x86-64-v3" */
    ve.contract_offset    = 17; /* JSON contract */
    ve.priority           = 0;
    ve.binary_offset      = 0;
    ve.binary_size        = 1;  /* 1 dummy byte */
    ve.entry_point_offset = 34; /* "matmul_entry" */
    ve.reserved           = 0;

    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(&ke,  sizeof(ke),  1, f);
    fwrite(&ve,  sizeof(ve),  1, f);
    fwrite(STRTAB, STRTAB_LEN, 1, f);
    fputc(0x90, f); /* 1-byte dummy binary (NOP) */
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* CSV output helpers                                                   */
/* ------------------------------------------------------------------ */

static void print_csv_header(void)
{
    printf("phase,target,mean_ns,median_ns,p99_ns,min_ns,max_ns\n");
}

static void print_csv_row(const char *phase, const char *target,
                          bench_stats s)
{
    printf("%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f\n",
           phase, target,
           s.mean_ns, s.median_ns, s.p99_ns, s.min_ns, s.max_ns);
}

/* ------------------------------------------------------------------ */
/* CUDA direct-launch baseline (optional, loaded at runtime)           */
/* ------------------------------------------------------------------ */

typedef int (*cu_init_fn)(unsigned int);
typedef int (*cu_dev_get_fn)(int *, int);
typedef int (*cu_ctx_create_fn)(void **, unsigned int, int);
typedef int (*cu_mod_load_data_fn)(void **, const void *);
typedef int (*cu_mod_get_fn_fn)(void **, void *, const char *);
typedef int (*cu_launch_fn)(void *, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, void *,
                            void **, void **);
typedef int (*cu_stream_sync_fn)(void *);

typedef struct {
    void *lib;
    cu_init_fn         cuInit;
    cu_dev_get_fn      cuDeviceGet;
    cu_ctx_create_fn   cuCtxCreate;
    cu_launch_fn       cuLaunchKernel;
    cu_stream_sync_fn  cuStreamSynchronize;
    int available;
} cuda_fns;

static cuda_fns load_cuda(void)
{
    cuda_fns c;
    memset(&c, 0, sizeof(c));
    c.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!c.lib) return c;

    c.cuInit               = dlsym(c.lib, "cuInit");
    c.cuDeviceGet          = dlsym(c.lib, "cuDeviceGet");
    c.cuCtxCreate          = dlsym(c.lib, "cuCtxCreate_v2");
    c.cuLaunchKernel       = dlsym(c.lib, "cuLaunchKernel");
    c.cuStreamSynchronize  = dlsym(c.lib, "cuStreamSynchronize");

    if (!c.cuInit || !c.cuLaunchKernel) { dlclose(c.lib); c.lib = NULL; return c; }

    int r = c.cuInit(0);
    if (r != 0) { dlclose(c.lib); c.lib = NULL; return c; }

    c.available = 1;
    return c;
}

/* ------------------------------------------------------------------ */
/* Benchmark phases                                                     */
/* ------------------------------------------------------------------ */

static void bench_init(bench_stats *out)
{
    double samples[N_ITER];
    for (int i = 0; i < N_ITER; i++) {
        kdl_ctx ctx = NULL;
        double t0 = now_ns();
        kdl_init(&ctx);
        double t1 = now_ns();
        samples[i] = t1 - t0;
        if (ctx) kdl_finalize(ctx);
    }
    *out = compute_stats(samples, N_ITER);
}

static void bench_load_bundle(const char *bundle_path, bench_stats *out)
{
    double samples[N_ITER];
    kdl_ctx ctx = NULL;
    if (kdl_init(&ctx) != KDL_SUCCESS) {
        fprintf(stderr, "bench_load_bundle: kdl_init failed\n");
        return;
    }
    for (int i = 0; i < N_ITER; i++) {
        kdl_bundle_t b = NULL;
        double t0 = now_ns();
        kdl_load_bundle(ctx, bundle_path, &b);
        double t1 = now_ns();
        samples[i] = t1 - t0;
        if (b) kdl_free_bundle(b);
    }
    kdl_finalize(ctx);
    *out = compute_stats(samples, N_ITER);
}

static void bench_select_cold(const char *bundle_path,
                              bench_stats *out)
{
    double samples[N_ITER];
    kdl_ctx ctx = NULL;
    if (kdl_init(&ctx) != KDL_SUCCESS) return;

    for (int i = 0; i < N_ITER; i++) {
        kdl_bundle_t b  = NULL;
        kdl_kernel_t k  = NULL;
        kdl_load_bundle(ctx, bundle_path, &b);
        double t0 = now_ns();
        kdl_select_kernel(ctx, b, "matmul", -1, &k);
        double t1 = now_ns();
        samples[i] = t1 - t0;
        if (b) kdl_free_bundle(b);
    }
    kdl_finalize(ctx);
    *out = compute_stats(samples, N_ITER);
}

static void bench_select_cached(const char *bundle_path,
                                bench_stats *out)
{
    double samples[N_ITER];
    kdl_ctx     ctx = NULL;
    kdl_bundle_t b  = NULL;
    kdl_kernel_t k  = NULL;

    if (kdl_init(&ctx) != KDL_SUCCESS) return;
    kdl_load_bundle(ctx, bundle_path, &b);
    /* warm the cache */
    kdl_select_kernel(ctx, b, "matmul", -1, &k);

    for (int i = 0; i < N_ITER; i++) {
        kdl_kernel_t k2 = NULL;
        double t0 = now_ns();
        kdl_select_kernel(ctx, b, "matmul", -1, &k2);
        double t1 = now_ns();
        samples[i] = t1 - t0;
    }
    if (b) kdl_free_bundle(b);
    kdl_finalize(ctx);
    *out = compute_stats(samples, N_ITER);
}

static void bench_launch(const char *bundle_path, bench_stats *out)
{
    double samples[N_ITER];
    kdl_ctx     ctx = NULL;
    kdl_bundle_t b  = NULL;
    kdl_kernel_t k  = NULL;

    if (kdl_init(&ctx) != KDL_SUCCESS) return;
    if (kdl_load_bundle(ctx, bundle_path, &b) != KDL_SUCCESS) {
        kdl_finalize(ctx);
        return;
    }
    kdl_select_kernel(ctx, b, "matmul", -1, &k);
    if (!k) {
        if (b) kdl_free_bundle(b);
        kdl_finalize(ctx);
        return;
    }

    void *args[1] = {NULL};
    for (int i = 0; i < N_ITER; i++) {
        double t0 = now_ns();
        kdl_launch(k, 1, 1, 1, 1, 1, 1, 0, args);
        double t1 = now_ns();
        samples[i] = t1 - t0;
    }
    if (b) kdl_free_bundle(b);
    kdl_finalize(ctx);
    *out = compute_stats(samples, N_ITER);
}

/* CUDA direct-launch baseline (measures only cuLaunchKernel + cuStreamSync) */
static void bench_cuda_direct(cuda_fns *cu, bench_stats *out)
{
    double samples[N_ITER];

    for (int i = 0; i < N_ITER; i++) {
        double t0 = now_ns();
        /* NULL stream, NULL module fn -- this measures the raw driver API
         * overhead for the call itself; real cubin load is excluded. */
        cu->cuLaunchKernel(NULL, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        cu->cuStreamSynchronize(NULL);
        double t1 = now_ns();
        samples[i] = t1 - t0;
    }
    *out = compute_stats(samples, N_ITER);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    /* Write demo bundle to /tmp */
    const char *bundle_path = "/tmp/kdl_bench_demo.mtb";
    if (write_demo_bundle(bundle_path) != 0) {
        fprintf(stderr, "Failed to write demo bundle to %s\n", bundle_path);
        return 1;
    }

    fprintf(stderr, "mlir-hetero-dispatch: dispatch overhead microbenchmark\n");
    fprintf(stderr, "Iterations per phase: %d\n", N_ITER);
    fprintf(stderr, "Bundle: %s\n\n", bundle_path);

    print_csv_header();

    bench_stats s;

    /* -- kdl_init -- */
    bench_init(&s);
    print_csv_row("kdl_init", "cpu", s);
    fprintf(stderr, "kdl_init        mean=%.0f ns  median=%.0f ns  p99=%.0f ns\n",
            s.mean_ns, s.median_ns, s.p99_ns);

    /* -- kdl_load_bundle -- */
    bench_load_bundle(bundle_path, &s);
    print_csv_row("kdl_load_bundle", "cpu", s);
    fprintf(stderr, "kdl_load_bundle mean=%.0f ns  median=%.0f ns  p99=%.0f ns\n",
            s.mean_ns, s.median_ns, s.p99_ns);

    /* -- kdl_select_kernel (cold) -- */
    bench_select_cold(bundle_path, &s);
    print_csv_row("kdl_select_cold", "cpu", s);
    fprintf(stderr, "kdl_select(cold) mean=%.0f ns  median=%.0f ns  p99=%.0f ns\n",
            s.mean_ns, s.median_ns, s.p99_ns);

    /* -- kdl_select_kernel (cached) -- */
    bench_select_cached(bundle_path, &s);
    print_csv_row("kdl_select_cached", "cpu", s);
    fprintf(stderr, "kdl_select(hit)  mean=%.0f ns  median=%.0f ns  p99=%.0f ns\n",
            s.mean_ns, s.median_ns, s.p99_ns);

    /* -- kdl_launch -- */
    bench_launch(bundle_path, &s);
    print_csv_row("kdl_launch", "cpu", s);
    fprintf(stderr, "kdl_launch       mean=%.0f ns  median=%.0f ns  p99=%.0f ns\n",
            s.mean_ns, s.median_ns, s.p99_ns);

    /* -- CUDA direct baseline (optional) -- */
    cuda_fns cu = load_cuda();
    if (cu.available) {
        bench_cuda_direct(&cu, &s);
        print_csv_row("cuda_direct_launch", "cuda", s);
        fprintf(stderr, "cuda_direct_launch mean=%.0f ns  median=%.0f ns  p99=%.0f ns\n",
                s.mean_ns, s.median_ns, s.p99_ns);

        /*
         * Overhead ratio: how much extra does kdl add on top of the
         * unavoidable CUDA launch floor?
         */
        bench_stats kdl_s;
        bench_launch(bundle_path, &kdl_s);
        double overhead_pct = ((kdl_s.median_ns - s.median_ns) / s.median_ns) * 100.0;
        fprintf(stderr, "\nkdl dispatch overhead vs direct CUDA launch: %.2f%%\n",
                overhead_pct < 0.0 ? 0.0 : overhead_pct);
        if (cu.lib) dlclose(cu.lib);
    } else {
        fprintf(stderr, "(libcuda.so.1 not found -- CUDA baseline skipped)\n");
    }

    fprintf(stderr, "\nCSV written to stdout.\n");
    return 0;
}
