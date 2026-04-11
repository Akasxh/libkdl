/*
 * bench_gemm.c -- GEMM performance benchmark
 *
 * Measures floating-point throughput (GFLOPS) for matrix multiplication:
 *   - CPU naive GEMM baseline (always available)
 *   - kdl-dispatched GEMM (when an MTB bundle is present)
 *   - Direct cuBLAS SGEMM (when CUDA is available)
 *
 * Matrix sizes tested: 256, 512, 1024, 2048 (square, single-precision).
 * GFLOPS = 2*N^3 / time_seconds / 1e9  (standard multiply-accumulate count)
 *
 * Output: CSV to stdout with columns:
 *   size, target, gflops, time_ms, bandwidth_gbps
 *
 * Compile:
 *   gcc -O2 -Wall -I../src -L.. -lkdl -ldl -lm -o bench_gemm bench_gemm.c
 *
 * Run:
 *   LD_LIBRARY_PATH=.. ./bench_gemm [bundle.mtb]
 *
 * If no bundle path is given, only the CPU naive baseline runs.
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
/* Timing                                                               */
/* ------------------------------------------------------------------ */

static inline double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* CPU naive GEMM  C = A * B  (row-major, single-precision)            */
/* ------------------------------------------------------------------ */

/*
 * Simple triple-loop GEMM, cache-oblivious ordering (i-k-j) which gives
 * better cache locality than the textbook i-j-k order on row-major arrays.
 */
static void gemm_cpu_naive(const float *A, const float *B, float *C,
                            int N)
{
    /* Zero C */
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* cuBLAS dynamic binding                                               */
/* ------------------------------------------------------------------ */

typedef int   CUresult;
typedef void *CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *cublasHandle;

typedef int  (*cu_init_fn)(unsigned int);
typedef int  (*cu_dev_get_fn)(void *, int);
typedef int  (*cu_ctx_create_fn)(void **, unsigned int, void *);
typedef int  (*cu_mem_alloc_fn)(void **, size_t);
typedef int  (*cu_mem_free_fn)(void *);
typedef int  (*cu_memcpy_h2d_fn)(void *, const void *, size_t);
typedef int  (*cu_memcpy_d2h_fn)(void *, const void *, size_t);
typedef int  (*cu_dev_sync_fn)(void);


typedef int  (*cublas_create_fn)(cublasHandle *);
typedef int  (*cublas_destroy_fn)(cublasHandle);
typedef int  (*cublas_sgemm_fn)(cublasHandle,
                                 int, int,           /* transa, transb */
                                 int, int, int,       /* m, n, k */
                                 const float *,       /* alpha */
                                 const float *, int,  /* A, lda */
                                 const float *, int,  /* B, ldb */
                                 const float *,       /* beta */
                                 float *, int);       /* C, ldc */

typedef struct {
    void *libcuda;
    void *libcublas;
    cu_init_fn       cuInit;
    cu_dev_get_fn    cuDeviceGet;
    cu_ctx_create_fn cuCtxCreate;
    cu_mem_alloc_fn  cuMemAlloc;
    cu_mem_free_fn   cuMemFree;
    cu_memcpy_h2d_fn cuMemcpyHtoD;
    cu_memcpy_d2h_fn cuMemcpyDtoH;
    cu_dev_sync_fn   cuCtxSynchronize;
    cublas_create_fn  cublasCreate;
    cublas_destroy_fn cublasDestroy;
    cublas_sgemm_fn   cublasSgemm;
    cublasHandle      handle;
    int available;
} gpu_ctx;

static gpu_ctx gpu_init(void)
{
    gpu_ctx g;
    memset(&g, 0, sizeof(g));

    g.libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!g.libcuda) return g;

    g.cuInit           = dlsym(g.libcuda, "cuInit");
    g.cuDeviceGet      = dlsym(g.libcuda, "cuDeviceGet");
    g.cuCtxCreate      = dlsym(g.libcuda, "cuCtxCreate_v2");
    g.cuMemAlloc       = dlsym(g.libcuda, "cuMemAlloc_v2");
    g.cuMemFree        = dlsym(g.libcuda, "cuMemFree_v2");
    g.cuMemcpyHtoD     = dlsym(g.libcuda, "cuMemcpyHtoD_v2");
    g.cuMemcpyDtoH     = dlsym(g.libcuda, "cuMemcpyDtoH_v2");
    g.cuCtxSynchronize = dlsym(g.libcuda, "cuCtxSynchronize");

    if (!g.cuInit || !g.cuMemAlloc) { dlclose(g.libcuda); g.libcuda = NULL; return g; }
    if (g.cuInit(0) != 0) { dlclose(g.libcuda); g.libcuda = NULL; return g; }

    void *dev = NULL;
    if (g.cuDeviceGet(&dev, 0) != 0) { dlclose(g.libcuda); g.libcuda = NULL; return g; }

    void *ctx_ptr = NULL;
    g.cuCtxCreate(&ctx_ptr, 0, dev);

    /* cuBLAS */
    g.libcublas = dlopen("libcublas.so", RTLD_LAZY);
    if (!g.libcublas) g.libcublas = dlopen("libcublas.so.12", RTLD_LAZY);
    if (!g.libcublas) g.libcublas = dlopen("libcublas.so.11", RTLD_LAZY);
    if (!g.libcublas) { dlclose(g.libcuda); g.libcuda = NULL; return g; }

    g.cublasCreate  = dlsym(g.libcublas, "cublasCreate_v2");
    g.cublasDestroy = dlsym(g.libcublas, "cublasDestroy_v2");
    g.cublasSgemm   = dlsym(g.libcublas, "cublasSgemm_v2");

    if (!g.cublasCreate || !g.cublasSgemm) {
        dlclose(g.libcublas); g.libcublas = NULL;
        dlclose(g.libcuda);   g.libcuda   = NULL;
        return g;
    }

    if (g.cublasCreate(&g.handle) != 0) {
        dlclose(g.libcublas); dlclose(g.libcuda);
        g.libcublas = g.libcuda = NULL;
        return g;
    }

    g.available = 1;
    return g;
}

static void gpu_destroy(gpu_ctx *g)
{
    if (g->available && g->cublasDestroy) g->cublasDestroy(g->handle);
    if (g->libcublas) dlclose(g->libcublas);
    if (g->libcuda)   dlclose(g->libcuda);
    g->available = 0;
}

/* ------------------------------------------------------------------ */
/* GFLOPS calculation                                                   */
/* ------------------------------------------------------------------ */

/*
 * For square N x N GEMM: multiply-accumulate pairs = N^3
 * Each pair = 1 multiply + 1 add = 2 FLOPs.
 */
static double gflops(int N, double time_sec)
{
    double flops = 2.0 * (double)N * (double)N * (double)N;
    return flops / time_sec / 1e9;
}

/*
 * Bandwidth: reads A (N^2) + B (N^2) + C (N^2), writes C (N^2).
 * In practice naive CPU only reads C once per i-k pass, but we report
 * the theoretical roofline bandwidth to contextualise arithmetic intensity.
 */
static double bandwidth_gbps(int N, double time_sec)
{
    double bytes = 4.0 * 4.0 * (double)N * (double)N; /* 4 matrices * float */
    return bytes / time_sec / 1e9;
}

/* ------------------------------------------------------------------ */
/* CSV output                                                           */
/* ------------------------------------------------------------------ */

static void print_csv_header(void)
{
    printf("size,target,gflops,time_ms,bandwidth_gbps\n");
}

static void print_csv_row(int N, const char *target,
                           double gf, double time_sec)
{
    printf("%d,%s,%.3f,%.3f,%.3f\n",
           N, target, gf, time_sec * 1000.0,
           bandwidth_gbps(N, time_sec));
}

/* ------------------------------------------------------------------ */
/* Benchmark runners                                                    */
/* ------------------------------------------------------------------ */

/* Allocate and fill a matrix with deterministic values (no rand()). */
static float *alloc_matrix(int N)
{
    float *m = malloc((size_t)N * N * sizeof(float));
    if (!m) return NULL;
    for (int i = 0; i < N * N; i++)
        m[i] = (float)(i % 97) * 0.01f + 0.1f;
    return m;
}

static void run_cpu_naive(int N)
{
    float *A = alloc_matrix(N);
    float *B = alloc_matrix(N);
    float *C = malloc((size_t)N * N * sizeof(float));
    if (!A || !B || !C) { free(A); free(B); free(C); return; }

    /* Warm-up pass (excluded from timing). */
    if (N <= 512) gemm_cpu_naive(A, B, C, N);

    /* Timed pass — for large N use 1 iteration; small N use 3 to reduce noise. */
    int iters = (N >= 1024) ? 1 : 3;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) gemm_cpu_naive(A, B, C, N);
    double elapsed = (now_sec() - t0) / iters;

    double gf = gflops(N, elapsed);
    print_csv_row(N, "cpu_naive", gf, elapsed);
    fprintf(stderr, "  cpu_naive N=%4d  %.3f GFLOPS  %.3f ms\n",
            N, gf, elapsed * 1000.0);

    free(A); free(B); free(C);
}

static void run_kdl_gemm(int N, kdl_ctx ctx, kdl_bundle_t bundle)
{
    if (!ctx || !bundle) return;

    kdl_kernel_t k = NULL;
    if (kdl_select_kernel(ctx, bundle, "matmul", -1, &k) != KDL_SUCCESS || !k) {
        fprintf(stderr, "  kdl: no matching variant for N=%d\n", N);
        return;
    }

    /* Allocate device memory via kdl memory management. */
    void *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t sz = (size_t)N * N * sizeof(float);
    if (kdl_malloc(k, sz, &d_A) != KDL_SUCCESS) return;
    if (kdl_malloc(k, sz, &d_B) != KDL_SUCCESS) { kdl_free_mem(k, d_A); return; }
    if (kdl_malloc(k, sz, &d_C) != KDL_SUCCESS) {
        kdl_free_mem(k, d_A); kdl_free_mem(k, d_B); return;
    }

    float *h_A = alloc_matrix(N);
    float *h_B = alloc_matrix(N);
    if (!h_A || !h_B) { free(h_A); free(h_B); return; }

    kdl_memcpy_h2d(k, d_A, h_A, sz);
    kdl_memcpy_h2d(k, d_B, h_B, sz);

    /*
     * Grid/block: tile 16x16 thread blocks covering the N x N output.
     * Actual launch behaviour depends on which variant kdl selected.
     */
    uint32_t tile = 16;
    uint32_t grid = ((uint32_t)N + tile - 1) / tile;
    void *args[] = {&d_A, &d_B, &d_C, &N};

    /* Warm-up */
    kdl_launch(k, grid, grid, 1, tile, tile, 1, 0, args);
    kdl_sync(k);

    double t0 = now_sec();
    kdl_launch(k, grid, grid, 1, tile, tile, 1, 0, args);
    kdl_sync(k);
    double elapsed = now_sec() - t0;

    double gf = gflops(N, elapsed);
    print_csv_row(N, "kdl_dispatch", gf, elapsed);
    fprintf(stderr, "  kdl_dispatch N=%4d  %.3f GFLOPS  %.3f ms\n",
            N, gf, elapsed * 1000.0);

    kdl_free_mem(k, d_A);
    kdl_free_mem(k, d_B);
    kdl_free_mem(k, d_C);
    free(h_A);
    free(h_B);
}

static void run_cublas(int N, gpu_ctx *g)
{
    if (!g->available) return;

    size_t sz = (size_t)N * N * sizeof(float);
    void *d_A = NULL, *d_B = NULL, *d_C = NULL;

    if (g->cuMemAlloc(&d_A, sz) != 0) return;
    if (g->cuMemAlloc(&d_B, sz) != 0) { g->cuMemFree(d_A); return; }
    if (g->cuMemAlloc(&d_C, sz) != 0) { g->cuMemFree(d_A); g->cuMemFree(d_B); return; }

    float *h_A = alloc_matrix(N);
    float *h_B = alloc_matrix(N);
    if (!h_A || !h_B) { free(h_A); free(h_B); return; }

    g->cuMemcpyHtoD(d_A, h_A, sz);
    g->cuMemcpyHtoD(d_B, h_B, sz);
    free(h_A); free(h_B);

    /* cuBLAS is column-major; pass B, A (swap) to compute A*B row-major. */
    float alpha = 1.0f, beta = 0.0f;

    /* Warm-up */
    g->cublasSgemm(g->handle,
                   111 /* CUBLAS_OP_N */, 111 /* CUBLAS_OP_N */,
                   N, N, N, &alpha,
                   d_B, N, d_A, N, &beta, d_C, N);
    g->cuCtxSynchronize();

    double t0 = now_sec();
    g->cublasSgemm(g->handle,
                   111 /* CUBLAS_OP_N */, 111 /* CUBLAS_OP_N */,
                   N, N, N, &alpha,
                   d_B, N, d_A, N, &beta, d_C, N);
    g->cuCtxSynchronize();
    double elapsed = now_sec() - t0;

    double gf = gflops(N, elapsed);
    print_csv_row(N, "cublas", gf, elapsed);
    fprintf(stderr, "  cublas       N=%4d  %.3f GFLOPS  %.3f ms\n",
            N, gf, elapsed * 1000.0);

    g->cuMemFree(d_A);
    g->cuMemFree(d_B);
    g->cuMemFree(d_C);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *bundle_path = (argc > 1) ? argv[1] : NULL;

    fprintf(stderr, "mlir-hetero-dispatch: GEMM performance benchmark\n");
    fprintf(stderr, "Matrix sizes: 256, 512, 1024, 2048 (square, fp32)\n\n");

    /* Initialise kdl if a bundle was supplied. */
    kdl_ctx     ctx    = NULL;
    kdl_bundle_t bundle = NULL;
    if (bundle_path) {
        if (kdl_init(&ctx) == KDL_SUCCESS)
            kdl_load_bundle(ctx, bundle_path, &bundle);
        if (!bundle)
            fprintf(stderr, "Warning: could not load bundle %s\n", bundle_path);
    }

    /* Initialise GPU (optional). */
    gpu_ctx gpu = gpu_init();
    if (gpu.available)
        fprintf(stderr, "GPU (cuBLAS): available\n");
    else
        fprintf(stderr, "GPU (cuBLAS): not available -- CPU-only run\n");

    if (bundle)
        fprintf(stderr, "kdl bundle: %s\n", bundle_path);
    else
        fprintf(stderr, "kdl bundle: none (pass path as argument)\n");

    fprintf(stderr, "\n");

    print_csv_header();

    static const int sizes[] = {256, 512, 1024, 2048};
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        fprintf(stderr, "N = %d\n", N);
        run_cpu_naive(N);
        run_kdl_gemm(N, ctx, bundle);
        run_cublas(N, &gpu);
    }

    /* Cleanup */
    if (bundle) kdl_free_bundle(bundle);
    if (ctx)    kdl_finalize(ctx);
    gpu_destroy(&gpu);

    fprintf(stderr, "\nCSV written to stdout. Feed into plot_results.py.\n");
    return 0;
}
