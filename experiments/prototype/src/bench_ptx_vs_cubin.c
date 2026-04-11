/*
 * bench_ptx_vs_cubin.c -- PTX (JIT) vs CUBIN (pre-compiled) load benchmark
 *
 * Quantifies the JIT cold-start cost that pre-compiled dispatch avoids.
 * Loads a null kernel in both PTX and CUBIN form via cuModuleLoadDataEx,
 * with JIT caching explicitly disabled (CU_JIT_CACHE_OPTION_NONE).
 *
 * Three measurements:
 *   1. PTX cold first-load (cache cleared, single measurement)
 *   2. PTX steady-state (100 load+unload cycles, cache disabled)
 *   3. CUBIN steady-state (100 load+unload cycles)
 *
 * The ratio PTX/CUBIN is the "JIT cost multiplier".
 *
 * Requires: libcuda.so.1 (NVIDIA driver), no compile-time CUDA dep.
 *
 * Build:
 *   cc -O2 -Wall -std=c11 -o bench_ptx_vs_cubin bench_ptx_vs_cubin.c -ldl
 *
 * Usage:
 *   ./bench_ptx_vs_cubin <cubin_path> <ptx_path>
 *   ./bench_ptx_vs_cubin /tmp/null_sm75.cubin /tmp/null_sm75.ptx
 */

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define WARMUP_ITERS   5
#define MEASURE_ITERS  100
#define COLD_TRIALS    20   /* child-process cold PTX loads */

/* ------------------------------------------------------------------ */
/*  CUDA driver API types                                              */
/* ------------------------------------------------------------------ */

typedef int   CUresult;
typedef int   CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;

#define CUDA_SUCCESS 0

/* CU_JIT_CACHE_MODE = 7, CU_JIT_CACHE_OPTION_NONE = 1 */
#define CU_JIT_CACHE_MODE        7
#define CU_JIT_CACHE_OPTION_NONE 1

/* ------------------------------------------------------------------ */
/*  CUDA driver function pointer table                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    void *lib;
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleLoadDataEx)(CUmodule *, const void *,
                                   unsigned int, int *, void **);
    CUresult (*cuModuleUnload)(CUmodule);
} cuda_api;

/* ------------------------------------------------------------------ */
/*  Timing                                                             */
/* ------------------------------------------------------------------ */

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/*  Statistics                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    double mean_ns;
    double median_ns;
    double p5_ns;
    double p95_ns;
    double min_ns;
    double max_ns;
} bench_stats;

static int cmp_u64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static bench_stats compute_stats(uint64_t *samples, int n)
{
    bench_stats s;
    memset(&s, 0, sizeof(s));
    if (n <= 0) return s;

    uint64_t *sorted = malloc((size_t)n * sizeof(uint64_t));
    if (!sorted) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    memcpy(sorted, samples, (size_t)n * sizeof(uint64_t));
    qsort(sorted, (size_t)n, sizeof(uint64_t), cmp_u64);

    uint64_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += sorted[i];

    s.mean_ns   = (double)sum / n;
    s.median_ns = (double)sorted[n / 2];
    s.p5_ns     = (double)sorted[(int)(n * 0.05)];
    s.p95_ns    = (double)sorted[(int)(n * 0.95)];
    s.min_ns    = (double)sorted[0];
    s.max_ns    = (double)sorted[n - 1];

    free(sorted);
    return s;
}

/* ------------------------------------------------------------------ */
/*  File I/O                                                           */
/* ------------------------------------------------------------------ */

static void *read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s: ", path);
        perror("");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *buf = malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)len, f);
    fclose(f);

    /* null-terminate for PTX (text format) */
    ((char *)buf)[rd] = '\0';
    *out_size = rd;
    return buf;
}

/* ------------------------------------------------------------------ */
/*  Load libcuda.so.1                                                  */
/* ------------------------------------------------------------------ */

static int load_cuda_api(cuda_api *api)
{
    memset(api, 0, sizeof(*api));
    api->lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!api->lib) {
        fprintf(stderr, "dlopen(libcuda.so.1): %s\n", dlerror());
        return -1;
    }

#define LOAD(sym) do { \
    *(void **)(&api->sym) = dlsym(api->lib, #sym); \
    if (!api->sym) { \
        fprintf(stderr, "dlsym(%s) failed\n", #sym); \
        return -1; \
    } \
} while (0)

    LOAD(cuInit);
    LOAD(cuDeviceGet);
    LOAD(cuCtxCreate);
    LOAD(cuCtxDestroy);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleLoadDataEx);
    LOAD(cuModuleUnload);
#undef LOAD

    return 0;
}

/* ------------------------------------------------------------------ */
/*  PTX load with JIT cache disabled                                   */
/* ------------------------------------------------------------------ */

static CUresult load_ptx_no_cache(const cuda_api *api, CUmodule *mod,
                                   const void *data)
{
    int    options[] = { CU_JIT_CACHE_MODE };
    void  *values[]  = { (void *)(uintptr_t)CU_JIT_CACHE_OPTION_NONE };
    return api->cuModuleLoadDataEx(mod, data, 1, options, values);
}

/* ------------------------------------------------------------------ */
/*  Cold PTX load via child process (defeats driver JIT cache)         */
/*                                                                     */
/*  CUDA is not fork-safe after cuInit, so we use execve to get a      */
/*  pristine address space per trial.  The child writes its timing     */
/*  result (ns) to a pipe.                                             */
/* ------------------------------------------------------------------ */

#define COLD_CHILD_ENV "BENCH_PTX_COLD_CHILD"

static int run_cold_ptx_child(const char *self_exe, const char *ptx_path,
                               uint64_t *out_ns)
{
    int pfd[2];
    if (pipe(pfd) < 0) {
        perror("pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* child: exec self with env marker */
        close(pfd[0]);
        char fd_str[32];
        snprintf(fd_str, sizeof(fd_str), "%d", pfd[1]);
        setenv(COLD_CHILD_ENV, fd_str, 1);

        execl(self_exe, self_exe, ptx_path, (char *)NULL);
        perror("execl");
        _exit(127);
    }

    /* parent: read result */
    close(pfd[1]);
    uint64_t ns = 0;
    ssize_t rd = read(pfd[0], &ns, sizeof(ns));
    close(pfd[0]);

    int status;
    waitpid(pid, &status, 0);

    if (rd != sizeof(ns) || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "cold child failed (rd=%zd, status=%d)\n",
                rd, status);
        return -1;
    }

    *out_ns = ns;
    return 0;
}

/*
 * When invoked as cold child: init CUDA, load PTX once, report time, exit.
 */
static int cold_child_main(int argc, char **argv, int write_fd)
{
    (void)argc;
    const char *ptx_path = argv[1];

    size_t ptx_size;
    void *ptx_data = read_file(ptx_path, &ptx_size);
    if (!ptx_data) return 1;

    cuda_api api;
    if (load_cuda_api(&api) != 0) return 1;

    if (api.cuInit(0) != CUDA_SUCCESS) return 1;

    CUdevice dev;
    if (api.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 1;

    CUcontext ctx;
    if (api.cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) return 1;

    /* measure single cold PTX load (no prior JIT in this process) */
    CUmodule mod;
    uint64_t t0 = now_ns();
    CUresult rc = load_ptx_no_cache(&api, &mod, ptx_data);
    uint64_t t1 = now_ns();

    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cold PTX load failed: %d\n", rc);
        return 1;
    }
    api.cuModuleUnload(mod);

    uint64_t elapsed = t1 - t0;
    if (write(write_fd, &elapsed, sizeof(elapsed)) != sizeof(elapsed))
        return 1;
    close(write_fd);

    api.cuCtxDestroy(ctx);
    free(ptx_data);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Benchmark: CUBIN load+unload cycle                                 */
/* ------------------------------------------------------------------ */

static int bench_cubin_cycle(const cuda_api *api, const void *data,
                              bench_stats *out)
{
    uint64_t samples[MEASURE_ITERS];
    CUmodule mod;
    CUresult rc;

    /* warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        rc = api->cuModuleLoadData(&mod, data);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[CUBIN] warmup failed: %d\n", rc);
            return -1;
        }
        api->cuModuleUnload(mod);
    }

    /* measure */
    for (int i = 0; i < MEASURE_ITERS; i++) {
        uint64_t t0 = now_ns();
        rc = api->cuModuleLoadData(&mod, data);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[CUBIN] iter %d failed: %d\n", i, rc);
            return -1;
        }
        api->cuModuleUnload(mod);
        uint64_t t1 = now_ns();
        samples[i] = t1 - t0;
    }

    *out = compute_stats(samples, MEASURE_ITERS);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Benchmark: PTX load+unload cycle (cache disabled in-process)       */
/* ------------------------------------------------------------------ */

static int bench_ptx_cycle(const cuda_api *api, const void *data,
                            bench_stats *out)
{
    uint64_t samples[MEASURE_ITERS];
    CUmodule mod;
    CUresult rc;

    /* warmup (still JIT compiles each time with cache disabled) */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        rc = load_ptx_no_cache(api, &mod, data);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[PTX] warmup failed: %d\n", rc);
            return -1;
        }
        api->cuModuleUnload(mod);
    }

    /* measure */
    for (int i = 0; i < MEASURE_ITERS; i++) {
        uint64_t t0 = now_ns();
        rc = load_ptx_no_cache(api, &mod, data);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[PTX] iter %d failed: %d\n", i, rc);
            return -1;
        }
        api->cuModuleUnload(mod);
        uint64_t t1 = now_ns();
        samples[i] = t1 - t0;
    }

    *out = compute_stats(samples, MEASURE_ITERS);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Print helpers                                                      */
/* ------------------------------------------------------------------ */

static void print_stats_row(const char *label, bench_stats s)
{
    printf("%-16s  %10.1f  %10.1f  %10.1f  %10.1f  %10.1f  %10.1f\n",
           label,
           s.mean_ns / 1000.0,
           s.median_ns / 1000.0,
           s.p5_ns / 1000.0,
           s.p95_ns / 1000.0,
           s.min_ns / 1000.0,
           s.max_ns / 1000.0);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    /* child-process mode for cold PTX measurement */
    const char *child_env = getenv(COLD_CHILD_ENV);
    if (child_env) {
        int fd = atoi(child_env);
        return cold_child_main(argc, argv, fd);
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <cubin_path> <ptx_path>\n", argv[0]);
        return 1;
    }

    const char *cubin_path = argv[1];
    const char *ptx_path   = argv[2];

    /* load binary files */
    size_t cubin_size, ptx_size;
    void *cubin_data = read_file(cubin_path, &cubin_size);
    void *ptx_data   = read_file(ptx_path, &ptx_size);
    if (!cubin_data || !ptx_data) return 1;

    printf("=== PTX vs CUBIN Load Benchmark ===\n");
    printf("CUBIN: %s (%zu bytes)\n", cubin_path, cubin_size);
    printf("PTX:   %s (%zu bytes)\n", ptx_path, ptx_size);
    printf("Iterations: %d steady-state, %d cold PTX trials\n\n",
           MEASURE_ITERS, COLD_TRIALS);

    /* ---- Phase 1: Cold PTX loads via child processes ---- */
    printf("--- Phase 1: Cold PTX loads (child process per trial) ---\n");
    uint64_t cold_samples[COLD_TRIALS];
    int cold_ok = 0;

    /* resolve self path for execl */
    char self_exe[4096];
    ssize_t self_len = readlink("/proc/self/exe", self_exe,
                                 sizeof(self_exe) - 1);
    if (self_len <= 0) {
        fprintf(stderr, "cannot resolve /proc/self/exe\n");
        return 1;
    }
    self_exe[self_len] = '\0';

    for (int i = 0; i < COLD_TRIALS; i++) {
        if (run_cold_ptx_child(self_exe, ptx_path, &cold_samples[i]) == 0)
            cold_ok++;
        else
            fprintf(stderr, "  cold trial %d failed\n", i);
    }

    bench_stats cold_stats = {0};
    if (cold_ok > 0) {
        cold_stats = compute_stats(cold_samples, cold_ok);
        printf("  %d/%d trials succeeded\n\n", cold_ok, COLD_TRIALS);
    }

    /* ---- Phase 2: In-process steady-state ---- */
    printf("--- Phase 2: Steady-state (in-process, %d iters) ---\n",
           MEASURE_ITERS);

    /* init CUDA driver */
    cuda_api api;
    if (load_cuda_api(&api) != 0) return 1;

    CUresult rc = api.cuInit(0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", rc);
        return 1;
    }

    CUdevice dev;
    rc = api.cuDeviceGet(&dev, 0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", rc);
        return 1;
    }

    CUcontext ctx;
    rc = api.cuCtxCreate(&ctx, 0, dev);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", rc);
        return 1;
    }

    bench_stats cubin_stats;
    if (bench_cubin_cycle(&api, cubin_data, &cubin_stats) != 0)
        return 1;

    bench_stats ptx_warm_stats;
    if (bench_ptx_cycle(&api, ptx_data, &ptx_warm_stats) != 0)
        return 1;

    /* ---- Results ---- */
    printf("\n%-16s  %10s  %10s  %10s  %10s  %10s  %10s\n",
           "mode", "mean_us", "median_us", "p5_us", "p95_us",
           "min_us", "max_us");
    printf("%-16s  %10s  %10s  %10s  %10s  %10s  %10s\n",
           "----------------", "----------", "----------", "----------",
           "----------", "----------", "----------");

    print_stats_row("CUBIN", cubin_stats);
    print_stats_row("PTX (warm)", ptx_warm_stats);
    if (cold_ok > 0)
        print_stats_row("PTX (cold)", cold_stats);

    printf("\n");
    printf("=== JIT Cost Multiplier (PTX / CUBIN) ===\n");
    double ratio_warm = ptx_warm_stats.median_ns / cubin_stats.median_ns;
    printf("  warm PTX / CUBIN:  %6.1fx (median)\n", ratio_warm);

    if (cold_ok > 0) {
        double ratio_cold = cold_stats.median_ns / cubin_stats.median_ns;
        printf("  cold PTX / CUBIN:  %6.1fx (median)\n", ratio_cold);
    }

    printf("\n");
    if (cold_ok > 0) {
        double ratio_cold = cold_stats.median_ns / cubin_stats.median_ns;
        if (ratio_cold > 2.0) {
            printf("CONCLUSION: Cold PTX loading is %.0fx slower than "
                   "CUBIN.\n", ratio_cold);
            printf("Pre-compiled dispatch avoids this JIT overhead "
                   "entirely.\n");
        } else {
            printf("NOTE: Cold PTX JIT overhead lower than expected "
                   "(%.1fx).\n", ratio_cold);
            printf("May indicate aggressive driver-level caching.\n");
        }
    }

    /* cleanup */
    api.cuCtxDestroy(ctx);
    dlclose(api.lib);
    free(cubin_data);
    free(ptx_data);
    return 0;
}
