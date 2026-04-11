/*
 * bench_layers.c -- LLVM GPU dispatch stack layer-by-layer benchmark
 *
 * Measures each layer of the CUDA driver path independently so that
 * per-layer overhead can be subtracted to isolate liboffload's own cost.
 * This is "Approach B" from proposal-v2.md: separate baseline measurements.
 *
 * Layers measured:
 *   Layer 1: cuDeviceGet (warm, in-process proxy for init overhead)
 *   Layer 2: cuModuleLoadData (cold via exec-child, and warm in same ctx)
 *   Layer 3: cuModuleGetFunction (symbol lookup)
 *   Layer 4: cuLaunchKernel (submit to GPU, host-side time only)
 *   Layer 5: cuStreamSynchronize (wait for GPU completion)
 *
 * Cold path design: CUDA is not fork-safe (cuInit in a forked child fails
 * with CUDA_ERROR_NOT_INITIALIZED when the parent has already called it).
 * We use execve(/proc/self/exe) per trial so each child starts with a
 * pristine address space. The child detects it was launched as a cold-path
 * helper via the BENCH_LAYERS_COLD_CHILD environment variable.
 *
 * Requires: libcuda.so.1 and /tmp/null_sm75.cubin (SM 7.5 cubin).
 *           Falls back to an embedded PTX stub if the cubin is absent.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 \
 *      -o bench_layers bench_layers.c -ldl
 */

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define WARMUP_ITERS   100
#define MEASURE_ITERS  10000
#define COLD_TRIALS    100

#define CUBIN_PATH     "/tmp/null_sm75.cubin"

/* env var used to launch cold-path child processes */
#define COLD_CHILD_ENV "BENCH_LAYERS_COLD_CHILD"

/* ------------------------------------------------------------------ */
/*  CUDA driver API types                                              */
/* ------------------------------------------------------------------ */

typedef int      CUresult;
typedef int      CUdevice;
typedef void    *CUcontext;
typedef void    *CUmodule;
typedef void    *CUfunction;
typedef void    *CUstream;

#define CUDA_SUCCESS 0

/* ------------------------------------------------------------------ */
/*  CUDA driver function pointer table                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    void *lib;
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuLaunchKernel)(CUfunction,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream,
                               void **, void **);
    CUresult (*cuStreamSynchronize)(CUstream);
} cuda_api;

/* ------------------------------------------------------------------ */
/*  Embedded minimal PTX stub (fallback when no cubin is available)   */
/* ------------------------------------------------------------------ */

static const char g_null_ptx[] =
    ".version 7.5\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .entry null_kernel() { ret; }\n";

/* ------------------------------------------------------------------ */
/*  Globals: cubin blob                                                */
/* ------------------------------------------------------------------ */

static void  *g_cubin_data = NULL;
static size_t g_cubin_size = 0;
static int    g_cubin_is_ptx = 0;  /* 1 = PTX text, 0 = raw cubin */

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
    double p99_ns;
    double min_ns;
    double max_ns;
} bench_stats;

/* insertion sort: fine for N = 10 000 */
static void sort_u64(uint64_t *arr, int n)
{
    for (int i = 1; i < n; i++) {
        uint64_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static bench_stats compute_stats(uint64_t *samples, int n)
{
    bench_stats s;
    memset(&s, 0, sizeof(s));
    if (n <= 0) return s;

    uint64_t sum = 0;
    s.min_ns = (double)samples[0];
    s.max_ns = (double)samples[0];
    for (int i = 0; i < n; i++) {
        sum += samples[i];
        if ((double)samples[i] < s.min_ns) s.min_ns = (double)samples[i];
        if ((double)samples[i] > s.max_ns) s.max_ns = (double)samples[i];
    }
    s.mean_ns = (double)sum / n;

    uint64_t *sorted = malloc((size_t)n * sizeof(uint64_t));
    if (!sorted) return s;
    memcpy(sorted, samples, (size_t)n * sizeof(uint64_t));
    sort_u64(sorted, n);
    s.median_ns = (double)sorted[n / 2];
    s.p99_ns    = (double)sorted[(int)(n * 0.99)];
    free(sorted);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Print helpers                                                      */
/* ------------------------------------------------------------------ */

static void print_layer_header(void)
{
    printf("%-42s  %9s  %9s  %9s  %9s  %9s\n",
           "layer", "mean_ns", "median_ns", "p99_ns", "min_ns", "max_ns");
    printf("%-42s  %9s  %9s  %9s  %9s  %9s\n",
           "------------------------------------------",
           "---------", "---------", "---------", "---------", "---------");
}

static void print_layer(const char *name, bench_stats s)
{
    printf("%-42s  %9.1f  %9.1f  %9.1f  %9.1f  %9.1f\n",
           name, s.mean_ns, s.median_ns, s.p99_ns, s.min_ns, s.max_ns);
}

static void print_flamegraph(const char *stack, bench_stats s)
{
    printf("%s %.0f\n", stack, s.mean_ns);
}

/* ------------------------------------------------------------------ */
/*  Load libcuda.so.1                                                  */
/* ------------------------------------------------------------------ */

static int load_cuda_api(cuda_api *api)
{
    memset(api, 0, sizeof(*api));
    api->lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!api->lib) {
        fprintf(stderr, "[bench_layers] dlopen(libcuda.so.1) failed: %s\n",
                dlerror());
        return -1;
    }

#define LOAD(sym) \
    *(void **)(&api->sym) = dlsym(api->lib, #sym); \
    if (!api->sym) { \
        fprintf(stderr, "[bench_layers] dlsym(%s) failed\n", #sym); \
        return -1; \
    }

    LOAD(cuInit)
    LOAD(cuDeviceGet)
    LOAD(cuDeviceGetCount)
    LOAD(cuModuleGetFunction)
    LOAD(cuModuleUnload)
    LOAD(cuLaunchKernel)
    LOAD(cuStreamSynchronize)
    LOAD(cuModuleLoadData)
#undef LOAD

    /* cuCtxCreate may be v2 */
    *(void **)(&api->cuCtxCreate) = dlsym(api->lib, "cuCtxCreate_v2");
    if (!api->cuCtxCreate)
        *(void **)(&api->cuCtxCreate) = dlsym(api->lib, "cuCtxCreate");
    if (!api->cuCtxCreate) {
        fprintf(stderr, "[bench_layers] dlsym(cuCtxCreate) failed\n");
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Load cubin blob (file or embedded PTX stub)                       */
/* ------------------------------------------------------------------ */

static int load_cubin_blob(void)
{
    FILE *f = fopen(CUBIN_PATH, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz > 0) {
            g_cubin_data = malloc((size_t)sz);
            if (g_cubin_data &&
                fread(g_cubin_data, 1, (size_t)sz, f) == (size_t)sz) {
                g_cubin_size = (size_t)sz;
                g_cubin_is_ptx = 0;
                fclose(f);
                printf("[bench_layers] loaded cubin from %s (%zu bytes)\n",
                       CUBIN_PATH, g_cubin_size);
                return 0;
            }
            free(g_cubin_data);
            g_cubin_data = NULL;
        }
        fclose(f);
    }

    /* Fallback: embedded PTX stub */
    fprintf(stderr, "[bench_layers] %s not found, using embedded PTX stub\n",
            CUBIN_PATH);
    g_cubin_data = (void *)g_null_ptx;
    g_cubin_size = sizeof(g_null_ptx);
    g_cubin_is_ptx = 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Cold-child entry point                                             */
/*                                                                     */
/*  Called when BENCH_LAYERS_COLD_CHILD=<fd>:<cubin_path> is set.     */
/*  Performs cuInit → cuModuleLoadData in a fresh address space,      */
/*  writes elapsed ns to the pipe fd, then exits.                      */
/* ------------------------------------------------------------------ */

static void cold_child_run(const char *env_val)
{
    /* parse "write_fd:cubin_path" */
    char buf[512];
    strncpy(buf, env_val, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *sep = strchr(buf, ':');
    if (!sep) _exit(1);
    *sep = '\0';
    int write_fd = atoi(buf);
    const char *cubin_path = sep + 1;

    /* load cubin */
    FILE *f = fopen(cubin_path, "rb");
    if (!f) _exit(1);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); _exit(1); }
    void *blob = malloc((size_t)sz);
    if (!blob) { fclose(f); _exit(1); }
    if (fread(blob, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(blob); _exit(1);
    }
    fclose(f);

    /* fresh CUDA API in clean address space */
    void *lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!lib) _exit(1);

    CUresult (*p_cuInit)(unsigned int) = dlsym(lib, "cuInit");
    CUresult (*p_cuDeviceGet)(CUdevice *, int) = dlsym(lib, "cuDeviceGet");
    CUresult (*p_cuDeviceGetCount)(int *) = dlsym(lib, "cuDeviceGetCount");
    CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice) =
        dlsym(lib, "cuCtxCreate_v2");
    if (!p_cuCtxCreate)
        p_cuCtxCreate = dlsym(lib, "cuCtxCreate");
    CUresult (*p_cuModuleLoadData)(CUmodule *, const void *) =
        dlsym(lib, "cuModuleLoadData");

    if (!p_cuInit || !p_cuDeviceGet || !p_cuDeviceGetCount ||
        !p_cuCtxCreate || !p_cuModuleLoadData)
        _exit(1);

    if (p_cuInit(0) != CUDA_SUCCESS) _exit(1);

    int count = 0;
    p_cuDeviceGetCount(&count);
    if (count == 0) _exit(1);

    CUdevice dev;
    if (p_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) _exit(1);

    CUcontext ctx;
    if (p_cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) _exit(1);

    uint64_t t0 = now_ns();
    CUmodule mod;
    CUresult r = p_cuModuleLoadData(&mod, blob);
    uint64_t elapsed = now_ns() - t0;

    if (r != CUDA_SUCCESS) elapsed = 0;

    ssize_t wr = write(write_fd, &elapsed, sizeof(elapsed));
    (void)wr;
    close(write_fd);
    _exit(r == CUDA_SUCCESS ? 0 : 1);
}

/* ------------------------------------------------------------------ */
/*  Layer 1: cuDeviceGet (hot, in-process)                            */
/* ------------------------------------------------------------------ */

static bench_stats bench_layer1_init(cuda_api *api)
{
    uint64_t *samples = malloc(MEASURE_ITERS * sizeof(uint64_t));
    if (!samples) { bench_stats s; memset(&s, 0, sizeof(s)); return s; }

    /*
     * cuInit and cuCtxCreate are one-shot operations. We measure the
     * cheapest re-entrant init-path call (cuDeviceGet) as a proxy for
     * the driver's per-call overhead once the context exists.
     * The true cold init cost is captured by the exec-child loop below.
     */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUdevice d;
        api->cuDeviceGet(&d, 0);
    }
    for (int i = 0; i < MEASURE_ITERS; i++) {
        uint64_t t0 = now_ns();
        CUdevice d;
        api->cuDeviceGet(&d, 0);
        samples[i] = now_ns() - t0;
    }

    bench_stats s = compute_stats(samples, MEASURE_ITERS);
    free(samples);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Layer 2 cold: execve per trial                                     */
/* ------------------------------------------------------------------ */

static bench_stats bench_layer2_module_load_cold(void)
{
    /* resolve our own executable path */
    char self_exe[512];
    ssize_t self_len = readlink("/proc/self/exe", self_exe,
                                sizeof(self_exe) - 1);
    if (self_len < 0) {
        fprintf(stderr,
                "[bench_layers] readlink /proc/self/exe failed: %s\n",
                strerror(errno));
        bench_stats s; memset(&s, 0, sizeof(s)); return s;
    }
    self_exe[self_len] = '\0';

    /*
     * The child needs to read the cubin from a file.
     * If g_cubin_is_ptx, write the embedded PTX to a temp file.
     */
    const char *cubin_path_for_child = CUBIN_PATH;
    char tmp_cubin[64] = "";
    if (g_cubin_is_ptx) {
        snprintf(tmp_cubin, sizeof(tmp_cubin),
                 "/tmp/bench_layers_ptx_%d.ptx", (int)getpid());
        FILE *tf = fopen(tmp_cubin, "wb");
        if (tf) {
            fwrite(g_cubin_data, 1, g_cubin_size, tf);
            fclose(tf);
            cubin_path_for_child = tmp_cubin;
        }
    }

    uint64_t *samples = malloc(COLD_TRIALS * sizeof(uint64_t));
    if (!samples) { bench_stats s; memset(&s, 0, sizeof(s)); return s; }

    int n_valid = 0;

    for (int trial = 0; trial < COLD_TRIALS; trial++) {
        int pipefd[2];
        if (pipe(pipefd) != 0) continue;

        /* Keep write end open across execve (clear FD_CLOEXEC) */
        int flags = fcntl(pipefd[1], F_GETFD);
        if (flags >= 0) fcntl(pipefd[1], F_SETFD, flags & ~FD_CLOEXEC);

        /* Build BENCH_LAYERS_COLD_CHILD=fd:cubin_path */
        char env_val[600];
        snprintf(env_val, sizeof(env_val), "%s=%d:%s",
                 COLD_CHILD_ENV, pipefd[1], cubin_path_for_child);

        char *envp[] = { env_val, NULL };
        char *argv_child[] = { self_exe, NULL };

        pid_t pid = fork();
        if (pid < 0) {
            close(pipefd[0]);
            close(pipefd[1]);
            continue;
        }

        if (pid == 0) {
            /* child: exec fresh copy of self */
            close(pipefd[0]);
            execve(self_exe, argv_child, envp);
            _exit(1);  /* execve failed */
        }

        /* parent */
        close(pipefd[1]);
        uint64_t elapsed = 0;
        ssize_t bytes = read(pipefd[0], &elapsed, sizeof(elapsed));
        close(pipefd[0]);
        waitpid(pid, NULL, 0);

        if (bytes == (ssize_t)sizeof(elapsed) && elapsed > 0) {
            samples[n_valid++] = elapsed;
        }
    }

    bench_stats s;
    memset(&s, 0, sizeof(s));
    if (n_valid > 0)
        s = compute_stats(samples, n_valid);
    free(samples);

    if (tmp_cubin[0]) unlink(tmp_cubin);

    printf("[bench_layers] layer2 cold: %d/%d trials succeeded\n",
           n_valid, COLD_TRIALS);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Layer 2 warm: cuModuleLoadData repeated in existing context       */
/* ------------------------------------------------------------------ */

static bench_stats bench_layer2_module_load_warm(cuda_api *api)
{
    uint64_t *samples = malloc(MEASURE_ITERS * sizeof(uint64_t));
    if (!samples) { bench_stats s; memset(&s, 0, sizeof(s)); return s; }

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUmodule mod;
        CUresult r = api->cuModuleLoadData(&mod, g_cubin_data);
        if (r == CUDA_SUCCESS) api->cuModuleUnload(mod);
    }
    for (int i = 0; i < MEASURE_ITERS; i++) {
        CUmodule mod;
        uint64_t t0 = now_ns();
        CUresult r = api->cuModuleLoadData(&mod, g_cubin_data);
        samples[i] = now_ns() - t0;
        if (r == CUDA_SUCCESS) api->cuModuleUnload(mod);
    }

    bench_stats s = compute_stats(samples, MEASURE_ITERS);
    free(samples);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Layer 3: cuModuleGetFunction                                       */
/* ------------------------------------------------------------------ */

static bench_stats bench_layer3_get_function(cuda_api *api, CUmodule mod)
{
    uint64_t *samples = malloc(MEASURE_ITERS * sizeof(uint64_t));
    if (!samples) { bench_stats s; memset(&s, 0, sizeof(s)); return s; }

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUfunction fn;
        api->cuModuleGetFunction(&fn, mod, "null_kernel");
    }
    for (int i = 0; i < MEASURE_ITERS; i++) {
        CUfunction fn;
        uint64_t t0 = now_ns();
        api->cuModuleGetFunction(&fn, mod, "null_kernel");
        samples[i] = now_ns() - t0;
    }

    bench_stats s = compute_stats(samples, MEASURE_ITERS);
    free(samples);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Layer 4: cuLaunchKernel (host-side submit, sync outside window)   */
/* ------------------------------------------------------------------ */

static bench_stats bench_layer4_launch(cuda_api *api, CUfunction fn)
{
    uint64_t *samples = malloc(MEASURE_ITERS * sizeof(uint64_t));
    if (!samples) { bench_stats s; memset(&s, 0, sizeof(s)); return s; }

    for (int i = 0; i < WARMUP_ITERS; i++) {
        api->cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        api->cuStreamSynchronize(NULL);
    }
    for (int i = 0; i < MEASURE_ITERS; i++) {
        uint64_t t0 = now_ns();
        api->cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        samples[i] = now_ns() - t0;
        api->cuStreamSynchronize(NULL);  /* sync outside measurement window */
    }

    bench_stats s = compute_stats(samples, MEASURE_ITERS);
    free(samples);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Layer 5: cuStreamSynchronize (GPU round-trip)                     */
/* ------------------------------------------------------------------ */

static bench_stats bench_layer5_sync(cuda_api *api, CUfunction fn)
{
    uint64_t *samples = malloc(MEASURE_ITERS * sizeof(uint64_t));
    if (!samples) { bench_stats s; memset(&s, 0, sizeof(s)); return s; }

    for (int i = 0; i < WARMUP_ITERS; i++) {
        api->cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        api->cuStreamSynchronize(NULL);
    }
    for (int i = 0; i < MEASURE_ITERS; i++) {
        api->cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        uint64_t t0 = now_ns();
        api->cuStreamSynchronize(NULL);
        samples[i] = now_ns() - t0;
    }

    bench_stats s = compute_stats(samples, MEASURE_ITERS);
    free(samples);
    return s;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    /*
     * Cold-child detection: if BENCH_LAYERS_COLD_CHILD is set, we are
     * a measurement helper launched by execve. Run the cold-path
     * measurement and exit without entering the benchmark main loop.
     */
    const char *cold_env = getenv(COLD_CHILD_ENV);
    if (cold_env) {
        cold_child_run(cold_env);
        /* cold_child_run calls _exit; should never reach here */
        return 1;
    }

    printf("=== bench_layers: LLVM GPU dispatch stack layer benchmark ===\n");
    printf("warmup=%d  measure=%d  cold_trials=%d\n\n",
           WARMUP_ITERS, MEASURE_ITERS, COLD_TRIALS);

    /* Load cubin blob */
    if (load_cubin_blob() != 0) {
        fprintf(stderr, "Failed to load cubin blob\n");
        return 1;
    }

    /* Load CUDA driver API */
    cuda_api api;
    if (load_cuda_api(&api) != 0) {
        fprintf(stderr, "CUDA driver not available\n");
        return 1;
    }

    CUresult r = api.cuInit(0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed (%d)\n", r);
        return 1;
    }

    int count = 0;
    api.cuDeviceGetCount(&count);
    printf("CUDA devices found: %d\n\n", count);
    if (count == 0) {
        fprintf(stderr, "No CUDA devices -- cannot benchmark GPU layers\n");
        return 1;
    }

    CUdevice dev;
    r = api.cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed (%d)\n", r);
        return 1;
    }

    CUcontext ctx;
    r = api.cuCtxCreate(&ctx, 0, dev);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed (%d)\n", r);
        return 1;
    }

    /* Load module once for use in layers 3-5 */
    CUmodule mod = NULL;
    CUresult mod_r = api.cuModuleLoadData(&mod, g_cubin_data);

    /* ---- Layer 1 ---- */
    printf("--- Layer 1: cuDeviceGet (hot-path, in-process) ---\n");
    bench_stats l1 = bench_layer1_init(&api);

    /* ---- Layer 2 cold (exec-per-trial) ---- */
    printf("--- Layer 2 (cold): cuModuleLoadData via exec-child ---\n");
    bench_stats l2c = bench_layer2_module_load_cold();

    bench_stats l2w = {0};
    bench_stats l3  = {0};
    bench_stats l4  = {0};
    bench_stats l5  = {0};

    if (mod_r == CUDA_SUCCESS) {
        /* ---- Layer 2 warm ---- */
        printf("--- Layer 2 (warm): cuModuleLoadData in existing context ---\n");
        l2w = bench_layer2_module_load_warm(&api);

        /* ---- Layer 3 ---- */
        printf("--- Layer 3: cuModuleGetFunction ---\n");
        l3 = bench_layer3_get_function(&api, mod);

        /* Resolve kernel handle for layers 4-5 */
        CUfunction fn = NULL;
        if (api.cuModuleGetFunction(&fn, mod, "null_kernel") == CUDA_SUCCESS && fn) {
            /* ---- Layer 4 ---- */
            printf("--- Layer 4: cuLaunchKernel ---\n");
            l4 = bench_layer4_launch(&api, fn);

            /* ---- Layer 5 ---- */
            printf("--- Layer 5: cuStreamSynchronize ---\n");
            l5 = bench_layer5_sync(&api, fn);
        } else {
            fprintf(stderr,
                    "[bench_layers] cuModuleGetFunction failed -- skipping layers 4-5\n"
                    "  (hint: check that cubin SM matches the device)\n");
        }

        api.cuModuleUnload(mod);
    } else {
        fprintf(stderr,
                "[bench_layers] cuModuleLoadData failed (%d) -- skipping layers 2w-5\n"
                "  (hint: /tmp/null_sm75.cubin may target a different SM)\n",
                mod_r);
    }

    /* ---------------------------------------------------------------- */
    /*  Results table                                                    */
    /* ---------------------------------------------------------------- */

    printf("\n=== RESULTS (%d warm iterations, sorted percentiles) ===\n\n",
           MEASURE_ITERS);
    print_layer_header();
    print_layer("layer1:cuDeviceGet (warm/in-process)",       l1);
    print_layer("layer2:cuModuleLoadData (cold/exec-child)",  l2c);
    if (mod_r == CUDA_SUCCESS) {
        print_layer("layer2:cuModuleLoadData (warm/same-ctx)", l2w);
        print_layer("layer3:cuModuleGetFunction",               l3);
        if (l4.mean_ns > 0) {
            print_layer("layer4:cuLaunchKernel (submit)",       l4);
            print_layer("layer5:cuStreamSynchronize (GPU RTT)", l5);
        }
    }

    /* ---------------------------------------------------------------- */
    /*  Flame-graph folded output (mean ns)                             */
    /* ---------------------------------------------------------------- */

    printf("\n=== FLAME-GRAPH (folded format, mean_ns) ===\n");
    print_flamegraph("cuInit;cuDeviceGet;cuCtxCreate", l1);
    if (mod_r == CUDA_SUCCESS && l2w.mean_ns > 0) {
        print_flamegraph("cuModuleLoadData", l2w);
        print_flamegraph("cuModuleGetFunction", l3);
        if (l4.mean_ns > 0) {
            print_flamegraph("cuLaunchKernel", l4);
            print_flamegraph("cuStreamSynchronize", l5);
        }
    } else {
        /* fall back to cold measurement for the ModuleLoad line */
        print_flamegraph("cuModuleLoadData", l2c);
    }

    /* ---------------------------------------------------------------- */
    /*  Summary                                                          */
    /* ---------------------------------------------------------------- */

    printf("\n=== SUMMARY ===\n");
    if (l4.mean_ns > 0 && l5.mean_ns > 0) {
        double hot_path_ns = l4.mean_ns + l5.mean_ns;
        printf("Hot-path dispatch (launch+sync):       %8.0f ns  (%6.2f us)\n",
               hot_path_ns, hot_path_ns / 1000.0);
    }
    if (l2c.mean_ns > 0) {
        printf("Cold module load (exec-child/trial):   %8.0f ns  (%6.2f us)\n",
               l2c.mean_ns, l2c.mean_ns / 1000.0);
    }
    if (l2w.mean_ns > 0) {
        printf("Warm module load (same context):       %8.0f ns  (%6.2f us)\n",
               l2w.mean_ns, l2w.mean_ns / 1000.0);
    }
    if (l3.mean_ns > 0) {
        printf("Symbol lookup (cuModuleGetFunction):   %8.0f ns  (%6.2f us)\n",
               l3.mean_ns, l3.mean_ns / 1000.0);
    }
    if (l4.mean_ns > 0 && l2w.mean_ns > 0) {
        printf("ModuleLoad/Launch overhead ratio:      %8.1f x\n",
               l2w.mean_ns / l4.mean_ns);
    }

    printf("\ndone.\n");
    return 0;
}
