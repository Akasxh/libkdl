/*
 * kdl.h -- Kernel Dynamic Linker public API
 *
 * Lightweight vendor-agnostic runtime dispatch for MLIR-compiled GPU kernels.
 * Analogous to ld.so: resolves kernel symbols at runtime based on available
 * GPU hardware rather than library search paths.
 *
 * Part of mlir-hetero-dispatch (LLVM Developers' Meeting, Dublin 2026).
 */

#ifndef KDL_H
#define KDL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Opaque handles ---------- */

typedef struct kdl_context *kdl_ctx;
typedef struct kdl_bundle  *kdl_bundle_t;
typedef struct kdl_kernel  *kdl_kernel_t;

/* ---------- Status codes ---------- */

typedef enum {
    KDL_SUCCESS = 0,
    KDL_ERROR_NO_DEVICES,
    KDL_ERROR_NO_MATCHING_VARIANT,
    KDL_ERROR_LOAD_FAILED,
    KDL_ERROR_LAUNCH_FAILED,
    KDL_ERROR_INVALID_BUNDLE,
    KDL_ERROR_VRAM_INSUFFICIENT,
    KDL_ERROR_CACHE_INVALID,
    KDL_ERROR_CALIBRATION_FAILED,
    KDL_ERROR_POOL_EXHAUSTED,
    KDL_ERROR_BACKEND_NOT_FOUND,
    KDL_ERROR_INVALID_ARGUMENT,
    KDL_ERROR_NOT_SUPPORTED,
} kdl_status;

/* ---------- Vendor IDs ---------- */

enum {
    KDL_VENDOR_NVIDIA = 0,
    KDL_VENDOR_AMD    = 1,
    KDL_VENDOR_INTEL  = 2,
    KDL_VENDOR_CPU    = 3,
};

/* ---------- Target kinds (in MTB variant table) ---------- */

enum {
    KDL_TARGET_NVPTX   = 0,
    KDL_TARGET_AMDGCN  = 1,
    KDL_TARGET_SPIRV   = 2,
    KDL_TARGET_X86_64  = 3,
};

/* ---------- Device descriptor ---------- */

typedef struct {
    uint32_t vendor;           /* KDL_VENDOR_* */
    char     name[256];
    char     arch[64];         /* "sm_80", "gfx90a", "x86-64-v3" */
    uint64_t vram_bytes;
    uint32_t compute_units;    /* SMs / CUs / cores */
    uint32_t max_shared_mem;   /* bytes per block */
    uint32_t warp_size;        /* 32 / 64 / 1 */
    double   peak_tflops_f32;
    double   peak_bw_gbps;
    uint32_t driver_version;
    int      device_index;
} kdl_device_info;

/* ---------- Cache statistics (Iteration 5) ---------- */

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t collisions;
    int      occupied_slots;
    int      total_slots;
} kdl_cache_stats_t;

/* ---------- Verbose selection report (Iteration 10) ---------- */

#define KDL_MAX_CANDIDATES 64

typedef struct {
    int         device_index;
    uint32_t    variant_index;
    const char *variant_chip;
    double      cost;
    int         contract_pass;   /* 1=pass, 0=fail */
    const char *reject_reason;   /* NULL if passed */
} kdl_candidate_info;

typedef struct {
    /* Selected result */
    int             selected_device;
    uint32_t        selected_variant;
    double          selected_cost;

    /* All considered (device, variant) pairs */
    kdl_candidate_info candidates[KDL_MAX_CANDIDATES];
    int                num_candidates;
} kdl_selection_report;

/* ---------- Lifecycle ---------- */

/**
 * @brief Initialise a KDL context and discover available compute devices.
 *
 * Attempts to load libcuda and libamdhip64 via dlopen.  At least one
 * device (including the CPU fallback) will always be present.  The
 * returned context must be released with kdl_finalize().
 *
 * @param[out] out_ctx  Receives the newly created context pointer.
 * @return KDL_SUCCESS on success, KDL_ERROR_INVALID_ARGUMENT if out_ctx
 *         is NULL, or KDL_ERROR_NO_DEVICES if no usable device was found.
 */
kdl_status kdl_init(kdl_ctx *out_ctx);

/**
 * @brief Release all resources associated with a context.
 *
 * Frees cached kernels, closes dlopen handles, destroys internal streams,
 * and frees the context object.  Safe to call with NULL (no-op).
 *
 * @param ctx  Context to destroy.
 */
void       kdl_finalize(kdl_ctx ctx);

/* ---------- Device discovery ---------- */

/**
 * @brief Return the number of compute devices visible to the context.
 *
 * Always >= 1 after a successful kdl_init() because the CPU fallback
 * device is always enumerated.
 *
 * @param ctx  Initialised KDL context.
 * @return Number of devices, or 0 if ctx is NULL.
 */
int        kdl_get_device_count(kdl_ctx ctx);

/**
 * @brief Populate a device descriptor for the device at @p index.
 *
 * @param ctx    Initialised KDL context.
 * @param index  Zero-based device index (< kdl_get_device_count()).
 * @param[out] out  Receives the device descriptor.
 * @return KDL_SUCCESS, KDL_ERROR_INVALID_ARGUMENT, or KDL_ERROR_NO_DEVICES.
 */
kdl_status kdl_get_device_info(kdl_ctx ctx, int index, kdl_device_info *out);

/* ---------- Bundle loading ---------- */

/**
 * @brief Load a Multi-Target Bundle (MTB) file from disk.
 *
 * Reads and validates the binary MTB format.  The bundle object is
 * independent of the context and may be shared across calls to
 * kdl_select_kernel().  Release with kdl_free_bundle().
 *
 * @param ctx   Initialised KDL context.
 * @param path  Filesystem path to the .mtb file.
 * @param[out] out  Receives the bundle handle.
 * @return KDL_SUCCESS, KDL_ERROR_INVALID_BUNDLE, or KDL_ERROR_LOAD_FAILED.
 */
kdl_status kdl_load_bundle(kdl_ctx ctx, const char *path, kdl_bundle_t *out);

/**
 * @brief Free a bundle previously loaded by kdl_load_bundle().
 *
 * All kdl_kernel_t handles resolved from this bundle become invalid
 * after this call.  Safe to call with NULL (no-op).
 *
 * @param bundle  Bundle handle to release.
 */
void       kdl_free_bundle(kdl_bundle_t bundle);

/* ---------- Kernel selection (device_index=-1 for auto-select) ---------- */

/**
 * @brief Select the best kernel variant for the given device.
 *
 * Iterates over all variants in the bundle, checks capability contracts,
 * scores each candidate via the roofline cost model, and returns a
 * resolved kernel handle ready for kdl_launch().
 *
 * Pass @p device_index = -1 to let KDL choose the optimal device
 * automatically according to the active dispatch policy.
 *
 * @param ctx           Initialised KDL context.
 * @param bundle        Loaded MTB bundle.
 * @param kernel_name   Null-terminated name of the desired kernel.
 * @param device_index  Target device index, or -1 for auto-selection.
 * @param[out] out      Receives the resolved kernel handle.
 * @return KDL_SUCCESS or KDL_ERROR_NO_MATCHING_VARIANT.
 */
kdl_status kdl_select_kernel(kdl_ctx ctx, kdl_bundle_t bundle,
                             const char *kernel_name,
                             int device_index, kdl_kernel_t *out);

/* ---------- Verbose kernel selection (Iteration 10) ---------- */

/**
 * @brief Select a kernel variant and return a full selection report.
 *
 * Identical to kdl_select_kernel() but additionally populates @p report
 * with the list of all (device, variant) pairs considered, their costs,
 * and the reason each non-selected candidate was rejected.
 *
 * @param ctx           Initialised KDL context.
 * @param bundle        Loaded MTB bundle.
 * @param kernel_name   Null-terminated kernel name.
 * @param device_index  Target device index, or -1 for auto.
 * @param[out] out      Receives the resolved kernel handle.
 * @param[out] report   Receives the detailed selection report (may be NULL).
 * @return KDL_SUCCESS or KDL_ERROR_NO_MATCHING_VARIANT.
 */
kdl_status kdl_select_kernel_verbose(kdl_ctx ctx, kdl_bundle_t bundle,
                                     const char *kernel_name,
                                     int device_index, kdl_kernel_t *out,
                                     kdl_selection_report *report);

/* ---------- Kernel launch ---------- */

/**
 * @brief Launch a resolved kernel synchronously.
 *
 * Blocks until the kernel has completed execution.  Grid and block
 * dimensions follow the same convention as CUDA/HIP: the total thread
 * count is grid_x * grid_y * grid_z * block_x * block_y * block_z.
 *
 * @param kernel            Resolved kernel handle.
 * @param grid_x/y/z        Grid dimensions in blocks.
 * @param block_x/y/z       Block dimensions in threads.
 * @param shared_mem_bytes  Dynamic shared memory per block (bytes).
 * @param args              NULL-terminated array of kernel argument pointers.
 * @return KDL_SUCCESS or KDL_ERROR_LAUNCH_FAILED.
 */
kdl_status kdl_launch(kdl_kernel_t kernel,
                      uint32_t grid_x,  uint32_t grid_y,  uint32_t grid_z,
                      uint32_t block_x, uint32_t block_y, uint32_t block_z,
                      uint32_t shared_mem_bytes, void **args);

/* ---------- Async launch (Iteration 6) ---------- */

/**
 * @brief Enqueue a kernel launch without waiting for completion.
 *
 * The kernel is submitted to the device's internal stream.  Use
 * kdl_sync() to wait for completion.
 *
 * @param kernel            Resolved kernel handle.
 * @param grid_x/y/z        Grid dimensions in blocks.
 * @param block_x/y/z       Block dimensions in threads.
 * @param shared_mem_bytes  Dynamic shared memory per block (bytes).
 * @param args              NULL-terminated array of kernel argument pointers.
 * @return KDL_SUCCESS or KDL_ERROR_LAUNCH_FAILED.
 */
kdl_status kdl_launch_async(kdl_kernel_t kernel,
                            uint32_t grid_x,  uint32_t grid_y,  uint32_t grid_z,
                            uint32_t block_x, uint32_t block_y, uint32_t block_z,
                            uint32_t shared_mem_bytes, void **args);

/**
 * @brief Wait for all previously enqueued async launches to complete.
 *
 * @param kernel  Kernel whose stream to synchronise.
 * @return KDL_SUCCESS or KDL_ERROR_LAUNCH_FAILED.
 */
kdl_status kdl_sync(kdl_kernel_t kernel);

/* ---------- Cache statistics (Iteration 5) ---------- */

/**
 * @brief Retrieve dispatch-cache statistics for the context.
 *
 * The cache avoids repeated selection walks for (kernel_name, device)
 * pairs that have already been resolved.
 *
 * @param ctx    Initialised KDL context.
 * @param[out] out  Populated with hit/miss/eviction/collision counts.
 * @return KDL_SUCCESS or KDL_ERROR_INVALID_ARGUMENT.
 */
kdl_status kdl_cache_stats(kdl_ctx ctx, kdl_cache_stats_t *out);

/* ---------- Memory management ---------- */

/**
 * @brief Allocate device memory associated with a kernel's device.
 *
 * @param kernel    Resolved kernel handle (determines target device).
 * @param bytes     Number of bytes to allocate (must be > 0).
 * @param[out] out_ptr  Receives the device pointer.
 * @return KDL_SUCCESS, KDL_ERROR_INVALID_ARGUMENT, or KDL_ERROR_VRAM_INSUFFICIENT.
 */
kdl_status kdl_malloc(kdl_kernel_t kernel, size_t bytes, void **out_ptr);

/**
 * @brief Free a device allocation made by kdl_malloc().
 *
 * @param kernel  Kernel whose device owns the allocation.
 * @param ptr     Device pointer to free.
 * @return KDL_SUCCESS or KDL_ERROR_INVALID_ARGUMENT.
 */
kdl_status kdl_free_mem(kdl_kernel_t kernel, void *ptr);

/**
 * @brief Copy data from host memory to device memory.
 *
 * @param kernel  Kernel whose device owns the destination buffer.
 * @param dst     Destination device pointer.
 * @param src     Source host pointer.
 * @param bytes   Number of bytes to copy.
 * @return KDL_SUCCESS or an error code.
 */
kdl_status kdl_memcpy_h2d(kdl_kernel_t kernel, void *dst,
                          const void *src, size_t bytes);

/**
 * @brief Copy data from device memory to host memory.
 *
 * @param kernel  Kernel whose device owns the source buffer.
 * @param dst     Destination host pointer.
 * @param src     Source device pointer.
 * @param bytes   Number of bytes to copy.
 * @return KDL_SUCCESS or an error code.
 */
kdl_status kdl_memcpy_d2h(kdl_kernel_t kernel, void *dst,
                          const void *src, size_t bytes);

/* ---------- Iteration 11: Multi-kernel graph dispatch ---------- */

typedef struct kdl_graph *kdl_graph_t;

kdl_status kdl_create_graph(kdl_ctx ctx, kdl_graph_t *out);
kdl_status kdl_graph_add_kernel(kdl_graph_t graph, kdl_kernel_t kernel,
                                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                uint32_t shared_mem_bytes, void **args);
kdl_status kdl_graph_dispatch(kdl_graph_t graph);
void       kdl_graph_destroy(kdl_graph_t graph);

/* ---------- Iteration 12: Weighted multi-criteria cost model ---------- */

typedef struct {
    double compute;    /* weight for compute score (default 0.4) */
    double memory;     /* weight for memory score (default 0.4) */
    double overhead;   /* weight for launch overhead (default 0.1) */
    double locality;   /* weight for data locality (default 0.1) */
} kdl_cost_weights;

kdl_status kdl_set_cost_weights(kdl_ctx ctx, const kdl_cost_weights *weights);
kdl_status kdl_get_cost_weights(kdl_ctx ctx, kdl_cost_weights *out);

/* ---------- Iteration 13: Persistent kernel cache on disk ---------- */

kdl_status kdl_save_cache(kdl_ctx ctx, const char *path);
kdl_status kdl_load_cache(kdl_ctx ctx, const char *path);

/* ---------- Iteration 15: Auto-benchmark calibration ---------- */

kdl_status kdl_calibrate(kdl_ctx ctx);

/* ---------- Iteration 16: Multi-device split dispatch ---------- */

#define KDL_MAX_SPLIT 16

typedef struct {
    kdl_kernel_t kernel;
    int          device_index;
    uint64_t     work_offset;
    uint64_t     work_size;
} kdl_split_entry;

typedef struct {
    kdl_split_entry entries[KDL_MAX_SPLIT];
    int             num_entries;
    uint64_t        total_work;
} kdl_split_plan;

kdl_status kdl_select_kernel_split(kdl_ctx ctx, kdl_bundle_t bundle,
                                   const char *kernel_name,
                                   uint64_t total_work,
                                   kdl_split_plan *out);

/* ---------- Iteration 17: Memory pool allocator ---------- */

typedef struct kdl_pool *kdl_pool_t;

kdl_status kdl_pool_create(kdl_kernel_t kernel, size_t pool_size,
                           kdl_pool_t *out);
kdl_status kdl_pool_alloc(kdl_pool_t pool, size_t bytes, void **out_ptr);
kdl_status kdl_pool_free(kdl_pool_t pool, void *ptr);
void       kdl_pool_destroy(kdl_pool_t pool);

/* ---------- Iteration 18: Kernel fusion hints ---------- */

kdl_status kdl_set_fusion_group(kdl_kernel_t kernel, uint32_t group_id);
kdl_status kdl_launch_fused(kdl_kernel_t kernel,
                            uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                            uint32_t block_x, uint32_t block_y, uint32_t block_z,
                            uint32_t shared_mem_bytes, void **args);

/* ---------- Iteration 19: Telemetry and profiling ---------- */

typedef struct {
    char     kernel_name[128];
    int      device_index;
    uint64_t launch_count;
    double   total_time_ms;
    double   avg_time_ms;
    double   min_time_ms;
    double   max_time_ms;
    uint64_t cache_hits;
} kdl_profile_entry;

#define KDL_MAX_PROFILE_ENTRIES 256

typedef struct {
    kdl_profile_entry entries[KDL_MAX_PROFILE_ENTRIES];
    int               num_entries;
    uint64_t          total_launches;
    double            total_time_ms;
    double            cache_hit_rate;
} kdl_profile_report;

kdl_status kdl_enable_profiling(kdl_ctx ctx, int enable);
kdl_status kdl_get_profile(kdl_ctx ctx, kdl_profile_report *out);
kdl_status kdl_reset_profile(kdl_ctx ctx);

/* ---------- Iteration 21: Error string API ---------- */

const char *kdl_status_string(kdl_status status);
const char *kdl_get_last_error(kdl_ctx ctx);

/* ---------- Iteration 41: Status-to-string (verbose) ---------- */

const char *kdl_status_to_string(kdl_status status);

/* ---------- Iteration 42: KDL_ASSERT macro ---------- */

/*
 * KDL_ASSERT(cond, ctx, retval): logs the failing condition at ERROR level
 * and returns retval from the enclosing function. Unlike assert(3), it never
 * aborts the process, making it safe for use inside library code.
 */
#define KDL_ASSERT(cond, ctx, retval) \
    do { \
        if (!(cond)) { \
            kdl_assert_fail((ctx), __FILE__, __LINE__, #cond); \
            return (retval); \
        } \
    } while (0)

/* Internal helper called by KDL_ASSERT; do not call directly. */
void kdl_assert_fail(kdl_ctx ctx, const char *file, int line,
                     const char *cond_str);

/* ---------- Iteration 43: Bundle introspection ---------- */

uint32_t    kdl_bundle_get_kernel_count(kdl_bundle_t bundle);
const char *kdl_bundle_get_kernel_name(kdl_bundle_t bundle, uint32_t index);

/* ---------- Iteration 44: Device info formatting ---------- */

/*
 * Writes a human-readable description of the device into buf (at most bufsz
 * bytes including the NUL terminator).  Returns buf on success, NULL on error.
 */
const char *kdl_device_info_to_string(const kdl_device_info *info,
                                      char *buf, size_t bufsz);

/* ---------- Iteration 45: Default device override ---------- */

kdl_status kdl_set_default_device(kdl_ctx ctx, int device_index);

/* ---------- Iteration 46: Query selected device after select_kernel ---------- */

/*
 * Returns the device_index that was chosen by the most recent call to
 * kdl_select_kernel() on the given context.  Returns -1 if no kernel has
 * been selected yet.
 */
int kdl_get_selected_device_index(kdl_ctx ctx);

/* ---------- Iteration 47: KDL_NO_CPU_FALLBACK flag ---------- */

/*
 * Pass as the flags argument of kdl_select_kernel_ex() to prevent the
 * dispatcher from falling back to a CPU variant when no GPU variant matches.
 */
#define KDL_SELECT_NO_CPU_FALLBACK  (1u << 0)

kdl_status kdl_select_kernel_ex(kdl_ctx ctx, kdl_bundle_t bundle,
                                 const char *kernel_name,
                                 int device_index, uint32_t flags,
                                 kdl_kernel_t *out);

/* ---------- Iteration 48: Benchmark helper ---------- */

typedef struct {
    double min_ms;
    double mean_ms;
    double max_ms;
    int    iterations;
} kdl_benchmark_result;

kdl_status kdl_benchmark_kernel(kdl_kernel_t kernel,
                                uint32_t grid_x,  uint32_t grid_y,  uint32_t grid_z,
                                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                uint32_t shared_mem_bytes, void **args,
                                int num_iterations,
                                kdl_benchmark_result *out);

/* ---------- Iteration 49: Context JSON serialisation ---------- */

/*
 * Serialises the full runtime context (devices, cache stats, cost weights,
 * dispatch policy, resource limits, profiling state) as a JSON string.
 * The returned pointer is heap-allocated; the caller must free() it.
 * Returns NULL on allocation failure.
 */
char *kdl_context_to_json(kdl_ctx ctx);

/* ---------- Iteration 22: Bundle validation ---------- */

kdl_status kdl_validate_bundle(kdl_bundle_t bundle);

/* ---------- Iteration 23: Device preference API ---------- */

#define KDL_MAX_PREFERENCES 16

typedef struct {
    uint32_t vendor;        /* KDL_VENDOR_* to prefer or exclude */
    int      prefer;        /* 1=prefer, 0=exclude */
    double   bias;          /* multiplier: <1.0 favors, >1.0 penalizes */
} kdl_device_preference;

kdl_status kdl_set_device_preference(kdl_ctx ctx,
                                     const kdl_device_preference *prefs,
                                     int num_prefs);

/* ---------- Iteration 24: Kernel argument descriptor ---------- */

typedef struct {
    char     name[64];
    uint32_t size_bytes;
    uint32_t offset;
    uint32_t kind;          /* 0=pointer, 1=scalar, 2=struct */
} kdl_arg_info;

int        kdl_kernel_get_arg_count(kdl_bundle_t bundle, const char *kernel_name);
kdl_status kdl_kernel_get_arg_info(kdl_bundle_t bundle, const char *kernel_name,
                                   int arg_index, kdl_arg_info *out);

/* ---------- Iteration 25: Event-based timing ---------- */

typedef struct kdl_event *kdl_event_t;

kdl_status kdl_event_create(kdl_kernel_t kernel, kdl_event_t *out);
kdl_status kdl_event_record(kdl_event_t event);
kdl_status kdl_event_elapsed(kdl_event_t start, kdl_event_t end,
                             float *out_ms);
void       kdl_event_destroy(kdl_event_t event);

/* ---------- Iteration 26: Occupancy query ---------- */

kdl_status kdl_get_max_active_blocks(kdl_kernel_t kernel,
                                     uint32_t block_size,
                                     uint32_t shared_mem_bytes,
                                     int *out_blocks);

/* ---------- Iteration 27: Multi-stream concurrent dispatch ---------- */

typedef struct kdl_stream *kdl_stream_t;

kdl_status kdl_create_stream(kdl_ctx ctx, int device_index, kdl_stream_t *out);
kdl_status kdl_launch_on_stream(kdl_kernel_t kernel, kdl_stream_t stream,
                                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                uint32_t shared_mem_bytes, void **args);
kdl_status kdl_stream_sync(kdl_stream_t stream);
void       kdl_stream_destroy(kdl_stream_t stream);

/* ---------- Iteration 28: Shared memory configuration ---------- */

enum {
    KDL_SHMEM_PREFER_EQUAL  = 0,
    KDL_SHMEM_PREFER_SHARED = 1,
    KDL_SHMEM_PREFER_L1     = 2,
};

kdl_status kdl_set_shared_mem_config(kdl_kernel_t kernel, int config);

/* ---------- Iteration 29: Module unload and hot-reload ---------- */

kdl_status kdl_reload_bundle(kdl_ctx ctx, kdl_bundle_t *bundle,
                             const char *path);

/* ---------- Iteration 30: Version API and ABI stability ---------- */

#define KDL_VERSION_MAJOR 0
#define KDL_VERSION_MINOR 3
#define KDL_VERSION_PATCH 0

const char *kdl_version_string(void);
uint32_t    kdl_abi_version(void);

/* ---------- Iteration 51: API version constant and query ---------- */

/**
 * @def KDL_API_VERSION
 * @brief Monotonically increasing integer that identifies the API revision.
 *
 * Incremented once per iteration cycle that changes the public header.
 * Callers may compare against this constant to gate features at compile
 * time; use kdl_get_api_version() for a runtime check.
 */
#define KDL_API_VERSION 60

/**
 * @brief Returns the API version compiled into the library.
 *
 * Always equals KDL_API_VERSION.  Useful when the caller links against
 * a shared library that may differ from the headers it was compiled with.
 *
 * @return KDL_API_VERSION (currently 60).
 */
uint32_t kdl_get_api_version(void);

/* ---------- Iteration 31: Dispatch policy API ---------- */

typedef enum {
    KDL_POLICY_FASTEST      = 0,
    KDL_POLICY_LOWEST_POWER = 1,
    KDL_POLICY_PREFER_GPU   = 2,
    KDL_POLICY_PREFER_CPU   = 3,
    KDL_POLICY_ROUND_ROBIN  = 4,
} kdl_dispatch_policy;

kdl_status kdl_set_dispatch_policy(kdl_ctx ctx, kdl_dispatch_policy policy);

/* ---------- Iteration 32: Kernel variant versioning ---------- */

kdl_status kdl_select_kernel_versioned(kdl_ctx ctx, kdl_bundle_t bundle,
                                       const char *kernel_name,
                                       int device_index,
                                       uint32_t max_version,
                                       kdl_kernel_t *out);

/* ---------- Iteration 33: Async bundle loading ---------- */

typedef void (*kdl_bundle_callback)(kdl_status status, kdl_bundle_t bundle,
                                    void *user_data);

kdl_status kdl_load_bundle_async(kdl_ctx ctx, const char *path,
                                 kdl_bundle_callback callback,
                                 void *user_data);

/* ---------- Iteration 34: Device groups ---------- */

#define KDL_MAX_GROUP_DEVICES 16

typedef struct kdl_device_group *kdl_device_group_t;

kdl_status kdl_create_device_group(kdl_ctx ctx, const int *device_indices,
                                   int num_devices,
                                   kdl_device_group_t *out);
int        kdl_device_group_count(kdl_device_group_t group);
kdl_status kdl_device_group_launch(kdl_device_group_t group,
                                   kdl_bundle_t bundle,
                                   const char *kernel_name,
                                   uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                   uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                   uint32_t shared_mem_bytes, void **args);
void       kdl_device_group_destroy(kdl_device_group_t group);

/* ---------- Iteration 35: Memory transfer optimization ---------- */

kdl_status kdl_memcpy_peer(kdl_ctx ctx, int dst_device, void *dst,
                           int src_device, const void *src, size_t bytes);

/* ---------- Iteration 36: Kernel launch with dependency ---------- */

#define KDL_MAX_DEPS 32

kdl_status kdl_launch_after(kdl_kernel_t kernel,
                            kdl_event_t *deps, int num_deps,
                            uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                            uint32_t block_x, uint32_t block_y, uint32_t block_z,
                            uint32_t shared_mem_bytes, void **args);

/* ---------- Iteration 37: Resource limits ---------- */

typedef enum {
    KDL_LIMIT_VRAM_BYTES          = 0,
    KDL_LIMIT_MAX_CONCURRENT      = 1,
    KDL_LIMIT_MAX_STREAMS         = 2,
} kdl_resource_limit_kind;

kdl_status kdl_set_resource_limit(kdl_ctx ctx, int device_index,
                                  kdl_resource_limit_kind kind,
                                  uint64_t value);
kdl_status kdl_get_resource_limit(kdl_ctx ctx, int device_index,
                                  kdl_resource_limit_kind kind,
                                  uint64_t *out_value);

/* ---------- Iteration 38: Telemetry export ---------- */

kdl_status kdl_export_telemetry_json(kdl_ctx ctx, const char *path);

/* ---------- Iteration 39: Contract negotiation ---------- */

#define KDL_MAX_SUGGESTIONS 8

typedef struct {
    int         device_index;
    uint32_t    variant_index;
    const char *variant_chip;
    const char *mismatch_field;     /* e.g. "min_arch" */
    uint32_t    required_value;
    uint32_t    available_value;
    double      estimated_perf_ratio; /* 0.0-1.0, how much perf is lost */
} kdl_fallback_suggestion;

typedef struct {
    kdl_fallback_suggestion suggestions[KDL_MAX_SUGGESTIONS];
    int                     num_suggestions;
} kdl_negotiation_result;

kdl_status kdl_negotiate_contract(kdl_ctx ctx, kdl_bundle_t bundle,
                                  const char *kernel_name,
                                  kdl_negotiation_result *out);

/* ---------- Iteration 40: Dispatch trace replay ---------- */

typedef struct kdl_trace *kdl_trace_t;

kdl_status kdl_record_trace(kdl_ctx ctx, kdl_trace_t *out);
kdl_status kdl_trace_add(kdl_trace_t trace, kdl_kernel_t kernel,
                         uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t block_x, uint32_t block_y, uint32_t block_z,
                         uint32_t shared_mem_bytes, void **args);
kdl_status kdl_stop_trace(kdl_trace_t trace);
kdl_status kdl_replay_trace(kdl_trace_t trace, int num_iterations,
                            double *out_avg_ms);
void       kdl_trace_destroy(kdl_trace_t trace);

/* ---------- Iteration 20: Plugin backend system ---------- */

typedef struct {
    const char *name;
    int  (*discover)(void *backend_ctx, kdl_device_info *devices, int max_devices);
    int  (*load_module)(void *backend_ctx, const void *binary, size_t size,
                        void **out_module);
    int  (*get_function)(void *backend_ctx, void *module, const char *name,
                         void **out_func);
    int  (*launch)(void *backend_ctx, void *function,
                   uint32_t gx, uint32_t gy, uint32_t gz,
                   uint32_t bx, uint32_t by, uint32_t bz,
                   uint32_t shared_mem, void *stream, void **args);
    int  (*sync)(void *backend_ctx, void *stream);
    int  (*mem_alloc)(void *backend_ctx, size_t bytes, void **out);
    int  (*mem_free)(void *backend_ctx, void *ptr);
    int  (*memcpy_h2d)(void *backend_ctx, void *dst, const void *src, size_t bytes);
    int  (*memcpy_d2h)(void *backend_ctx, void *dst, const void *src, size_t bytes);
    void (*destroy)(void *backend_ctx);
} kdl_backend_vtable;

kdl_status kdl_register_backend(kdl_ctx ctx, uint32_t vendor_id,
                                const kdl_backend_vtable *vtable,
                                void *backend_ctx);
int        kdl_get_backend_count(kdl_ctx ctx);

/* ---------- Iteration 53: Device feature query ---------- */

/**
 * @brief Feature flags for kdl_device_supports_feature().
 *
 * Each flag corresponds to a hardware capability that may or may not be
 * present on a given device.  Flags are independent bit values so that
 * future additions are backwards-compatible.
 */
typedef enum {
    KDL_FEATURE_TENSOR_CORES  = 1 << 0,  /**< NVIDIA Tensor Cores / AMD Matrix Cores */
    KDL_FEATURE_FP16          = 1 << 1,  /**< Native half-precision (FP16) arithmetic */
    KDL_FEATURE_INT8          = 1 << 2,  /**< Native INT8 arithmetic */
    KDL_FEATURE_FP64          = 1 << 3,  /**< Native double-precision (FP64) arithmetic */
    KDL_FEATURE_BF16          = 1 << 4,  /**< Brain float (BF16) arithmetic */
    KDL_FEATURE_MANAGED_MEM   = 1 << 5,  /**< Unified/managed memory support */
    KDL_FEATURE_PEER_TRANSFER = 1 << 6,  /**< Direct GPU-to-GPU peer transfers */
} kdl_feature_flag;

/**
 * @brief Query whether a device supports a specific hardware feature.
 *
 * @param ctx          Initialised KDL context.
 * @param device_index Index of the device to query (0-based).
 * @param feature      One of the ::kdl_feature_flag values.
 * @param[out] out_supported  Set to 1 if the feature is available, 0 otherwise.
 * @return KDL_SUCCESS, KDL_ERROR_INVALID_ARGUMENT, or KDL_ERROR_NO_DEVICES.
 */
kdl_status kdl_device_supports_feature(kdl_ctx ctx, int device_index,
                                       kdl_feature_flag feature,
                                       int *out_supported);

/* ---------- Iteration 54: Dispatch latency measurement ---------- */

/**
 * @brief Measure the raw host-side overhead of a kernel dispatch call.
 *
 * Performs a calibration micro-benchmark: calls the launch machinery
 * without meaningful grid work and measures wall-clock time with
 * clock_gettime(CLOCK_MONOTONIC).  The result represents scheduling
 * and API call overhead, not kernel execution time.
 *
 * @param ctx           Initialised KDL context.
 * @param device_index  Device to measure (-1 for the default device).
 * @param[out] out_ns   Estimated dispatch overhead in nanoseconds.
 * @return KDL_SUCCESS or an error code.
 */
kdl_status kdl_get_dispatch_latency_ns(kdl_ctx ctx, int device_index,
                                       uint64_t *out_ns);

/* ---------- Iteration 55: Bundle integrity validation ---------- */

/**
 * @brief Perform a thorough integrity check on a loaded MTB bundle.
 *
 * Verifies the magic bytes, header version, kernel entry offsets, variant
 * binary data ranges, and string-table null-termination.  Stricter than
 * the lightweight validation done during kdl_load_bundle().
 *
 * @param bundle  Bundle handle returned by kdl_load_bundle().
 * @return KDL_SUCCESS if the bundle is internally consistent,
 *         KDL_ERROR_INVALID_BUNDLE if any check fails,
 *         KDL_ERROR_INVALID_ARGUMENT if bundle is NULL.
 */
kdl_status kdl_bundle_validate(kdl_bundle_t bundle);

/* ---------- Iteration 56: User-defined log callback ---------- */

/**
 * @brief Prototype for a user-supplied log sink.
 *
 * @param level      Log level: 1=ERROR, 2=INFO, 3=DEBUG.
 * @param message    NUL-terminated log message (no trailing newline).
 * @param user_data  Opaque pointer supplied to kdl_set_log_callback().
 */
typedef void (*kdl_log_fn)(int level, const char *message, void *user_data);

/**
 * @brief Redirect KDL log output to a caller-supplied function.
 *
 * Once set, KDL will call @p fn instead of writing to stderr.  Pass
 * NULL to restore the default stderr behaviour.  The callback is
 * invoked from whichever thread generates the log entry; the
 * implementation must be thread-safe.
 *
 * @param fn         Log callback, or NULL to reset to default.
 * @param user_data  Opaque pointer forwarded verbatim to @p fn.
 */
void kdl_set_log_callback(kdl_log_fn fn, void *user_data);

/* ---------- Iteration 57: Backend name query ---------- */

/**
 * @brief Return a short ASCII string identifying the backend for a device.
 *
 * Returned strings are static and must not be freed.  Examples:
 *   - "cuda"   for NVIDIA devices via the CUDA Driver API
 *   - "hip"    for AMD devices via the HIP runtime
 *   - "opencl" for devices dispatched through OpenCL / SPIR-V
 *   - "cpu"    for CPU/host fallback
 *   - "plugin" for devices registered via kdl_register_backend()
 *
 * @param ctx          Initialised KDL context.
 * @param device_index Device index (0-based).
 * @return Static string, or NULL if device_index is out of range.
 */
const char *kdl_get_backend_name(kdl_ctx ctx, int device_index);

/* ---------- Iteration 58: Query device for a resolved kernel ---------- */

/**
 * @brief Return the device index targeted by a resolved kernel handle.
 *
 * The device was chosen at kdl_select_kernel() time.  Use this to confirm
 * which physical device will execute the kernel before launching it.
 *
 * @param kernel  Resolved kernel handle.
 * @return Device index (>= 0), or -1 if kernel is NULL.
 */
int kdl_kernel_get_device(kdl_kernel_t kernel);

/* ---------- Iteration 59: Context reset ---------- */

/**
 * @brief Clear all caches and re-discover devices.
 *
 * Flushes the in-memory dispatch cache, resets profiling counters, clears
 * device preferences, and re-runs device discovery.  Useful after a GPU
 * is hot-plugged or a driver is updated without restarting the process.
 * All previously resolved kdl_kernel_t handles become invalid after this
 * call; the caller must re-select kernels.
 *
 * @param ctx  Context to reset.
 * @return KDL_SUCCESS or KDL_ERROR_INVALID_ARGUMENT.
 */
kdl_status kdl_context_reset(kdl_ctx ctx);

/* ---------- Iteration 60: Self-test ---------- */

/**
 * @brief Exercise all core APIs and report pass/fail.
 *
 * Creates an ephemeral context, exercises lifecycle, device query,
 * bundle loading (in-memory synthetic MTB), cache ops, cost weights,
 * profiling, backend registration, and error-string APIs.  Each
 * sub-test is logged at INFO level.  No GPU hardware is required; all
 * GPU-specific paths are exercised via the CPU fallback variant.
 *
 * @param[out] out_tests_run    Number of sub-tests exercised (may be NULL).
 * @param[out] out_tests_failed Number of sub-tests that failed (may be NULL).
 * @return KDL_SUCCESS if every sub-test passed, KDL_ERROR_LOAD_FAILED
 *         otherwise.
 */
kdl_status kdl_self_test(int *out_tests_run, int *out_tests_failed);

#ifdef __cplusplus
}
#endif

#endif /* KDL_H */
