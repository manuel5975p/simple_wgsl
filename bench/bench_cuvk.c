/*
 * bench_cuvk.c - Benchmark cuvk (CUDA-on-Vulkan) versus real CUDA
 *
 * Usage: ./bench_cuvk <path-to-libcuda.so> [label]
 *
 * Examples:
 *   ./bench_cuvk /usr/lib/libcuda.so.1          "Real CUDA"
 *   ./bench_cuvk ../build/cuvk_runtime/libcuda.so.1  "cuvk"
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <math.h>

/* ========================================================================== */
/* Minimal CUDA type definitions (avoid including cuda.h)                     */
/* ========================================================================== */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

/* ========================================================================== */
/* Function pointer types                                                     */
/* ========================================================================== */

typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuDeviceGetCount)(int *);
typedef CUresult (*pfn_cuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*pfn_cuCtxCreate_v4)(CUcontext *, void *, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxDestroy_v2)(CUcontext);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuMemAlloc_v2)(CUdeviceptr *, size_t);
typedef CUresult (*pfn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemAllocHost_v2)(void **, size_t);
typedef CUresult (*pfn_cuMemFreeHost)(void *);
typedef CUresult (*pfn_cuMemsetD32_v2)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleUnload)(CUmodule);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void **, void **);
typedef CUresult (*pfn_cuEventCreate)(CUevent *, unsigned int);
typedef CUresult (*pfn_cuEventDestroy_v2)(CUevent);
typedef CUresult (*pfn_cuEventRecord)(CUevent, CUstream);
typedef CUresult (*pfn_cuEventSynchronize)(CUevent);
typedef CUresult (*pfn_cuEventElapsedTime_v2)(float *, CUevent, CUevent);

/* ========================================================================== */
/* Global state                                                               */
/* ========================================================================== */

static void *g_lib;
static pfn_cuInit                  cu_Init;
static pfn_cuDeviceGet             cu_DeviceGet;
static pfn_cuDeviceGetCount        cu_DeviceGetCount;
static pfn_cuDeviceGetName         cu_DeviceGetName;
static pfn_cuCtxCreate_v4          cu_CtxCreate;
static pfn_cuCtxDestroy_v2         cu_CtxDestroy;
static pfn_cuCtxSynchronize        cu_CtxSynchronize;
static pfn_cuMemAlloc_v2           cu_MemAlloc;
static pfn_cuMemFree_v2            cu_MemFree;
static pfn_cuMemcpyHtoD_v2        cu_MemcpyHtoD;
static pfn_cuMemcpyDtoH_v2        cu_MemcpyDtoH;
static pfn_cuMemcpyDtoD_v2        cu_MemcpyDtoD;
static pfn_cuMemAllocHost_v2      cu_MemAllocHost;
static pfn_cuMemFreeHost           cu_MemFreeHost;
static pfn_cuMemsetD32_v2          cu_MemsetD32;
static pfn_cuModuleLoadData        cu_ModuleLoadData;
static pfn_cuModuleUnload          cu_ModuleUnload;
static pfn_cuModuleGetFunction     cu_ModuleGetFunction;
static pfn_cuLaunchKernel          cu_LaunchKernel;
static pfn_cuEventCreate           cu_EventCreate;
static pfn_cuEventDestroy_v2       cu_EventDestroy;
static pfn_cuEventRecord           cu_EventRecord;
static pfn_cuEventSynchronize      cu_EventSynchronize;
static pfn_cuEventElapsedTime_v2   cu_EventElapsedTime;

static CUcontext g_ctx;
static CUdevice  g_dev;

/* ========================================================================== */
/* PTX kernels                                                                */
/* ========================================================================== */

/* Empty kernel for measuring launch overhead */
static const char *EMPTY_KERNEL_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    ".visible .entry emptyKernel()\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    ret;\n"
    "}\n";

/* Vector add: 1 thread per block, index by ctaid.x (works on both backends) */
static const char *VECADD_1T_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    ".visible .entry vecAdd1T(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<7>;\n"
    "    .reg .f32 %f<3>;\n"
    "    ld.param.u64 %rd0, [A];\n"
    "    ld.param.u64 %rd1, [B];\n"
    "    ld.param.u64 %rd2, [C];\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd3, %r0;\n"
    "    mul.lo.u64 %rd3, %rd3, 4;\n"
    "    add.u64 %rd4, %rd0, %rd3;\n"
    "    ld.global.f32 %f0, [%rd4];\n"
    "    add.u64 %rd5, %rd1, %rd3;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    add.f32 %f2, %f0, %f1;\n"
    "    add.u64 %rd6, %rd2, %rd3;\n"
    "    st.global.f32 [%rd6], %f2;\n"
    "    ret;\n"
    "}\n";

/* Vector add: multi-threaded blocks (blockIdx*blockDim+threadIdx) */
static const char *VECADD_MT_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    ".visible .entry vecAddMT(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C,\n"
    "    .param .u32 N\n"
    ")\n"
    "{\n"
    "    .reg .u32 %r<6>;\n"
    "    .reg .u64 %rd<8>;\n"
    "    .reg .f32 %f<3>;\n"
    "    .reg .pred %p0;\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    mov.u32 %r1, %ntid.x;\n"
    "    mov.u32 %r2, %tid.x;\n"
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n"
    "    ld.param.u32 %r4, [N];\n"
    "    setp.ge.u32 %p0, %r3, %r4;\n"
    "    @%p0 bra DONE;\n"
    "    cvt.u64.u32 %rd0, %r3;\n"
    "    shl.b64 %rd0, %rd0, 2;\n"
    "    ld.param.u64 %rd1, [A];\n"
    "    ld.param.u64 %rd2, [B];\n"
    "    ld.param.u64 %rd3, [C];\n"
    "    add.u64 %rd4, %rd1, %rd0;\n"
    "    ld.global.f32 %f0, [%rd4];\n"
    "    add.u64 %rd5, %rd2, %rd0;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    add.f32 %f2, %f0, %f1;\n"
    "    add.u64 %rd6, %rd3, %rd0;\n"
    "    st.global.f32 [%rd6], %f2;\n"
    "DONE:\n"
    "    ret;\n"
    "}\n";

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    if (n % 2 == 0)
        return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static void print_header(const char *label) {
    printf("\n========================================\n");
    printf("  %s\n", label);
    printf("========================================\n\n");
}

static void print_bw(const char *name, size_t bytes, double ms) {
    double gbps = (double)bytes / (ms / 1000.0) / 1e9;
    printf("  %-40s %10.3f ms  (%6.2f GB/s)\n", name, ms, gbps);
}

static void print_time(const char *name, double ms) {
    printf("  %-40s %10.3f ms\n", name, ms);
}

static void print_rate(const char *name, int count, double ms) {
    double rate = count / (ms / 1000.0) / 1e6;
    printf("  %-40s %10.3f ms  (%6.2f Melem/s)\n", name, ms, rate);
}

/* ========================================================================== */
/* Library loading                                                            */
/* ========================================================================== */

static int load_library(const char *path) {
    g_lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        return -1;
    }

    #define LOAD(name, sym) do { \
        *(void **)&name = dlsym(g_lib, sym); \
        if (!name) { fprintf(stderr, "dlsym(%s): %s\n", sym, dlerror()); return -1; } \
    } while(0)

    LOAD(cu_Init,             "cuInit");
    LOAD(cu_DeviceGet,        "cuDeviceGet");
    LOAD(cu_DeviceGetCount,   "cuDeviceGetCount");
    LOAD(cu_DeviceGetName,    "cuDeviceGetName");
    LOAD(cu_CtxCreate,        "cuCtxCreate_v4");
    LOAD(cu_CtxDestroy,       "cuCtxDestroy_v2");
    LOAD(cu_CtxSynchronize,   "cuCtxSynchronize");
    LOAD(cu_MemAlloc,         "cuMemAlloc_v2");
    LOAD(cu_MemFree,          "cuMemFree_v2");
    LOAD(cu_MemcpyHtoD,      "cuMemcpyHtoD_v2");
    LOAD(cu_MemcpyDtoH,      "cuMemcpyDtoH_v2");
    LOAD(cu_MemcpyDtoD,      "cuMemcpyDtoD_v2");
    LOAD(cu_MemAllocHost,    "cuMemAllocHost_v2");
    LOAD(cu_MemFreeHost,     "cuMemFreeHost");
    LOAD(cu_MemsetD32,        "cuMemsetD32_v2");
    LOAD(cu_ModuleLoadData,   "cuModuleLoadData");
    LOAD(cu_ModuleUnload,     "cuModuleUnload");
    LOAD(cu_ModuleGetFunction,"cuModuleGetFunction");
    LOAD(cu_LaunchKernel,     "cuLaunchKernel");
    LOAD(cu_EventCreate,      "cuEventCreate");
    LOAD(cu_EventDestroy,     "cuEventDestroy_v2");
    LOAD(cu_EventRecord,      "cuEventRecord");
    LOAD(cu_EventSynchronize, "cuEventSynchronize");
    LOAD(cu_EventElapsedTime, "cuEventElapsedTime_v2");

    #undef LOAD
    return 0;
}

/* ========================================================================== */
/* Benchmarks                                                                 */
/* ========================================================================== */

#define WARMUP  3
#define ITERS  10

static void bench_init(void) {
    /* Already initialized, just report device info */
    char name[256];
    cu_DeviceGetName(name, sizeof(name), g_dev);
    int count = 0;
    cu_DeviceGetCount(&count);
    printf("  Device:       %s\n", name);
    printf("  Device count: %d\n", count);
}

static void bench_module_load(void) {
    print_header("Module Load (PTX Compilation)");

    const struct { const char *name; const char *ptx; } modules[] = {
        { "Empty kernel",          EMPTY_KERNEL_PTX },
        { "VecAdd (1-thread/blk)", VECADD_1T_PTX },
        { "VecAdd (multi-thread)", VECADD_MT_PTX },
    };

    for (int m = 0; m < 3; m++) {
        double times[ITERS];

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) {
            CUmodule mod;
            CHECK_CU(cu_ModuleLoadData(&mod, modules[m].ptx));
            cu_ModuleUnload(mod);
        }

        for (int i = 0; i < ITERS; i++) {
            CUmodule mod;
            double t0 = now_ms();
            CHECK_CU(cu_ModuleLoadData(&mod, modules[m].ptx));
            double t1 = now_ms();
            cu_ModuleUnload(mod);
            times[i] = t1 - t0;
        }

        char buf[128];
        snprintf(buf, sizeof(buf), "Module load: %s", modules[m].name);
        print_time(buf, median(times, ITERS));
    }
}

static void bench_memory_alloc(void) {
    print_header("Memory Allocation");

    size_t sizes[] = { 4096, 65536, 1<<20, 16<<20, 64<<20 };
    const char *names[] = { "4 KB", "64 KB", "1 MB", "16 MB", "64 MB" };

    for (int s = 0; s < 5; s++) {
        double times[ITERS];

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) {
            CUdeviceptr p;
            CHECK_CU(cu_MemAlloc(&p, sizes[s]));
            cu_MemFree(p);
        }

        for (int i = 0; i < ITERS; i++) {
            CUdeviceptr p;
            double t0 = now_ms();
            CHECK_CU(cu_MemAlloc(&p, sizes[s]));
            double t1 = now_ms();
            cu_MemFree(p);
            times[i] = t1 - t0;
        }

        char buf[128];
        snprintf(buf, sizeof(buf), "Alloc %s", names[s]);
        print_time(buf, median(times, ITERS));
    }
}

static void bench_transfer(void) {
    print_header("Memory Transfer Bandwidth");

    size_t sizes[] = { 4096, 65536, 1<<20, 16<<20, 64<<20 };
    const char *names[] = { "4 KB", "64 KB", "1 MB", "16 MB", "64 MB" };

    for (int s = 0; s < 5; s++) {
        size_t sz = sizes[s];
        void *host = NULL;
        CHECK_CU(cu_MemAllocHost(&host, sz));
        memset(host, 0xAB, sz);

        CUdeviceptr d_a, d_b;
        CHECK_CU(cu_MemAlloc(&d_a, sz));
        CHECK_CU(cu_MemAlloc(&d_b, sz));

        /* ---------- H2D ---------- */
        double times[ITERS];
        for (int i = 0; i < WARMUP; i++) {
            CHECK_CU(cu_MemcpyHtoD(d_a, host, sz));
            cu_CtxSynchronize();
        }
        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            CHECK_CU(cu_MemcpyHtoD(d_a, host, sz));
            cu_CtxSynchronize();
            double t1 = now_ms();
            times[i] = t1 - t0;
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "H2D %s", names[s]);
        print_bw(buf, sz, median(times, ITERS));

        /* ---------- D2H ---------- */
        for (int i = 0; i < WARMUP; i++) {
            CHECK_CU(cu_MemcpyDtoH(host, d_a, sz));
            cu_CtxSynchronize();
        }
        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            CHECK_CU(cu_MemcpyDtoH(host, d_a, sz));
            cu_CtxSynchronize();
            double t1 = now_ms();
            times[i] = t1 - t0;
        }
        snprintf(buf, sizeof(buf), "D2H %s", names[s]);
        print_bw(buf, sz, median(times, ITERS));

        /* ---------- D2D ---------- */
        for (int i = 0; i < WARMUP; i++) {
            CHECK_CU(cu_MemcpyDtoD(d_b, d_a, sz));
            cu_CtxSynchronize();
        }
        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            CHECK_CU(cu_MemcpyDtoD(d_b, d_a, sz));
            cu_CtxSynchronize();
            double t1 = now_ms();
            times[i] = t1 - t0;
        }
        snprintf(buf, sizeof(buf), "D2D %s", names[s]);
        print_bw(buf, sz, median(times, ITERS));

        printf("\n");

        cu_MemFree(d_a);
        cu_MemFree(d_b);
        cu_MemFreeHost(host);
    }
}

static void bench_memset(void) {
    print_header("Memset Bandwidth");

    size_t sizes[] = { 4096, 65536, 1<<20, 16<<20, 64<<20 };
    const char *names[] = { "4 KB", "64 KB", "1 MB", "16 MB", "64 MB" };

    for (int s = 0; s < 5; s++) {
        size_t sz = sizes[s];
        size_t count = sz / 4;  /* D32 = 4 bytes per element */

        CUdeviceptr d_a;
        CHECK_CU(cu_MemAlloc(&d_a, sz));

        double times[ITERS];
        for (int i = 0; i < WARMUP; i++) {
            CHECK_CU(cu_MemsetD32(d_a, 0xDEADBEEF, count));
            cu_CtxSynchronize();
        }
        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            CHECK_CU(cu_MemsetD32(d_a, 0xDEADBEEF, count));
            cu_CtxSynchronize();
            double t1 = now_ms();
            times[i] = t1 - t0;
        }

        char buf[128];
        snprintf(buf, sizeof(buf), "Memset D32 %s", names[s]);
        print_bw(buf, sz, median(times, ITERS));

        cu_MemFree(d_a);
    }
}

static void bench_launch_latency(void) {
    print_header("Kernel Launch Latency");

    CUmodule mod;
    CHECK_CU(cu_ModuleLoadData(&mod, EMPTY_KERNEL_PTX));
    CUfunction func;
    CHECK_CU(cu_ModuleGetFunction(&func, mod, "emptyKernel"));

    /* Single launch + sync */
    double times[ITERS];
    for (int i = 0; i < WARMUP; i++) {
        CHECK_CU(cu_LaunchKernel(func, 1,1,1, 1,1,1, 0, NULL, NULL, NULL));
        cu_CtxSynchronize();
    }
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_ms();
        CHECK_CU(cu_LaunchKernel(func, 1,1,1, 1,1,1, 0, NULL, NULL, NULL));
        cu_CtxSynchronize();
        double t1 = now_ms();
        times[i] = t1 - t0;
    }
    print_time("Single empty kernel + sync", median(times, ITERS));

    /* Batch: 100 launches then sync */
    for (int i = 0; i < WARMUP; i++) {
        for (int j = 0; j < 100; j++)
            CHECK_CU(cu_LaunchKernel(func, 1,1,1, 1,1,1, 0, NULL, NULL, NULL));
        cu_CtxSynchronize();
    }
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_ms();
        for (int j = 0; j < 100; j++)
            CHECK_CU(cu_LaunchKernel(func, 1,1,1, 1,1,1, 0, NULL, NULL, NULL));
        cu_CtxSynchronize();
        double t1 = now_ms();
        times[i] = t1 - t0;
    }
    double med = median(times, ITERS);
    char buf[128];
    snprintf(buf, sizeof(buf), "100x empty kernel + sync (%.3f us/launch)", med * 1000.0 / 100.0);
    print_time(buf, med);

    cu_ModuleUnload(mod);
}

static void bench_vecadd_1t(void) {
    print_header("Vector Add (1 thread/block) - Compute");

    CUmodule mod;
    CHECK_CU(cu_ModuleLoadData(&mod, VECADD_1T_PTX));
    CUfunction func;
    CHECK_CU(cu_ModuleGetFunction(&func, mod, "vecAdd1T"));

    int counts[] = { 1<<16, 1<<18, 1<<20, 1<<22 };
    const char *names[] = { "64K", "256K", "1M", "4M" };

    for (int c = 0; c < 4; c++) {
        int N = counts[c];
        size_t bytes = N * sizeof(float);

        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_c = (float *)malloc(bytes);

        for (int i = 0; i < N; i++) {
            h_a[i] = (float)i * 0.5f;
            h_b[i] = (float)(N - i) * 0.25f;
        }

        CUdeviceptr d_a, d_b, d_c;
        CHECK_CU(cu_MemAlloc(&d_a, bytes));
        CHECK_CU(cu_MemAlloc(&d_b, bytes));
        CHECK_CU(cu_MemAlloc(&d_c, bytes));
        CHECK_CU(cu_MemcpyHtoD(d_a, h_a, bytes));
        CHECK_CU(cu_MemcpyHtoD(d_b, h_b, bytes));

        void *params[] = { &d_a, &d_b, &d_c };

        /* GPU timing with events */
        CUevent ev_start, ev_end;
        CHECK_CU(cu_EventCreate(&ev_start, 0));
        CHECK_CU(cu_EventCreate(&ev_end, 0));

        double gpu_times[ITERS];
        double host_times[ITERS];

        for (int i = 0; i < WARMUP; i++) {
            CHECK_CU(cu_LaunchKernel(func, N,1,1, 1,1,1, 0, NULL, params, NULL));
            cu_CtxSynchronize();
        }

        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            CHECK_CU(cu_EventRecord(ev_start, NULL));
            CHECK_CU(cu_LaunchKernel(func, N,1,1, 1,1,1, 0, NULL, params, NULL));
            CHECK_CU(cu_EventRecord(ev_end, NULL));
            CHECK_CU(cu_EventSynchronize(ev_end));
            double t1 = now_ms();

            float ms = 0;
            CHECK_CU(cu_EventElapsedTime(&ms, ev_start, ev_end));
            gpu_times[i] = ms;
            host_times[i] = t1 - t0;
        }

        /* Verify correctness (spot check) */
        CHECK_CU(cu_MemcpyDtoH(h_c, d_c, bytes));
        int errors = 0;
        for (int i = 0; i < N && errors < 5; i++) {
            float expected = h_a[i] + h_b[i];
            if (fabsf(h_c[i] - expected) > 1e-3f) errors++;
        }

        char buf[128];
        double gmed = median(gpu_times, ITERS);
        double hmed = median(host_times, ITERS);
        snprintf(buf, sizeof(buf), "VecAdd 1T N=%s (GPU)", names[c]);
        print_rate(buf, N, gmed);
        snprintf(buf, sizeof(buf), "VecAdd 1T N=%s (host wall)", names[c]);
        print_rate(buf, N, hmed);
        if (errors > 0)
            printf("  *** VERIFICATION FAILED: %d errors ***\n", errors);

        cu_EventDestroy(ev_start);
        cu_EventDestroy(ev_end);
        cu_MemFree(d_a);
        cu_MemFree(d_b);
        cu_MemFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
    }

    cu_ModuleUnload(mod);
}

static void bench_vecadd_mt(void) {
    print_header("Vector Add (multi-thread blocks) - Compute");

    CUmodule mod;
    CUresult r = cu_ModuleLoadData(&mod, VECADD_MT_PTX);
    if (r != CUDA_SUCCESS) {
        printf("  SKIPPED (module load failed with error %d)\n", r);
        printf("  (This backend may not support multi-threaded blocks)\n");
        return;
    }
    CUfunction func;
    CHECK_CU(cu_ModuleGetFunction(&func, mod, "vecAddMT"));

    int counts[] = { 1<<16, 1<<18, 1<<20, 1<<22 };
    const char *names[] = { "64K", "256K", "1M", "4M" };
    int blockSize = 256;

    for (int c = 0; c < 4; c++) {
        int N = counts[c];
        int gridSize = (N + blockSize - 1) / blockSize;
        size_t bytes = N * sizeof(float);

        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_c = (float *)malloc(bytes);

        for (int i = 0; i < N; i++) {
            h_a[i] = (float)i * 0.5f;
            h_b[i] = (float)(N - i) * 0.25f;
        }

        CUdeviceptr d_a, d_b, d_c;
        CHECK_CU(cu_MemAlloc(&d_a, bytes));
        CHECK_CU(cu_MemAlloc(&d_b, bytes));
        CHECK_CU(cu_MemAlloc(&d_c, bytes));
        CHECK_CU(cu_MemcpyHtoD(d_a, h_a, bytes));
        CHECK_CU(cu_MemcpyHtoD(d_b, h_b, bytes));

        unsigned int n = (unsigned int)N;
        void *params[] = { &d_a, &d_b, &d_c, &n };

        CUevent ev_start, ev_end;
        CHECK_CU(cu_EventCreate(&ev_start, 0));
        CHECK_CU(cu_EventCreate(&ev_end, 0));

        double gpu_times[ITERS];
        double host_times[ITERS];

        for (int i = 0; i < WARMUP; i++) {
            CHECK_CU(cu_LaunchKernel(func, gridSize,1,1, blockSize,1,1, 0, NULL, params, NULL));
            cu_CtxSynchronize();
        }

        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            CHECK_CU(cu_EventRecord(ev_start, NULL));
            CHECK_CU(cu_LaunchKernel(func, gridSize,1,1, blockSize,1,1, 0, NULL, params, NULL));
            CHECK_CU(cu_EventRecord(ev_end, NULL));
            CHECK_CU(cu_EventSynchronize(ev_end));
            double t1 = now_ms();

            float ms = 0;
            CHECK_CU(cu_EventElapsedTime(&ms, ev_start, ev_end));
            gpu_times[i] = ms;
            host_times[i] = t1 - t0;
        }

        /* Verify */
        CHECK_CU(cu_MemcpyDtoH(h_c, d_c, bytes));
        int errors = 0;
        for (int i = 0; i < N && errors < 5; i++) {
            float expected = h_a[i] + h_b[i];
            if (fabsf(h_c[i] - expected) > 1e-3f) errors++;
        }

        char buf[128];
        double gmed = median(gpu_times, ITERS);
        double hmed = median(host_times, ITERS);
        snprintf(buf, sizeof(buf), "VecAdd MT N=%s (GPU)", names[c]);
        print_rate(buf, N, gmed);
        snprintf(buf, sizeof(buf), "VecAdd MT N=%s (host wall)", names[c]);
        print_rate(buf, N, hmed);
        if (errors > 0)
            printf("  *** VERIFICATION FAILED: %d errors ***\n", errors);

        cu_EventDestroy(ev_start);
        cu_EventDestroy(ev_end);
        cu_MemFree(d_a);
        cu_MemFree(d_b);
        cu_MemFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
    }

    cu_ModuleUnload(mod);
}

static void bench_end_to_end(void) {
    print_header("End-to-End: Alloc + H2D + Compute + D2H");

    CUmodule mod;
    CHECK_CU(cu_ModuleLoadData(&mod, VECADD_1T_PTX));
    CUfunction func;
    CHECK_CU(cu_ModuleGetFunction(&func, mod, "vecAdd1T"));

    int counts[] = { 1<<16, 1<<18, 1<<20 };
    const char *names[] = { "64K", "256K", "1M" };

    for (int c = 0; c < 3; c++) {
        int N = counts[c];
        size_t bytes = N * sizeof(float);

        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_c = (float *)malloc(bytes);
        for (int i = 0; i < N; i++) {
            h_a[i] = (float)i;
            h_b[i] = (float)(i * 2);
        }

        double times[ITERS];

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            CUdeviceptr d_a, d_b, d_c;
            cu_MemAlloc(&d_a, bytes);
            cu_MemAlloc(&d_b, bytes);
            cu_MemAlloc(&d_c, bytes);
            cu_MemcpyHtoD(d_a, h_a, bytes);
            cu_MemcpyHtoD(d_b, h_b, bytes);
            void *params[] = { &d_a, &d_b, &d_c };
            cu_LaunchKernel(func, N,1,1, 1,1,1, 0, NULL, params, NULL);
            cu_CtxSynchronize();
            cu_MemcpyDtoH(h_c, d_c, bytes);
            cu_MemFree(d_a);
            cu_MemFree(d_b);
            cu_MemFree(d_c);
        }

        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();

            CUdeviceptr d_a, d_b, d_c;
            cu_MemAlloc(&d_a, bytes);
            cu_MemAlloc(&d_b, bytes);
            cu_MemAlloc(&d_c, bytes);
            cu_MemcpyHtoD(d_a, h_a, bytes);
            cu_MemcpyHtoD(d_b, h_b, bytes);
            void *params[] = { &d_a, &d_b, &d_c };
            cu_LaunchKernel(func, N,1,1, 1,1,1, 0, NULL, params, NULL);
            cu_CtxSynchronize();
            cu_MemcpyDtoH(h_c, d_c, bytes);

            double t1 = now_ms();

            cu_MemFree(d_a);
            cu_MemFree(d_b);
            cu_MemFree(d_c);
            times[i] = t1 - t0;
        }

        char buf[128];
        snprintf(buf, sizeof(buf), "E2E VecAdd N=%s", names[c]);
        print_time(buf, median(times, ITERS));

        free(h_a);
        free(h_b);
        free(h_c);
    }

    cu_ModuleUnload(mod);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-libcuda.so> [label]\n", argv[0]);
        fprintf(stderr, "\nExamples:\n");
        fprintf(stderr, "  %s /usr/lib/libcuda.so.1               \"Real CUDA\"\n", argv[0]);
        fprintf(stderr, "  %s ./build/cuvk_runtime/libcuda.so.1   \"cuvk\"\n", argv[0]);
        return 1;
    }

    const char *lib_path = argv[1];
    const char *label = argc > 2 ? argv[2] : lib_path;

    printf("============================================================\n");
    printf("  CUDA Benchmark: %s\n", label);
    printf("  Library: %s\n", lib_path);
    printf("============================================================\n");

    if (load_library(lib_path) != 0)
        return 1;

    CHECK_CU(cu_Init(0));

    int count = 0;
    CHECK_CU(cu_DeviceGetCount(&count));
    if (count == 0) {
        fprintf(stderr, "No devices found\n");
        return 1;
    }

    CHECK_CU(cu_DeviceGet(&g_dev, 0));
    CHECK_CU(cu_CtxCreate(&g_ctx, NULL, 0, g_dev));

    bench_init();
    bench_module_load();
    bench_memory_alloc();
    bench_transfer();
    bench_memset();
    bench_launch_latency();
    bench_vecadd_1t();
    bench_vecadd_mt();
    bench_end_to_end();

    cu_CtxDestroy(g_ctx);
    dlclose(g_lib);

    printf("\n============================================================\n");
    printf("  Done.\n");
    printf("============================================================\n");
    return 0;
}
