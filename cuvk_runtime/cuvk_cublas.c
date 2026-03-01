/*
 * cuvk_cublas.c - cuBLAS implementation for the CUDA-on-Vulkan runtime
 *
 * Provides a libcublas.so replacement that implements BLAS operations using
 * CPU fallback: device data is copied to host via cuMemcpyDtoH, the BLAS
 * operation runs on the CPU, and results are copied back via cuMemcpyHtoD.
 *
 * Supports: handle management, pointer mode, Level 1 (saxpy, sscal, sdot,
 * snrm2, sasum, isamax, scopy, sswap + double variants), Level 2 (sgemv,
 * dgemv), Level 3 (sgemm, dgemm).
 */

#include "cuvk_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * cuBLAS types (must match cublas_api.h)
 * ============================================================================ */

typedef enum {
    CUBLAS_STATUS_SUCCESS          = 0,
    CUBLAS_STATUS_NOT_INITIALIZED  = 1,
    CUBLAS_STATUS_ALLOC_FAILED     = 3,
    CUBLAS_STATUS_INVALID_VALUE    = 7,
    CUBLAS_STATUS_ARCH_MISMATCH    = 8,
    CUBLAS_STATUS_MAPPING_ERROR    = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR   = 14,
    CUBLAS_STATUS_NOT_SUPPORTED    = 15,
    CUBLAS_STATUS_LICENSE_ERROR    = 16
} cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_CONJG = 3
} cublasOperation_t;

typedef enum {
    CUBLAS_POINTER_MODE_HOST   = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL  = 2
} cublasFillMode_t;

typedef enum {
    CUBLAS_SIDE_LEFT  = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT     = 1
} cublasDiagType_t;

typedef enum {
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 0,
    CUBLAS_DEFAULT_MATH                               = 0
} cublasMath_t;

typedef struct cublasContext {
    struct CUctx_st *ctx;
    CUstream         stream;
    cublasPointerMode_t pointer_mode;
} cublasContext;

typedef cublasContext *cublasHandle_t;

/* ============================================================================
 * Helpers
 * ============================================================================ */

static cublasStatus_t cu_to_blas(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS:             return CUBLAS_STATUS_SUCCESS;
    case CUDA_ERROR_INVALID_VALUE: return CUBLAS_STATUS_INVALID_VALUE;
    case CUDA_ERROR_OUT_OF_MEMORY: return CUBLAS_STATUS_ALLOC_FAILED;
    default:                       return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}

/* Copy device memory to a newly malloc'd host buffer */
static cublasStatus_t dev_to_host(const void *dptr, void **out, size_t bytes) {
    void *h = malloc(bytes);
    if (!h) return CUBLAS_STATUS_ALLOC_FAILED;
    CUresult r = cuMemcpyDtoH_v2(h, (CUdeviceptr)dptr, bytes);
    if (r != CUDA_SUCCESS) { free(h); return cu_to_blas(r); }
    *out = h;
    return CUBLAS_STATUS_SUCCESS;
}

/* Copy host buffer to device and free the host buffer */
static cublasStatus_t host_to_dev(void *dptr, void *hbuf, size_t bytes) {
    CUresult r = cuMemcpyHtoD_v2((CUdeviceptr)dptr, hbuf, bytes);
    free(hbuf);
    return cu_to_blas(r);
}

/* Read a scalar — either from host pointer or device pointer */
static cublasStatus_t read_scalar_f(const cublasContext *h,
                                     const float *ptr, float *out) {
    if (h->pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        *out = *ptr;
        return CUBLAS_STATUS_SUCCESS;
    }
    CUresult r = cuMemcpyDtoH_v2(out, (CUdeviceptr)ptr, sizeof(float));
    return cu_to_blas(r);
}

static cublasStatus_t read_scalar_d(const cublasContext *h,
                                     const double *ptr, double *out) {
    if (h->pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        *out = *ptr;
        return CUBLAS_STATUS_SUCCESS;
    }
    CUresult r = cuMemcpyDtoH_v2(out, (CUdeviceptr)ptr, sizeof(double));
    return cu_to_blas(r);
}

/* Write a scalar result — either to host pointer or device pointer */
static cublasStatus_t write_scalar_f(const cublasContext *h,
                                      float *ptr, float val) {
    if (h->pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        *ptr = val;
        return CUBLAS_STATUS_SUCCESS;
    }
    CUresult r = cuMemcpyHtoD_v2((CUdeviceptr)ptr, &val, sizeof(float));
    return cu_to_blas(r);
}

static cublasStatus_t write_scalar_d(const cublasContext *h,
                                      double *ptr, double val) {
    if (h->pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        *ptr = val;
        return CUBLAS_STATUS_SUCCESS;
    }
    CUresult r = cuMemcpyHtoD_v2((CUdeviceptr)ptr, &val, sizeof(double));
    return cu_to_blas(r);
}

static cublasStatus_t write_scalar_i(const cublasContext *h,
                                      int *ptr, int val) {
    if (h->pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        *ptr = val;
        return CUBLAS_STATUS_SUCCESS;
    }
    CUresult r = cuMemcpyHtoD_v2((CUdeviceptr)ptr, &val, sizeof(int));
    return cu_to_blas(r);
}

/* ============================================================================
 * Handle management
 * ============================================================================ */

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;

    /* Ensure cuvk is initialized */
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return cu_to_blas(r);

    if (!g_cuvk.current_ctx) {
        CUcontext ctx;
        r = cuDevicePrimaryCtxRetain(&ctx, 0);
        if (r != CUDA_SUCCESS) return cu_to_blas(r);
    }

    cublasContext *h = (cublasContext *)calloc(1, sizeof(*h));
    if (!h) return CUBLAS_STATUS_ALLOC_FAILED;

    h->ctx = g_cuvk.current_ctx;
    h->stream = NULL;
    h->pointer_mode = CUBLAS_POINTER_MODE_HOST;

    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    free(handle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, CUstream streamId) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, CUstream *streamId) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!streamId) return CUBLAS_STATUS_INVALID_VALUE;
    *streamId = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                        cublasPointerMode_t mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->pointer_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle,
                                        cublasPointerMode_t *mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = handle->pointer_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    (void)handle;
    if (!version) return CUBLAS_STATUS_INVALID_VALUE;
    *version = 130000;  /* 13.0.0 */
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    (void)handle; (void)mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    (void)handle;
    if (mode) *mode = CUBLAS_DEFAULT_MATH;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace,
                                      size_t workspaceSizeInBytes) {
    (void)handle; (void)workspace; (void)workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

/* ============================================================================
 * Level 1 BLAS — Float
 * ============================================================================ */

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
                               const float *alpha,
                               const float *x, int incx,
                               float *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    float a;
    cublasStatus_t s = read_scalar_f(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    /* Download x and y */
    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(float);
    float *hx = NULL, *hy = NULL;

    s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    for (int i = 0; i < n; i++)
        hy[i * incy] += a * hx[i * incx];

    free(hx);
    return host_to_dev(y, hy, y_bytes);
}

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n,
                               const float *alpha,
                               float *x, int incx) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    float a;
    cublasStatus_t s = read_scalar_f(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    float *hx = NULL;
    s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    for (int i = 0; i < n; i++)
        hx[i * incx] *= a;

    return host_to_dev(x, hx, bytes);
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
                              const float *x, int incx,
                              const float *y, int incy,
                              float *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_f(handle, result, 0.0f);

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(float);
    float *hx = NULL, *hy = NULL;

    cublasStatus_t s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    double dot = 0.0;
    for (int i = 0; i < n; i++)
        dot += (double)hx[i * incx] * (double)hy[i * incy];

    free(hx);
    free(hy);
    return write_scalar_f(handle, result, (float)dot);
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n,
                                const float *x, int incx,
                                float *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_f(handle, result, 0.0f);

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    float *hx = NULL;
    cublasStatus_t s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double v = (double)hx[i * incx];
        sum += v * v;
    }

    free(hx);
    return write_scalar_f(handle, result, (float)sqrt(sum));
}

cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n,
                               const float *x, int incx,
                               float *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_f(handle, result, 0.0f);

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    float *hx = NULL;
    cublasStatus_t s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += fabs((double)hx[i * incx]);

    free(hx);
    return write_scalar_f(handle, result, (float)sum);
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n,
                                const float *x, int incx,
                                int *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_i(handle, result, 0);

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    float *hx = NULL;
    cublasStatus_t s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    int max_idx = 0;
    float max_val = fabsf(hx[0]);
    for (int i = 1; i < n; i++) {
        float v = fabsf(hx[i * incx]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    free(hx);
    /* cuBLAS returns 1-based index */
    return write_scalar_i(handle, result, max_idx + 1);
}

cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n,
                               const float *x, int incx,
                               float *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(float);
    float *hx = NULL, *hy = NULL;

    cublasStatus_t s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    for (int i = 0; i < n; i++)
        hy[i * incy] = hx[i * incx];

    free(hx);
    return host_to_dev(y, hy, y_bytes);
}

cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n,
                               float *x, int incx,
                               float *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(float);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(float);
    float *hx = NULL, *hy = NULL;

    cublasStatus_t s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    for (int i = 0; i < n; i++) {
        float tmp = hx[i * incx];
        hx[i * incx] = hy[i * incy];
        hy[i * incy] = tmp;
    }

    s = host_to_dev(x, hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hy); return s; }
    return host_to_dev(y, hy, y_bytes);
}

/* ============================================================================
 * Level 1 BLAS — Double
 * ============================================================================ */

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n,
                               const double *alpha,
                               const double *x, int incx,
                               double *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    double a;
    cublasStatus_t s = read_scalar_d(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(double);
    double *hx = NULL, *hy = NULL;

    s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    for (int i = 0; i < n; i++)
        hy[i * incy] += a * hx[i * incx];

    free(hx);
    return host_to_dev(y, hy, y_bytes);
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n,
                               const double *alpha,
                               double *x, int incx) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    double a;
    cublasStatus_t s = read_scalar_d(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    double *hx = NULL;
    s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    for (int i = 0; i < n; i++)
        hx[i * incx] *= a;

    return host_to_dev(x, hx, bytes);
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n,
                              const double *x, int incx,
                              const double *y, int incy,
                              double *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_d(handle, result, 0.0);

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(double);
    double *hx = NULL, *hy = NULL;

    cublasStatus_t s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    double dot = 0.0;
    for (int i = 0; i < n; i++)
        dot += hx[i * incx] * hy[i * incy];

    free(hx);
    free(hy);
    return write_scalar_d(handle, result, dot);
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n,
                                const double *x, int incx,
                                double *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_d(handle, result, 0.0);

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    double *hx = NULL;
    cublasStatus_t s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double v = hx[i * incx];
        sum += v * v;
    }

    free(hx);
    return write_scalar_d(handle, result, sqrt(sum));
}

cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n,
                               const double *x, int incx,
                               double *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_d(handle, result, 0.0);

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    double *hx = NULL;
    cublasStatus_t s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += fabs(hx[i * incx]);

    free(hx);
    return write_scalar_d(handle, result, sum);
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n,
                                const double *x, int incx,
                                int *result) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return write_scalar_i(handle, result, 0);

    size_t bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    double *hx = NULL;
    cublasStatus_t s = dev_to_host(x, (void **)&hx, bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    int max_idx = 0;
    double max_val = fabs(hx[0]);
    for (int i = 1; i < n; i++) {
        double v = fabs(hx[i * incx]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    free(hx);
    return write_scalar_i(handle, result, max_idx + 1);
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n,
                               const double *x, int incx,
                               double *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(double);
    double *hx = NULL, *hy = NULL;

    cublasStatus_t s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    for (int i = 0; i < n; i++)
        hy[i * incy] = hx[i * incx];

    free(hx);
    return host_to_dev(y, hy, y_bytes);
}

cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n,
                               double *x, int incx,
                               double *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;

    size_t x_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incx)) * sizeof(double);
    size_t y_bytes = (size_t)(1 + ((size_t)(n - 1)) * abs(incy)) * sizeof(double);
    double *hx = NULL, *hy = NULL;

    cublasStatus_t s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hx); return s; }

    for (int i = 0; i < n; i++) {
        double tmp = hx[i * incx];
        hx[i * incx] = hy[i * incy];
        hy[i * incy] = tmp;
    }

    s = host_to_dev(x, hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hy); return s; }
    return host_to_dev(y, hy, y_bytes);
}

/* ============================================================================
 * Level 2 BLAS — Sgemv / Dgemv
 * ============================================================================ */

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
                               cublasOperation_t trans,
                               int m, int n,
                               const float *alpha,
                               const float *A, int lda,
                               const float *x, int incx,
                               const float *beta,
                               float *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m <= 0 || n <= 0) return CUBLAS_STATUS_SUCCESS;

    float a, b;
    cublasStatus_t s = read_scalar_f(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = read_scalar_f(handle, beta, &b);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    /* A is m x n (column-major) in lda rows */
    size_t A_bytes = (size_t)lda * (size_t)n * sizeof(float);
    int x_len = (trans == CUBLAS_OP_N) ? n : m;
    int y_len = (trans == CUBLAS_OP_N) ? m : n;
    size_t x_bytes = (size_t)(1 + ((size_t)(x_len - 1)) * abs(incx)) * sizeof(float);
    size_t y_bytes = (size_t)(1 + ((size_t)(y_len - 1)) * abs(incy)) * sizeof(float);

    float *hA = NULL, *hx = NULL, *hy = NULL;
    s = dev_to_host(A, (void **)&hA, A_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); return s; }
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); free(hx); return s; }

    /* y = alpha * op(A) * x + beta * y */
    for (int i = 0; i < y_len; i++) {
        double sum = 0.0;
        for (int j = 0; j < x_len; j++) {
            float Aij;
            if (trans == CUBLAS_OP_N)
                Aij = hA[j * lda + i];  /* A[i][j] column-major */
            else
                Aij = hA[i * lda + j];  /* A^T[i][j] = A[j][i] */
            sum += (double)Aij * (double)hx[j * incx];
        }
        hy[i * incy] = (float)((double)a * sum + (double)b * (double)hy[i * incy]);
    }

    free(hA);
    free(hx);
    return host_to_dev(y, hy, y_bytes);
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
                               cublasOperation_t trans,
                               int m, int n,
                               const double *alpha,
                               const double *A, int lda,
                               const double *x, int incx,
                               const double *beta,
                               double *y, int incy) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m <= 0 || n <= 0) return CUBLAS_STATUS_SUCCESS;

    double a, b;
    cublasStatus_t s = read_scalar_d(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = read_scalar_d(handle, beta, &b);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    size_t A_bytes = (size_t)lda * (size_t)n * sizeof(double);
    int x_len = (trans == CUBLAS_OP_N) ? n : m;
    int y_len = (trans == CUBLAS_OP_N) ? m : n;
    size_t x_bytes = (size_t)(1 + ((size_t)(x_len - 1)) * abs(incx)) * sizeof(double);
    size_t y_bytes = (size_t)(1 + ((size_t)(y_len - 1)) * abs(incy)) * sizeof(double);

    double *hA = NULL, *hx = NULL, *hy = NULL;
    s = dev_to_host(A, (void **)&hA, A_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(x, (void **)&hx, x_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); return s; }
    s = dev_to_host(y, (void **)&hy, y_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); free(hx); return s; }

    for (int i = 0; i < y_len; i++) {
        double sum = 0.0;
        for (int j = 0; j < x_len; j++) {
            double Aij;
            if (trans == CUBLAS_OP_N)
                Aij = hA[j * lda + i];
            else
                Aij = hA[i * lda + j];
            sum += Aij * hx[j * incx];
        }
        hy[i * incy] = a * sum + b * hy[i * incy];
    }

    free(hA);
    free(hx);
    return host_to_dev(y, hy, y_bytes);
}

/* ============================================================================
 * Level 3 BLAS — Sgemm / Dgemm
 * ============================================================================ */

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m, int n, int k,
                               const float *alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               const float *beta,
                               float *C, int ldc) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;

    float a, b;
    cublasStatus_t s = read_scalar_f(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = read_scalar_f(handle, beta, &b);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    /* Column-major: A is lda x Ka, B is ldb x Kb, C is ldc x n */
    int Ka = (transa == CUBLAS_OP_N) ? k : m;
    int Kb = (transb == CUBLAS_OP_N) ? n : k;
    size_t A_bytes = (size_t)lda * (size_t)Ka * sizeof(float);
    size_t B_bytes = (size_t)ldb * (size_t)Kb * sizeof(float);
    size_t C_bytes = (size_t)ldc * (size_t)n * sizeof(float);

    float *hA = NULL, *hB = NULL, *hC = NULL;
    s = dev_to_host(A, (void **)&hA, A_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(B, (void **)&hB, B_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); return s; }
    s = dev_to_host(C, (void **)&hC, C_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); free(hB); return s; }

    /* C = alpha * op(A) * op(B) + beta * C */
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                float Aval, Bval;
                if (transa == CUBLAS_OP_N)
                    Aval = hA[p * lda + row];
                else
                    Aval = hA[row * lda + p];
                if (transb == CUBLAS_OP_N)
                    Bval = hB[col * ldb + p];
                else
                    Bval = hB[p * ldb + col];
                sum += (double)Aval * (double)Bval;
            }
            hC[col * ldc + row] = (float)((double)a * sum +
                                           (double)b * (double)hC[col * ldc + row]);
        }
    }

    free(hA);
    free(hB);
    return host_to_dev(C, hC, C_bytes);
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m, int n, int k,
                               const double *alpha,
                               const double *A, int lda,
                               const double *B, int ldb,
                               const double *beta,
                               double *C, int ldc) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;

    double a, b;
    cublasStatus_t s = read_scalar_d(handle, alpha, &a);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = read_scalar_d(handle, beta, &b);
    if (s != CUBLAS_STATUS_SUCCESS) return s;

    int Ka = (transa == CUBLAS_OP_N) ? k : m;
    int Kb = (transb == CUBLAS_OP_N) ? n : k;
    size_t A_bytes = (size_t)lda * (size_t)Ka * sizeof(double);
    size_t B_bytes = (size_t)ldb * (size_t)Kb * sizeof(double);
    size_t C_bytes = (size_t)ldc * (size_t)n * sizeof(double);

    double *hA = NULL, *hB = NULL, *hC = NULL;
    s = dev_to_host(A, (void **)&hA, A_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = dev_to_host(B, (void **)&hB, B_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); return s; }
    s = dev_to_host(C, (void **)&hC, C_bytes);
    if (s != CUBLAS_STATUS_SUCCESS) { free(hA); free(hB); return s; }

    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                double Aval, Bval;
                if (transa == CUBLAS_OP_N)
                    Aval = hA[p * lda + row];
                else
                    Aval = hA[row * lda + p];
                if (transb == CUBLAS_OP_N)
                    Bval = hB[col * ldb + p];
                else
                    Bval = hB[p * ldb + col];
                sum += Aval * Bval;
            }
            hC[col * ldc + row] = a * sum + b * hC[col * ldc + row];
        }
    }

    free(hA);
    free(hB);
    return host_to_dev(C, hC, C_bytes);
}

/* ============================================================================
 * 64-bit integer variants (forward to 32-bit for now)
 * ============================================================================ */

cublasStatus_t cublasSaxpy_v2_64(cublasHandle_t h, int64_t n, const float *a,
                                  const float *x, int64_t incx,
                                  float *y, int64_t incy) {
    return cublasSaxpy_v2(h, (int)n, a, x, (int)incx, y, (int)incy);
}

cublasStatus_t cublasDaxpy_v2_64(cublasHandle_t h, int64_t n, const double *a,
                                  const double *x, int64_t incx,
                                  double *y, int64_t incy) {
    return cublasDaxpy_v2(h, (int)n, a, x, (int)incx, y, (int)incy);
}

cublasStatus_t cublasSscal_v2_64(cublasHandle_t h, int64_t n, const float *a,
                                  float *x, int64_t incx) {
    return cublasSscal_v2(h, (int)n, a, x, (int)incx);
}

cublasStatus_t cublasDscal_v2_64(cublasHandle_t h, int64_t n, const double *a,
                                  double *x, int64_t incx) {
    return cublasDscal_v2(h, (int)n, a, x, (int)incx);
}

cublasStatus_t cublasSdot_v2_64(cublasHandle_t h, int64_t n,
                                 const float *x, int64_t incx,
                                 const float *y, int64_t incy,
                                 float *result) {
    return cublasSdot_v2(h, (int)n, x, (int)incx, y, (int)incy, result);
}

cublasStatus_t cublasDdot_v2_64(cublasHandle_t h, int64_t n,
                                 const double *x, int64_t incx,
                                 const double *y, int64_t incy,
                                 double *result) {
    return cublasDdot_v2(h, (int)n, x, (int)incx, y, (int)incy, result);
}

cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t h, int64_t n,
                                   const float *x, int64_t incx,
                                   float *result) {
    return cublasSnrm2_v2(h, (int)n, x, (int)incx, result);
}

cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t h, int64_t n,
                                   const double *x, int64_t incx,
                                   double *result) {
    return cublasDnrm2_v2(h, (int)n, x, (int)incx, result);
}

cublasStatus_t cublasSasum_v2_64(cublasHandle_t h, int64_t n,
                                  const float *x, int64_t incx,
                                  float *result) {
    return cublasSasum_v2(h, (int)n, x, (int)incx, result);
}

cublasStatus_t cublasDasum_v2_64(cublasHandle_t h, int64_t n,
                                  const double *x, int64_t incx,
                                  double *result) {
    return cublasDasum_v2(h, (int)n, x, (int)incx, result);
}

cublasStatus_t cublasIsamax_v2_64(cublasHandle_t h, int64_t n,
                                   const float *x, int64_t incx,
                                   int64_t *result) {
    int r;
    cublasStatus_t s = cublasIsamax_v2(h, (int)n, x, (int)incx, &r);
    if (s == CUBLAS_STATUS_SUCCESS && result) *result = r;
    return s;
}

cublasStatus_t cublasIdamax_v2_64(cublasHandle_t h, int64_t n,
                                   const double *x, int64_t incx,
                                   int64_t *result) {
    int r;
    cublasStatus_t s = cublasIdamax_v2(h, (int)n, x, (int)incx, &r);
    if (s == CUBLAS_STATUS_SUCCESS && result) *result = r;
    return s;
}

cublasStatus_t cublasScopy_v2_64(cublasHandle_t h, int64_t n,
                                  const float *x, int64_t incx,
                                  float *y, int64_t incy) {
    return cublasScopy_v2(h, (int)n, x, (int)incx, y, (int)incy);
}

cublasStatus_t cublasDcopy_v2_64(cublasHandle_t h, int64_t n,
                                  const double *x, int64_t incx,
                                  double *y, int64_t incy) {
    return cublasDcopy_v2(h, (int)n, x, (int)incx, y, (int)incy);
}

cublasStatus_t cublasSgemm_v2_64(cublasHandle_t h,
                                  cublasOperation_t ta, cublasOperation_t tb,
                                  int64_t m, int64_t n, int64_t k,
                                  const float *alpha,
                                  const float *A, int64_t lda,
                                  const float *B, int64_t ldb,
                                  const float *beta,
                                  float *C, int64_t ldc) {
    return cublasSgemm_v2(h, ta, tb, (int)m, (int)n, (int)k,
                           alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

cublasStatus_t cublasDgemm_v2_64(cublasHandle_t h,
                                  cublasOperation_t ta, cublasOperation_t tb,
                                  int64_t m, int64_t n, int64_t k,
                                  const double *alpha,
                                  const double *A, int64_t lda,
                                  const double *B, int64_t ldb,
                                  const double *beta,
                                  double *C, int64_t ldc) {
    return cublasDgemm_v2(h, ta, tb, (int)m, (int)n, (int)k,
                           alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

cublasStatus_t cublasSgemv_v2_64(cublasHandle_t h, cublasOperation_t trans,
                                  int64_t m, int64_t n,
                                  const float *alpha,
                                  const float *A, int64_t lda,
                                  const float *x, int64_t incx,
                                  const float *beta,
                                  float *y, int64_t incy) {
    return cublasSgemv_v2(h, trans, (int)m, (int)n, alpha,
                           A, (int)lda, x, (int)incx, beta, y, (int)incy);
}

cublasStatus_t cublasDgemv_v2_64(cublasHandle_t h, cublasOperation_t trans,
                                  int64_t m, int64_t n,
                                  const double *alpha,
                                  const double *A, int64_t lda,
                                  const double *x, int64_t incx,
                                  const double *beta,
                                  double *y, int64_t incy) {
    return cublasDgemv_v2(h, trans, (int)m, (int)n, alpha,
                           A, (int)lda, x, (int)incx, beta, y, (int)incy);
}

/* ============================================================================
 * Stubs for less common functions that programs may reference
 * ============================================================================ */

cublasStatus_t cublasGetStatusName(cublasStatus_t status, const char **name) {
    static const char *names[] = {
        "CUBLAS_STATUS_SUCCESS",
        "CUBLAS_STATUS_NOT_INITIALIZED",
        "", "",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "", "", "",
        "CUBLAS_STATUS_INVALID_VALUE",
    };
    if (name) *name = (status >= 0 && status <= 7) ? names[status] : "CUBLAS_STATUS_UNKNOWN";
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStatusString(cublasStatus_t status, const char **str) {
    return cublasGetStatusName(status, str);
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut,
                                      int logToStdErr, const char *logFileName) {
    (void)logIsOn; (void)logToStdOut; (void)logToStdErr; (void)logFileName;
    return CUBLAS_STATUS_SUCCESS;
}
