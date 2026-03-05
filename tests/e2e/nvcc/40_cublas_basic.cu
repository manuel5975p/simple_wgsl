// 40. cuBLAS basic operations: Level 1, 2, 3 BLAS
//     Tests saxpy, sscal, sdot, snrm2, sasum, isamax, scopy,
//     sgemv, sgemm, dgemm against known results.
//     Compiled by nvcc, runs against real cuBLAS or our Vulkan replacement.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", \
                e, cudaGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

static int check_close_f(const char *label, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %f expected %f (diff %e)\n",
                label, got, expected, diff);
        return 1;
    }
    return 0;
}

static int check_close_d(const char *label, double got, double expected, double tol) {
    double diff = fabs(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %f expected %f (diff %e)\n",
                label, got, expected, diff);
        return 1;
    }
    return 0;
}

// Test 1: cublasSaxpy — y = alpha*x + y
static int test_saxpy() {
    const int N = 256;
    float h_x[N], h_y[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(N - i);
    }

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 2.0f;
    CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1));

    float h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float expected = 2.0f * (float)i + (float)(N - i);
        if (check_close_f("saxpy", h_result[i], expected, 1e-4f)) return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 2: cublasSscal — x = alpha*x
static int test_sscal() {
    const int N = 128;
    float h_x[N];
    for (int i = 0; i < N; i++) h_x[i] = (float)(i + 1);

    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 0.5f;
    CHECK_CUBLAS(cublasSscal(handle, N, &alpha, d_x, 1));

    float h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float expected = 0.5f * (float)(i + 1);
        if (check_close_f("sscal", h_result[i], expected, 1e-5f)) return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    return 0;
}

// Test 3: cublasSdot — dot product
static int test_sdot() {
    const int N = 64;
    float h_x[N], h_y[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = (float)(i + 1);
    }

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float result = 0.0f;
    CHECK_CUBLAS(cublasSdot(handle, N, d_x, 1, d_y, 1, &result));

    // sum(1..64) = 64*65/2 = 2080
    float expected = (float)(N * (N + 1) / 2);
    if (check_close_f("sdot", result, expected, 1e-2f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 4: cublasSnrm2 — Euclidean norm
static int test_snrm2() {
    const int N = 3;
    float h_x[] = {3.0f, 4.0f, 0.0f};  // norm = 5

    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float result = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle, N, d_x, 1, &result));
    if (check_close_f("snrm2", result, 5.0f, 1e-5f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    return 0;
}

// Test 5: cublasSasum — sum of absolute values
static int test_sasum() {
    const int N = 5;
    float h_x[] = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f};

    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float result = 0.0f;
    CHECK_CUBLAS(cublasSasum(handle, N, d_x, 1, &result));
    if (check_close_f("sasum", result, 15.0f, 1e-5f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    return 0;
}

// Test 6: cublasIsamax — index of max absolute value (1-based)
static int test_isamax() {
    const int N = 5;
    float h_x[] = {1.0f, -7.0f, 3.0f, -2.0f, 5.0f};

    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int result = 0;
    CHECK_CUBLAS(cublasIsamax(handle, N, d_x, 1, &result));
    // h_x[1] = -7.0 has max abs value, 1-based index = 2
    if (result != 2) {
        fprintf(stderr, "FAIL isamax: got %d expected 2\n", result);
        return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    return 0;
}

// Test 7: cublasScopy — copy vector
static int test_scopy() {
    const int N = 32;
    float h_x[N];
    for (int i = 0; i < N; i++) h_x[i] = (float)(i * 3 + 1);

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, N * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasScopy(handle, N, d_x, 1, d_y, 1));

    float h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        if (check_close_f("scopy", h_result[i], h_x[i], 1e-6f)) return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 8: cublasSgemv — matrix-vector multiply (y = alpha*A*x + beta*y)
static int test_sgemv() {
    // A = 2x3 matrix (column-major), x = 3-vector, y = 2-vector
    // A = [[1,2,3],[4,5,6]]  column-major: {1,4, 2,5, 3,6}
    const int M = 2, N = 3;
    float h_A[] = {1.0f, 4.0f,  2.0f, 5.0f,  3.0f, 6.0f};
    float h_x[] = {1.0f, 1.0f, 1.0f};
    float h_y[] = {0.0f, 0.0f};

    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, M * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, M * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, M, N,
                              &alpha, d_A, M, d_x, 1, &beta, d_y, 1));

    float h_result[M];
    CHECK_CUDA(cudaMemcpy(h_result, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    // y[0] = 1*1 + 2*1 + 3*1 = 6
    // y[1] = 4*1 + 5*1 + 6*1 = 15
    if (check_close_f("sgemv[0]", h_result[0], 6.0f, 1e-4f)) return 1;
    if (check_close_f("sgemv[1]", h_result[1], 15.0f, 1e-4f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 9: cublasSgemv transposed
static int test_sgemv_t() {
    // Same A as above, but transposed: y = A^T * x
    // A^T is 3x2, x is 2-vector, y is 3-vector
    const int M = 2, N = 3;
    float h_A[] = {1.0f, 4.0f,  2.0f, 5.0f,  3.0f, 6.0f};
    float h_x[] = {1.0f, 2.0f};
    float h_y[] = {0.0f, 0.0f, 0.0f};

    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, M, N,
                              &alpha, d_A, M, d_x, 1, &beta, d_y, 1));

    float h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    // y[0] = 1*1 + 4*2 = 9
    // y[1] = 2*1 + 5*2 = 12
    // y[2] = 3*1 + 6*2 = 15
    if (check_close_f("sgemv_t[0]", h_result[0], 9.0f, 1e-4f)) return 1;
    if (check_close_f("sgemv_t[1]", h_result[1], 12.0f, 1e-4f)) return 1;
    if (check_close_f("sgemv_t[2]", h_result[2], 15.0f, 1e-4f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 10: cublasSgemm — C = alpha*A*B + beta*C
static int test_sgemm() {
    // A = 2x3, B = 3x2, C = 2x2
    // A = [[1,2,3],[4,5,6]] col-major: {1,4, 2,5, 3,6}
    // B = [[1,0],[0,1],[1,1]] col-major: {1,0,1, 0,1,1}
    const int M = 2, N = 2, K = 3;
    float h_A[] = {1.0f, 4.0f,  2.0f, 5.0f,  3.0f, 6.0f};
    float h_B[] = {1.0f, 0.0f, 1.0f,  0.0f, 1.0f, 1.0f};
    float h_C[4] = {0};

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));

    float h_result[4];
    CHECK_CUDA(cudaMemcpy(h_result, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    // C = A*B:
    // C[0,0] = 1*1 + 2*0 + 3*1 = 4
    // C[1,0] = 4*1 + 5*0 + 6*1 = 10
    // C[0,1] = 1*0 + 2*1 + 3*1 = 5
    // C[1,1] = 4*0 + 5*1 + 6*1 = 11
    if (check_close_f("sgemm[0,0]", h_result[0], 4.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm[1,0]", h_result[1], 10.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm[0,1]", h_result[2], 5.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm[1,1]", h_result[3], 11.0f, 1e-4f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}

// Test 11: cublasSgemm with alpha and beta scaling
static int test_sgemm_alphabeta() {
    // C = 2*I*I + 3*C  where I is 2x2 identity
    const int N = 2;
    float h_I[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float h_C[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_I, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_I, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 2.0f, beta = 3.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

    float h_result[4];
    CHECK_CUDA(cudaMemcpy(h_result, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    // C = 2*I + 3*ones = [[5,3],[3,5]]
    if (check_close_f("sgemm_ab[0,0]", h_result[0], 5.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm_ab[1,0]", h_result[1], 3.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm_ab[0,1]", h_result[2], 3.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm_ab[1,1]", h_result[3], 5.0f, 1e-4f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}

// Test 12: cublasSgemm with transpose
static int test_sgemm_transpose() {
    // C = A^T * B  where A is 3x2, B is 3x2, result is 2x2
    // A = [[1,2],[3,4],[5,6]] col-major: {1,3,5, 2,4,6}
    // B = [[1,0],[0,1],[1,0]] col-major: {1,0,1, 0,1,0}
    const int M = 2, N = 2, K = 3;
    float h_A[] = {1.0f, 3.0f, 5.0f,  2.0f, 4.0f, 6.0f};
    float h_B[] = {1.0f, 0.0f, 1.0f,  0.0f, 1.0f, 0.0f};
    float h_C[4] = {0};

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, K * M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, K * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    // A is K x M stored, but with OP_T it's treated as M x K
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              M, N, K, &alpha, d_A, K, d_B, K, &beta, d_C, M));

    float h_result[4];
    CHECK_CUDA(cudaMemcpy(h_result, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    // A^T = [[1,3,5],[2,4,6]]
    // C = A^T * B:
    // C[0,0] = 1*1+3*0+5*1 = 6
    // C[1,0] = 2*1+4*0+6*1 = 8
    // C[0,1] = 1*0+3*1+5*0 = 3
    // C[1,1] = 2*0+4*1+6*0 = 4
    if (check_close_f("sgemm_t[0,0]", h_result[0], 6.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm_t[1,0]", h_result[1], 8.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm_t[0,1]", h_result[2], 3.0f, 1e-4f)) return 1;
    if (check_close_f("sgemm_t[1,1]", h_result[3], 4.0f, 1e-4f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}

// Test 13: cublasDgemm — double precision
static int test_dgemm() {
    const int N = 2;
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]  col-major
    double h_A[] = {1.0, 3.0, 2.0, 4.0};
    double h_B[] = {5.0, 7.0, 6.0, 8.0};
    double h_C[4] = {0};

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, N * N * sizeof(double), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

    double h_result[4];
    CHECK_CUDA(cudaMemcpy(h_result, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    // C = A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    if (check_close_d("dgemm[0,0]", h_result[0], 19.0, 1e-10)) return 1;
    if (check_close_d("dgemm[1,0]", h_result[1], 43.0, 1e-10)) return 1;
    if (check_close_d("dgemm[0,1]", h_result[2], 22.0, 1e-10)) return 1;
    if (check_close_d("dgemm[1,1]", h_result[3], 50.0, 1e-10)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}

// Test 14: Larger sgemm (8x8)
static int test_sgemm_large() {
    const int N = 8;
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // A = identity, B = values 1..64
    memset(h_A, 0, sizeof(h_A));
    for (int i = 0; i < N; i++) h_A[i * N + i] = 1.0f;
    for (int i = 0; i < N * N; i++) h_B[i] = (float)(i + 1);
    memset(h_C, 0, sizeof(h_C));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

    float h_result[N * N];
    CHECK_CUDA(cudaMemcpy(h_result, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    // C = I * B = B
    for (int i = 0; i < N * N; i++) {
        if (check_close_f("sgemm_large", h_result[i], h_B[i], 1e-4f)) return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}

// Test 15: saxpy with stride (incx=2, incy=3)
static int test_saxpy_stride() {
    // x has stride 2: use elements at 0,2,4
    // y has stride 3: use elements at 0,3,6
    const int N = 3;
    float h_x[] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float h_y[] = {10.0f, 99.0f, 99.0f, 20.0f, 99.0f, 99.0f, 30.0f};

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, 5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, 7 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, 5 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, 7 * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 2.0f;
    CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 2, d_y, 3));

    float h_result[7];
    CHECK_CUDA(cudaMemcpy(h_result, d_y, 7 * sizeof(float), cudaMemcpyDeviceToHost));
    // y[0] += 2*x[0] = 10+2 = 12
    // y[3] += 2*x[2] = 20+4 = 24
    // y[6] += 2*x[4] = 30+6 = 36
    if (check_close_f("saxpy_stride[0]", h_result[0], 12.0f, 1e-4f)) return 1;
    if (check_close_f("saxpy_stride[3]", h_result[3], 24.0f, 1e-4f)) return 1;
    if (check_close_f("saxpy_stride[6]", h_result[6], 36.0f, 1e-4f)) return 1;
    // padding elements unchanged
    if (check_close_f("saxpy_stride[1]", h_result[1], 99.0f, 1e-4f)) return 1;
    if (check_close_f("saxpy_stride[4]", h_result[4], 99.0f, 1e-4f)) return 1;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 16: combined operations — scale then axpy
static int test_combined_ops() {
    const int N = 64;
    float h_x[N], h_y[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = 2.0f;
        h_y[i] = 1.0f;
    }

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // First: scale x by 3 -> x = 6
    float three = 3.0f;
    CHECK_CUBLAS(cublasSscal(handle, N, &three, d_x, 1));

    // Then: y = 0.5*x + y -> y = 3 + 1 = 4
    float half = 0.5f;
    CHECK_CUBLAS(cublasSaxpy(handle, N, &half, d_x, 1, d_y, 1));

    float h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        if (check_close_f("combined", h_result[i], 4.0f, 1e-4f)) return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

// Test 17: cublasSswap
static int test_sswap() {
    const int N = 4;
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[] = {10.0f, 20.0f, 30.0f, 40.0f};

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSswap(handle, N, d_x, 1, d_y, 1));

    float rx[N], ry[N];
    CHECK_CUDA(cudaMemcpy(rx, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ry, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (check_close_f("sswap_x", rx[i], h_y[i], 1e-6f)) return 1;
        if (check_close_f("sswap_y", ry[i], h_x[i], 1e-6f)) return 1;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}

int main() {
    struct { const char *name; int (*fn)(); } tests[] = {
        {"saxpy",              test_saxpy},
        {"sscal",              test_sscal},
        {"sdot",               test_sdot},
        {"snrm2",              test_snrm2},
        {"sasum",              test_sasum},
        {"isamax",             test_isamax},
        {"scopy",              test_scopy},
        {"sgemv",              test_sgemv},
        {"sgemv_t",            test_sgemv_t},
        {"sgemm",              test_sgemm},
        {"sgemm_alphabeta",    test_sgemm_alphabeta},
        {"sgemm_transpose",    test_sgemm_transpose},
        {"dgemm",              test_dgemm},
        {"sgemm_large",        test_sgemm_large},
        {"saxpy_stride",       test_saxpy_stride},
        {"combined_ops",       test_combined_ops},
        {"sswap",              test_sswap},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;

    for (int i = 0; i < n; i++) {
        printf("  [%2d/%d] %-25s ... ", i + 1, n, tests[i].name);
        fflush(stdout);
        int r = tests[i].fn();
        if (r == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
            failed++;
        }
    }

    printf("\nResults: %d passed, %d failed out of %d\n", passed, failed, n);
    if (failed) {
        fprintf(stderr, "FAIL: 40_cublas_basic\n");
        return 1;
    }
    printf("PASS: 40_cublas_basic (all %d tests)\n", n);
    return 0;
}
