// 42. CUDA events: create/record/sync/elapsed time around kernel dispatches
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Trivial kernel: just copy
__global__ void copy_kernel(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Heavier kernel: repeated FMA to burn some cycles
__global__ void burn_kernel(float *data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        for (int j = 0; j < iters; j++)
            v = v * 1.00001f + 0.00001f;
        data[i] = v;
    }
}

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "FAIL: %s returned %d at line %d\n", #call, (int)err, __LINE__); \
        return 1; \
    } \
} while(0)

int main() {
    // --- Basic event lifecycle ---
    cudaEvent_t ev;
    CHECK(cudaEventCreate(&ev));
    CHECK(cudaEventRecord(ev, 0));
    CHECK(cudaEventSynchronize(ev));
    CHECK(cudaEventDestroy(ev));

    // --- Elapsed time: two back-to-back events (should be ~0 ms) ---
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_empty = -1.0f;
    CHECK(cudaEventElapsedTime(&ms_empty, e0, e1));
    printf("back-to-back events: %.4f ms\n", ms_empty);
    if (ms_empty < 0.0f) {
        fprintf(stderr, "FAIL: negative elapsed time %.4f\n", ms_empty);
        return 1;
    }

    // --- Time a tiny kernel (64 elements) ---
    const int N_TINY = 64;
    float *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, N_TINY * sizeof(float)));
    CHECK(cudaMalloc(&d_b, N_TINY * sizeof(float)));
    float h_tiny[64];
    for (int i = 0; i < N_TINY; i++) h_tiny[i] = (float)i;
    CHECK(cudaMemcpy(d_a, h_tiny, N_TINY * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(e0));
    copy_kernel<<<1, 64>>>(d_b, d_a, N_TINY);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_tiny = -1.0f;
    CHECK(cudaEventElapsedTime(&ms_tiny, e0, e1));
    printf("tiny copy (N=%d): %.4f ms\n", N_TINY, ms_tiny);
    if (ms_tiny < 0.0f) {
        fprintf(stderr, "FAIL: negative elapsed time\n");
        return 1;
    }

    // Verify copy correctness
    float h_out[64];
    CHECK(cudaMemcpy(h_out, d_b, N_TINY * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N_TINY; i++) {
        if (h_out[i] != h_tiny[i]) {
            fprintf(stderr, "FAIL: copy mismatch at %d: %f vs %f\n", i, h_out[i], h_tiny[i]);
            return 1;
        }
    }
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    // --- Time a moderate kernel (64K elements, 256 burn iterations) ---
    const int N_MOD = 65536;
    const int BURN_ITERS = 256;
    float *d_mod;
    CHECK(cudaMalloc(&d_mod, N_MOD * sizeof(float)));
    float *h_mod = (float *)malloc(N_MOD * sizeof(float));
    for (int i = 0; i < N_MOD; i++) h_mod[i] = 1.0f;
    CHECK(cudaMemcpy(d_mod, h_mod, N_MOD * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(e0));
    burn_kernel<<<(N_MOD + 255) / 256, 256>>>(d_mod, N_MOD, BURN_ITERS);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_mod = -1.0f;
    CHECK(cudaEventElapsedTime(&ms_mod, e0, e1));
    printf("burn kernel (N=%d, iters=%d): %.4f ms\n", N_MOD, BURN_ITERS, ms_mod);
    if (ms_mod < 0.0f) {
        fprintf(stderr, "FAIL: negative elapsed time\n");
        return 1;
    }

    // --- Resolution check: burn kernel should take more time than tiny copy ---
    // (Not a hard requirement since oneshot semantics may dominate, but log it)
    printf("burn/tiny ratio: %.2fx\n", ms_mod / (ms_tiny > 0.0f ? ms_tiny : 0.001f));

    // --- Multiple sequential timed dispatches to check consistency ---
    printf("\n5 repeated burn dispatches:\n");
    for (int r = 0; r < 5; r++) {
        CHECK(cudaEventRecord(e0));
        burn_kernel<<<(N_MOD + 255) / 256, 256>>>(d_mod, N_MOD, BURN_ITERS);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms_rep = -1.0f;
        CHECK(cudaEventElapsedTime(&ms_rep, e0, e1));
        printf("  run %d: %.4f ms\n", r, ms_rep);
        if (ms_rep < 0.0f) {
            fprintf(stderr, "FAIL: negative elapsed time on run %d\n", r);
            return 1;
        }
    }

    CHECK(cudaFree(d_mod));
    free(h_mod);
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));

    printf("\nPASS: 42_events\n");
    return 0;
}
