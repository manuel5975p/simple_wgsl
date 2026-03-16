// 43. if/else branches: device functions with various conditional patterns
//     Tests simple if, if/else, if/else-if chains, nested conditions
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Simple if: clamp negative to zero
__device__ float relu(float x) {
    if (x < 0.0f)
        return 0.0f;
    return x;
}

// if/else: sign function
__device__ float sign(float x) {
    if (x > 0.0f)
        return 1.0f;
    else if (x < 0.0f)
        return -1.0f;
    else
        return 0.0f;
}

// if/else-if chain: piecewise linear
__device__ float piecewise(float x) {
    if (x < -2.0f)
        return -2.0f;
    else if (x < -1.0f)
        return x;
    else if (x < 1.0f)
        return 0.5f * x;
    else if (x < 2.0f)
        return x;
    else
        return 2.0f;
}

// Nested conditions: classify into bins
__device__ int classify(float x, float y) {
    if (x > 0.0f) {
        if (y > 0.0f)
            return 1;  // Q1
        else
            return 4;  // Q4
    } else {
        if (y > 0.0f)
            return 2;  // Q2
        else
            return 3;  // Q3
    }
}

// Boolean logic in conditions
__device__ float box_filter(float x, float lo, float hi) {
    if (x >= lo && x <= hi)
        return 1.0f;
    else
        return 0.0f;
}

__global__ void branch_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float y = in[(i + n / 2) % n];
        // Combine all branching functions
        float r = relu(x);
        float s = sign(x);
        float p = piecewise(x);
        int q = classify(x, y);
        float b = box_filter(x, -1.0f, 1.0f);
        out[i] = r + s + p + (float)q + b;
    }
}

// Host reference
static float host_relu(float x) { return x < 0.0f ? 0.0f : x; }
static float host_sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
static float host_piecewise(float x) {
    if (x < -2.0f) return -2.0f;
    else if (x < -1.0f) return x;
    else if (x < 1.0f) return 0.5f * x;
    else if (x < 2.0f) return x;
    else return 2.0f;
}
static int host_classify(float x, float y) {
    if (x > 0.0f) return y > 0.0f ? 1 : 4;
    else return y > 0.0f ? 2 : 3;
}
static float host_box_filter(float x, float lo, float hi) {
    return (x >= lo && x <= hi) ? 1.0f : 0.0f;
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i % 100) * 0.1f - 5.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    branch_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float y = h_in[(i + N / 2) % N];
        float expected = host_relu(x) + host_sign(x) + host_piecewise(x)
                       + (float)host_classify(x, y) + host_box_filter(x, -1.0f, 1.0f);
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 43_if_else_branches i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 43_if_else_branches (N=%d)\n", N);
    return 0;
}
