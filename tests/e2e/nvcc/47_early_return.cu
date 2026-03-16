// 47. Early returns: device functions with multiple return points
//     Tests guard clauses, validation-style returns, mid-function returns
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Guard clause pattern: multiple early exits at the top
__device__ float safe_divide(float a, float b) {
    if (b == 0.0f)
        return 0.0f;
    if (a == 0.0f)
        return 0.0f;
    if (isinf(b))
        return 0.0f;
    return a / b;
}

// Validation-style: check inputs, return error code
__device__ int validate_and_compute(int x, int y, float *result) {
    if (x < 0)
        return -1;
    if (y < 0)
        return -2;
    if (x == 0 && y == 0)
        return -3;
    *result = sqrtf((float)(x * x + y * y));
    return 0;
}

// Multiple return points in middle of logic
__device__ float categorize_value(float x) {
    if (x != x)  // NaN check
        return -999.0f;

    float abs_x = fabsf(x);

    if (abs_x < 0.001f)
        return 0.0f;  // treat as zero

    if (abs_x > 1000.0f)
        return x > 0.0f ? 1000.0f : -1000.0f;  // clamp

    // Normal range processing
    if (x > 0.0f) {
        if (x < 1.0f)
            return x * x;  // quadratic for small positive
        return sqrtf(x);   // sqrt for large positive
    }

    // Negative normal range
    if (x > -1.0f)
        return -(x * x);
    return -sqrtf(-x);
}

// Binary search with early return on exact match
__device__ int binary_search(const int *arr, int len, int target) {
    int lo = 0, hi = len - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target)
            return mid;  // early return on found
        if (arr[mid] < target)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;  // not found
}

// Recursive-style iterative with early return
__device__ int iterative_fib(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (n == 2) return 1;
    int a = 1, b = 1;
    for (int i = 3; i <= n; i++) {
        int t = a + b;
        a = b;
        b = t;
    }
    return b;
}

__global__ void early_return_kernel(float *out, const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = in[i];
        float x = (float)val * 0.01f;

        float div = safe_divide(x, (float)((val + 3) % 7 - 3));
        float vres = 0.0f;
        int vcode = validate_and_compute(val % 20 - 5, (val + 7) % 20 - 5, &vres);
        float cat = categorize_value(x);
        int fib = iterative_fib((val < 0 ? -val : val) % 20);

        out[i] = div + (vcode < 0 ? (float)vcode : vres) + cat + (float)fib;
    }
}

// Host references
static float host_safe_divide(float a, float b) {
    if (b == 0.0f || a == 0.0f || isinf(b)) return 0.0f;
    return a / b;
}
static int host_validate_and_compute(int x, int y, float *r) {
    if (x < 0) return -1;
    if (y < 0) return -2;
    if (x == 0 && y == 0) return -3;
    *r = sqrtf((float)(x * x + y * y));
    return 0;
}
static float host_categorize_value(float x) {
    float ax = fabsf(x);
    if (ax < 0.001f) return 0.0f;
    if (ax > 1000.0f) return x > 0.0f ? 1000.0f : -1000.0f;
    if (x > 0.0f) return x < 1.0f ? x * x : sqrtf(x);
    return x > -1.0f ? -(x * x) : -sqrtf(-x);
}
static int host_fib(int n) {
    if (n <= 0) return 0;
    if (n <= 2) return 1;
    int a = 1, b = 1;
    for (int i = 3; i <= n; i++) { int t = a + b; a = b; b = t; }
    return b;
}

int main() {
    const int N = 4096;
    size_t ibytes = N * sizeof(int);
    size_t fbytes = N * sizeof(float);

    int *h_in = (int *)malloc(ibytes);
    float *h_out = (float *)malloc(fbytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (i * 67 - 500) % 400;

    int *d_in; float *d_out;
    cudaMalloc(&d_in, ibytes);
    cudaMalloc(&d_out, fbytes);
    cudaMemcpy(d_in, h_in, ibytes, cudaMemcpyHostToDevice);

    early_return_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, fbytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int val = h_in[i];
        float x = (float)val * 0.01f;
        float div = host_safe_divide(x, (float)((val + 3) % 7 - 3));
        float vres = 0.0f;
        int vcode = host_validate_and_compute(val % 20 - 5, (val + 7) % 20 - 5, &vres);
        float cat = host_categorize_value(x);
        int fib = host_fib((val < 0 ? -val : val) % 20);
        float expected = div + (vcode < 0 ? (float)vcode : vres) + cat + (float)fib;
        if (fabsf(h_out[i] - expected) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 47_early_return i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 47_early_return (N=%d)\n", N);
    return 0;
}
