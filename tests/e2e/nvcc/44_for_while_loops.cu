// 44. Loop variants: for, while, do-while in device functions
//     Tests all three loop forms plus nested loops
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// for loop: compute factorial (clamped to avoid overflow)
__device__ float factorial_f(int n) {
    float result = 1.0f;
    for (int i = 2; i <= n; i++)
        result *= (float)i;
    return result;
}

// while loop: integer square root via Newton's method
__device__ int isqrt(int n) {
    if (n <= 1) return n;
    int x = n;
    int y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

// do-while loop: count digits
__device__ int count_digits(int n) {
    if (n < 0) n = -n;
    if (n == 0) return 1;
    int count = 0;
    do {
        count++;
        n /= 10;
    } while (n > 0);
    return count;
}

// Nested for loops: sum of divisors
__device__ int sum_of_divisors(int n) {
    if (n <= 0) return 0;
    int sum = 0;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i)
                sum += n / i;
        }
    }
    return sum;
}

// While loop with multiple exit conditions: GCD
__device__ int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void loop_kernel(float *out, const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = in[i];
        int abs_val = val < 0 ? -val : val;
        int clamped = (abs_val % 10) + 1;  // 1..10 for factorial

        float f = factorial_f(clamped);
        int sq = isqrt(abs_val);
        int digits = count_digits(val);
        int divsum = sum_of_divisors(clamped * 10);
        int g = gcd(abs_val + 1, clamped * 7 + 1);

        out[i] = f + (float)sq + (float)digits + (float)divsum + (float)g;
    }
}

// Host references
static float host_factorial(int n) {
    float r = 1.0f; for (int i = 2; i <= n; i++) r *= (float)i; return r;
}
static int host_isqrt(int n) {
    if (n <= 1) return n;
    int x = n, y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}
static int host_count_digits(int n) {
    if (n < 0) n = -n;
    if (n == 0) return 1;
    int c = 0; do { c++; n /= 10; } while (n > 0); return c;
}
static int host_sum_of_divisors(int n) {
    if (n <= 0) return 0;
    int s = 0;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) { s += i; if (i != n / i) s += n / i; }
    }
    return s;
}
static int host_gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; } return a;
}

int main() {
    const int N = 4096;
    size_t ibytes = N * sizeof(int);
    size_t fbytes = N * sizeof(float);

    int *h_in = (int *)malloc(ibytes);
    float *h_out = (float *)malloc(fbytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (i * 37 + 13) % 500 - 200;  // range -200..299

    int *d_in; float *d_out;
    cudaMalloc(&d_in, ibytes);
    cudaMalloc(&d_out, fbytes);
    cudaMemcpy(d_in, h_in, ibytes, cudaMemcpyHostToDevice);

    loop_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, fbytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int val = h_in[i];
        int abs_val = val < 0 ? -val : val;
        int clamped = (abs_val % 10) + 1;
        float expected = host_factorial(clamped) + (float)host_isqrt(abs_val)
                       + (float)host_count_digits(val)
                       + (float)host_sum_of_divisors(clamped * 10)
                       + (float)host_gcd(abs_val + 1, clamped * 7 + 1);
        if (fabsf(h_out[i] - expected) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 44_for_while_loops i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 44_for_while_loops (N=%d)\n", N);
    return 0;
}
