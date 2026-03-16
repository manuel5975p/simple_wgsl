// 46. continue and break: loop control flow
//     Tests continue, break, and labeled-like patterns in various loop forms
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// for loop with continue: sum only odd numbers
__device__ int sum_odds(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        if (i % 2 == 0)
            continue;
        sum += i;
    }
    return sum;
}

// for loop with break: find first divisor > 1
__device__ int smallest_factor(int n) {
    if (n <= 1) return n;
    int factor = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            factor = i;
            break;
        }
    }
    return factor;
}

// while loop with continue: count non-zero bits in a range
__device__ int count_set_bits_skip_even_pos(int x) {
    // Count bits at odd positions only (positions 1,3,5,...)
    int count = 0;
    int pos = 0;
    int val = x;
    while (val != 0) {
        if (pos % 2 == 0) {
            val >>= 1;
            pos++;
            continue;
        }
        count += val & 1;
        val >>= 1;
        pos++;
    }
    return count;
}

// Nested loop with break on inner, continue on outer
__device__ int count_primes_up_to(int n) {
    int count = 0;
    for (int i = 2; i <= n; i++) {
        int is_prime = 1;
        for (int j = 2; j * j <= i; j++) {
            if (i % j == 0) {
                is_prime = 0;
                break;  // break inner
            }
        }
        if (!is_prime)
            continue;  // continue outer
        count++;
    }
    return count;
}

// do-while with break: find collatz sequence length
__device__ int collatz_len(int n) {
    if (n <= 0) return 0;
    int steps = 0;
    int x = n;
    do {
        if (x == 1)
            break;
        if (x % 2 == 0)
            x = x / 2;
        else
            x = 3 * x + 1;
        steps++;
        if (steps > 200)  // safety cap
            break;
    } while (1);
    return steps;
}

__global__ void continue_break_kernel(float *out, const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = in[i];
        int abs_val = val < 0 ? -val : val;
        int small = (abs_val % 30) + 2;

        int odds = sum_odds(small);
        int sf = smallest_factor(small);
        int bits = count_set_bits_skip_even_pos(abs_val & 0xFFFF);
        int primes = count_primes_up_to(small);
        int coll = collatz_len((abs_val % 100) + 1);

        out[i] = (float)odds + (float)sf + (float)bits
               + (float)primes + (float)coll;
    }
}

// Host references
static int host_sum_odds(int n) {
    int s = 0; for (int i = 1; i <= n; i++) { if (i % 2 == 0) continue; s += i; } return s;
}
static int host_smallest_factor(int n) {
    if (n <= 1) return n;
    for (int i = 2; i * i <= n; i++) { if (n % i == 0) return i; } return n;
}
static int host_count_set_bits_skip_even_pos(int x) {
    int c = 0, pos = 0, val = x;
    while (val != 0) {
        if (pos % 2 == 0) { val >>= 1; pos++; continue; }
        c += val & 1; val >>= 1; pos++;
    }
    return c;
}
static int host_count_primes_up_to(int n) {
    int c = 0;
    for (int i = 2; i <= n; i++) {
        int ok = 1;
        for (int j = 2; j * j <= i; j++) { if (i % j == 0) { ok = 0; break; } }
        if (!ok) continue;
        c++;
    }
    return c;
}
static int host_collatz_len(int n) {
    if (n <= 0) return 0;
    int s = 0, x = n;
    do {
        if (x == 1) break;
        x = (x % 2 == 0) ? x / 2 : 3 * x + 1;
        s++;
        if (s > 200) break;
    } while (1);
    return s;
}

int main() {
    const int N = 4096;
    size_t ibytes = N * sizeof(int);
    size_t fbytes = N * sizeof(float);

    int *h_in = (int *)malloc(ibytes);
    float *h_out = (float *)malloc(fbytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (i * 53 + 19) % 500 - 100;

    int *d_in; float *d_out;
    cudaMalloc(&d_in, ibytes);
    cudaMalloc(&d_out, fbytes);
    cudaMemcpy(d_in, h_in, ibytes, cudaMemcpyHostToDevice);

    continue_break_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, fbytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int val = h_in[i];
        int abs_val = val < 0 ? -val : val;
        int small = (abs_val % 30) + 2;
        float expected = (float)host_sum_odds(small)
                       + (float)host_smallest_factor(small)
                       + (float)host_count_set_bits_skip_even_pos(abs_val & 0xFFFF)
                       + (float)host_count_primes_up_to(small)
                       + (float)host_collatz_len((abs_val % 100) + 1);
        if (fabsf(h_out[i] - expected) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 46_continue_break i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 46_continue_break (N=%d)\n", N);
    return 0;
}
