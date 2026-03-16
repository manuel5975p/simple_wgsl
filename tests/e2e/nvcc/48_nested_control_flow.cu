// 48. Deeply nested control flow: mixed branches, loops, switches, break/continue
//     Tests complex nesting of all control flow types
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// State machine: process a "command sequence" encoded as bits
// Deeply nested switch inside loops with break/continue
__device__ float state_machine(int input) {
    float accum = 0.0f;
    int state = 0;  // 0=idle, 1=add, 2=mul, 3=done
    int bits = input;

    for (int step = 0; step < 16; step++) {
        int cmd = (bits >> (step * 2)) & 0x3;

        switch (state) {
            case 0:  // idle
                if (cmd == 0) continue;  // stay idle
                if (cmd == 1) { state = 1; accum = 1.0f; }
                else if (cmd == 2) { state = 2; accum = 1.0f; }
                else { state = 3; }
                break;
            case 1:  // adding
                switch (cmd) {
                    case 0: accum += 1.0f; break;
                    case 1: accum += 2.0f; break;
                    case 2: state = 2; break;
                    case 3: state = 3; break;
                }
                break;
            case 2:  // multiplying
                if (cmd == 3) {
                    state = 3;
                    break;
                }
                accum *= (float)(cmd + 1);
                if (accum > 1000.0f) {
                    accum = 1000.0f;
                    state = 3;
                }
                break;
            case 3:  // done
                break;
        }

        if (state == 3)
            break;
    }

    return accum;
}

// Nested loops with mixed conditions: sparse matrix-vector multiply simulation
__device__ float sparse_dot(int row, int n) {
    float sum = 0.0f;
    for (int blk = 0; blk < 4; blk++) {
        int start = blk * (n / 4);
        int end = (blk + 1) * (n / 4);
        if (end > n) end = n;

        for (int col = start; col < end; col++) {
            // Simulate sparsity: skip elements where (row+col) is not divisible by 3
            if ((row + col) % 3 != 0)
                continue;

            float val;
            if (row == col)
                val = 2.0f;  // diagonal
            else if (col == row + 1 || col == row - 1)
                val = -1.0f;  // adjacent
            else if ((row + col) % 7 == 0)
                val = 0.5f;
            else
                continue;  // truly zero

            // Apply value with branch-dependent scaling
            if (val > 0.0f) {
                sum += val * (float)(col + 1);
            } else {
                sum += val * (float)(col + 1) * 0.5f;
            }
        }
    }
    return sum;
}

// If-else inside switch inside while inside if
__device__ int complex_classify(int x) {
    if (x <= 0)
        return 0;

    int result = 0;
    int iter = 0;
    int val = x;

    while (iter < 10) {
        int digit = val % 10;
        val /= 10;

        switch (digit) {
            case 0: case 1:
                if (iter == 0)
                    result += 1;
                else
                    result += 2;
                break;
            case 2: case 3: case 4:
                if (val > 0) {
                    result += digit;
                } else {
                    result += digit * 2;
                    if (iter > 3)
                        return result;  // early return from nested context
                }
                break;
            case 5:
                result *= 2;
                if (result > 100) {
                    result = 100;
                    break;
                }
                continue;  // skip iter increment below
            default:
                result += 1;
                break;
        }

        iter++;
        if (val == 0)
            break;
    }
    return result;
}

__global__ void nested_kernel(float *out, const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = in[i];
        int abs_val = val < 0 ? -val : val;

        float sm = state_machine(abs_val);
        float sd = sparse_dot(i % 32, 32);
        int cc = complex_classify(abs_val % 10000);

        out[i] = sm + sd + (float)cc;
    }
}

// Host references
static float host_state_machine(int input) {
    float accum = 0.0f;
    int state = 0, bits = input;
    for (int step = 0; step < 16; step++) {
        int cmd = (bits >> (step * 2)) & 0x3;
        switch (state) {
            case 0:
                if (cmd == 0) continue;
                if (cmd == 1) { state = 1; accum = 1.0f; }
                else if (cmd == 2) { state = 2; accum = 1.0f; }
                else { state = 3; }
                break;
            case 1:
                switch (cmd) {
                    case 0: accum += 1.0f; break; case 1: accum += 2.0f; break;
                    case 2: state = 2; break; case 3: state = 3; break;
                }
                break;
            case 2:
                if (cmd == 3) { state = 3; break; }
                accum *= (float)(cmd + 1);
                if (accum > 1000.0f) { accum = 1000.0f; state = 3; }
                break;
            case 3: break;
        }
        if (state == 3) break;
    }
    return accum;
}
static float host_sparse_dot(int row, int n) {
    float sum = 0.0f;
    for (int blk = 0; blk < 4; blk++) {
        int start = blk * (n / 4), end = (blk + 1) * (n / 4);
        if (end > n) end = n;
        for (int col = start; col < end; col++) {
            if ((row + col) % 3 != 0) continue;
            float val;
            if (row == col) val = 2.0f;
            else if (col == row + 1 || col == row - 1) val = -1.0f;
            else if ((row + col) % 7 == 0) val = 0.5f;
            else continue;
            if (val > 0.0f) sum += val * (float)(col + 1);
            else sum += val * (float)(col + 1) * 0.5f;
        }
    }
    return sum;
}
static int host_complex_classify(int x) {
    if (x <= 0) return 0;
    int result = 0, iter = 0, val = x;
    while (iter < 10) {
        int digit = val % 10; val /= 10;
        switch (digit) {
            case 0: case 1:
                result += (iter == 0) ? 1 : 2; break;
            case 2: case 3: case 4:
                if (val > 0) { result += digit; }
                else { result += digit * 2; if (iter > 3) return result; }
                break;
            case 5:
                result *= 2;
                if (result > 100) { result = 100; break; }
                continue;
            default: result += 1; break;
        }
        iter++;
        if (val == 0) break;
    }
    return result;
}

int main() {
    const int N = 4096;
    size_t ibytes = N * sizeof(int);
    size_t fbytes = N * sizeof(float);

    int *h_in = (int *)malloc(ibytes);
    float *h_out = (float *)malloc(fbytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (i * 97 + 31) % 800 - 200;

    int *d_in; float *d_out;
    cudaMalloc(&d_in, ibytes);
    cudaMalloc(&d_out, fbytes);
    cudaMemcpy(d_in, h_in, ibytes, cudaMemcpyHostToDevice);

    nested_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, fbytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int val = h_in[i];
        int abs_val = val < 0 ? -val : val;
        float expected = host_state_machine(abs_val)
                       + host_sparse_dot(i % 32, 32)
                       + (float)host_complex_classify(abs_val % 10000);
        if (fabsf(h_out[i] - expected) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 48_nested_control_flow i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 48_nested_control_flow (N=%d)\n", N);
    return 0;
}
