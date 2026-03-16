// 45. Switch statements: cases, default, fallthrough
//     Tests switch in device functions with various patterns
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Basic switch with all cases having break
__device__ float color_weight(int channel) {
    float w;
    switch (channel) {
        case 0: w = 0.2126f; break;  // R (luminance)
        case 1: w = 0.7152f; break;  // G
        case 2: w = 0.0722f; break;  // B
        default: w = 0.0f; break;
    }
    return w;
}

// Switch with fallthrough (intentional)
__device__ int cumulative_days(int month) {
    // Days remaining from month to end of year (non-leap), via fallthrough
    int days = 0;
    switch (month) {
        case 1:  days += 31; // Jan
        case 2:  days += 28; // Feb
        case 3:  days += 31; // Mar
        case 4:  days += 30; // Apr
        case 5:  days += 31; // May
        case 6:  days += 30; // Jun
        case 7:  days += 31; // Jul
        case 8:  days += 31; // Aug
        case 9:  days += 30; // Sep
        case 10: days += 31; // Oct
        case 11: days += 30; // Nov
        case 12: days += 31; // Dec
        default: break;
    }
    return days;
}

// Switch with grouped cases
__device__ int season(int month) {
    switch (month) {
        case 12: case 1: case 2:
            return 0;  // winter
        case 3: case 4: case 5:
            return 1;  // spring
        case 6: case 7: case 8:
            return 2;  // summer
        case 9: case 10: case 11:
            return 3;  // autumn
        default:
            return -1;
    }
}

// Switch on computed value with expressions in cases
__device__ float apply_op(float a, float b, int op) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return (b != 0.0f) ? a / b : 0.0f;
        case 4: return fminf(a, b);
        case 5: return fmaxf(a, b);
        default: return 0.0f;
    }
}

__global__ void switch_kernel(float *out, const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = in[i];
        int month = (val % 12) + 1;
        int channel = val % 3;
        int op = val % 6;
        float a = (float)(val % 50) * 0.1f;
        float b = (float)((val + 17) % 50) * 0.1f + 0.1f;

        float w = color_weight(channel);
        int cd = cumulative_days(month);
        int s = season(month);
        float r = apply_op(a, b, op);

        out[i] = w * 100.0f + (float)cd + (float)s + r;
    }
}

// Host references
static float host_color_weight(int c) {
    switch (c) { case 0: return 0.2126f; case 1: return 0.7152f; case 2: return 0.0722f; default: return 0.0f; }
}
static int host_cumulative_days(int m) {
    int d = 0;
    switch (m) {
        case 1:  d += 31; case 2:  d += 28; case 3:  d += 31;
        case 4:  d += 30; case 5:  d += 31; case 6:  d += 30;
        case 7:  d += 31; case 8:  d += 31; case 9:  d += 30;
        case 10: d += 31; case 11: d += 30; case 12: d += 31;
        default: break;
    }
    return d;
}
static int host_season(int m) {
    switch (m) {
        case 12: case 1: case 2: return 0; case 3: case 4: case 5: return 1;
        case 6: case 7: case 8: return 2; case 9: case 10: case 11: return 3;
        default: return -1;
    }
}
static float host_apply_op(float a, float b, int op) {
    switch (op) {
        case 0: return a + b; case 1: return a - b; case 2: return a * b;
        case 3: return (b != 0.0f) ? a / b : 0.0f;
        case 4: return fminf(a, b); case 5: return fmaxf(a, b);
        default: return 0.0f;
    }
}

int main() {
    const int N = 4096;
    size_t ibytes = N * sizeof(int);
    size_t fbytes = N * sizeof(float);

    int *h_in = (int *)malloc(ibytes);
    float *h_out = (float *)malloc(fbytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (i * 41 + 7) % 300;

    int *d_in; float *d_out;
    cudaMalloc(&d_in, ibytes);
    cudaMalloc(&d_out, fbytes);
    cudaMemcpy(d_in, h_in, ibytes, cudaMemcpyHostToDevice);

    switch_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, fbytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int val = h_in[i];
        int month = (val % 12) + 1;
        int channel = val % 3;
        int op = val % 6;
        float a = (float)(val % 50) * 0.1f;
        float b = (float)((val + 17) % 50) * 0.1f + 0.1f;
        float expected = host_color_weight(channel) * 100.0f
                       + (float)host_cumulative_days(month)
                       + (float)host_season(month)
                       + host_apply_op(a, b, op);
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 45_switch_statement i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 45_switch_statement (N=%d)\n", N);
    return 0;
}
