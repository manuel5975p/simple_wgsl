// 49. Deep device call trees with mixed control flow throughout
//     Tests: function pointers via switch dispatch, mutual-call patterns,
//     multi-level returns, loops calling functions calling loops
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// --- Layer 0: leaf functions with various control flow ---

__device__ float leaf_abs_clamp(float x, float lo, float hi) {
    float ax = fabsf(x);
    if (ax < lo) return lo;
    if (ax > hi) return hi;
    return ax;
}

__device__ int leaf_popcount(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

__device__ float leaf_harmonic(int n) {
    if (n <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 1; i <= n; i++)
        sum += 1.0f / (float)i;
    return sum;
}

// --- Layer 1: functions calling layer 0 with their own control flow ---

__device__ float mid_weighted_sum(float a, float b, float c, int mode) {
    float wa = leaf_abs_clamp(a, 0.1f, 10.0f);
    float wb = leaf_abs_clamp(b, 0.1f, 10.0f);
    float wc = leaf_abs_clamp(c, 0.1f, 10.0f);

    switch (mode % 3) {
        case 0: return wa + wb + wc;
        case 1: return wa * 0.5f + wb * 0.3f + wc * 0.2f;
        case 2: {
            float mx = fmaxf(wa, fmaxf(wb, wc));
            float mn = fminf(wa, fminf(wb, wc));
            return mx - mn;
        }
        default: return 0.0f;
    }
}

__device__ int mid_bit_score(int x) {
    int pc = leaf_popcount((unsigned int)(x < 0 ? -x : x));
    int score = 0;
    for (int i = 0; i < pc; i++) {
        if (i % 2 == 0)
            score += i + 1;
        else
            score -= 1;
    }
    if (score < 0) return 0;
    return score;
}

__device__ float mid_scaled_harmonic(int n, float scale) {
    if (scale <= 0.0f)
        return 0.0f;
    float h = leaf_harmonic(n);
    // Iterative refinement with early exit
    float result = h * scale;
    for (int i = 0; i < 5; i++) {
        if (result > 100.0f)
            return 100.0f;
        if (result < 0.01f)
            return 0.0f;
        result = result * 0.9f + 0.1f * h;
    }
    return result;
}

// --- Layer 2: functions calling layer 1, deeper nesting ---

__device__ float top_evaluate(int idx, float x, float y) {
    // Multiple dispatch paths calling different mid-level functions
    int category = idx % 5;
    float result;

    if (category < 2) {
        // Path A: weighted sum with bit scoring
        float a = x, b = y, c = x + y;
        result = mid_weighted_sum(a, b, c, idx);
        int bs = mid_bit_score(idx);
        if (bs > 10) {
            result += mid_scaled_harmonic(bs, 0.5f);
        } else {
            result -= (float)bs * 0.1f;
        }
    } else if (category == 2) {
        // Path B: loop calling mid-level functions
        result = 0.0f;
        for (int j = 0; j < 3; j++) {
            float contribution = mid_weighted_sum(
                x + (float)j, y - (float)j, x * y, j);
            if (contribution > 20.0f) {
                result += 20.0f;
                continue;
            }
            result += contribution;
        }
    } else if (category == 3) {
        // Path C: nested conditions with function calls at each level
        int bs = mid_bit_score(idx * 3 + 1);
        if (bs == 0) {
            result = mid_scaled_harmonic(5, fabsf(x));
        } else if (bs < 5) {
            result = mid_weighted_sum(x, y, (float)bs, 0);
            if (result < 1.0f)
                result = 1.0f;
        } else {
            float h1 = mid_scaled_harmonic(bs, 1.0f);
            float h2 = mid_scaled_harmonic(bs / 2, 2.0f);
            result = fmaxf(h1, h2);
        }
    } else {
        // Path D: switch inside loop with function calls
        result = 0.0f;
        for (int step = 0; step < 4; step++) {
            float val;
            switch ((idx + step) % 4) {
                case 0:
                    val = mid_weighted_sum(x, y, 1.0f, step);
                    break;
                case 1:
                    val = (float)mid_bit_score(idx + step);
                    break;
                case 2:
                    val = mid_scaled_harmonic(step + 2, fabsf(x) + 0.1f);
                    break;
                default:
                    val = leaf_abs_clamp(x * y, 0.0f, 50.0f);
                    break;
            }
            result += val;
            if (result > 200.0f)
                break;
        }
    }

    return result;
}

__global__ void deep_call_kernel(float *out, const float *in_x,
                                  const float *in_y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = top_evaluate(i, in_x[i], in_y[i]);
    }
}

// --- Host references (same logic) ---
static float h_leaf_abs_clamp(float x, float lo, float hi) {
    float ax = fabsf(x);
    if (ax < lo) return lo; if (ax > hi) return hi; return ax;
}
static int h_leaf_popcount(unsigned int x) {
    int c = 0; while (x) { c += x & 1; x >>= 1; } return c;
}
static float h_leaf_harmonic(int n) {
    if (n <= 0) return 0.0f;
    float s = 0.0f; for (int i = 1; i <= n; i++) s += 1.0f / (float)i; return s;
}
static float h_mid_weighted_sum(float a, float b, float c, int mode) {
    float wa = h_leaf_abs_clamp(a, 0.1f, 10.0f);
    float wb = h_leaf_abs_clamp(b, 0.1f, 10.0f);
    float wc = h_leaf_abs_clamp(c, 0.1f, 10.0f);
    switch (mode % 3) {
        case 0: return wa + wb + wc;
        case 1: return wa * 0.5f + wb * 0.3f + wc * 0.2f;
        case 2: return fmaxf(wa, fmaxf(wb, wc)) - fminf(wa, fminf(wb, wc));
        default: return 0.0f;
    }
}
static int h_mid_bit_score(int x) {
    int pc = h_leaf_popcount((unsigned int)(x < 0 ? -x : x));
    int s = 0;
    for (int i = 0; i < pc; i++) s += (i % 2 == 0) ? i + 1 : -1;
    return s < 0 ? 0 : s;
}
static float h_mid_scaled_harmonic(int n, float scale) {
    if (scale <= 0.0f) return 0.0f;
    float h = h_leaf_harmonic(n);
    float r = h * scale;
    for (int i = 0; i < 5; i++) {
        if (r > 100.0f) return 100.0f;
        if (r < 0.01f) return 0.0f;
        r = r * 0.9f + 0.1f * h;
    }
    return r;
}
static float h_top_evaluate(int idx, float x, float y) {
    int cat = idx % 5;
    float result;
    if (cat < 2) {
        result = h_mid_weighted_sum(x, y, x + y, idx);
        int bs = h_mid_bit_score(idx);
        if (bs > 10) result += h_mid_scaled_harmonic(bs, 0.5f);
        else result -= (float)bs * 0.1f;
    } else if (cat == 2) {
        result = 0.0f;
        for (int j = 0; j < 3; j++) {
            float c = h_mid_weighted_sum(x + (float)j, y - (float)j, x * y, j);
            if (c > 20.0f) { result += 20.0f; continue; }
            result += c;
        }
    } else if (cat == 3) {
        int bs = h_mid_bit_score(idx * 3 + 1);
        if (bs == 0) {
            result = h_mid_scaled_harmonic(5, fabsf(x));
        } else if (bs < 5) {
            result = h_mid_weighted_sum(x, y, (float)bs, 0);
            if (result < 1.0f) result = 1.0f;
        } else {
            float h1 = h_mid_scaled_harmonic(bs, 1.0f);
            float h2 = h_mid_scaled_harmonic(bs / 2, 2.0f);
            result = fmaxf(h1, h2);
        }
    } else {
        result = 0.0f;
        for (int step = 0; step < 4; step++) {
            float val;
            switch ((idx + step) % 4) {
                case 0: val = h_mid_weighted_sum(x, y, 1.0f, step); break;
                case 1: val = (float)h_mid_bit_score(idx + step); break;
                case 2: val = h_mid_scaled_harmonic(step + 2, fabsf(x) + 0.1f); break;
                default: val = h_leaf_abs_clamp(x * y, 0.0f, 50.0f); break;
            }
            result += val;
            if (result > 200.0f) break;
        }
    }
    return result;
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)((i * 43 + 11) % 200) * 0.1f - 10.0f;
        h_y[i] = (float)((i * 71 + 23) % 200) * 0.1f - 10.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    deep_call_kernel<<<(N + 255) / 256, 256>>>(d_out, d_x, d_y, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_top_evaluate(i, h_x[i], h_y[i]);
        if (fabsf(h_out[i] - expected) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: 49_deep_call_tree i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_out);

    if (errors) { fprintf(stderr, "%d errors total\n", errors); return 1; }
    printf("PASS: 49_deep_call_tree (N=%d)\n", N);
    return 0;
}
