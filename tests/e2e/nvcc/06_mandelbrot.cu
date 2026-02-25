// 6. Mandelbrot set with __device__ iteration, 2D grid
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ int mandelbrot_iter(float cx, float cy, int max_iter) {
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (iter < max_iter) {
        float zx2 = zx * zx;
        float zy2 = zy * zy;
        if (zx2 + zy2 > 4.0f) break;
        float new_zx = zx2 - zy2 + cx;
        zy = 2.0f * zx * zy + cy;
        zx = new_zx;
        iter++;
    }
    return iter;
}

__global__ void mandelbrot_kernel(int *out, int width, int height,
                                  float x0, float y0, float dx, float dy,
                                  int max_iter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < width && py < height) {
        float cx = x0 + px * dx;
        float cy = y0 + py * dy;
        out[py * width + px] = mandelbrot_iter(cx, cy, max_iter);
    }
}

int main() {
    const int W = 64, H = 64;
    const int MAX_ITER = 100;
    const float x0 = -2.0f, y0 = -1.5f;
    const float dx = 3.0f / W, dy = 3.0f / H;
    size_t bytes = W * H * sizeof(int);

    int *h_out = (int *)malloc(bytes);
    int *d_out;
    cudaMalloc(&d_out, bytes);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    mandelbrot_kernel<<<grid, block>>>(d_out, W, H, x0, y0, dx, dy, MAX_ITER);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify known points
    // Center of set (-0.5, 0) should be max_iter (inside)
    int cx = (int)((-0.5f - x0) / dx);
    int cy = (int)((0.0f - y0) / dy);
    int center_val = h_out[cy * W + cx];

    // Corner (-2, -1.5) should escape quickly
    int corner_val = h_out[0];

    // Reference: compute a few on host
    int host_center = 0;
    {
        float zx = 0, zy = 0;
        float pcx = x0 + cx * dx, pcy = y0 + cy * dy;
        for (int i = 0; i < MAX_ITER; i++) {
            float zx2 = zx * zx, zy2 = zy * zy;
            if (zx2 + zy2 > 4.0f) break;
            float nzx = zx2 - zy2 + pcx;
            zy = 2.0f * zx * zy + pcy;
            zx = nzx;
            host_center++;
        }
    }

    cudaFree(d_out);

    if (center_val != host_center) {
        fprintf(stderr, "FAIL: 06_mandelbrot center: got %d expected %d\n",
                center_val, host_center);
        free(h_out);
        return 1;
    }
    if (corner_val >= MAX_ITER) {
        fprintf(stderr, "FAIL: 06_mandelbrot corner should escape (got %d)\n",
                corner_val);
        free(h_out);
        return 1;
    }

    free(h_out);
    printf("PASS: 06_mandelbrot (%dx%d, max_iter=%d)\n", W, H, MAX_ITER);
    return 0;
}
