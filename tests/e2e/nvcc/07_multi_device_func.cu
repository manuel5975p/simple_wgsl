// 7. Multiple __device__ functions: vector math library
//    Tests deep __device__ call trees and struct-like multi-output
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float vec3_dot(float ax, float ay, float az,
                          float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ float vec3_length(float x, float y, float z) {
    return sqrtf(vec3_dot(x, y, z, x, y, z));
}

__device__ void vec3_normalize(float *ox, float *oy, float *oz,
                               float x, float y, float z) {
    float len = vec3_length(x, y, z);
    float inv = (len > 1e-8f) ? (1.0f / len) : 0.0f;
    *ox = x * inv;
    *oy = y * inv;
    *oz = z * inv;
}

__device__ float vec3_reflect_dot(float vx, float vy, float vz,
                                  float nx, float ny, float nz) {
    // reflect v around n: v - 2*dot(v,n)*n, then dot with n
    float d = vec3_dot(vx, vy, vz, nx, ny, nz);
    float rx = vx - 2.0f * d * nx;
    float ry = vy - 2.0f * d * ny;
    float rz = vz - 2.0f * d * nz;
    return vec3_dot(rx, ry, rz, nx, ny, nz);
}

// For each element i, compute the normalize + reflect_dot of two vectors
__global__ void vecOps(float *out,
                       const float *vx, const float *vy, const float *vz,
                       const float *nx, const float *ny, const float *nz,
                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float nnx, nny, nnz;
        vec3_normalize(&nnx, &nny, &nnz, nx[i], ny[i], nz[i]);
        out[i] = vec3_reflect_dot(vx[i], vy[i], vz[i], nnx, nny, nnz);
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_vx = (float *)malloc(bytes);
    float *h_vy = (float *)malloc(bytes);
    float *h_vz = (float *)malloc(bytes);
    float *h_nx = (float *)malloc(bytes);
    float *h_ny = (float *)malloc(bytes);
    float *h_nz = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_vx[i] = (float)(i % 7) - 3.0f;
        h_vy[i] = (float)(i % 11) - 5.0f;
        h_vz[i] = (float)(i % 5) - 2.0f;
        h_nx[i] = 1.0f;
        h_ny[i] = (float)(i % 3);
        h_nz[i] = 0.5f;
    }

    float *d_vx, *d_vy, *d_vz, *d_nx, *d_ny, *d_nz, *d_out;
    cudaMalloc(&d_vx, bytes); cudaMalloc(&d_vy, bytes); cudaMalloc(&d_vz, bytes);
    cudaMalloc(&d_nx, bytes); cudaMalloc(&d_ny, bytes); cudaMalloc(&d_nz, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nx, h_nx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ny, h_ny, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nz, h_nz, bytes, cudaMemcpyHostToDevice);

    vecOps<<<(N + 255) / 256, 256>>>(d_out, d_vx, d_vy, d_vz,
                                      d_nx, d_ny, d_nz, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify: reflect_dot(v, n_hat) == -dot(v, n_hat)
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float len = sqrtf(h_nx[i]*h_nx[i] + h_ny[i]*h_ny[i] + h_nz[i]*h_nz[i]);
        float inv = (len > 1e-8f) ? (1.0f / len) : 0.0f;
        float nnx = h_nx[i]*inv, nny = h_ny[i]*inv, nnz = h_nz[i]*inv;
        float d = h_vx[i]*nnx + h_vy[i]*nny + h_vz[i]*nnz;
        float expected = -d;  // reflect_dot always equals -dot(v, n_hat)
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            fprintf(stderr, "FAIL: 07_multi_device_func i=%d got %f expected %f\n",
                    i, h_out[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_nx); cudaFree(d_ny); cudaFree(d_nz);
    cudaFree(d_out);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_nx); free(h_ny); free(h_nz); free(h_out);

    if (errors) return 1;
    printf("PASS: 07_multi_device_func (N=%d)\n", N);
    return 0;
}
