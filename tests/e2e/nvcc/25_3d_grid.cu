// 25. 3D grid/block indexing: threadIdx.z, blockIdx.z, 3D stencil
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void stencil_3d(float *out, const float *in,
                           int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= 1 && x < nx-1 && y >= 1 && y < ny-1 && z >= 1 && z < nz-1) {
        int idx = z * ny * nx + y * nx + x;
        float val = in[idx] * 6.0f
            - in[idx - 1] - in[idx + 1]
            - in[idx - nx] - in[idx + nx]
            - in[idx - ny*nx] - in[idx + ny*nx];
        out[idx] = val;
    }
}

int main() {
    const int NX = 32, NY = 32, NZ = 32;
    const int TOTAL = NX * NY * NZ;
    size_t bytes = TOTAL * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < TOTAL; i++)
        h_in[i] = (float)(i % 100) * 0.01f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bytes);

    dim3 block(8, 8, 4);
    dim3 grid((NX + 7) / 8, (NY + 7) / 8, (NZ + 3) / 4);
    stencil_3d<<<grid, block>>>(d_out, d_in, NX, NY, NZ);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int z = 1; z < NZ-1; z++) {
        for (int y = 1; y < NY-1; y++) {
            for (int x = 1; x < NX-1; x++) {
                int idx = z * NY * NX + y * NX + x;
                float expected = h_in[idx] * 6.0f
                    - h_in[idx-1] - h_in[idx+1]
                    - h_in[idx-NX] - h_in[idx+NX]
                    - h_in[idx-NY*NX] - h_in[idx+NY*NX];
                if (fabsf(h_out[idx] - expected) > 1e-4f) {
                    if (errors == 0)
                        fprintf(stderr, "FAIL: 25_3d_grid at (%d,%d,%d) got %f expected %f\n",
                                x, y, z, h_out[idx], expected);
                    errors++;
                }
            }
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 25_3d_grid (%dx%dx%d stencil)\n", NX, NY, NZ);
    return 0;
}
