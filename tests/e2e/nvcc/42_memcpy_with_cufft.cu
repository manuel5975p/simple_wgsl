#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>

int main() {
    const int N = 64;
    cufftComplex h_data[N];
    memset(h_data, 0, sizeof(h_data));
    h_data[0].x = 1.0f;

    cufftComplex *d_data;
    cudaMalloc(&d_data, N * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    cufftComplex h_result[N];
    cudaMemcpy(h_result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    printf("h_result[0] = (%f, %f)\n", h_result[0].x, h_result[0].y);

    if (h_result[0].x < 0.9f || h_result[0].x > 1.1f) {
        printf("FAIL: expected ~1.0, got %f\n", h_result[0].x);
        cufftDestroy(plan);
        cudaFree(d_data);
        return 1;
    }
    printf("PASS\n");
    cufftDestroy(plan);
    cudaFree(d_data);
    return 0;
}
