// 9. Multiple CUDA streams with async memcpy and kernel launches
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float process(float x, float scale) {
    return x * scale + sqrtf(fabsf(x));
}

__global__ void transform(float *out, const float *in, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = process(in[i], scale);
}

int main() {
    const int NSTREAMS = 4;
    const int CHUNK = 1024;
    const int N = NSTREAMS * CHUNK;
    const float SCALE = 2.5f;
    size_t total_bytes = N * sizeof(float);
    size_t chunk_bytes = CHUNK * sizeof(float);

    float *h_in = (float *)malloc(total_bytes);
    float *h_out = (float *)malloc(total_bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i % 200) * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, total_bytes);
    cudaMalloc(&d_out, total_bytes);

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamCreate(&streams[s]);

    // Launch work on each stream: copy in -> kernel -> copy out
    for (int s = 0; s < NSTREAMS; s++) {
        int offset = s * CHUNK;
        cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_bytes,
                        cudaMemcpyHostToDevice, streams[s]);
        transform<<<(CHUNK + 255) / 256, 256, 0, streams[s]>>>(
            d_out + offset, d_in + offset, SCALE, CHUNK);
        cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_bytes,
                        cudaMemcpyDeviceToHost, streams[s]);
    }

    // Sync all streams
    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamSynchronize(streams[s]);

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float expected = x * SCALE + sqrtf(fabsf(x));
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            fprintf(stderr, "FAIL: 09_streams i=%d got %f expected %f\n",
                    i, h_out[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamDestroy(streams[s]);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) return 1;
    printf("PASS: 09_streams (%d streams, N=%d)\n", NSTREAMS, N);
    return 0;
}
