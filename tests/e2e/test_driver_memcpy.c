#include <stdio.h>
#include <string.h>
#include "cuvk_internal.h"

int main() {
    CUresult r;
    r = cuInit(0);
    if (r != CUDA_SUCCESS) { printf("cuInit failed: %d\n", r); return 1; }

    CUdevice dev;
    r = cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) { printf("cuDeviceGet failed: %d\n", r); return 1; }

    CUcontext ctx;
    r = cuCtxCreate_v4(&ctx, NULL, 0, dev);
    if (r != CUDA_SUCCESS) { printf("cuCtxCreate failed: %d\n", r); return 1; }

    CUdeviceptr d_buf;
    size_t size = 512;
    r = cuMemAlloc_v2(&d_buf, size);
    if (r != CUDA_SUCCESS) { printf("cuMemAlloc failed: %d\n", r); return 1; }
    printf("Allocated device buffer at 0x%llx, size %zu\n",
           (unsigned long long)d_buf, size);

    float h_src[128];
    memset(h_src, 0, sizeof(h_src));
    h_src[0] = 1.0f;
    h_src[1] = 0.0f;

    r = cuMemcpyHtoD_v2(d_buf, h_src, size);
    if (r != CUDA_SUCCESS) { printf("cuMemcpyHtoD failed: %d\n", r); return 1; }
    printf("H2D: uploaded [%f %f ...]\n", h_src[0], h_src[1]);

    float h_dst[128];
    memset(h_dst, 0xAA, sizeof(h_dst));
    r = cuMemcpyDtoH_v2(h_dst, d_buf, size);
    if (r != CUDA_SUCCESS) { printf("cuMemcpyDtoH failed: %d\n", r); return 1; }
    printf("D2H: got [%f %f %f %f]\n", h_dst[0], h_dst[1], h_dst[2], h_dst[3]);

    if (h_dst[0] != 1.0f) {
        printf("FAIL: expected 1.0, got %f\n", h_dst[0]);
        return 1;
    }
    printf("PASS: driver API memcpy roundtrip\n");

    cuMemFree_v2(d_buf);
    cuCtxDestroy_v2(ctx);
    return 0;
}
