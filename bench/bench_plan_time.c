#include "cufft.h"
#include "cuda.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

int main(void) {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, NULL, 0, dev);

    /* Warmup: plan+destroy to amortize one-time init */
    { cufftHandle h; cufftPlan1d(&h, 4, CUFFT_C2C, 1); cufftDestroy(h); }

    int batches[] = {1, 256, 16384};
    for (int j = 0; j < 3; j++) {
        int64_t t0 = now_ns();
        cufftHandle plan;
        cufftPlan1d(&plan, 128, CUFFT_C2C, batches[j]);
        int64_t t1 = now_ns();
        printf("N=128  batch=%-6d  %.1f ms\n", batches[j], (double)(t1 - t0) / 1e6);
        cufftDestroy(plan);
    }

    return 0;
}
