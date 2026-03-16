/*
 * bench_cufft_2d_events.cu - 2D cuFFT batch=1, CUDA events (GPU-only time)
 */
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d at %s:%d\n", (int)e, __FILE__, __LINE__); exit(1); } } while(0)
#define WARMUP 10
#define ITERS 50

static double median(double *a, int n) { std::sort(a, a+n); return a[n/2]; }

static void bench(int nx, int ny) {
    size_t bytes = (size_t)nx * ny * sizeof(cufftComplex);
    cufftComplex *d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes));
    cufftHandle plan; CK(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));

    for (int w = 0; w < WARMUP; w++) CK(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0)); CK(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float pilot; CK(cudaEventElapsedTime(&pilot, t0, t1));
    int reps = (pilot > 0.001f) ? (int)(50.0f / pilot) : 50000;
    if (reps < 10) reps = 10; if (reps > 100000) reps = 100000;

    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        CK(cudaEventRecord(t0));
        for (int r = 0; r < reps; r++) CK(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); times[it] = (double)ms / reps;
    }
    double ms = median(times, ITERS);
    int l2x = 0, l2y = 0; { int t=nx; while(t>1){t>>=1;l2x++;} } { int t=ny; while(t>1){t>>=1;l2y++;} }
    double flops = 5.0*ny*l2y*nx + 5.0*nx*l2x*ny;
    printf("%-4d  %-4d  %6d  %10.4f  %10.2f\n", nx, ny, reps, ms*1000, flops/(ms*1e6));
    fflush(stdout);
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    CK(cufftDestroy(plan)); CK(cudaFree(d));
}

static void bench_r2c(int nx, int ny) {
    size_t in_bytes = (size_t)nx * ny * sizeof(cufftReal);
    int padded_y = ny / 2 + 1;
    size_t out_bytes = (size_t)nx * padded_y * sizeof(cufftComplex);
    cufftReal *d_in; CK(cudaMalloc(&d_in, in_bytes)); CK(cudaMemset(d_in, 0, in_bytes));
    cufftComplex *d_out; CK(cudaMalloc(&d_out, out_bytes));
    cufftHandle plan; CK(cufftPlan2d(&plan, nx, ny, CUFFT_R2C));

    for (int w = 0; w < WARMUP; w++) CK(cufftExecR2C(plan, d_in, d_out));
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0)); CK(cufftExecR2C(plan, d_in, d_out));
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float pilot; CK(cudaEventElapsedTime(&pilot, t0, t1));
    int reps = (pilot > 0.001f) ? (int)(50.0f / pilot) : 50000;
    if (reps < 10) reps = 10; if (reps > 100000) reps = 100000;

    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        CK(cudaEventRecord(t0));
        for (int r = 0; r < reps; r++) CK(cufftExecR2C(plan, d_in, d_out));
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); times[it] = (double)ms / reps;
    }
    double ms = median(times, ITERS);
    int l2x = 0, l2y = 0; { int t=nx; while(t>1){t>>=1;l2x++;} } { int t=ny; while(t>1){t>>=1;l2y++;} }
    double flops = 2.5*ny*l2y*nx + 5.0*nx*l2x*padded_y;
    printf("%-4d  %-4d  %6d  %10.4f  %10.2f\n", nx, ny, reps, ms*1000, flops/(ms*1e6));
    fflush(stdout);
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    CK(cufftDestroy(plan)); CK(cudaFree(d_in)); CK(cudaFree(d_out));
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Real cuFFT 2D, batch=1, CUDA events -- %s\n\n", p.name);
    printf("%-4s  %-4s  %6s  %10s  %10s\n", "NX", "NY", "reps", "us/fft", "GFLOP/s");
    printf("----  ----  ------  ----------  ----------\n");
    struct { int x,y; } c[] = {{64,64},{128,128},{256,256},{256,128},{128,256},{512,128},{128,512},{512,256},{256,512}};
    for (int i = 0; i < 9; i++) bench(c[i].x, c[i].y);

    printf("\n  [R2C]\n");
    struct { int x,y; } r[] = {{64,64},{128,128},{256,256},{512,128},{128,512},{256,128},{128,256},{512,256},{256,512}};
    for (int i = 0; i < 9; i++) bench_r2c(r[i].x, r[i].y);
    printf("\n"); return 0;
}
