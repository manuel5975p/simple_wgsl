/* Partial dot product: each block reduces its chunk, writes one partial sum.
   Host sums the partial results. */
extern "C" __global__ void dotPartial(float *A, float *B, float *partials, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread writes A[i]*B[i] to partials[i], host sums them.
       Simple approach (no shared memory reduction). */
    if (i < N) {
        partials[i] = A[i] * B[i];
    }
}
