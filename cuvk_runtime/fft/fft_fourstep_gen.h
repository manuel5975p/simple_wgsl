#ifndef FFT_FOURSTEP_GEN_H
#define FFT_FOURSTEP_GEN_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generate twiddle multiply + transpose kernel.
 * Reads N1×N2 complex matrix (row-major, interleaved f32 pairs) from src,
 * multiplies by W_N^(dir×i×j), writes transposed N2×N1 matrix to dst.
 *
 * Bindings:
 *   @group(0) @binding(0) var<storage, read>       src
 *   @group(0) @binding(1) var<storage, read_write>  dst
 *
 * Dispatch: (ceil(n1*n2 / workgroup_size), batch_count, 1)
 *
 * direction: 1 = forward (angle = -2π·i·j/N), -1 = inverse (+2π·i·j/N)
 *
 * Returns: malloc'd WGSL source (caller frees), or NULL on error.
 */
char *gen_fft_twiddle_transpose(int n1, int n2, int direction, int workgroup_size);

/*
 * Generate plain transpose kernel (no twiddle).
 * Reads N1×N2, writes N2×N1. Same bindings and dispatch as above.
 */
char *gen_fft_transpose(int n1, int n2, int workgroup_size);

/*
 * Decompose N into N1 × N2 for four-step FFT.
 * Both factors ≤ max_sub_n and fused-friendly ({2,3,4,5,7,8,16} factors).
 * Prefers balanced splits (near √N).
 * If N ≤ max_sub_n and fused-friendly, sets n1=1, n2=N (no split needed).
 *
 * Returns: 1 on success (out_n1, out_n2 set), 0 if N can't be decomposed.
 */
int fft_fourstep_decompose(int n, int max_sub_n, int *out_n1, int *out_n2);

#ifdef __cplusplus
}
#endif

#endif
