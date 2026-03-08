#ifndef FFT_FUSED_GEN_H
#define FFT_FUSED_GEN_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generate a single-dispatch fused FFT compute shader using workgroup
 * shared memory.  All radix stages run in one kernel.
 *
 * Optimizations:
 *   - Twiddle LUT: precomputed twiddles from @binding(2)
 *   - Bank-conflict padding on shared memory for power-of-2 N.
 *   - Specialized radix-3/5/7/16 butterflies.
 *   - Intra-workgroup batching: multiple FFTs per workgroup for small N.
 *   - Rader's algorithm for prime radices (11, 13, 17, 19, 29, 31, ...).
 *
 * Bindings:
 *   @group(0) @binding(0) var<storage, read>       src { d: array<f32> }
 *   @group(0) @binding(1) var<storage, read_write>  dst { d: array<f32> }
 *   @group(0) @binding(2) var<storage, read>       lut { d: array<f32> }
 *                          (only present when fft_fused_lut_size() > 0)
 *
 * Dispatch: (ceil(batch_count / batch_per_wg), 1, 1)
 *   Caller must pad input/output buffers to batch_per_wg boundary.
 *
 * Parameters:
 *   n         - FFT size (must factor into supported radices/primes)
 *   direction - 1 = forward, -1 = inverse
 *
 * Returns: malloc'd WGSL source string (caller frees), or NULL on error.
 */
char *gen_fft_fused(int n, int direction);

/* Total workgroup size (threads per workgroup). */
int fft_fused_workgroup_size(int n);

/* Number of FFTs computed per workgroup (>=1, larger for small N). */
int fft_fused_batch_per_wg(int n);

/* Number of vec2<f32> twiddle+kernel entries needed (0 means no LUT). */
int fft_fused_lut_size(int n, int direction);

/* Compute twiddle LUT on the host.  Returns malloc'd float array with
 * 2 * fft_fused_lut_size() floats (cos, sin pairs).  Caller frees. */
float *fft_fused_compute_lut(int n, int direction);

/*
 * Extended API: explicit max_radix control for planner convergence.
 *
 * max_radix controls the factorization strategy:
 *   0      - auto (uses built-in heuristic)
 *   2..16  - cap radix to this value; smaller = more threads per FFT
 *            (better memory coalescing, more barriers)
 *
 * The planner should try max_radix = {2, 4, 8, 16} for each N,
 * benchmark each, and pick the fastest for the target GPU.
 */
char *gen_fft_fused_ex(int n, int direction, int max_radix);
int   fft_fused_workgroup_size_ex(int n, int max_radix);
int   fft_fused_batch_per_wg_ex(int n, int max_radix);
int   fft_fused_lut_size_ex(int n, int direction, int max_radix);
float *fft_fused_compute_lut_ex(int n, int direction, int max_radix);

#ifdef __cplusplus
}
#endif

#endif /* FFT_FUSED_GEN_H */
