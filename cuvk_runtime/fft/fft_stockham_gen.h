#ifndef FFT_STOCKHAM_GEN_H
#define FFT_STOCKHAM_GEN_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generate a WGSL compute shader for one Stockham FFT stage.
 *
 * The shader reads `radix` elements at stride `stride` from src,
 * applies inter-stage twiddle factors, performs a radix-N FFT with
 * baked intra-radix twiddles, and writes results in Stockham order to dst.
 *
 * Bindings:
 *   @group(0) @binding(0) var<storage, read>       src: SrcBuf { d: array<f32> };
 *   @group(0) @binding(1) var<storage, read_write>  dst: DstBuf { d: array<f32> };
 *
 * Dispatch: (ceil(n_total/radix / workgroup_size), batch_count, 1)
 *   gid.x = butterfly index within one FFT (0 .. n_total/radix - 1)
 *   gid.y = batch index
 *
 * Parameters:
 *   radix          - small FFT size (2..32)
 *   stride         - L = product of previous radices (source element stride)
 *   n_total        - overall FFT size (must be divisible by radix)
 *   direction      - 1 = forward, -1 = inverse
 *   workgroup_size - @workgroup_size(N), tunable per device
 *
 * Returns: malloc'd WGSL source string (caller frees), or NULL on error.
 */
char *gen_fft_stockham(int radix, int stride, int n_total,
                       int direction, int workgroup_size);

/*
 * Strided variant for multi-dimensional FFT.
 *
 * Additional parameters:
 *   element_stride - physical distance between consecutive logical FFT elements.
 *                    1 for contiguous (row-wise), M for column-wise on NxM data.
 *   batch_stride   - physical distance between start of consecutive FFTs.
 *                    n_total for contiguous, 1 for column-wise.
 *
 * gen_fft_stockham() is equivalent to gen_fft_stockham_strided() with
 * element_stride=1, batch_stride=n_total.
 */
char *gen_fft_stockham_strided(int radix, int stride, int n_total,
                               int direction, int workgroup_size,
                               int element_stride, int batch_stride);

/*
 * 2D-batched strided variant for 3D FFT.
 *
 * Adds a second batch dimension via gid.z:
 *   batch_offset = gid.y * batch_stride + gid.z * batch_stride2
 *
 * Dispatch: (ceil(n_total/radix / workgroup_size), batch_count, batch_count2)
 */
char *gen_fft_stockham_strided2(int radix, int stride, int n_total,
                                int direction, int workgroup_size,
                                int element_stride, int batch_stride,
                                int batch_stride2);

/*
 * Generate R2C post-processing shader.
 *
 * Takes N/2 complex values from a N/2-point C2C forward FFT and produces
 * N/2+1 complex output bins (the real-valued FFT spectrum).
 *
 * Bindings: same 2-buffer layout (src = C2C output, dst = R2C output).
 * Dispatch: (ceil((N/2+1) / workgroup_size), batch_count, 1)
 *
 * batch_stride: physical distance between consecutive N/2-point FFT results
 *               in the source, and N/2+1-point results in the destination.
 *               For 1D: src_batch_stride = N/2, dst_batch_stride = N/2+1.
 */
char *gen_fft_r2c_postprocess(int n, int workgroup_size);

/*
 * Generate C2R pre-processing shader.
 *
 * Takes N/2+1 complex frequency bins and produces N/2 complex values
 * ready for a N/2-point C2C inverse FFT.
 *
 * Bindings: same 2-buffer layout (src = frequency bins, dst = pre-processed).
 * Dispatch: (ceil((N/2) / workgroup_size), batch_count, 1)
 */
char *gen_fft_c2r_preprocess(int n, int workgroup_size);

/* ========================================================================== */
/* Extended API: planner-friendly multi-stage FFT                             */
/* ========================================================================== */

#define FFT_STOCKHAM_MAX_STAGES 20

/*
 * Factorize N into radices [2..max_radix], greedy largest-first.
 *
 *   max_radix - cap largest radix (2..32, 0 = default 32).
 *               Smaller max_radix = more stages, fewer registers per stage.
 *
 * NOTE: Currently factorization is greedy largest-first with a radix cap.
 * A future extension could accept an explicit radix sequence (e.g.
 * [8,8,8,8] for N=4096) giving the planner full control over the
 * decomposition, at the cost of a larger search space.
 *
 * Returns: number of stages, or 0 if N cannot be factorized.
 */
int fft_stockham_factorize(int n, int max_radix, int *radices);

/*
 * Query the number of stages for a given (N, max_radix) configuration.
 */
int fft_stockham_num_stages(int n, int max_radix);

/*
 * Compute dispatch_x for a given stage.
 *   n_total        - FFT size
 *   radix          - this stage's radix
 *   workgroup_size - threads per workgroup
 */
int fft_stockham_dispatch_x(int n_total, int radix, int workgroup_size);

#ifdef __cplusplus
}
#endif

#endif /* FFT_STOCKHAM_GEN_H */
