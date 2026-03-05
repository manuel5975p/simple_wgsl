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
 *   @group(0) @binding(0) var<storage, read>       src: array<vec2<f32>>;
 *   @group(0) @binding(1) var<storage, read_write>  dst: array<vec2<f32>>;
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

#ifdef __cplusplus
}
#endif

#endif /* FFT_STOCKHAM_GEN_H */
