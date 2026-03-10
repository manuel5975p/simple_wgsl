#ifndef FFT_2D_GEN_H
#define FFT_2D_GEN_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generate a single-dispatch 2D fused FFT compute shader.
 * Performs row FFTs (ny-point) then column FFTs (nx-point) entirely in
 * workgroup shared memory. One workgroup per 2D FFT.
 *
 * Constraint: nx * ny must fit in shared memory.
 *
 * Bindings:
 *   @group(0) @binding(0) var<storage, read>       src { d: array<f32> }
 *   @group(0) @binding(1) var<storage, read_write>  dst { d: array<f32> }
 *   @group(0) @binding(2) var<storage, read>       lut { d: array<f32> }
 *                          (only when lut_size > 0)
 *
 * Data layout: row-major, nx rows x ny columns, interleaved f32 (re, im).
 * Dispatch: (batch_count, 1, 1)
 *
 * max_radix: 0 = auto, 2..16 = cap radix for both axes.
 */
char *gen_fft_2d_fused(int nx, int ny, int direction, int max_radix);
int fft_2d_fused_workgroup_size(int nx, int ny, int max_radix);
int fft_2d_fused_lut_size(int nx, int ny, int direction, int max_radix);
float *fft_2d_fused_compute_lut(int nx, int ny, int direction, int max_radix);

/*
 * Looped variant: reads/writes binding(0) in a loop.
 * Repeat count is read from binding(2) element 0 (bitcast f32→u32).
 * For repeated in-place FFTs without per-dispatch overhead.
 * Binding 0 is read_write.  Data stays in shared memory across iterations.
 */
char *gen_fft_2d_fused_looped(int nx, int ny, int direction, int max_radix);

/*
 * Generate a tiled matrix transpose compute shader.
 * Transposes nx x ny row-major complex matrix to ny x nx.
 * Uses tile_dim x tile_dim shared-memory tiles with +1 padding.
 *
 * Bindings: same 2-buffer layout.
 * Dispatch: (ceil(ny / tile_dim), ceil(nx / tile_dim), batch_count)
 */
char *gen_transpose_tiled(int nx, int ny, int tile_dim);
int transpose_tiled_workgroup_size(int tile_dim);

#ifdef __cplusplus
}
#endif
#endif
