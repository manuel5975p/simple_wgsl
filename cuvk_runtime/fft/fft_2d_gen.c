/*
 * fft_2d_gen.c - Generate 2D fused FFT and tiled transpose WGSL shaders
 *
 * 2D fused FFT: row FFTs (ny-point) then column FFTs (nx-point) entirely
 * in workgroup shared memory. One workgroup per 2D FFT.
 *
 * Tiled transpose: shared-memory tiled transpose with +1 padding to
 * avoid bank conflicts.
 */

#include "fft_2d_gen.h"
#include "fft_strbuf.h"
#include "fft_butterfly.h"
#include "fft_bda.h"

#include <math.h>
#include <stdlib.h>

#define MAX_SHARED_BYTES 32768

/* ========================================================================== */
/* Tiled transpose                                                            */
/* ========================================================================== */

int transpose_tiled_workgroup_size(int tile_dim) {
  return tile_dim * tile_dim;
}

char *gen_transpose_tiled(int nx, int ny, int tile_dim) {
  if (nx < 1 || ny < 1 || tile_dim < 2) return NULL;
  if ((tile_dim & (tile_dim - 1)) != 0) return NULL; /* must be po2 */

  int padded = tile_dim * (tile_dim + 1);
  int wg_size = tile_dim * tile_dim;

  StrBuf sb;
  sb_init_cap(&sb, 8192);

  sb_emit_bda_src_dst(&sb, 0);
  sb_printf(&sb, "\n");

  sb_printf(&sb, "var<workgroup> tile_re: array<f32, %d>;\n", padded);
  sb_printf(&sb, "var<workgroup> tile_im: array<f32, %d>;\n\n", padded);

  sb_printf(&sb, "@compute @workgroup_size(%d)\n", wg_size);
  sb_printf(&sb, "fn main(\n");
  sb_printf(&sb, "  @builtin(local_invocation_id) lid: vec3<u32>,\n");
  sb_printf(&sb, "  @builtin(workgroup_id) wid: vec3<u32>\n");
  sb_printf(&sb, ") {\n");
  sb_printf(&sb, "  let lx: u32 = lid.x %% %uu;\n", tile_dim);
  sb_printf(&sb, "  let ly: u32 = lid.x / %uu;\n", tile_dim);
  sb_printf(&sb, "  let batch_off: u32 = wid.z * %uu;\n", nx * ny * 2);
  sb_printf(&sb, "\n");

  /* Load: src is nx rows x ny cols */
  sb_printf(&sb, "  let src_row: u32 = wid.y * %uu + ly;\n", tile_dim);
  sb_printf(&sb, "  let src_col: u32 = wid.x * %uu + lx;\n", tile_dim);
  sb_printf(&sb, "  if (src_row < %uu && src_col < %uu) {\n", nx, ny);
  sb_printf(&sb, "    let si: u32 = batch_off + (src_row * %uu + src_col) * 2u;\n", ny);
  sb_printf(&sb, "    tile_re[ly * %uu + lx] = src.d[si];\n", tile_dim + 1);
  sb_printf(&sb, "    tile_im[ly * %uu + lx] = src.d[si + 1u];\n", tile_dim + 1);
  sb_printf(&sb, "  }\n");
  sb_printf(&sb, "  workgroupBarrier();\n\n");

  /* Write transposed: dst is ny rows x nx cols */
  sb_printf(&sb, "  let dst_row: u32 = wid.x * %uu + ly;\n", tile_dim);
  sb_printf(&sb, "  let dst_col: u32 = wid.y * %uu + lx;\n", tile_dim);
  sb_printf(&sb, "  if (dst_row < %uu && dst_col < %uu) {\n", ny, nx);
  sb_printf(&sb, "    let di: u32 = batch_off + (dst_row * %uu + dst_col) * 2u;\n", nx);
  sb_printf(&sb, "    dst.d[di] = tile_re[lx * %uu + ly];\n", tile_dim + 1);
  sb_printf(&sb, "    dst.d[di + 1u] = tile_im[lx * %uu + ly];\n", tile_dim + 1);
  sb_printf(&sb, "  }\n");

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* ========================================================================== */
/* 2D fused FFT helpers                                                       */
/* ========================================================================== */

/* threads_per_fft is now wg_per_fft_mr in fft_butterfly.h */

/* LUT size for one axis (same as lut_size_mr in fft_fused_gen.c but without
 * the direct-path check since we always use shared-memory stages for 2D). */
static int axis_lut_size(int n, int max_r) {
  int radices[MAX_STAGES];
  int ns = factorize_mr(n, radices, max_r);
  if (ns == 0) return 0;

  int total = 0;
  int stride = 1;
  for (int s = 0; s < ns; s++) {
    int R = radices[s];
    if (stride > 1)
      total += (R - 1) * stride;
    stride *= R;
  }
  /* Rader kernel entries */
  for (int s = 0; s < ns; s++) {
    int R = radices[s];
    if (is_prime(R) && rader_supported(R))
      total += R - 1;
  }
  return total;
}

/* Compute LUT for one axis. Returns malloc'd float array (2 floats per entry). */
static float *axis_compute_lut(int n, int direction, int max_r) {
  int radices[MAX_STAGES];
  int ns = factorize_mr(n, radices, max_r);
  if (ns == 0) return NULL;

  int total = axis_lut_size(n, max_r);
  if (total == 0) return NULL;

  float *lut = (float *)calloc((size_t)total * 2, sizeof(float));
  int idx = 0;
  int stride = 1;

  for (int s = 0; s < ns; s++) {
    int R = radices[s];
    if (stride > 1) {
      for (int k = 1; k < R; k++) {
        for (int pos = 0; pos < stride; pos++) {
          double angle = (double)direction * -2.0 * M_PI * k * pos /
                         (double)(stride * R);
          lut[idx++] = (float)cos(angle);
          lut[idx++] = (float)sin(angle);
        }
      }
    }
    stride *= R;
  }

  /* Rader kernels */
  for (int s = 0; s < ns; s++) {
    int p = radices[s];
    if (!is_prime(p) || !rader_supported(p)) continue;

    int M = p - 1;
    int g = primitive_root(p);

    double *b_re = (double *)calloc(M, sizeof(double));
    double *b_im = (double *)calloc(M, sizeof(double));

    int *g_pow = (int *)malloc(M * sizeof(int));
    g_pow[0] = 1;
    for (int t = 1; t < M; t++)
      g_pow[t] = (int)((long long)g_pow[t - 1] * g % p);

    for (int t = 0; t < M; t++) {
      int neg_t = (M - t) % M;
      int exp_val = g_pow[neg_t];
      double angle = (double)direction * -2.0 * M_PI * exp_val / (double)p;
      b_re[t] = cos(angle);
      b_im[t] = sin(angle);
    }
    free(g_pow);

    /* Host DFT of length M */
    double *B_re = (double *)calloc(M, sizeof(double));
    double *B_im = (double *)calloc(M, sizeof(double));
    for (int k = 0; k < M; k++) {
      double re = 0, im = 0;
      for (int j = 0; j < M; j++) {
        double a = -2.0 * M_PI * k * j / M;
        double wr = cos(a), wi = sin(a);
        re += b_re[j] * wr - b_im[j] * wi;
        im += b_re[j] * wi + b_im[j] * wr;
      }
      B_re[k] = re;
      B_im[k] = im;
    }
    free(b_re);
    free(b_im);

    for (int k = 0; k < M; k++) {
      lut[idx++] = (float)B_re[k];
      lut[idx++] = (float)B_im[k];
    }
    free(B_re);
    free(B_im);
  }

  return lut;
}

/* ========================================================================== */
/* 2D fused FFT public queries                                                */
/* ========================================================================== */

int fft_2d_fused_workgroup_size(int nx, int ny, int max_radix) {
  int mr_row = effective_max_r(ny, max_radix);
  int mr_col = effective_max_r(nx, max_radix);

  int wpf_row = wg_per_fft_mr(ny, mr_row);
  int wpf_col = wg_per_fft_mr(nx, mr_col);
  if (wpf_row == 0 || wpf_col == 0) return 0;

  int t_rows = nx * wpf_row;
  int t_cols = ny * wpf_col;
  int wg = t_rows > t_cols ? t_rows : t_cols;
  return wg;
}

int fft_2d_fused_lut_size(int nx, int ny, int direction, int max_radix) {
  (void)direction;
  int mr_row = effective_max_r(ny, max_radix);
  int mr_col = effective_max_r(nx, max_radix);
  return axis_lut_size(ny, mr_row) + axis_lut_size(nx, mr_col);
}

float *fft_2d_fused_compute_lut(int nx, int ny, int direction, int max_radix) {
  int mr_row = effective_max_r(ny, max_radix);
  int mr_col = effective_max_r(nx, max_radix);

  int row_lut = axis_lut_size(ny, mr_row);
  int col_lut = axis_lut_size(nx, mr_col);
  int total = row_lut + col_lut;
  if (total == 0) return NULL;

  float *lut = (float *)calloc((size_t)total * 2, sizeof(float));

  /* Row twiddles first */
  if (row_lut > 0) {
    float *row = axis_compute_lut(ny, direction, mr_row);
    if (row) {
      memcpy(lut, row, (size_t)row_lut * 2 * sizeof(float));
      free(row);
    }
  }

  /* Column twiddles after */
  if (col_lut > 0) {
    float *col = axis_compute_lut(nx, direction, mr_col);
    if (col) {
      memcpy(lut + row_lut * 2, col, (size_t)col_lut * 2 * sizeof(float));
      free(col);
    }
  }

  return lut;
}

/* ========================================================================== */
/* Stage emission helper for one axis of the 2D FFT                           */
/* ========================================================================== */

/*
 * Emit Stockham FFT stages for one axis operating in 2D shared memory.
 *
 * n       = FFT size along this axis
 * n_ffts  = number of FFTs (= dimension along other axis)
 * wpf     = threads per FFT
 * max_r   = max radix for this axis
 * direction = 1 or -1
 * axis_stride = element stride in shared memory between consecutive elements
 *               along this axis:
 *   row FFTs:  axis_stride = 1      (consecutive elements along row)
 *   col FFTs:  axis_stride = padded_ny  (stride between rows)
 * cross_stride = stride between different FFTs in shared memory:
 *   row FFTs:  cross_stride = padded_ny (each row offset)
 *   col FFTs:  cross_stride = 1      (each column offset)
 * lut_base_offset = starting index in the LUT for this axis's twiddles
 * sb = string buffer
 *
 * Assumes variables already declared:
 *   t = lid.x (thread index within workgroup)
 */
static void emit_fft_stages_2d(StrBuf *sb, int n, int n_ffts, int wpf,
                                int max_r, int direction,
                                int axis_stride, int cross_stride,
                                int lut_base_offset, int has_lut,
                                int axis_id /* 0=row,1=col for unique names */) {
  int radices[MAX_STAGES];
  int n_stages = factorize_mr(n, radices, max_r);
  if (n_stages == 0) return;

  int max_radix = 0;
  for (int i = 0; i < n_stages; i++)
    if (radices[i] > max_radix) max_radix = radices[i];

  /* Max registers per butterfly (for Rader we need extra) */
  int max_regs_per_bf = max_radix;
  for (int s = 0; s < n_stages; s++) {
    int R = radices[s];
    if (is_prime(R) && rader_supported(R)) {
      int need = R + (R - 1);
      if (need > max_regs_per_bf) max_regs_per_bf = need;
    }
  }

  int epr = n / wpf; /* elements per thread */

  /* Compute LUT offsets for stage twiddles */
  int lut_offsets[MAX_STAGES];
  int lut_acc = 0;
  {
    int stride = 1;
    for (int s = 0; s < n_stages; s++) {
      lut_offsets[s] = lut_base_offset + lut_acc;
      int R = radices[s];
      if (stride > 1)
        lut_acc += (R - 1) * stride;
      stride *= R;
    }
  }

  /* Rader kernel LUT offsets */
  int rader_offsets[MAX_STAGES];
  {
    int roff = lut_base_offset + lut_acc;
    for (int s = 0; s < n_stages; s++) {
      rader_offsets[s] = roff;
      int R = radices[s];
      if (is_prime(R) && rader_supported(R))
        roff += R - 1;
    }
  }

  int total_threads = n_ffts * wpf;

  const char *ax = (axis_id == 0) ? "row" : "col";

  sb_printf(sb, "\n  // %s FFTs: %d-point x %d\n", ax, n, n_ffts);
  sb_printf(sb, "  {\n");

  /* Thread assignment for this axis */
  sb_printf(sb, "    let %s_active: bool = t < %uu;\n", ax, total_threads);
  sb_printf(sb, "    let %s_fft_id: u32 = t / %uu;\n", ax, wpf);
  sb_printf(sb, "    let %s_fft_t: u32 = t %% %uu;\n", ax, wpf);

  /* Stockham stages */
  int stride = 1;
  for (int stage = 0; stage < n_stages; stage++) {
    int R = radices[stage];
    int n_bf = n / R;
    int bpt = (n_bf + wpf - 1) / wpf;
    int read_stride = n / R;
    int total_regs = bpt * max_regs_per_bf;

    sb_printf(sb, "\n    // %s stage %d: radix-%d, stride=%d\n", ax, stage, R, stride);

    for (int i = 0; i < total_regs; i++)
      sb_printf(sb, "    var %s_v%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", ax, i);

    /* Phase 1: Read from shared memory */
    for (int b = 0; b < bpt; b++) {
      int rb = b * max_regs_per_bf;
      sb_printf(sb, "    {\n");
      if (bpt == 1)
        sb_printf(sb, "      let bf: u32 = %s_fft_t;\n", ax);
      else
        sb_printf(sb, "      let bf: u32 = %s_fft_t * %uu + %uu;\n", ax, bpt, b);

      /* Always guard: some threads may be idle in this phase */
      sb_printf(sb, "      if (%s_active && bf < %uu) {\n", ax, n_bf);

      if (stride == 1) {
        for (int k = 0; k < R; k++) {
          sb_printf(sb, "        { let idx: u32 = %s_fft_id * %uu + (bf + %uu) * %uu;\n",
                    ax, cross_stride, k * read_stride, axis_stride);
          sb_printf(sb, "          %s_v%d = vec2<f32>(s_re[idx], s_im[idx]); }\n",
                    ax, rb + k);
        }
      } else {
        sb_printf(sb, "        let grp: u32 = bf / %uu;\n", stride);
        sb_printf(sb, "        let pos: u32 = bf %% %uu;\n", stride);
        for (int k = 0; k < R; k++) {
          sb_printf(sb, "        { let idx: u32 = %s_fft_id * %uu + "
                    "(grp * %uu + pos + %uu) * %uu;\n",
                    ax, cross_stride,
                    stride, k * read_stride, axis_stride);
          sb_printf(sb, "          %s_v%d = vec2<f32>(s_re[idx], s_im[idx]); }\n",
                    ax, rb + k);
        }
      }

      sb_printf(sb, "      }\n");
      sb_printf(sb, "    }\n");
    }

    sb_printf(sb, "    workgroupBarrier();\n");

    /* Phase 2: Twiddle + butterfly + write back */
    for (int b = 0; b < bpt; b++) {
      int rb = b * max_regs_per_bf;
      sb_printf(sb, "    {\n");
      if (bpt == 1)
        sb_printf(sb, "      let bf: u32 = %s_fft_t;\n", ax);
      else
        sb_printf(sb, "      let bf: u32 = %s_fft_t * %uu + %uu;\n", ax, bpt, b);

      sb_printf(sb, "      if (%s_active && bf < %uu) {\n", ax, n_bf);

      if (stride > 1) {
        sb_printf(sb, "        let grp: u32 = bf / %uu;\n", stride);
        sb_printf(sb, "        let pos: u32 = bf %% %uu;\n", stride);
      }

      /* Twiddles */
      if (stride > 1 && has_lut) {
        for (int k = 1; k < R; k++) {
          int lut_base = lut_offsets[stage] + (k - 1) * stride;
          sb_printf(sb, "        { let tw: vec2<f32> = vec2<f32>("
                    "lut.d[%uu + pos * 2u], lut.d[%uu + pos * 2u + 1u]);\n",
                    lut_base * 2, lut_base * 2);
          sb_printf(sb, "          %s_v%d = vec2<f32>("
                    "%s_v%d.x*tw.x - %s_v%d.y*tw.y, "
                    "%s_v%d.x*tw.y + %s_v%d.y*tw.x); }\n",
                    ax, rb + k, ax, rb + k, ax, rb + k,
                    ax, rb + k, ax, rb + k);
        }
      } else if (stride > 1) {
        /* Compute twiddles inline */
        for (int k = 1; k < R; k++) {
          double tw_angle = (double)direction * -2.0 * M_PI * k / (double)(stride * R);
          sb_printf(sb, "        { let tw_a: f32 = ");
          sb_float(sb, tw_angle);
          sb_printf(sb, " * f32(pos);\n");
          sb_printf(sb, "          let tw: vec2<f32> = vec2<f32>(cos(tw_a), sin(tw_a));\n");
          sb_printf(sb, "          %s_v%d = vec2<f32>("
                    "%s_v%d.x*tw.x - %s_v%d.y*tw.y, "
                    "%s_v%d.x*tw.y + %s_v%d.y*tw.x); }\n",
                    ax, rb + k, ax, rb + k, ax, rb + k,
                    ax, rb + k, ax, rb + k);
        }
      }

      /* Butterfly */
      {
        /* We need to use a temporary prefix for the butterfly emitter.
         * Copy from ax_v registers to temp "v" registers, butterfly, copy back.
         * Actually, emit_radix_butterfly uses a prefix string directly, so
         * we can create the right prefix. But it expects prefix like "v" and
         * constructs "v0", "v1", etc. We need "row_v0", "col_v0".
         * Since emit_radix_butterfly uses printf with "%s%d", we can use
         * "row_v" or "col_v" as the prefix. */
        char bf_prefix[16];
        snprintf(bf_prefix, sizeof(bf_prefix), "%s_v", ax);
        emit_radix_butterfly(sb, R, direction, bf_prefix, rb,
                             axis_id * 1000 + stage * 100 + b,
                             rader_offsets[stage]);
      }

      /* Stockham write back to shared */
      if (stride == 1) {
        for (int k = 0; k < R; k++) {
          sb_printf(sb, "        { let idx: u32 = %s_fft_id * %uu + "
                    "(bf * %uu + %uu) * %uu;\n",
                    ax, cross_stride, R, k, axis_stride);
          sb_printf(sb, "          s_re[idx] = %s_v%d.x; s_im[idx] = %s_v%d.y; }\n",
                    ax, rb + k, ax, rb + k);
        }
      } else {
        for (int k = 0; k < R; k++) {
          sb_printf(sb, "        { let idx: u32 = %s_fft_id * %uu + "
                    "(grp * %uu + pos + %uu) * %uu;\n",
                    ax, cross_stride, stride * R, k * stride, axis_stride);
          sb_printf(sb, "          s_re[idx] = %s_v%d.x; s_im[idx] = %s_v%d.y; }\n",
                    ax, rb + k, ax, rb + k);
        }
      }

      sb_printf(sb, "      }\n");
      sb_printf(sb, "    }\n");
    }

    sb_printf(sb, "    workgroupBarrier();\n");
    stride *= R;
  }

  sb_printf(sb, "  }\n");
  (void)epr;
}

/* ========================================================================== */
/* 2D fused FFT generator                                                     */
/* ========================================================================== */

char *gen_fft_2d_fused(int nx, int ny, int direction, int max_radix) {
  if (nx < 2 || ny < 2) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  int mr_row = effective_max_r(ny, max_radix);
  int mr_col = effective_max_r(nx, max_radix);

  /* Factorize both axes */
  int row_radices[MAX_STAGES], col_radices[MAX_STAGES];
  int row_stages = factorize_mr(ny, row_radices, mr_row);
  int col_stages = factorize_mr(nx, col_radices, mr_col);
  if (row_stages == 0 || col_stages == 0) return NULL;

  /* Compute thread counts */
  int wpf_row = wg_per_fft_mr(ny, mr_row);
  int wpf_col = wg_per_fft_mr(nx, mr_col);
  if (wpf_row == 0 || wpf_col == 0) return NULL;

  int t_rows = nx * wpf_row; /* threads needed for row phase */
  int t_cols = ny * wpf_col; /* threads needed for col phase */
  int wg_size = t_rows > t_cols ? t_rows : t_cols;

  /* Check shared memory fits */
  int total_elems = nx * ny;
  if (total_elems * 8 > MAX_SHARED_BYTES) return NULL; /* 8 = 2 arrays x 4 bytes */

  /* LUT */
  int row_lut = axis_lut_size(ny, mr_row);
  int col_lut = axis_lut_size(nx, mr_col);
  int has_lut = (row_lut + col_lut) > 0;

  StrBuf sb;
  sb_init_cap(&sb, 131072);

  /* Bindings */
  sb_emit_bda_src_dst(&sb, has_lut);
  sb_printf(&sb, "\n");

  /* Shared memory: nx rows x ny cols, no padding for simplicity in 2D */
  sb_printf(&sb, "var<workgroup> s_re: array<f32, %d>;\n", total_elems);
  sb_printf(&sb, "var<workgroup> s_im: array<f32, %d>;\n\n", total_elems);

  /* Entry point */
  sb_printf(&sb, "@compute @workgroup_size(%d)\n", wg_size);
  sb_printf(&sb, "fn main(\n");
  sb_printf(&sb, "  @builtin(local_invocation_id) lid: vec3<u32>,\n");
  sb_printf(&sb, "  @builtin(workgroup_id) wid: vec3<u32>\n");
  sb_printf(&sb, ") {\n");
  sb_printf(&sb, "  let t: u32 = lid.x;\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n\n", total_elems);

  /* Global load -> shared memory (coalesced) */
  {
    int loads_per_thread = (total_elems + wg_size - 1) / wg_size;
    for (int e = 0; e < loads_per_thread; e++) {
      int off = e * wg_size;
      if (off == 0) {
        sb_printf(&sb, "  { let gi: u32 = t;\n");
      } else {
        sb_printf(&sb, "  { let gi: u32 = t + %uu;\n", off);
      }
      if (off + wg_size > total_elems) {
        sb_printf(&sb, "    if (gi < %uu) {\n", total_elems);
        sb_printf(&sb, "      s_re[gi] = src.d[(base + gi) * 2u];\n");
        sb_printf(&sb, "      s_im[gi] = src.d[(base + gi) * 2u + 1u];\n");
        sb_printf(&sb, "    }\n");
      } else {
        sb_printf(&sb, "    s_re[gi] = src.d[(base + gi) * 2u];\n");
        sb_printf(&sb, "    s_im[gi] = src.d[(base + gi) * 2u + 1u];\n");
      }
      sb_printf(&sb, "  }\n");
    }
  }
  sb_printf(&sb, "  workgroupBarrier();\n");

  /* Row FFTs: nx row-FFTs of ny-point each.
   * Shared memory layout: row r, col c => s_re[r * ny + c]
   * axis_stride = 1 (consecutive elements along row)
   * cross_stride = ny (stride between rows = between different FFTs) */
  emit_fft_stages_2d(&sb, ny, nx, wpf_row, mr_row, direction,
                      1 /* axis_stride */, ny /* cross_stride */,
                      0 /* lut offset */, has_lut, 0 /* axis_id=row */);

  /* Column FFTs: ny col-FFTs of nx-point each.
   * axis_stride = ny (stride between rows for column elements)
   * cross_stride = 1 (stride between columns = between different FFTs) */
  emit_fft_stages_2d(&sb, nx, ny, wpf_col, mr_col, direction,
                      ny /* axis_stride */, 1 /* cross_stride */,
                      row_lut /* lut offset */, has_lut, 1 /* axis_id=col */);

  /* Global store from shared memory (coalesced) */
  sb_printf(&sb, "\n");
  {
    int loads_per_thread = (total_elems + wg_size - 1) / wg_size;
    for (int e = 0; e < loads_per_thread; e++) {
      int off = e * wg_size;
      if (off == 0) {
        sb_printf(&sb, "  { let gi: u32 = t;\n");
      } else {
        sb_printf(&sb, "  { let gi: u32 = t + %uu;\n", off);
      }
      if (off + wg_size > total_elems) {
        sb_printf(&sb, "    if (gi < %uu) {\n", total_elems);
        sb_printf(&sb, "      dst.d[(base + gi) * 2u] = s_re[gi];\n");
        sb_printf(&sb, "      dst.d[(base + gi) * 2u + 1u] = s_im[gi];\n");
        sb_printf(&sb, "    }\n");
      } else {
        sb_printf(&sb, "    dst.d[(base + gi) * 2u] = s_re[gi];\n");
        sb_printf(&sb, "    dst.d[(base + gi) * 2u + 1u] = s_im[gi];\n");
      }
      sb_printf(&sb, "  }\n");
    }
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* ========================================================================== */
/* Looped 2D fused FFT (push-constant repeat count, single dispatch)          */
/* ========================================================================== */

char *gen_fft_2d_fused_looped(int nx, int ny, int direction, int max_radix) {
  if (nx < 2 || ny < 2) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  int mr_row = effective_max_r(ny, max_radix);
  int mr_col = effective_max_r(nx, max_radix);

  int row_radices[MAX_STAGES], col_radices[MAX_STAGES];
  int row_stages = factorize_mr(ny, row_radices, mr_row);
  int col_stages = factorize_mr(nx, col_radices, mr_col);
  if (row_stages == 0 || col_stages == 0) return NULL;

  int wpf_row = wg_per_fft_mr(ny, mr_row);
  int wpf_col = wg_per_fft_mr(nx, mr_col);
  if (wpf_row == 0 || wpf_col == 0) return NULL;

  int t_rows = nx * wpf_row;
  int t_cols = ny * wpf_col;
  int wg_size = t_rows > t_cols ? t_rows : t_cols;

  int total_elems = nx * ny;
  if (total_elems * 8 > MAX_SHARED_BYTES) return NULL;

  int row_lut = axis_lut_size(ny, mr_row);
  int col_lut = axis_lut_size(nx, mr_col);
  int has_lut = (row_lut + col_lut) > 0;

  /* Only support no-LUT sizes for the looped variant (covers 4×4..16×16) */
  if (has_lut) return NULL;

  StrBuf sb;
  sb_init_cap(&sb, 131072);

  /* Bindings — data is read_write for in-place looping,
   * ctl: element 0 = repeat count (bitcast from f32) */
  sb_emit_bda_inplace_ctl(&sb);
  sb_printf(&sb, "\n");

  /* Shared memory */
  sb_printf(&sb, "var<workgroup> s_re: array<f32, %d>;\n", total_elems);
  sb_printf(&sb, "var<workgroup> s_im: array<f32, %d>;\n\n", total_elems);

  /* Entry point */
  sb_printf(&sb, "@compute @workgroup_size(%d)\n", wg_size);
  sb_printf(&sb, "fn main(\n");
  sb_printf(&sb, "  @builtin(local_invocation_id) lid: vec3<u32>,\n");
  sb_printf(&sb, "  @builtin(workgroup_id) wid: vec3<u32>\n");
  sb_printf(&sb, ") {\n");
  sb_printf(&sb, "  let t: u32 = lid.x;\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", total_elems);
  sb_printf(&sb, "  let repeat_count: u32 = bitcast<u32>(ctl.d[0]);\n\n");

  /* Global load -> shared memory (coalesced) — once, outside the loop */
  {
    int loads_per_thread = (total_elems + wg_size - 1) / wg_size;
    for (int e = 0; e < loads_per_thread; e++) {
      int off = e * wg_size;
      if (off == 0)
        sb_printf(&sb, "  { let gi: u32 = t;\n");
      else
        sb_printf(&sb, "  { let gi: u32 = t + %uu;\n", off);
      if (off + wg_size > total_elems) {
        sb_printf(&sb, "    if (gi < %uu) {\n", total_elems);
        sb_printf(&sb, "      s_re[gi] = data.d[(base + gi) * 2u];\n");
        sb_printf(&sb, "      s_im[gi] = data.d[(base + gi) * 2u + 1u];\n");
        sb_printf(&sb, "    }\n");
      } else {
        sb_printf(&sb, "    s_re[gi] = data.d[(base + gi) * 2u];\n");
        sb_printf(&sb, "    s_im[gi] = data.d[(base + gi) * 2u + 1u];\n");
      }
      sb_printf(&sb, "  }\n");
    }
  }
  sb_printf(&sb, "  workgroupBarrier();\n\n");

  /* Repeat loop — FFT stages only, data stays in shared memory */
  sb_printf(&sb, "  for (var iter: u32 = 0u; iter < repeat_count; iter = iter + 1u) {\n");

  /* Row FFTs */
  emit_fft_stages_2d(&sb, ny, nx, wpf_row, mr_row, direction,
                      1, ny, 0, has_lut, 0);

  /* Column FFTs */
  emit_fft_stages_2d(&sb, nx, ny, wpf_col, mr_col, direction,
                      ny, 1, row_lut, has_lut, 1);

  sb_printf(&sb, "  }\n\n");

  /* Global store from shared memory (coalesced) — once, after the loop */
  {
    int loads_per_thread = (total_elems + wg_size - 1) / wg_size;
    for (int e = 0; e < loads_per_thread; e++) {
      int off = e * wg_size;
      if (off == 0)
        sb_printf(&sb, "  { let gi: u32 = t;\n");
      else
        sb_printf(&sb, "  { let gi: u32 = t + %uu;\n", off);
      if (off + wg_size > total_elems) {
        sb_printf(&sb, "    if (gi < %uu) {\n", total_elems);
        sb_printf(&sb, "      data.d[(base + gi) * 2u] = s_re[gi];\n");
        sb_printf(&sb, "      data.d[(base + gi) * 2u + 1u] = s_im[gi];\n");
        sb_printf(&sb, "    }\n");
      } else {
        sb_printf(&sb, "      data.d[(base + gi) * 2u] = s_re[gi];\n");
        sb_printf(&sb, "      data.d[(base + gi) * 2u + 1u] = s_im[gi];\n");
      }
      sb_printf(&sb, "  }\n");
    }
  }
  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}
