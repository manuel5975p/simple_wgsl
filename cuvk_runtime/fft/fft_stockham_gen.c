/*
 * fft_stockham_gen.c - Generate WGSL compute shaders for Stockham FFT stages
 *
 * Each generated shader performs one stage of a multi-stage Stockham FFT:
 * reads radix elements at stride from src, applies inter-stage twiddles,
 * performs a radix-N FFT with baked intra-radix twiddles, and writes
 * results in Stockham order to dst.
 */

#include "fft_stockham_gen.h"
#include "fft_butterfly.h"
#include "fft_bda.h"

#include <math.h>

/* ========================================================================== */
/* FFT helpers                                                                */
/* ========================================================================== */

/* ========================================================================== */
/* Prologue: bindings, main entry, source indexing, element loads              */
/* ========================================================================== */

static void emit_prologue(StrBuf *sb, int radix, int stride,
                           int n_total, int workgroup_size,
                           int element_stride, int batch_stride,
                           int batch_stride2) {
  sb_emit_bda_src_dst(sb, 0);

  /* Entry point */
  sb_printf(sb, "@compute @workgroup_size(%d)\n", workgroup_size);
  sb_printf(sb, "fn main(@builtin(global_invocation_id) "
                "gid: vec3<u32>) {\n");

  /* Butterfly and batch indexing.
   * batch_offset and src_base are in units of f32 pairs (complex elements).
   * Multiply by 2 when indexing into the f32 array.
   * element_stride: physical distance between consecutive logical elements.
   * batch_stride: physical distance between consecutive FFTs (gid.y).
   * batch_stride2: physical distance for outer batch dimension (gid.z). */
  sb_printf(sb, "  let bf_id: u32 = gid.x;\n");
  sb_printf(sb, "  if (bf_id >= %uu) { return; }\n", n_total / radix);
  if (batch_stride2 > 0) {
    sb_printf(sb, "  let batch_offset: u32 = gid.y * %uu + gid.z * %uu;\n",
              batch_stride, batch_stride2);
  } else {
    sb_printf(sb, "  let batch_offset: u32 = gid.y * %uu;\n", batch_stride);
  }

  /* Stockham auto-sort source indexing (in complex-element units).
   *
   * bf_id = group * stride + pos, where:
   *   group = bf_id / stride   (which sub-FFT block)
   *   pos   = bf_id % stride   (position within the block)
   *
   * Read R elements at stride N/R (half-array distance for radix-2,
   * generalised for arbitrary radix):
   *   logical_base = group * stride + pos
   *   v[k]  = src[(batch_offset + (logical_base + k * (N/R)) * E) * 2]
   * where E = element_stride.
   */
  sb_printf(sb, "  let group: u32 = bf_id / %uu;\n", stride);
  sb_printf(sb, "  let pos: u32 = bf_id %% %uu;\n", stride);
  sb_printf(sb, "  let logical_base: u32 = group * %uu + pos;\n", stride);

  /* Load radix elements at stride n_total/radix, scaled by element_stride */
  int read_stride = n_total / radix;
  for (int k = 0; k < radix; k++) {
    int logical_off = k * read_stride;
    if (element_stride == 1) {
      sb_printf(sb, "  var v%d: vec2<f32> = vec2<f32>("
                    "src.d[(batch_offset + logical_base + %uu) * 2u], "
                    "src.d[(batch_offset + logical_base + %uu) * 2u + 1u]);\n",
                k, logical_off, logical_off);
    } else {
      sb_printf(sb, "  var v%d: vec2<f32> = vec2<f32>("
                    "src.d[(batch_offset + (logical_base + %uu) * %uu) * 2u], "
                    "src.d[(batch_offset + (logical_base + %uu) * %uu) * 2u + 1u]);\n",
                k, logical_off, element_stride, logical_off, element_stride);
    }
  }
}

/* ========================================================================== */
/* Inter-stage twiddles                                                       */
/* ========================================================================== */

static void emit_inter_stage_twiddles(StrBuf *sb, int radix, int stride,
                                       int direction) {
  if (stride <= 1) return; /* first stage: all twiddles are 1 */

  sb_printf(sb, "  let pos_f: f32 = f32(pos);\n");

  double base_angle = (double)direction * -2.0 * M_PI / (double)(stride * radix);

  for (int k = 1; k < radix; k++) {
    sb_printf(sb, "  let tw_angle%d: f32 = ", k);
    sb_float(sb, base_angle * k);
    sb_printf(sb, " * pos_f;\n");
    sb_printf(sb, "  let tw%d: vec2<f32> = vec2<f32>(cos(tw_angle%d), "
                  "sin(tw_angle%d));\n", k, k, k);
    sb_printf(sb, "  v%d = vec2<f32>(v%d.x*tw%d.x - v%d.y*tw%d.y, "
                  "v%d.x*tw%d.y + v%d.y*tw%d.x);\n",
              k, k, k, k, k, k, k, k, k);
  }
}

/* ========================================================================== */
/* Radix-N FFT: power-of-2 Cooley-Tukey (fully unrolled, vec2 ops)           */
/* ========================================================================== */

static void emit_fft_po2(StrBuf *sb, int radix, int direction) {
  int log2r = ilog2(radix);

  /* Bit-reverse permutation via swaps */
  int swap_id = 0;
  for (int i = 0; i < radix; i++) {
    int j = bit_reverse(i, log2r);
    if (j > i) {
      sb_printf(sb, "  let s%d = v%d; v%d = v%d; v%d = s%d;\n",
                swap_id, i, i, j, j, swap_id);
      swap_id++;
    }
  }

  /* Unrolled butterfly stages */
  int tmp_id = 0;
  for (int stage = 0; stage < log2r; stage++) {
    int half_size = 1 << stage;
    int group_size = half_size * 2;
    int n_groups = radix / group_size;

    for (int g = 0; g < n_groups; g++) {
      for (int b = 0; b < half_size; b++) {
        int even = g * group_size + b;
        int odd = even + half_size;
        int tw_k = b * (radix / group_size);
        double angle = (double)direction * -2.0 * M_PI * tw_k / radix;
        double tw_re = cos(angle);
        double tw_im = sin(angle);

        if (fabs(tw_re) < 1e-15) tw_re = 0.0;
        if (fabs(tw_im) < 1e-15) tw_im = 0.0;

        /* Compute twiddle * v_odd */
        if (tw_re == 1.0 && tw_im == 0.0) {
          /* W = 1: no multiply needed */
          sb_printf(sb, "  let t%d = v%d; let e%d = v%d; "
                        "v%d = e%d + t%d; v%d = e%d - t%d;\n",
                    tmp_id, odd, tmp_id, even,
                    even, tmp_id, tmp_id, odd, tmp_id, tmp_id);
        } else if (tw_re == -1.0 && tw_im == 0.0) {
          /* W = -1: negate */
          sb_printf(sb, "  let t%d = -v%d; let e%d = v%d; "
                        "v%d = e%d + t%d; v%d = e%d - t%d;\n",
                    tmp_id, odd, tmp_id, even,
                    even, tmp_id, tmp_id, odd, tmp_id, tmp_id);
        } else if (tw_re == 0.0 && tw_im == -1.0) {
          /* W = -i (forward rotation by -pi/2): (a+bi)*(-i) = b - ai
             vec2(y, -x) */
          sb_printf(sb, "  let t%d = vec2<f32>(v%d.y, -v%d.x); "
                        "let e%d = v%d; "
                        "v%d = e%d + t%d; v%d = e%d - t%d;\n",
                    tmp_id, odd, odd, tmp_id, even,
                    even, tmp_id, tmp_id, odd, tmp_id, tmp_id);
        } else if (tw_re == 0.0 && tw_im == 1.0) {
          /* W = i: (a+bi)*(i) = -b + ai
             vec2(-y, x) */
          sb_printf(sb, "  let t%d = vec2<f32>(-v%d.y, v%d.x); "
                        "let e%d = v%d; "
                        "v%d = e%d + t%d; v%d = e%d - t%d;\n",
                    tmp_id, odd, odd, tmp_id, even,
                    even, tmp_id, tmp_id, odd, tmp_id, tmp_id);
        } else {
          /* General twiddle: inline cmul */
          char vname[16];
          snprintf(vname, sizeof(vname), "v%d", odd);
          sb_printf(sb, "  let t%d = ", tmp_id);
          sb_cmul(sb, vname, tw_re, tw_im);
          sb_printf(sb, "; let e%d = v%d; "
                        "v%d = e%d + t%d; v%d = e%d - t%d;\n",
                    tmp_id, even,
                    even, tmp_id, tmp_id, odd, tmp_id, tmp_id);
        }
        tmp_id++;
      }
    }
  }
}

/* ========================================================================== */
/* Radix-N FFT: direct DFT O(R^2) for non-power-of-2 radices                 */
/* ========================================================================== */

static void emit_fft_dft(StrBuf *sb, int radix, int direction) {
  /* Compute each output */
  for (int k = 0; k < radix; k++) {
    sb_printf(sb, "  var o%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", k);
    for (int j = 0; j < radix; j++) {
      int tw_idx = (k * j) % radix;
      double angle = (double)direction * -2.0 * M_PI * tw_idx / radix;
      double wr = cos(angle);
      double wi = sin(angle);
      if (fabs(wr) < 1e-15) wr = 0.0;
      if (fabs(wi) < 1e-15) wi = 0.0;

      if (wr == 1.0 && wi == 0.0) {
        /* W = 1 */
        sb_printf(sb, "  o%d = o%d + v%d;\n", k, k, j);
      } else if (wr == -1.0 && wi == 0.0) {
        /* W = -1 */
        sb_printf(sb, "  o%d = o%d - v%d;\n", k, k, j);
      } else if (wr == 0.0 && wi == -1.0) {
        /* W = -i: (a+bi)*(-i) = b - ai -> vec2(y, -x) */
        sb_printf(sb, "  o%d = o%d + vec2<f32>(v%d.y, -v%d.x);\n",
                  k, k, j, j);
      } else if (wr == 0.0 && wi == 1.0) {
        /* W = i: (a+bi)*(i) = -b + ai -> vec2(-y, x) */
        sb_printf(sb, "  o%d = o%d + vec2<f32>(-v%d.y, v%d.x);\n",
                  k, k, j, j);
      } else {
        /* General twiddle: inline cmul */
        char vname[16];
        snprintf(vname, sizeof(vname), "v%d", j);
        sb_printf(sb, "  o%d = o%d + ", k, k);
        sb_cmul(sb, vname, wr, wi);
        sb_printf(sb, ";\n");
      }
    }
  }

  /* Copy back: vK = oK */
  for (int k = 0; k < radix; k++) {
    sb_printf(sb, "  v%d = o%d;\n", k, k);
  }
}

/* ========================================================================== */
/* Epilogue: Stockham output writes                                           */
/* ========================================================================== */

static void emit_epilogue(StrBuf *sb, int radix, int stride, int n_total,
                          int element_stride) {
  /* Stockham auto-sort output write.
   *
   * For a butterfly at (group, pos) with radix R and stride L:
   *   logical_idx = group * L * R + pos + k * L   for k = 0..R-1
   *   dst[(batch_offset + logical_idx * element_stride) * 2]
   *
   * This arranges the output so that the next stage (with stride L*R) reads
   * its R elements at stride N/R_next from the correct positions. */
  (void)n_total;

  for (int k = 0; k < radix; k++) {
    int logical_off = k * stride;
    if (element_stride == 1) {
      sb_printf(sb, "  dst.d[(batch_offset + group * %uu + pos + %uu) * 2u] = v%d.x;\n",
                stride * radix, logical_off, k);
      sb_printf(sb, "  dst.d[(batch_offset + group * %uu + pos + %uu) * 2u + 1u] = v%d.y;\n",
                stride * radix, logical_off, k);
    } else {
      sb_printf(sb, "  dst.d[(batch_offset + (group * %uu + pos + %uu) * %uu) * 2u] = v%d.x;\n",
                stride * radix, logical_off, element_stride, k);
      sb_printf(sb, "  dst.d[(batch_offset + (group * %uu + pos + %uu) * %uu) * 2u + 1u] = v%d.y;\n",
                stride * radix, logical_off, element_stride, k);
    }
  }

  sb_printf(sb, "}\n");
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

char *gen_fft_stockham(int radix, int stride, int n_total,
                       int direction, int workgroup_size) {
  return gen_fft_stockham_strided2(radix, stride, n_total, direction,
                                   workgroup_size, 1, n_total, 0);
}

char *gen_fft_stockham_strided(int radix, int stride, int n_total,
                               int direction, int workgroup_size,
                               int element_stride, int batch_stride) {
  return gen_fft_stockham_strided2(radix, stride, n_total, direction,
                                   workgroup_size, element_stride,
                                   batch_stride, 0);
}

char *gen_fft_stockham_strided2(int radix, int stride, int n_total,
                                int direction, int workgroup_size,
                                int element_stride, int batch_stride,
                                int batch_stride2) {
  if (radix < 2 || radix > 32) return NULL;
  if (stride < 1) return NULL;
  if (n_total < radix) return NULL;
  if (n_total % radix != 0) return NULL;
  if (direction != 1 && direction != -1) return NULL;
  if (workgroup_size < 1) return NULL;
  if (element_stride < 1) return NULL;
  if (batch_stride < 1) return NULL;

  StrBuf sb;
  sb_init(&sb);

  emit_prologue(&sb, radix, stride, n_total, workgroup_size,
                element_stride, batch_stride, batch_stride2);
  emit_inter_stage_twiddles(&sb, radix, stride, direction);

  if (is_po2(radix))
    emit_fft_po2(&sb, radix, direction);
  else
    emit_fft_dft(&sb, radix, direction);

  emit_epilogue(&sb, radix, stride, n_total, element_stride);

  return sb_finish(&sb);
}

/* ========================================================================== */
/* Extended API: factorization + plan queries                                 */
/* ========================================================================== */

int fft_stockham_factorize(int n, int max_radix, int *radices) {
  if (max_radix <= 0 || max_radix > 32) max_radix = 32;
  if (max_radix < 2) max_radix = 2;
  int count = 0, rem = n;
  while (rem > 1 && count < FFT_STOCKHAM_MAX_STAGES) {
    int best = 0;
    for (int r = max_radix; r >= 2; r--) {
      if (rem % r == 0) { best = r; break; }
    }
    if (best == 0) return 0;
    radices[count++] = best;
    rem /= best;
  }
  return (rem == 1) ? count : 0;
}

int fft_stockham_num_stages(int n, int max_radix) {
  int radices[FFT_STOCKHAM_MAX_STAGES];
  return fft_stockham_factorize(n, max_radix, radices);
}

int fft_stockham_dispatch_x(int n_total, int radix, int workgroup_size) {
  int n_bf = n_total / radix;
  return (n_bf + workgroup_size - 1) / workgroup_size;
}

/* ========================================================================== */
/* R2C post-processing shader                                                 */
/* ========================================================================== */

char *gen_fft_r2c_postprocess(int n, int workgroup_size) {
  if (n < 2 || n % 2 != 0 || workgroup_size < 1) return NULL;
  int half = n / 2;

  StrBuf sb;
  sb_init(&sb);

  sb_emit_bda_src_dst(&sb, 0);

  sb_printf(&sb, "@compute @workgroup_size(%d)\n", workgroup_size);
  sb_printf(&sb, "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
  sb_printf(&sb, "  let k: u32 = gid.x;\n");
  sb_printf(&sb, "  if (k >= %uu) { return; }\n", half);
  /* batch_offset: src has N/2 complex values per batch, dst has N/2+1 */
  sb_printf(&sb, "  let src_batch: u32 = gid.y * %uu;\n", half);
  sb_printf(&sb, "  let dst_batch: u32 = gid.y * %uu;\n", half + 1);

  /* DC and Nyquist bins */
  sb_printf(&sb, "  if (k == 0u) {\n");
  sb_printf(&sb, "    let z0r: f32 = src.d[src_batch * 2u];\n");
  sb_printf(&sb, "    let z0i: f32 = src.d[src_batch * 2u + 1u];\n");
  sb_printf(&sb, "    dst.d[dst_batch * 2u] = z0r + z0i;\n");
  sb_printf(&sb, "    dst.d[dst_batch * 2u + 1u] = 0.0;\n");
  sb_printf(&sb, "    dst.d[(dst_batch + %uu) * 2u] = z0r - z0i;\n", half);
  sb_printf(&sb, "    dst.d[(dst_batch + %uu) * 2u + 1u] = 0.0;\n", half);
  sb_printf(&sb, "    return;\n");
  sb_printf(&sb, "  }\n");

  /* General bin k: X[k] = 0.5*(Z[k]+conj(Z[N/2-k]))
   *                      - 0.5i*W^k*(Z[k]-conj(Z[N/2-k]))
   * where W = exp(-2*pi*i/N) */
  sb_printf(&sb, "  let nk: u32 = %uu - k;\n", half);
  sb_printf(&sb, "  let zk_r: f32 = src.d[(src_batch + k) * 2u];\n");
  sb_printf(&sb, "  let zk_i: f32 = src.d[(src_batch + k) * 2u + 1u];\n");
  sb_printf(&sb, "  let znk_r: f32 = src.d[(src_batch + nk) * 2u];\n");
  sb_printf(&sb, "  let znk_i: f32 = -src.d[(src_batch + nk) * 2u + 1u];\n");
  /* E = 0.5*(Z[k] + conj(Z[N/2-k])) */
  sb_printf(&sb, "  let er: f32 = 0.5 * (zk_r + znk_r);\n");
  sb_printf(&sb, "  let ei: f32 = 0.5 * (zk_i + znk_i);\n");
  /* D = 0.5*(Z[k] - conj(Z[N/2-k])) */
  sb_printf(&sb, "  let dr: f32 = 0.5 * (zk_r - znk_r);\n");
  sb_printf(&sb, "  let di: f32 = 0.5 * (zk_i - znk_i);\n");
  /* W^k = exp(-2*pi*i*k/N), compute as cos/sin */
  sb_printf(&sb, "  let angle: f32 = ");
  sb_float(&sb, -2.0 * M_PI / n);
  sb_printf(&sb, " * f32(k);\n");
  sb_printf(&sb, "  let wr: f32 = cos(angle);\n");
  sb_printf(&sb, "  let wi: f32 = sin(angle);\n");
  /* O = -i * W^k * D = (-i)*(wr+i*wi)*(dr+i*di)
   *   = (-i)*(wr*dr - wi*di + i*(wr*di + wi*dr))
   *   = (wr*di + wi*dr) + i*(-(wr*dr - wi*di))
   *   = (wr*di + wi*dr) - i*(wr*dr - wi*di) */
  sb_printf(&sb, "  let or_: f32 = wr * di + wi * dr;\n");
  sb_printf(&sb, "  let oi: f32 = -(wr * dr - wi * di);\n");
  /* X[k] = E + O */
  sb_printf(&sb, "  dst.d[(dst_batch + k) * 2u] = er + or_;\n");
  sb_printf(&sb, "  dst.d[(dst_batch + k) * 2u + 1u] = ei + oi;\n");

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* ========================================================================== */
/* C2R pre-processing shader                                                  */
/* ========================================================================== */

char *gen_fft_c2r_preprocess(int n, int workgroup_size) {
  if (n < 2 || n % 2 != 0 || workgroup_size < 1) return NULL;
  int half = n / 2;

  StrBuf sb;
  sb_init(&sb);

  sb_emit_bda_src_dst(&sb, 0);

  sb_printf(&sb, "@compute @workgroup_size(%d)\n", workgroup_size);
  sb_printf(&sb, "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
  sb_printf(&sb, "  let k: u32 = gid.x;\n");
  sb_printf(&sb, "  if (k >= %uu) { return; }\n", half);
  /* src has N/2+1 complex bins per batch, dst has N/2 complex values */
  sb_printf(&sb, "  let src_batch: u32 = gid.y * %uu;\n", half + 1);
  sb_printf(&sb, "  let dst_batch: u32 = gid.y * %uu;\n", half);

  /* DC bin: Z[0] = (X[0].re + X[N/2].re, X[0].re - X[N/2].re)
   * Note: no 0.5 factor — we need 2x scaling so that the N/2-point
   * inverse C2C produces N*x (matching cuFFT's unnormalized convention). */
  sb_printf(&sb, "  if (k == 0u) {\n");
  sb_printf(&sb, "    let x0r: f32 = src.d[src_batch * 2u];\n");
  sb_printf(&sb, "    let xnr: f32 = src.d[(src_batch + %uu) * 2u];\n", half);
  sb_printf(&sb, "    dst.d[dst_batch * 2u] = x0r + xnr;\n");
  sb_printf(&sb, "    dst.d[dst_batch * 2u + 1u] = x0r - xnr;\n");
  sb_printf(&sb, "    return;\n");
  sb_printf(&sb, "  }\n");

  /* General bin k: Z[k] = (X[k]+conj(X[N/2-k]))
   *                      + i*W^(-k)*(X[k]-conj(X[N/2-k]))
   * where W = exp(-2*pi*i/N), so W^(-k) = exp(+2*pi*i*k/N)
   * Note: no 0.5 — doubled to match cuFFT scaling. */
  sb_printf(&sb, "  let nk: u32 = %uu - k;\n", half);
  sb_printf(&sb, "  let xk_r: f32 = src.d[(src_batch + k) * 2u];\n");
  sb_printf(&sb, "  let xk_i: f32 = src.d[(src_batch + k) * 2u + 1u];\n");
  sb_printf(&sb, "  let xnk_r: f32 = src.d[(src_batch + nk) * 2u];\n");
  sb_printf(&sb, "  let xnk_i: f32 = -src.d[(src_batch + nk) * 2u + 1u];\n");
  /* E = X[k] + conj(X[N/2-k]) */
  sb_printf(&sb, "  let er: f32 = xk_r + xnk_r;\n");
  sb_printf(&sb, "  let ei: f32 = xk_i + xnk_i;\n");
  /* D = X[k] - conj(X[N/2-k]) */
  sb_printf(&sb, "  let dr: f32 = xk_r - xnk_r;\n");
  sb_printf(&sb, "  let di: f32 = xk_i - xnk_i;\n");
  /* W^(-k) = exp(+2*pi*i*k/N) */
  sb_printf(&sb, "  let angle: f32 = ");
  sb_float(&sb, 2.0 * M_PI / n);
  sb_printf(&sb, " * f32(k);\n");
  sb_printf(&sb, "  let wr: f32 = cos(angle);\n");
  sb_printf(&sb, "  let wi: f32 = sin(angle);\n");
  /* O = i * W^(-k) * D = i*(wr+i*wi)*(dr+i*di)
   *   = i*(wr*dr - wi*di + i*(wr*di + wi*dr))
   *   = -(wr*di + wi*dr) + i*(wr*dr - wi*di) */
  sb_printf(&sb, "  let or_: f32 = -(wr * di + wi * dr);\n");
  sb_printf(&sb, "  let oi: f32 = wr * dr - wi * di;\n");
  /* Z[k] = E + O */
  sb_printf(&sb, "  dst.d[(dst_batch + k) * 2u] = er + or_;\n");
  sb_printf(&sb, "  dst.d[(dst_batch + k) * 2u + 1u] = ei + oi;\n");

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}
