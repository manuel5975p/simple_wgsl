/*
 * fft_stockham_gen.c - Generate WGSL compute shaders for Stockham FFT stages
 *
 * Each generated shader performs one stage of a multi-stage Stockham FFT:
 * reads radix elements at stride from src, applies inter-stage twiddles,
 * performs a radix-N FFT with baked intra-radix twiddles, and writes
 * results in Stockham order to dst.
 */

#include "fft_stockham_gen.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================== */
/* String builder                                                             */
/* ========================================================================== */

typedef struct {
  char *buf;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
  sb->cap = 4096;
  sb->buf = (char *)malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int needed = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (needed < 0) return;
  while (sb->len + (size_t)needed + 1 > sb->cap) {
    sb->cap *= 2;
    sb->buf = (char *)realloc(sb->buf, sb->cap);
  }
  va_start(ap, fmt);
  vsnprintf(sb->buf + sb->len, sb->cap - sb->len, fmt, ap);
  va_end(ap);
  sb->len += (size_t)needed;
}

static char *sb_finish(StrBuf *sb) { return sb->buf; }

/*
 * sb_float: emit a float literal that WGSL will parse as f32, not i32.
 *
 * %.17g can format 1.0 as "1" and 0.0 as "0". WGSL treats these as integer
 * literals. When multiplied with f32, the compiler emits OpIMul on float bit
 * patterns instead of OpFMul. We append ".0" when the formatted string has
 * no decimal point and no exponent notation.
 */
static void sb_float(StrBuf *sb, double v) {
  char tmp[64];
  snprintf(tmp, sizeof(tmp), "%.17g", v);
  sb_printf(sb, "%s", tmp);
  if (!strchr(tmp, '.') && !strchr(tmp, 'e') && !strchr(tmp, 'E'))
    sb_printf(sb, ".0");
}

/* ========================================================================== */
/* FFT helpers                                                                */
/* ========================================================================== */

static int is_po2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static int ilog2(int n) {
  int r = 0;
  while (n > 1) { n >>= 1; r++; }
  return r;
}

static int bit_reverse(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) { r = (r << 1) | (v & 1); v >>= 1; }
  return r;
}

/* ========================================================================== */
/* Inline complex multiply helper                                             */
/* ========================================================================== */

/*
 * Emit inline complex multiply: vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
 * where 'a' is a variable name and 'b' is a baked constant vec2.
 */
static void sb_cmul(StrBuf *sb, const char *a, double br, double bi) {
  sb_printf(sb, "vec2<f32>(%s.x*", a);
  sb_float(sb, br);
  sb_printf(sb, " - %s.y*", a);
  sb_float(sb, bi);
  sb_printf(sb, ", %s.x*", a);
  sb_float(sb, bi);
  sb_printf(sb, " + %s.y*", a);
  sb_float(sb, br);
  sb_printf(sb, ")");
}

/* ========================================================================== */
/* Prologue: bindings, main entry, source indexing, element loads              */
/* ========================================================================== */

static void emit_prologue(StrBuf *sb, int radix, int stride,
                           int n_total, int workgroup_size) {
  /* Storage bindings: use array<f32> (proven reliable) with vec2 locals.
   * Two separate struct types to avoid NonWritable decoration bleeding. */
  sb_printf(sb, "struct SrcBuf { d: array<f32> };\n");
  sb_printf(sb, "struct DstBuf { d: array<f32> };\n");
  sb_printf(sb, "@group(0) @binding(0) var<storage, read> src: SrcBuf;\n");
  sb_printf(sb, "@group(0) @binding(1) var<storage, read_write> dst: DstBuf;\n\n");

  /* Entry point */
  sb_printf(sb, "@compute @workgroup_size(%d)\n", workgroup_size);
  sb_printf(sb, "fn main(@builtin(global_invocation_id) "
                "gid: vec3<u32>) {\n");

  /* Butterfly and batch indexing.
   * batch_offset and src_base are in units of complex elements.
   * Multiply by 2 when indexing into the f32 array. */
  sb_printf(sb, "  let bf_id: u32 = gid.x;\n");
  sb_printf(sb, "  let batch_offset: u32 = gid.y * %uu;\n", n_total);

  /* Source indexing (strided read, in complex-element units) */
  sb_printf(sb, "  let group: u32 = bf_id / %uu;\n", stride);
  sb_printf(sb, "  let pos: u32 = bf_id %% %uu;\n", stride);
  sb_printf(sb, "  let src_base: u32 = batch_offset + "
                "group * %uu + pos;\n", stride * radix);

  /* Load radix elements with stride: read f32 pairs into vec2 locals */
  for (int k = 0; k < radix; k++) {
    int off = k * stride;
    sb_printf(sb, "  var v%d: vec2<f32> = vec2<f32>("
                  "src.d[(src_base + %uu) * 2u], "
                  "src.d[(src_base + %uu) * 2u + 1u]);\n",
              k, off, off);
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

static void emit_epilogue(StrBuf *sb, int radix, int n_total) {
  int dst_stride = n_total / radix;

  for (int k = 0; k < radix; k++) {
    int off = k * dst_stride;
    sb_printf(sb, "  dst.d[(batch_offset + bf_id + %uu) * 2u] = v%d.x;\n",
              off, k);
    sb_printf(sb, "  dst.d[(batch_offset + bf_id + %uu) * 2u + 1u] = v%d.y;\n",
              off, k);
  }

  sb_printf(sb, "}\n");
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

char *gen_fft_stockham(int radix, int stride, int n_total,
                       int direction, int workgroup_size) {
  /* Validate parameters */
  if (radix < 2 || radix > 32) return NULL;
  if (stride < 1) return NULL;
  if (n_total < radix) return NULL;
  if (n_total % radix != 0) return NULL;
  if (direction != 1 && direction != -1) return NULL;
  if (workgroup_size < 1) return NULL;

  StrBuf sb;
  sb_init(&sb);

  emit_prologue(&sb, radix, stride, n_total, workgroup_size);
  emit_inter_stage_twiddles(&sb, radix, stride, direction);

  if (is_po2(radix))
    emit_fft_po2(&sb, radix, direction);
  else
    emit_fft_dft(&sb, radix, direction);

  emit_epilogue(&sb, radix, n_total);

  return sb_finish(&sb);
}
