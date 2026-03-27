/*
 * fft_fourstep_gen.c - Generate twiddle+transpose and transpose WGSL shaders
 *                      for the four-step FFT algorithm.
 *
 * The four-step FFT decomposes an N-point FFT (N = N1 * N2) into:
 *   1. N1 independent N2-point column FFTs
 *   2. Twiddle multiply + transpose (N1xN2 -> N2xN1)
 *   3. N2 independent N1-point row FFTs
 *   4. Final transpose (N2xN1 -> N1xN2)
 *
 * This file generates the kernels for steps 2 and 4.
 */

#include "fft_fourstep_gen.h"
#include "fft_strbuf.h"
#include "fft_bda.h"

#include <math.h>

/* ========================================================================== */
/* Twiddle + Transpose kernel                                                 */
/* ========================================================================== */

char *gen_fft_twiddle_transpose(int n1, int n2, int direction, int workgroup_size) {
  int n = n1 * n2;
  double angle_scale = (direction == 1) ? -2.0 * M_PI / n : 2.0 * M_PI / n;

  StrBuf sb;
  sb_init(&sb);

  sb_emit_bda_src_dst(&sb, 0);

  sb_printf(&sb, "const N1: u32 = %du;\n", n1);
  sb_printf(&sb, "const N2: u32 = %du;\n", n2);
  sb_printf(&sb, "const N: u32 = %du;\n", n);
  sb_printf(&sb, "const ANGLE_SCALE: f32 = ");
  sb_float(&sb, angle_scale);
  sb_printf(&sb, ";\n\n");

  sb_printf(&sb, "@compute @workgroup_size(%d)\n", workgroup_size);
  sb_printf(&sb, "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
  sb_printf(&sb, "  let batch = gid.y;\n");
  sb_printf(&sb, "  let idx = gid.x;\n");
  sb_printf(&sb, "  if (idx >= N) { return; }\n");
  sb_printf(&sb, "  let base = batch * N * 2u;\n");
  sb_printf(&sb, "  let i = idx / N2;\n");
  sb_printf(&sb, "  let j = idx %% N2;\n");
  sb_printf(&sb, "  let src_off = base + idx * 2u;\n");
  sb_printf(&sb, "  let re = src.d[src_off];\n");
  sb_printf(&sb, "  let im = src.d[src_off + 1u];\n");
  sb_printf(&sb, "  let angle = ANGLE_SCALE * f32(i) * f32(j);\n");
  sb_printf(&sb, "  let tw_re = cos(angle);\n");
  sb_printf(&sb, "  let tw_im = sin(angle);\n");
  sb_printf(&sb, "  let out_re = re * tw_re - im * tw_im;\n");
  sb_printf(&sb, "  let out_im = re * tw_im + im * tw_re;\n");
  sb_printf(&sb, "  let dst_off = base + (j * N1 + i) * 2u;\n");
  sb_printf(&sb, "  dst.d[dst_off] = out_re;\n");
  sb_printf(&sb, "  dst.d[dst_off + 1u] = out_im;\n");
  sb_printf(&sb, "}\n");

  return sb_finish(&sb);
}

/* ========================================================================== */
/* Pure Transpose kernel                                                      */
/* ========================================================================== */

char *gen_fft_transpose(int n1, int n2, int workgroup_size) {
  int n = n1 * n2;

  StrBuf sb;
  sb_init(&sb);

  sb_emit_bda_src_dst(&sb, 0);

  sb_printf(&sb, "const N1: u32 = %du;\n", n1);
  sb_printf(&sb, "const N2: u32 = %du;\n", n2);
  sb_printf(&sb, "const N: u32 = %du;\n\n", n);

  sb_printf(&sb, "@compute @workgroup_size(%d)\n", workgroup_size);
  sb_printf(&sb, "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
  sb_printf(&sb, "  let batch = gid.y;\n");
  sb_printf(&sb, "  let idx = gid.x;\n");
  sb_printf(&sb, "  if (idx >= N) { return; }\n");
  sb_printf(&sb, "  let base = batch * N * 2u;\n");
  sb_printf(&sb, "  let i = idx / N2;\n");
  sb_printf(&sb, "  let j = idx %% N2;\n");
  sb_printf(&sb, "  let src_off = base + idx * 2u;\n");
  sb_printf(&sb, "  let re = src.d[src_off];\n");
  sb_printf(&sb, "  let im = src.d[src_off + 1u];\n");
  sb_printf(&sb, "  let dst_off = base + (j * N1 + i) * 2u;\n");
  sb_printf(&sb, "  dst.d[dst_off] = re;\n");
  sb_printf(&sb, "  dst.d[dst_off + 1u] = im;\n");
  sb_printf(&sb, "}\n");

  return sb_finish(&sb);
}

/* ========================================================================== */
/* Decomposition: N -> N1 x N2                                               */
/* ========================================================================== */

/* Fused-friendly radices: {2, 3, 4, 5, 7, 8, 16} */
static const int fused_radices[] = {16, 8, 7, 5, 4, 3, 2};
static const int n_fused_radices = sizeof(fused_radices) / sizeof(fused_radices[0]);

/* Check if m factors entirely into fused-friendly radices */
static int is_fused_friendly(int m) {
  if (m <= 0) return 0;
  int rem = m;
  for (int iter = 0; rem > 1 && iter < 64; iter++) {
    int found = 0;
    for (int i = 0; i < n_fused_radices; i++) {
      if (rem % fused_radices[i] == 0) {
        rem /= fused_radices[i];
        found = 1;
        break;
      }
    }
    if (!found) return 0;
  }
  return rem == 1;
}

int fft_fourstep_decompose(int n, int max_sub_n, int *out_n1, int *out_n2) {
  if (n <= 0 || max_sub_n <= 0) return 0;

  /* If N fits directly in a single fused kernel, no split needed */
  if (n <= max_sub_n && is_fused_friendly(n)) {
    *out_n1 = 1;
    *out_n2 = n;
    return 1;
  }

  /* Try to find N1 * N2 = N with both factors <= max_sub_n and fused-friendly.
   * Prefer balanced splits (closest to sqrt(N)). */
  int isqrt = (int)sqrt((double)n);
  int best_n1 = 0, best_n2 = 0;
  int best_balance = 0x7fffffff; /* |n1 - n2|, smaller is better */

  /* Search divisors from sqrt(N) downward and upward */
  for (int d = isqrt; d >= 2; d--) {
    if (n % d != 0) continue;
    int q = n / d;
    if (d > max_sub_n || q > max_sub_n) continue;
    if (!is_fused_friendly(d) || !is_fused_friendly(q)) continue;
    int balance = (q > d) ? q - d : d - q;
    if (balance < best_balance) {
      best_balance = balance;
      best_n1 = d;
      best_n2 = q;
    }
    break; /* first valid from sqrt is most balanced */
  }

  /* Also search upward from sqrt in case we missed */
  for (int d = isqrt + 1; (long long)d * d <= (long long)n; d++) {
    /* d is the smaller factor here, q = n/d is the larger */
    if (n % d != 0) continue;
    int q = n / d;
    if (d > max_sub_n || q > max_sub_n) continue;
    if (!is_fused_friendly(d) || !is_fused_friendly(q)) continue;
    int balance = q - d;
    if (balance < best_balance) {
      best_balance = balance;
      best_n1 = d;
      best_n2 = q;
    }
    break;
  }

  if (best_n1 == 0) {
    /* Fallback: brute-force all divisors */
    for (int d = 2; d <= n / 2; d++) {
      if (n % d != 0) continue;
      int q = n / d;
      if (d > max_sub_n || q > max_sub_n) continue;
      if (!is_fused_friendly(d) || !is_fused_friendly(q)) continue;
      int balance = (q > d) ? q - d : d - q;
      if (balance < best_balance) {
        best_balance = balance;
        best_n1 = d;
        best_n2 = q;
      }
    }
  }

  if (best_n1 == 0) return 0;

  *out_n1 = best_n1;
  *out_n2 = best_n2;
  return 1;
}
