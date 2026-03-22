/*
 * fft_optimal_gen.c - Generate a single-dispatch WGSL compute shader for a
 * complete FFT of size N, where one workgroup computes the entire FFT in
 * registers using recursive Cooley-Tukey decomposition.
 *
 * All code is inlined (no WGSL function calls) since the WGSL parser does
 * not support ptr<function, array<...>> parameter types. Each variable v0..vN
 * holds one complex value as vec2<f32>.
 */

#include "fft_optimal_gen.h"
#include "fft_butterfly.h"
#include "fft_bda.h"

#include <math.h>

/* ========================================================================== */
/* Inline DFT emitter                                                          */
/* ========================================================================== */

/*
 * Emit inline direct DFT of 'count' elements.
 * vars[0..count-1] are indices into the v0..v_{N-1} variable array.
 * perm_out[i] = which DFT bin is at position i after the DFT.
 * tc = temp counter (monotonically increasing, ensures unique names).
 */
static void emit_inline_dft(StrBuf *sb, int count, const int *vars,
                             int direction, int *tc, int *perm_out) {
  int base = *tc;

  /* Load inputs into temporaries */
  for (int j = 0; j < count; j++) {
    sb_printf(sb, "  let d%d: vec2<f32> = v%d;\n", base + j, vars[j]);
  }

  /* Compute DFT outputs */
  for (int k = 0; k < count; k++) {
    int oid = base + count + k;
    sb_printf(sb, "  var d%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", oid);
    for (int j = 0; j < count; j++) {
      int iid = base + j;
      int tw_idx = (k * j) % count;
      double angle = (double)direction * -2.0 * M_PI * tw_idx / count;
      double wr = cos(angle);
      double wi = sin(angle);
      if (fabs(wr) < 1e-15) wr = 0.0;
      if (fabs(wi) < 1e-15) wi = 0.0;

      if (wr == 1.0 && wi == 0.0) {
        sb_printf(sb, "  d%d = d%d + d%d;\n", oid, oid, iid);
      } else if (wr == -1.0 && wi == 0.0) {
        sb_printf(sb, "  d%d = d%d - d%d;\n", oid, oid, iid);
      } else if (wr == 0.0 && wi == -1.0) {
        sb_printf(sb, "  d%d = d%d + vec2<f32>(d%d.y, -d%d.x);\n",
                  oid, oid, iid, iid);
      } else if (wr == 0.0 && wi == 1.0) {
        sb_printf(sb, "  d%d = d%d + vec2<f32>(-d%d.y, d%d.x);\n",
                  oid, oid, iid, iid);
      } else {
        char vname[16];
        snprintf(vname, sizeof(vname), "d%d", iid);
        sb_printf(sb, "  d%d = d%d + ", oid, oid);
        sb_cmul(sb, vname, wr, wi);
        sb_printf(sb, ";\n");
      }
    }
  }

  /* Write outputs back to v variables */
  for (int k = 0; k < count; k++) {
    sb_printf(sb, "  v%d = d%d;\n", vars[k], base + count + k);
  }

  *tc = base + count * 2;

  /* Direct DFT: identity permutation */
  for (int i = 0; i < count; i++) perm_out[i] = i;
}

/* ========================================================================== */
/* Inline twiddle emitter                                                      */
/* ========================================================================== */

static void emit_twiddle(StrBuf *sb, int var_idx, double wr, double wi,
                          int *tc) {
  if (wr == 1.0 && wi == 0.0) return;
  if (wr == -1.0 && wi == 0.0) {
    sb_printf(sb, "  v%d = -v%d;\n", var_idx, var_idx);
  } else if (wr == 0.0 && wi == -1.0) {
    sb_printf(sb, "  v%d = vec2<f32>(v%d.y, -v%d.x);\n",
              var_idx, var_idx, var_idx);
  } else if (wr == 0.0 && wi == 1.0) {
    sb_printf(sb, "  v%d = vec2<f32>(-v%d.y, v%d.x);\n",
              var_idx, var_idx, var_idx);
  } else {
    int tid = (*tc)++;
    sb_printf(sb, "  let tw%d: vec2<f32> = v%d;\n", tid, var_idx);
    char vname[16];
    snprintf(vname, sizeof(vname), "tw%d", tid);
    sb_printf(sb, "  v%d = ", var_idx);
    sb_cmul(sb, vname, wr, wi);
    sb_printf(sb, ";\n");
  }
}

/* ========================================================================== */
/* Recursive inline FFT emitter                                                */
/* ========================================================================== */

/*
 * Emit inline FFT for 'count' elements at variable indices vars[0..count-1].
 * Operates in-place on the v0..v_{N-1} variables.
 *
 * perm_out[i] = which DFT frequency bin is at position i after the FFT.
 * tc = temp variable counter (ensures unique names across recursive calls).
 */
static void emit_inline_fft(StrBuf *sb, int count, const int *vars,
                             const FftOptPlan *plan_table, int direction,
                             int *tc, int *perm_out) {
  FftOptPlan plan = plan_table[ilog2(count)];

  if (plan.type == FFT_OPT_DIRECT) {
    emit_inline_dft(sb, count, vars, direction, tc, perm_out);
    return;
  }

  int R = plan.radix;
  int M = count / R;
  int n = R * M;

  /* Step 1: M sub-FFTs of size R on columns (stride M).
   * For row-major layout N = R*M, column DFTs come first. */
  int perm_R[256];
  for (int j = 0; j < M; j++) {
    int col_vars[256];
    for (int k = 0; k < R; k++) col_vars[k] = vars[j + k * M];
    emit_inline_fft(sb, R, col_vars, plan_table, direction, tc, perm_R);
  }

  /* Step 2: Twiddle factors.
   * For col j=1..M-1, row k=0..R-1:
   *   freq = perm_R[k] (actual DFT bin at position k after fft_R)
   *   angle = direction * -2*pi * j * freq / (R*M) */
  for (int j = 1; j < M; j++) {
    for (int k = 0; k < R; k++) {
      int freq = perm_R[k];
      if (freq == 0) continue;
      double angle = (double)direction * -2.0 * M_PI * j * freq / n;
      double wr = cos(angle);
      double wi = sin(angle);
      if (fabs(wr) < 1e-15) wr = 0.0;
      if (fabs(wi) < 1e-15) wi = 0.0;
      emit_twiddle(sb, vars[j + k * M], wr, wi, tc);
    }
  }

  /* Step 3: R sub-FFTs of size M on contiguous rows */
  int perm_M[256];
  for (int r = 0; r < R; r++) {
    int row_vars[256];
    for (int j = 0; j < M; j++) row_vars[j] = vars[r * M + j];
    emit_inline_fft(sb, M, row_vars, plan_table, direction, tc, perm_M);
  }

  /* Compute output permutation:
   * perm_N[j + k*M] = R * perm_M[j] + perm_R[k] */
  for (int j = 0; j < M; j++)
    for (int k = 0; k < R; k++)
      perm_out[j + k * M] = R * perm_M[j] + perm_R[k];
}

/* ========================================================================== */
/* Public API                                                                  */
/* ========================================================================== */

char *gen_fft_optimal(int n, const FftOptPlan *plan_table, int direction) {
  if (n < 2 || n > 256 || (n & (n - 1)) != 0) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  StrBuf sb;
  sb_init(&sb);

  /* Prologue: buffer binding + main entry */
  sb_printf(&sb, "enable device_address;\n");
  sb_printf(&sb, "struct Buf { d: array<f32> };\n");
  sb_printf(&sb, "var<device, read_write> data: Buf;\n\n");
  sb_printf(&sb, "@compute @workgroup_size(1)\n");
  sb_printf(&sb, "fn main(@builtin(workgroup_id) wid: vec3<u32>) {\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", n * 2);

  /* Load N complex values into individual var variables */
  for (int i = 0; i < n; i++) {
    sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                    "data.d[base + %uu], data.d[base + %uu]);\n",
              i, i * 2, i * 2 + 1);
  }

  /* Emit inline FFT */
  int vars[256];
  for (int i = 0; i < n; i++) vars[i] = i;
  int perm[256];
  int tc = 0;
  emit_inline_fft(&sb, n, vars, plan_table, direction, &tc, perm);

  /* Store with inverse permutation so output is in natural order */
  int inv_perm[256];
  for (int i = 0; i < n; i++) inv_perm[perm[i]] = i;
  for (int b = 0; b < n; b++) {
    sb_printf(&sb, "  data.d[base + %uu] = v%d.x;\n", b * 2, inv_perm[b]);
    sb_printf(&sb, "  data.d[base + %uu] = v%d.y;\n", b * 2 + 1, inv_perm[b]);
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

char *gen_fft_direct(int n, int direction) {
  if (n < 2 || n > 256) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  StrBuf sb;
  sb_init(&sb);

  sb_printf(&sb, "enable device_address;\n");
  sb_printf(&sb, "struct Buf { d: array<f32> };\n");
  sb_printf(&sb, "var<device, read_write> data: Buf;\n\n");
  sb_printf(&sb, "@compute @workgroup_size(1)\n");
  sb_printf(&sb, "fn main(@builtin(workgroup_id) wid: vec3<u32>) {\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", n * 2);

  for (int i = 0; i < n; i++) {
    sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                    "data.d[base + %uu], data.d[base + %uu]);\n",
              i, i * 2, i * 2 + 1);
  }

  int vars[256];
  for (int i = 0; i < n; i++) vars[i] = i;
  int perm[256];
  int tc = 0;
  emit_inline_dft(&sb, n, vars, direction, &tc, perm);

  /* Direct DFT produces identity permutation */
  for (int b = 0; b < n; b++) {
    sb_printf(&sb, "  data.d[base + %uu] = v%d.x;\n", b * 2, b);
    sb_printf(&sb, "  data.d[base + %uu] = v%d.y;\n", b * 2 + 1, b);
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* ========================================================================== */
/* Bluestein (chirp-z transform) for arbitrary sizes                          */
/* ========================================================================== */

static int next_pow2(int n) {
  int p = 1;
  while (p < n) p *= 2;
  return p;
}

char *gen_fft_bluestein(int n, const FftOptPlan *pow2_plan_table, int direction) {
  if (n < 2 || n > 256) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  int M = next_pow2(2 * n - 1);

  /* Precompute chirp[i] = W_N^{i²/2} = e^{dir * -πi * i² / N} */
  double chirp_re[256], chirp_im[256];
  for (int i = 0; i < n; i++) {
    double angle = (double)direction * -M_PI * (double)i * (double)i / (double)n;
    chirp_re[i] = cos(angle);
    chirp_im[i] = sin(angle);
    if (fabs(chirp_re[i]) < 1e-15) chirp_re[i] = 0.0;
    if (fabs(chirp_im[i]) < 1e-15) chirp_im[i] = 0.0;
  }

  /* Precompute h' (circular chirp sequence, length M).
   * h'[0] = 1, h'[m] = h'[M-m] = conj(chirp[m]) for m=1..N-1, rest zero. */
  double h_re[512], h_im[512];
  memset(h_re, 0, (size_t)M * sizeof(double));
  memset(h_im, 0, (size_t)M * sizeof(double));
  h_re[0] = 1.0;
  for (int m = 1; m < n; m++) {
    h_re[m] = chirp_re[m];
    h_im[m] = -chirp_im[m];
    h_re[M - m] = h_re[m];
    h_im[M - m] = h_im[m];
  }

  /* Precompute H = DFT_M(h') on CPU */
  double H_re[512], H_im[512];
  for (int k = 0; k < M; k++) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < M; j++) {
      double angle = -2.0 * M_PI * (double)k * (double)j / (double)M;
      double wr = cos(angle), wi = sin(angle);
      sr += h_re[j] * wr - h_im[j] * wi;
      si += h_re[j] * wi + h_im[j] * wr;
    }
    H_re[k] = sr;
    H_im[k] = si;
    if (fabs(H_re[k]) < 1e-12) H_re[k] = 0.0;
    if (fabs(H_im[k]) < 1e-12) H_im[k] = 0.0;
  }

  StrBuf sb;
  sb_init(&sb);
  int tc = 0;

  /* Prologue */
  sb_printf(&sb, "enable device_address;\n");
  sb_printf(&sb, "struct Buf { d: array<f32> };\n");
  sb_printf(&sb, "var<device, read_write> data: Buf;\n\n");
  sb_printf(&sb, "@compute @workgroup_size(1)\n");
  sb_printf(&sb, "fn main(@builtin(workgroup_id) wid: vec3<u32>) {\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", n * 2);

  /* Load N inputs */
  for (int i = 0; i < n; i++) {
    sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                    "data.d[base + %uu], data.d[base + %uu]);\n",
              i, i * 2, i * 2 + 1);
  }

  /* Multiply by input chirp */
  for (int i = 0; i < n; i++)
    emit_twiddle(&sb, i, chirp_re[i], chirp_im[i], &tc);

  /* Zero-pad to M */
  for (int i = n; i < M; i++)
    sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", i);

  /* Forward M-point FFT */
  int vars[512];
  for (int i = 0; i < M; i++) vars[i] = i;
  int perm[512];
  emit_inline_fft(&sb, M, vars, pow2_plan_table, 1, &tc, perm);

  /* Pointwise multiply with H (position i has DFT bin perm[i]) */
  for (int i = 0; i < M; i++)
    emit_twiddle(&sb, vars[i], H_re[perm[i]], H_im[perm[i]], &tc);

  /* Un-permute to natural order for the inverse FFT */
  int inv_perm[512];
  for (int i = 0; i < M; i++) inv_perm[perm[i]] = i;
  int tmp_base = tc;
  for (int k = 0; k < M; k++)
    sb_printf(&sb, "  let r%d: vec2<f32> = v%d;\n", tmp_base + k, inv_perm[k]);
  for (int k = 0; k < M; k++)
    sb_printf(&sb, "  v%d = r%d;\n", k, tmp_base + k);
  tc = tmp_base + M;

  /* Inverse M-point FFT */
  int perm2[512];
  emit_inline_fft(&sb, M, vars, pow2_plan_table, -1, &tc, perm2);

  /* Store: X[k] = v{inv_perm2[k]} * chirp[k] / M */
  int inv_perm2[512];
  for (int i = 0; i < M; i++) inv_perm2[perm2[i]] = i;
  double inv_M = 1.0 / (double)M;

  for (int k = 0; k < n; k++) {
    int src = inv_perm2[k];
    double cr = chirp_re[k] * inv_M;
    double ci = chirp_im[k] * inv_M;
    if (fabs(cr) < 1e-15) cr = 0.0;
    if (fabs(ci) < 1e-15) ci = 0.0;

    int oid = tc++;
    char vname[16];
    snprintf(vname, sizeof(vname), "v%d", src);
    sb_printf(&sb, "  let o%d: vec2<f32> = ", oid);
    sb_cmul(&sb, vname, cr, ci);
    sb_printf(&sb, ";\n");
    sb_printf(&sb, "  data.d[base + %uu] = o%d.x;\n", k * 2, oid);
    sb_printf(&sb, "  data.d[base + %uu] = o%d.y;\n", k * 2 + 1, oid);
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* ========================================================================== */
/* General FFT: mixed-radix CT + Bluestein + Rader + DFT for any size         */
/* ========================================================================== */

/* Forward declaration (mutual recursion with emit_bluestein_inline/rader) */
static void emit_general_fft(StrBuf *sb, int count, const int *vars,
                              const FftPlan *plans, int direction,
                              int *tc, int *next_var, int *perm_out);

/* ========================================================================== */
/* Number theory helpers for Rader's algorithm                                 */
/* ========================================================================== */

static int modpow_opt(int base, int exp, int m) {
  long long result = 1, b = base % m;
  while (exp > 0) {
    if (exp & 1) result = result * b % m;
    b = b * b % m;
    exp >>= 1;
  }
  return (int)result;
}

static int primitive_root_opt(int p) {
  if (p == 2) return 1;
  int pm1 = p - 1;
  int factors[32], nf = 0;
  int tmp = pm1;
  for (int d = 2; d * d <= tmp; d++) {
    if (tmp % d == 0) {
      factors[nf++] = d;
      while (tmp % d == 0) tmp /= d;
    }
  }
  if (tmp > 1) factors[nf++] = tmp;

  for (int g = 2; g < p; g++) {
    int ok = 1;
    for (int i = 0; i < nf; i++) {
      if (modpow_opt(g, pm1 / factors[i], p) == 1) { ok = 0; break; }
    }
    if (ok) return g;
  }
  return -1;
}

/* ========================================================================== */
/* Rader's algorithm — prime p via cyclic convolution of length p-1            */
/* ========================================================================== */

static void emit_rader_inline(StrBuf *sb, int p, const int *vars,
                               const FftPlan *plans, int direction,
                               int *tc, int *next_var, int *perm_out) {
  int M = p - 1;  /* convolution length */
  int g = primitive_root_opt(p);

  /* Compute g^j mod p for j=0..M-1 (permutation sequence) */
  int g_pow[FFT_PLAN_MAX_N + 1];
  g_pow[0] = 1;
  for (int j = 1; j < M; j++)
    g_pow[j] = (int)((long long)g_pow[j - 1] * g % p);

  /* Compute g^{-j} mod p for j=0..M-1 (inverse permutation) */
  int g_inv_pow[FFT_PLAN_MAX_N + 1];
  int g_inv = modpow_opt(g, p - 2, p);  /* g^{-1} mod p */
  g_inv_pow[0] = 1;
  for (int j = 1; j < M; j++)
    g_inv_pow[j] = (int)((long long)g_inv_pow[j - 1] * g_inv % p);

  /* Precompute Rader kernel b[j] = W_p^{g^{-j}} and its DFT B[k] */
  double b_re[FFT_PLAN_MAX_N + 1], b_im[FFT_PLAN_MAX_N + 1];
  for (int j = 0; j < M; j++) {
    double angle = (double)direction * -2.0 * M_PI * g_inv_pow[j] / (double)p;
    b_re[j] = cos(angle);
    b_im[j] = sin(angle);
    if (fabs(b_re[j]) < 1e-15) b_re[j] = 0.0;
    if (fabs(b_im[j]) < 1e-15) b_im[j] = 0.0;
  }

  /* DFT of kernel: B[k] = sum_j b[j] * W_M^{kj} */
  double B_re[FFT_PLAN_MAX_N + 1], B_im[FFT_PLAN_MAX_N + 1];
  for (int k = 0; k < M; k++) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < M; j++) {
      double angle = -2.0 * M_PI * (double)k * (double)j / (double)M;
      double wr = cos(angle), wi = sin(angle);
      sr += b_re[j] * wr - b_im[j] * wi;
      si += b_re[j] * wi + b_im[j] * wr;
    }
    B_re[k] = sr;  B_im[k] = si;
    if (fabs(B_re[k]) < 1e-12) B_re[k] = 0.0;
    if (fabs(B_im[k]) < 1e-12) B_im[k] = 0.0;
  }

  sb_printf(sb, "  { // Rader prime=%d, g=%d\n", p, g);

  /* Save x[0] (needed for non-DC outputs: X[k] = x[0] + conv[k]) */
  int xinp0 = (*tc)++;
  sb_printf(sb, "  let xinp%d: vec2<f32> = v%d;\n", xinp0, vars[0]);

  /* Step 1: x0 = sum of all p inputs (DC component) */
  int x0 = (*tc)++;
  sb_printf(sb, "  var x0_%d: vec2<f32> = v%d;\n", x0, vars[0]);
  for (int j = 1; j < p; j++)
    sb_printf(sb, "  x0_%d = x0_%d + v%d;\n", x0, x0, vars[j]);

  /* Step 2: Permute inputs by g^j: a[j] = input[g^j mod p], j=0..M-1 */
  int a_base = *next_var;
  *next_var += M;
  for (int j = 0; j < M; j++)
    sb_printf(sb, "  var v%d: vec2<f32> = v%d;\n", a_base + j, vars[g_pow[j]]);

  /* Step 3: Forward DFT of length M on a[] */
  int a_vars[FFT_PLAN_MAX_N + 1];
  for (int j = 0; j < M; j++) a_vars[j] = a_base + j;
  int perm_fwd[FFT_PLAN_MAX_N + 1];
  emit_general_fft(sb, M, a_vars, plans, 1, tc, next_var, perm_fwd);

  /* Step 4: Pointwise multiply with precomputed kernel B[] */
  sb_printf(sb, "  // Pointwise multiply with Rader kernel\n");
  for (int i = 0; i < M; i++) {
    int k = perm_fwd[i];  /* a_vars[i] holds DFT bin k */
    emit_twiddle(sb, a_vars[i], B_re[k], B_im[k], tc);
  }

  /* Step 5: Un-permute to natural order for inverse FFT */
  int inv_perm[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < M; i++) inv_perm[perm_fwd[i]] = i;
  int tmp_base = *tc;
  for (int k = 0; k < M; k++)
    sb_printf(sb, "  let r%d: vec2<f32> = v%d;\n",
              tmp_base + k, a_vars[inv_perm[k]]);
  for (int k = 0; k < M; k++)
    sb_printf(sb, "  v%d = r%d;\n", a_vars[k], tmp_base + k);
  *tc = tmp_base + M;

  /* Step 6: Inverse DFT of length M */
  int perm_inv[FFT_PLAN_MAX_N + 1];
  emit_general_fft(sb, M, a_vars, plans, -1, tc, next_var, perm_inv);

  /* Step 7: Scale by 1/M, add x[0], and unpermute to output */
  int inv_perm2[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < M; i++) inv_perm2[perm_inv[i]] = i;
  double inv_M = 1.0 / (double)M;

  /* output[0] = x0 */
  sb_printf(sb, "  v%d = x0_%d;\n", vars[0], x0);

  /* output[g^j mod p] = x0 + IDFT[j] / M, j=0..M-1 */
  for (int j = 0; j < M; j++) {
    int src = a_vars[inv_perm2[j]];
    int oid = (*tc)++;
    sb_printf(sb, "  let ro%d: vec2<f32> = xinp%d + v%d * vec2<f32>(", oid, xinp0, src);
    sb_float(sb, inv_M);
    sb_printf(sb, ", ");
    sb_float(sb, inv_M);
    sb_printf(sb, ");\n");
    sb_printf(sb, "  v%d = ro%d;\n", vars[g_inv_pow[j]], oid);
  }

  sb_printf(sb, "  } // end Rader prime=%d\n", p);

  /* Rader produces natural order */
  for (int i = 0; i < p; i++) perm_out[i] = i;
}

/* ========================================================================== */
/* Bluestein (chirp-z) for arbitrary sizes                                     */
/* ========================================================================== */

static void emit_bluestein_inline(StrBuf *sb, int count, const int *vars,
                                   const FftPlan *plans, int direction,
                                   int *tc, int *next_var, int *perm_out) {
  int M = next_pow2(2 * count - 1);

  /* Precompute chirp[i] = e^{dir * -pi*i * i^2 / N} */
  double chirp_re[FFT_PLAN_MAX_N + 1], chirp_im[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < count; i++) {
    double angle =
        (double)direction * -M_PI * (double)i * (double)i / (double)count;
    chirp_re[i] = cos(angle);
    chirp_im[i] = sin(angle);
    if (fabs(chirp_re[i]) < 1e-15) chirp_re[i] = 0.0;
    if (fabs(chirp_im[i]) < 1e-15) chirp_im[i] = 0.0;
  }

  /* Precompute h' and H = DFT_M(h') */
  double h_re[FFT_PLAN_MAX_N + 1], h_im[FFT_PLAN_MAX_N + 1];
  memset(h_re, 0, (size_t)M * sizeof(double));
  memset(h_im, 0, (size_t)M * sizeof(double));
  h_re[0] = 1.0;
  for (int m = 1; m < count; m++) {
    h_re[m] = chirp_re[m];  h_im[m] = -chirp_im[m];
    h_re[M - m] = h_re[m];  h_im[M - m] = h_im[m];
  }

  double H_re[FFT_PLAN_MAX_N + 1], H_im[FFT_PLAN_MAX_N + 1];
  for (int k = 0; k < M; k++) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < M; j++) {
      double angle = -2.0 * M_PI * (double)k * (double)j / (double)M;
      double wr = cos(angle), wi = sin(angle);
      sr += h_re[j] * wr - h_im[j] * wi;
      si += h_re[j] * wi + h_im[j] * wr;
    }
    H_re[k] = sr;  H_im[k] = si;
    if (fabs(H_re[k]) < 1e-12) H_re[k] = 0.0;
    if (fabs(H_im[k]) < 1e-12) H_im[k] = 0.0;
  }

  /* Allocate M fresh v-variables */
  int base = *next_var;
  *next_var += M;

  /* Copy inputs and multiply by chirp */
  for (int i = 0; i < count; i++) {
    sb_printf(sb, "  var v%d: vec2<f32> = v%d;\n", base + i, vars[i]);
    emit_twiddle(sb, base + i, chirp_re[i], chirp_im[i], tc);
  }

  /* Zero-pad to M */
  for (int i = count; i < M; i++)
    sb_printf(sb, "  var v%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", base + i);

  /* Forward M-point FFT (always direction=1 for convolution) */
  int fft_vars[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < M; i++) fft_vars[i] = base + i;
  int perm[FFT_PLAN_MAX_N + 1];
  emit_general_fft(sb, M, fft_vars, plans, 1, tc, next_var, perm);

  /* Pointwise multiply with H (position i has DFT bin perm[i]) */
  for (int i = 0; i < M; i++)
    emit_twiddle(sb, fft_vars[i], H_re[perm[i]], H_im[perm[i]], tc);

  /* Un-permute to natural order for the inverse FFT */
  int inv_perm[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < M; i++) inv_perm[perm[i]] = i;
  int tmp_base = *tc;
  for (int k = 0; k < M; k++)
    sb_printf(sb, "  let r%d: vec2<f32> = v%d;\n",
              tmp_base + k, fft_vars[inv_perm[k]]);
  for (int k = 0; k < M; k++)
    sb_printf(sb, "  v%d = r%d;\n", fft_vars[k], tmp_base + k);
  *tc = tmp_base + M;

  /* Inverse M-point FFT (always direction=-1) */
  int perm2[FFT_PLAN_MAX_N + 1];
  emit_general_fft(sb, M, fft_vars, plans, -1, tc, next_var, perm2);

  /* Output: chirp[k] * v[inv_perm2[k]] / M → vars[k] */
  int inv_perm2[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < M; i++) inv_perm2[perm2[i]] = i;
  double inv_M = 1.0 / (double)M;

  for (int k = 0; k < count; k++) {
    int src = fft_vars[inv_perm2[k]];
    double cr = chirp_re[k] * inv_M;
    double ci = chirp_im[k] * inv_M;
    if (fabs(cr) < 1e-15) cr = 0.0;
    if (fabs(ci) < 1e-15) ci = 0.0;

    int oid = (*tc)++;
    char vname[16];
    snprintf(vname, sizeof(vname), "v%d", src);
    sb_printf(sb, "  let o%d: vec2<f32> = ", oid);
    sb_cmul(sb, vname, cr, ci);
    sb_printf(sb, ";\n");
    sb_printf(sb, "  v%d = o%d;\n", vars[k], oid);
  }

  /* Bluestein always produces natural order */
  for (int i = 0; i < count; i++) perm_out[i] = i;
}

static void emit_general_fft(StrBuf *sb, int count, const int *vars,
                              const FftPlan *plans, int direction,
                              int *tc, int *next_var, int *perm_out) {
  FftPlan plan = plans[count];

  if (plan.type == FFT_PLAN_DFT) {
    emit_inline_dft(sb, count, vars, direction, tc, perm_out);
    return;
  }

  if (plan.type == FFT_PLAN_BLUESTEIN) {
    emit_bluestein_inline(sb, count, vars, plans, direction,
                           tc, next_var, perm_out);
    return;
  }

  if (plan.type == FFT_PLAN_RADER) {
    emit_rader_inline(sb, count, vars, plans, direction,
                       tc, next_var, perm_out);
    return;
  }

  /* FFT_PLAN_CT: N = R x M */
  int R = plan.radix;
  int M = count / R;

  /* Step 1: M column DFTs of size R (stride M) */
  int perm_R[FFT_PLAN_MAX_N + 1];
  for (int j = 0; j < M; j++) {
    int col_vars[FFT_PLAN_MAX_N + 1];
    for (int k = 0; k < R; k++) col_vars[k] = vars[j + k * M];
    emit_general_fft(sb, R, col_vars, plans, direction,
                      tc, next_var, perm_R);
  }

  /* Step 2: Twiddle factors */
  for (int j = 1; j < M; j++) {
    for (int k = 0; k < R; k++) {
      int freq = perm_R[k];
      if (freq == 0) continue;
      double angle =
          (double)direction * -2.0 * M_PI * j * freq / (double)(R * M);
      double wr = cos(angle), wi = sin(angle);
      if (fabs(wr) < 1e-15) wr = 0.0;
      if (fabs(wi) < 1e-15) wi = 0.0;
      emit_twiddle(sb, vars[j + k * M], wr, wi, tc);
    }
  }

  /* Step 3: R row DFTs of size M */
  int perm_M[FFT_PLAN_MAX_N + 1];
  for (int r = 0; r < R; r++) {
    int row_vars[FFT_PLAN_MAX_N + 1];
    for (int j = 0; j < M; j++) row_vars[j] = vars[r * M + j];
    emit_general_fft(sb, M, row_vars, plans, direction,
                      tc, next_var, perm_M);
  }

  /* Output permutation: perm_out[j + k*M] = R * perm_M[j] + perm_R[k] */
  for (int j = 0; j < M; j++)
    for (int k = 0; k < R; k++)
      perm_out[j + k * M] = R * perm_M[j] + perm_R[k];
}

char *gen_fft(int n, const FftPlan *plans, int direction) {
  if (n < 2 || n > FFT_PLAN_MAX_N) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  StrBuf sb;
  sb_init(&sb);

  sb_printf(&sb, "enable device_address;\n");
  sb_printf(&sb, "struct Buf { d: array<f32> };\n");
  sb_printf(&sb, "var<device, read_write> data: Buf;\n\n");
  sb_printf(&sb, "@compute @workgroup_size(1)\n");
  sb_printf(&sb, "fn main(@builtin(workgroup_id) wid: vec3<u32>) {\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", n * 2);

  for (int i = 0; i < n; i++) {
    sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                    "data.d[base + %uu], data.d[base + %uu]);\n",
              i, i * 2, i * 2 + 1);
  }

  int vars[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < n; i++) vars[i] = i;
  int perm_out[FFT_PLAN_MAX_N + 1];
  int tc = 0;
  int next_var = n;
  emit_general_fft(&sb, n, vars, plans, direction, &tc, &next_var, perm_out);

  /* Store with inverse permutation so output is in natural order */
  int inv_perm[FFT_PLAN_MAX_N + 1];
  for (int i = 0; i < n; i++) inv_perm[perm_out[i]] = i;
  for (int b = 0; b < n; b++) {
    sb_printf(&sb, "  data.d[base + %uu] = v%d.x;\n", b * 2, inv_perm[b]);
    sb_printf(&sb, "  data.d[base + %uu] = v%d.y;\n", b * 2 + 1, inv_perm[b]);
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}
