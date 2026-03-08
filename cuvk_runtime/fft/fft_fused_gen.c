/*
 * fft_fused_gen.c - Generate single-dispatch fused FFT WGSL shaders
 *
 * All radix stages execute in one compute dispatch using workgroup shared
 * memory with Stockham auto-sort addressing.
 *
 * Optimizations:
 *   1. Twiddle LUT buffer — precomputed cos/sin in GPU storage buffer
 *   2. Bank-conflict padding on shared memory for po2 N
 *   3. Specialized radix-3/5/7/16 butterflies
 *   4. Intra-workgroup batching — multiple FFTs per workgroup for small N
 *   5. Rader's algorithm — prime radices via cyclic convolution
 *
 * Supported radices: {2, 3, 4, 5, 7, 8, 16} + primes where p-1 factors
 * into these (11, 13, 17, 19, 23 skipped since 22=2*11 needs nested Rader,
 * 29, 31, 37, 41, 43, ...).
 */

#include "fft_fused_gen.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_STAGES 20
#define MAX_SHARED_BYTES 32768 /* conservative shared memory limit */
#define DIRECT_MAX_N 64 /* sizes ≤ this use register-only path (no shared mem) */

/* ========================================================================== */
/* String builder                                                             */
/* ========================================================================== */

typedef struct {
  char *buf;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
  sb->cap = 131072;
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

static void sb_float(StrBuf *sb, double v) {
  char tmp[64];
  snprintf(tmp, sizeof(tmp), "%.17g", v);
  sb_printf(sb, "%s", tmp);
  if (!strchr(tmp, '.') && !strchr(tmp, 'e') && !strchr(tmp, 'E'))
    sb_printf(sb, ".0");
}

/* ========================================================================== */
/* Primality / primitive root                                                 */
/* ========================================================================== */

static int is_prime(int n) {
  if (n < 2) return 0;
  if (n < 4) return 1;
  if (n % 2 == 0 || n % 3 == 0) return 0;
  for (int i = 5; i * i <= n; i += 6)
    if (n % i == 0 || n % (i + 2) == 0) return 0;
  return 1;
}

/* modular exponentiation: base^exp mod m */
static int modpow(int base, int exp, int m) {
  long long result = 1, b = base % m;
  while (exp > 0) {
    if (exp & 1) result = result * b % m;
    b = b * b % m;
    exp >>= 1;
  }
  return (int)result;
}

/* Find smallest primitive root modulo prime p */
static int primitive_root(int p) {
  if (p == 2) return 1;
  /* Factor p-1 */
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
      if (modpow(g, pm1 / factors[i], p) == 1) { ok = 0; break; }
    }
    if (ok) return g;
  }
  return -1;
}

/* ========================================================================== */
/* Factorization                                                              */
/* ========================================================================== */

static const int preferred_radices[] = {16, 8, 7, 5, 4, 3, 2};
static const int n_preferred = 7;

/* Check if m factors into our base radices (no Rader needed) */
static int factors_into_base(int m) {
  int rem = m;
  for (int count = 0; rem > 1 && count < MAX_STAGES; count++) {
    int found = 0;
    for (int i = 0; i < n_preferred; i++) {
      if (rem % preferred_radices[i] == 0) {
        rem /= preferred_radices[i];
        found = 1;
        break;
      }
    }
    if (!found) return 0;
  }
  return rem == 1;
}

/* Check if prime p can be used as Rader radix (p-1 must factor into base) */
static int rader_supported(int p) {
  return is_prime(p) && p > 7 && factors_into_base(p - 1);
}

/* Factorize n into radices with optional radix cap.
 * Returns stage count, 0 on failure.
 * Tries base radices first (up to max_r), then Rader-supported primes. */
static int factorize_ex(int n, int *radices, int max_r) {
  int count = 0;
  int rem = n;
  while (rem > 1 && count < MAX_STAGES) {
    int found = 0;
    /* Try base radices first (skip those > max_r) */
    for (int i = 0; i < n_preferred; i++) {
      if (preferred_radices[i] <= max_r && rem % preferred_radices[i] == 0) {
        radices[count++] = preferred_radices[i];
        rem /= preferred_radices[i];
        found = 1;
        break;
      }
    }
    if (found) continue;
    /* Try Rader primes */
    for (int p = 11; p <= rem; p += 2) {
      if (p <= max_r && rem % p == 0 && rader_supported(p)) {
        radices[count++] = p;
        rem /= p;
        found = 1;
        break;
      }
    }
    if (!found) return 0;
  }
  return (rem == 1) ? count : 0;
}

/* Radix cap for cooperative shared-memory processing of small N.
 * Smaller radices → more threads per FFT → better memory coalescing. */
static int radix_cap(int n) {
  if (n <= 4) return 2;    /* N=4: [2,2] → 2 threads/FFT */
  if (n <= 64) return 4;   /* N=8-64: radix-4 stages → more threads/FFT */
  return 16;               /* default */
}

/* Resolve effective max radix: 0 = auto (use heuristic), else as given. */
static int effective_max_r(int n, int max_r) {
  return (max_r > 0) ? max_r : radix_cap(n);
}

static int factorize_mr(int n, int *radices, int max_r) {
  return factorize_ex(n, radices, effective_max_r(n, max_r));
}


/* ========================================================================== */
/* Bank-conflict padded stride                                                */
/* ========================================================================== */

static int uses_direct_path_mr(int n, int max_r) {
  int eff = effective_max_r(n, max_r);
  /* If radix cap < n, the shared-memory path gives multiple threads/FFT */
  if (eff < n) return 0;
  return n <= DIRECT_MAX_N && factors_into_base(n);
}

static int padded_stride(int n) {
  if (n < 32 || n > 4096) return n;
  if ((n & (n - 1)) == 0)
    return n + (n / 16);
  return n;
}

/* ========================================================================== */
/* Workgroup size / batching                                                  */
/* ========================================================================== */

/* Threads per FFT (before batching) */
static int wg_per_fft_mr(int n, int max_r) {
  int radices[MAX_STAGES];
  if (n < 2) return 0;
  int ns = factorize_mr(n, radices, max_r);
  if (ns == 0) return 0;

  int mr = 0;
  for (int i = 0; i < ns; i++)
    if (radices[i] > mr) mr = radices[i];

  int limit = n / mr;
  if (limit > 256) limit = 256;

  int wg = limit;
  while (wg > 1 && n % wg != 0)
    wg--;
  return wg;
}

static int batch_per_wg_mr(int n, int max_r) {
  if (uses_direct_path_mr(n, max_r)) return 256;

  int wpf = wg_per_fft_mr(n, max_r);
  if (wpf == 0) return 0;

  int shr = padded_stride(n);
  int max_by_threads = 256 / wpf;
  int max_by_shmem = MAX_SHARED_BYTES / (shr * 8);
  if (max_by_shmem < 1) max_by_shmem = 1;

  int B = max_by_threads;
  if (B > max_by_shmem) B = max_by_shmem;
  if (B < 1) B = 1;
  return B;
}

static int workgroup_size_mr(int n, int max_r) {
  if (uses_direct_path_mr(n, max_r)) return 256;

  int wpf = wg_per_fft_mr(n, max_r);
  if (wpf == 0) return 0;
  return wpf * batch_per_wg_mr(n, max_r);
}

/* Public API — default (auto) */
int fft_fused_batch_per_wg(int n) { return batch_per_wg_mr(n, 0); }
int fft_fused_workgroup_size(int n) { return workgroup_size_mr(n, 0); }

/* Public API — explicit max_radix */
int fft_fused_batch_per_wg_ex(int n, int max_r) { return batch_per_wg_mr(n, max_r); }
int fft_fused_workgroup_size_ex(int n, int max_r) { return workgroup_size_mr(n, max_r); }

/* ========================================================================== */
/* Twiddle LUT computation (host-side)                                        */
/* ========================================================================== */

/* Count Rader kernel entries needed for all Rader radices in the plan */
static int rader_lut_count_mr(int n, int max_r) {
  int radices[MAX_STAGES];
  int ns = factorize_mr(n, radices, max_r);
  int total = 0;
  for (int s = 0; s < ns; s++) {
    int R = radices[s];
    if (is_prime(R) && rader_supported(R)) {
      total += R - 1;
    }
  }
  return total;
}

static int lut_size_mr(int n, int direction, int max_r) {
  (void)direction;
  int radices[MAX_STAGES];
  int ns = factorize_mr(n, radices, max_r);
  if (ns == 0) return 0;

  int total = 0;
  int stride = 1;
  for (int s = 0; s < ns; s++) {
    int R = radices[s];
    if (stride > 1) {
      total += (R - 1) * stride;
    }
    stride *= R;
  }
  total += rader_lut_count_mr(n, max_r);
  return total;
}

int fft_fused_lut_size(int n, int direction) { return lut_size_mr(n, direction, 0); }
int fft_fused_lut_size_ex(int n, int direction, int max_r) { return lut_size_mr(n, direction, max_r); }

/* Simple host-side DFT of length M (for precomputing Rader kernels) */
static void host_dft(const double *in_re, const double *in_im,
                     double *out_re, double *out_im, int M) {
  for (int k = 0; k < M; k++) {
    double re = 0, im = 0;
    for (int j = 0; j < M; j++) {
      double angle = -2.0 * M_PI * k * j / M;
      double wr = cos(angle), wi = sin(angle);
      re += in_re[j] * wr - in_im[j] * wi;
      im += in_re[j] * wi + in_im[j] * wr;
    }
    out_re[k] = re;
    out_im[k] = im;
  }
}

static float *compute_lut_mr(int n, int direction, int max_r) {
  int radices[MAX_STAGES];
  int ns = factorize_mr(n, radices, max_r);
  if (ns == 0) return NULL;

  int total = lut_size_mr(n, direction, max_r);
  if (total == 0) return NULL;

  float *lut = (float *)calloc((size_t)total * 2, sizeof(float));
  int idx = 0;
  int stride = 1;

  /* Stage twiddles */
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

  /* Rader kernels: for each Rader prime p, store FFT(b') of length p-1 */
  for (int s = 0; s < ns; s++) {
    int p = radices[s];
    if (!is_prime(p) || !rader_supported(p)) continue;

    int M = p - 1;
    int g = primitive_root(p);

    /* Compute b'[t] = W_p^{g^{-t mod M} mod p} */
    double *b_re = (double *)calloc(M, sizeof(double));
    double *b_im = (double *)calloc(M, sizeof(double));

    /* Compute g_inv_pow[t] = g^{-t} mod p = g^{M-t} mod p */
    int *g_pow = (int *)malloc(M * sizeof(int));
    g_pow[0] = 1;
    for (int t = 1; t < M; t++)
      g_pow[t] = (int)((long long)g_pow[t - 1] * g % p);

    for (int t = 0; t < M; t++) {
      int neg_t = (M - t) % M;
      int exp_val = g_pow[neg_t]; /* g^{-t} mod p */
      double angle = (double)direction * -2.0 * M_PI * exp_val / (double)p;
      b_re[t] = cos(angle);
      b_im[t] = sin(angle);
    }
    free(g_pow);

    /* FFT(b') */
    double *B_re = (double *)calloc(M, sizeof(double));
    double *B_im = (double *)calloc(M, sizeof(double));
    host_dft(b_re, b_im, B_re, B_im, M);
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

float *fft_fused_compute_lut(int n, int direction) { return compute_lut_mr(n, direction, 0); }
float *fft_fused_compute_lut_ex(int n, int direction, int max_r) { return compute_lut_mr(n, direction, max_r); }

/* ========================================================================== */
/* Specialized radix butterflies                                              */
/* ========================================================================== */

static void emit_radix2(StrBuf *sb, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  sb_printf(sb, "      { let bt_%d = %s%d; %s%d = bt_%d + %s%d; "
                "%s%d = bt_%d - %s%d; }\n",
            uid, R(0), R(0), uid, R(1), R(1), uid, R(1));
  #undef R
}

static void emit_radix3(StrBuf *sb, int dir, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  double s3 = 0.86602540378443864676;
  sb_printf(sb, "      {\n");
  sb_printf(sb, "        let s_%d = %s%d + %s%d;\n", uid, R(1), R(2));
  sb_printf(sb, "        let d_%d = %s%d - %s%d;\n", uid, R(1), R(2));
  sb_printf(sb, "        let t_%d = %s%d;\n", uid, R(0));
  sb_printf(sb, "        %s%d = t_%d + s_%d;\n", R(0), uid, uid);
  sb_printf(sb, "        %s%d = vec2<f32>(t_%d.x - 0.5*s_%d.x %c ",
            R(1), uid, uid, dir == 1 ? '+' : '-');
  sb_float(sb, s3); sb_printf(sb, "*d_%d.y, t_%d.y - 0.5*s_%d.y %c ",
                               uid, uid, uid, dir == 1 ? '-' : '+');
  sb_float(sb, s3); sb_printf(sb, "*d_%d.x);\n", uid);
  sb_printf(sb, "        %s%d = vec2<f32>(t_%d.x - 0.5*s_%d.x %c ",
            R(2), uid, uid, dir == 1 ? '-' : '+');
  sb_float(sb, s3); sb_printf(sb, "*d_%d.y, t_%d.y - 0.5*s_%d.y %c ",
                               uid, uid, uid, dir == 1 ? '+' : '-');
  sb_float(sb, s3); sb_printf(sb, "*d_%d.x);\n", uid);
  sb_printf(sb, "      }\n");
  #undef R
}

static void emit_radix4(StrBuf *sb, int dir, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  sb_printf(sb, "      {\n");
  sb_printf(sb, "        let a_%d = %s%d + %s%d;\n", uid, R(0), R(2));
  sb_printf(sb, "        let b_%d = %s%d - %s%d;\n", uid, R(0), R(2));
  sb_printf(sb, "        let c_%d = %s%d + %s%d;\n", uid, R(1), R(3));
  if (dir == 1)
    sb_printf(sb, "        let d_%d = vec2<f32>(%s%d.y - %s%d.y, "
                  "%s%d.x - %s%d.x);\n", uid, R(1), R(3), R(3), R(1));
  else
    sb_printf(sb, "        let d_%d = vec2<f32>(%s%d.y - %s%d.y, "
                  "%s%d.x - %s%d.x);\n", uid, R(3), R(1), R(1), R(3));
  sb_printf(sb, "        %s%d = a_%d + c_%d;\n", R(0), uid, uid);
  sb_printf(sb, "        %s%d = b_%d + d_%d;\n", R(1), uid, uid);
  sb_printf(sb, "        %s%d = a_%d - c_%d;\n", R(2), uid, uid);
  sb_printf(sb, "        %s%d = b_%d - d_%d;\n", R(3), uid, uid);
  sb_printf(sb, "      }\n");
  #undef R
}

static void emit_radix5(StrBuf *sb, int dir, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  double c1 = 0.30901699437494742410, c2 = -0.80901699437494742410;
  double s1 = 0.95105651629515357212, s2 = 0.58778525229247312917;
  sb_printf(sb, "      {\n");
  sb_printf(sb, "        let a1_%d = %s%d + %s%d;\n", uid, R(1), R(4));
  sb_printf(sb, "        let a2_%d = %s%d + %s%d;\n", uid, R(2), R(3));
  sb_printf(sb, "        let b1_%d = %s%d - %s%d;\n", uid, R(1), R(4));
  sb_printf(sb, "        let b2_%d = %s%d - %s%d;\n", uid, R(2), R(3));
  sb_printf(sb, "        %s%d = %s%d + a1_%d + a2_%d;\n", R(0), R(0), uid, uid);

  /* outputs 1-4: x0 + c*a + j*dir*(s*b) with conjugate pairs */
  double ca[4][2] = {{c1,c2},{c2,c1},{c2,c1},{c1,c2}};
  double sa[4][2] = {{s1,s2},{s2,-s1},{-s2,s1},{-s1,-s2}};
  for (int out = 0; out < 4; out++) {
    sb_printf(sb, "        %s%d = vec2<f32>(%s%d.x", R(out+1), R(0));
    sb_printf(sb, " + "); sb_float(sb, ca[out][0]); sb_printf(sb, "*a1_%d.x", uid);
    sb_printf(sb, " + "); sb_float(sb, ca[out][1]); sb_printf(sb, "*a2_%d.x", uid);
    double ds1 = (dir == 1) ? sa[out][0] : -sa[out][0];
    double ds2 = (dir == 1) ? sa[out][1] : -sa[out][1];
    if (ds1 >= 0) { sb_printf(sb, " + "); sb_float(sb, ds1); }
    else { sb_printf(sb, " - "); sb_float(sb, -ds1); }
    sb_printf(sb, "*b1_%d.y", uid);
    if (ds2 >= 0) { sb_printf(sb, " + "); sb_float(sb, ds2); }
    else { sb_printf(sb, " - "); sb_float(sb, -ds2); }
    sb_printf(sb, "*b2_%d.y, %s%d.y", uid, R(0));
    sb_printf(sb, " + "); sb_float(sb, ca[out][0]); sb_printf(sb, "*a1_%d.y", uid);
    sb_printf(sb, " + "); sb_float(sb, ca[out][1]); sb_printf(sb, "*a2_%d.y", uid);
    if (-ds1 >= 0) { sb_printf(sb, " + "); sb_float(sb, -ds1); }
    else { sb_printf(sb, " - "); sb_float(sb, ds1); }
    sb_printf(sb, "*b1_%d.x", uid);
    if (-ds2 >= 0) { sb_printf(sb, " + "); sb_float(sb, -ds2); }
    else { sb_printf(sb, " - "); sb_float(sb, ds2); }
    sb_printf(sb, "*b2_%d.x);\n", uid);
  }
  sb_printf(sb, "      }\n");
  #undef R
}

static void emit_radix7(StrBuf *sb, int dir, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  double cc[3] = {cos(2*M_PI/7), cos(4*M_PI/7), cos(6*M_PI/7)};
  double ss[3] = {sin(2*M_PI/7), sin(4*M_PI/7), sin(6*M_PI/7)};
  sb_printf(sb, "      {\n");
  sb_printf(sb, "        let a1_%d = %s%d + %s%d;\n", uid, R(1), R(6));
  sb_printf(sb, "        let a2_%d = %s%d + %s%d;\n", uid, R(2), R(5));
  sb_printf(sb, "        let a3_%d = %s%d + %s%d;\n", uid, R(3), R(4));
  sb_printf(sb, "        let b1_%d = %s%d - %s%d;\n", uid, R(1), R(6));
  sb_printf(sb, "        let b2_%d = %s%d - %s%d;\n", uid, R(2), R(5));
  sb_printf(sb, "        let b3_%d = %s%d - %s%d;\n", uid, R(3), R(4));
  sb_printf(sb, "        %s%d = %s%d + a1_%d + a2_%d + a3_%d;\n",
            R(0), R(0), uid, uid, uid);
  int ci[6][3] = {{0,1,2},{1,2,0},{2,0,1},{2,0,1},{1,2,0},{0,1,2}};
  int sign_s[6] = {1,1,1,-1,-1,-1};
  for (int out = 0; out < 6; out++) {
    int k = out + 1;
    int si = (dir == 1) ? sign_s[out] : -sign_s[out];
    sb_printf(sb, "        %s%d = vec2<f32>(%s%d.x", R(k), R(0));
    for (int j = 0; j < 3; j++) {
      sb_printf(sb, " + "); sb_float(sb, cc[ci[out][j]]);
      sb_printf(sb, "*a%d_%d.x", j+1, uid);
    }
    for (int j = 0; j < 3; j++) {
      sb_printf(sb, " %c ", si > 0 ? '+' : '-');
      sb_float(sb, ss[ci[out][j]]);
      sb_printf(sb, "*b%d_%d.y", j+1, uid);
    }
    sb_printf(sb, ", %s%d.y", R(0));
    for (int j = 0; j < 3; j++) {
      sb_printf(sb, " + "); sb_float(sb, cc[ci[out][j]]);
      sb_printf(sb, "*a%d_%d.y", j+1, uid);
    }
    for (int j = 0; j < 3; j++) {
      sb_printf(sb, " %c ", si > 0 ? '-' : '+');
      sb_float(sb, ss[ci[out][j]]);
      sb_printf(sb, "*b%d_%d.x", j+1, uid);
    }
    sb_printf(sb, ");\n");
  }
  sb_printf(sb, "      }\n");
  #undef R
}

static void emit_radix8(StrBuf *sb, int dir, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  double sq = 0.70710678118654752440;
  sb_printf(sb, "      {\n");
  sb_printf(sb, "        var e0_%d = %s%d + %s%d;\n", uid, R(0), R(4));
  sb_printf(sb, "        var e1_%d = %s%d - %s%d;\n", uid, R(0), R(4));
  sb_printf(sb, "        var e2_%d = %s%d + %s%d;\n", uid, R(2), R(6));
  sb_printf(sb, "        var e3_%d = %s%d - %s%d;\n", uid, R(2), R(6));
  if (dir == 1)
    sb_printf(sb, "        e3_%d = vec2<f32>(e3_%d.y, -e3_%d.x);\n", uid,uid,uid);
  else
    sb_printf(sb, "        e3_%d = vec2<f32>(-e3_%d.y, e3_%d.x);\n", uid,uid,uid);
  sb_printf(sb, "        let f0_%d = e0_%d + e2_%d;\n", uid, uid, uid);
  sb_printf(sb, "        let f1_%d = e1_%d + e3_%d;\n", uid, uid, uid);
  sb_printf(sb, "        let f2_%d = e0_%d - e2_%d;\n", uid, uid, uid);
  sb_printf(sb, "        let f3_%d = e1_%d - e3_%d;\n", uid, uid, uid);
  sb_printf(sb, "        var o0_%d = %s%d + %s%d;\n", uid, R(1), R(5));
  sb_printf(sb, "        var o1_%d = %s%d - %s%d;\n", uid, R(1), R(5));
  sb_printf(sb, "        var o2_%d = %s%d + %s%d;\n", uid, R(3), R(7));
  sb_printf(sb, "        var o3_%d = %s%d - %s%d;\n", uid, R(3), R(7));
  if (dir == 1)
    sb_printf(sb, "        o3_%d = vec2<f32>(o3_%d.y, -o3_%d.x);\n", uid,uid,uid);
  else
    sb_printf(sb, "        o3_%d = vec2<f32>(-o3_%d.y, o3_%d.x);\n", uid,uid,uid);
  sb_printf(sb, "        let g0_%d = o0_%d + o2_%d;\n", uid, uid, uid);
  sb_printf(sb, "        var g1_%d = o1_%d + o3_%d;\n", uid, uid, uid);
  sb_printf(sb, "        let g2_%d = o0_%d - o2_%d;\n", uid, uid, uid);
  sb_printf(sb, "        var g3_%d = o1_%d - o3_%d;\n", uid, uid, uid);
  if (dir == 1) {
    sb_printf(sb, "        g1_%d = vec2<f32>(", uid);
    sb_float(sb, sq); sb_printf(sb, "*(g1_%d.x+g1_%d.y), ", uid, uid);
    sb_float(sb, sq); sb_printf(sb, "*(g1_%d.y-g1_%d.x));\n", uid, uid);
    sb_printf(sb, "        g3_%d = vec2<f32>(", uid);
    sb_float(sb, -sq); sb_printf(sb, "*(g3_%d.x-g3_%d.y), ", uid, uid);
    sb_float(sb, -sq); sb_printf(sb, "*(g3_%d.x+g3_%d.y));\n", uid, uid);
  } else {
    sb_printf(sb, "        g1_%d = vec2<f32>(", uid);
    sb_float(sb, sq); sb_printf(sb, "*(g1_%d.x-g1_%d.y), ", uid, uid);
    sb_float(sb, sq); sb_printf(sb, "*(g1_%d.y+g1_%d.x));\n", uid, uid);
    sb_printf(sb, "        g3_%d = vec2<f32>(", uid);
    sb_float(sb, -sq); sb_printf(sb, "*(g3_%d.x+g3_%d.y), ", uid, uid);
    sb_float(sb, -sq); sb_printf(sb, "*(g3_%d.y-g3_%d.x));\n", uid, uid);
  }
  if (dir == 1)
    sb_printf(sb, "        let g2r_%d = vec2<f32>(g2_%d.y, -g2_%d.x);\n", uid,uid,uid);
  else
    sb_printf(sb, "        let g2r_%d = vec2<f32>(-g2_%d.y, g2_%d.x);\n", uid,uid,uid);
  sb_printf(sb, "        %s%d = f0_%d + g0_%d;\n", R(0), uid, uid);
  sb_printf(sb, "        %s%d = f1_%d + g1_%d;\n", R(1), uid, uid);
  sb_printf(sb, "        %s%d = f2_%d + g2r_%d;\n", R(2), uid, uid);
  sb_printf(sb, "        %s%d = f3_%d + g3_%d;\n", R(3), uid, uid);
  sb_printf(sb, "        %s%d = f0_%d - g0_%d;\n", R(4), uid, uid);
  sb_printf(sb, "        %s%d = f1_%d - g1_%d;\n", R(5), uid, uid);
  sb_printf(sb, "        %s%d = f2_%d - g2r_%d;\n", R(6), uid, uid);
  sb_printf(sb, "        %s%d = f3_%d - g3_%d;\n", R(7), uid, uid);
  sb_printf(sb, "      }\n");
  #undef R
}

static void emit_radix16(StrBuf *sb, int dir, const char *p, int b, int uid) {
  (void)uid;
  #define R(k) p, b + (k)
  double sq = 0.70710678118654752440;
  sb_printf(sb, "      {\n");
  for (int col = 0; col < 4; col++) {
    int i0=col,i1=col+4,i2=col+8,i3=col+12;
    sb_printf(sb, "        { let a_ = %s%d + %s%d; let b_ = %s%d - %s%d; "
              "let c_ = %s%d + %s%d; ", R(i0),R(i2), R(i0),R(i2), R(i1),R(i3));
    if (dir==1) sb_printf(sb, "let d_ = vec2<f32>(%s%d.y-%s%d.y, %s%d.x-%s%d.x); ",
                           R(i1),R(i3),R(i3),R(i1));
    else sb_printf(sb, "let d_ = vec2<f32>(%s%d.y-%s%d.y, %s%d.x-%s%d.x); ",
                   R(i3),R(i1),R(i1),R(i3));
    sb_printf(sb, "%s%d = a_+c_; %s%d = b_+d_; %s%d = a_-c_; %s%d = b_-d_; }\n",
              R(i0),R(i1),R(i2),R(i3));
  }
  /* Intermediate twiddles */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(5));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.x+%s%d.y), ", R(5),R(5));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.y-%s%d.x));\n", R(5),R(5));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(5));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.x-%s%d.y), ", R(5),R(5));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.y+%s%d.x));\n", R(5),R(5));
  }
  if (dir==1) sb_printf(sb, "        %s%d = vec2<f32>(%s%d.y, -%s%d.x);\n", R(6),R(6),R(6));
  else sb_printf(sb, "        %s%d = vec2<f32>(-%s%d.y, %s%d.x);\n", R(6),R(6),R(6));
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(7));
    sb_float(sb,-sq); sb_printf(sb, "*(%s%d.x-%s%d.y), ", R(7),R(7));
    sb_float(sb,-sq); sb_printf(sb, "*(%s%d.x+%s%d.y));\n", R(7),R(7));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(7));
    sb_float(sb,-sq); sb_printf(sb, "*(%s%d.x+%s%d.y), ", R(7),R(7));
    sb_float(sb, sq); sb_printf(sb, "*(%s%d.x-%s%d.y));\n", R(7),R(7));
  }
  if (dir==1) sb_printf(sb, "        %s%d = vec2<f32>(%s%d.y, -%s%d.x);\n", R(9),R(9),R(9));
  else sb_printf(sb, "        %s%d = vec2<f32>(-%s%d.y, %s%d.x);\n", R(9),R(9),R(9));
  sb_printf(sb, "        %s%d = vec2<f32>(-%s%d.x, -%s%d.y);\n", R(10),R(10),R(10));
  if (dir==1) sb_printf(sb, "        %s%d = vec2<f32>(-%s%d.y, %s%d.x);\n", R(11),R(11),R(11));
  else sb_printf(sb, "        %s%d = vec2<f32>(%s%d.y, -%s%d.x);\n", R(11),R(11),R(11));
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(13));
    sb_float(sb,-sq); sb_printf(sb, "*(%s%d.x-%s%d.y), ", R(13),R(13));
    sb_float(sb,-sq); sb_printf(sb, "*(%s%d.x+%s%d.y));\n", R(13),R(13));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(13));
    sb_float(sb,-sq); sb_printf(sb, "*(%s%d.x+%s%d.y), ", R(13),R(13));
    sb_float(sb, sq); sb_printf(sb, "*(%s%d.x-%s%d.y));\n", R(13),R(13));
  }
  if (dir==1) sb_printf(sb, "        %s%d = vec2<f32>(-%s%d.y, %s%d.x);\n", R(14),R(14),R(14));
  else sb_printf(sb, "        %s%d = vec2<f32>(%s%d.y, -%s%d.x);\n", R(14),R(14),R(14));
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(15));
    sb_float(sb, sq); sb_printf(sb, "*(%s%d.x-%s%d.y), ", R(15),R(15));
    sb_float(sb, sq); sb_printf(sb, "*(%s%d.y+%s%d.x));\n", R(15),R(15));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(15));
    sb_float(sb, sq); sb_printf(sb, "*(%s%d.x+%s%d.y), ", R(15),R(15));
    sb_float(sb, sq); sb_printf(sb, "*(%s%d.y-%s%d.x));\n", R(15),R(15));
  }
  for (int row = 0; row < 4; row++) {
    int i0=row*4,i1=row*4+1,i2=row*4+2,i3=row*4+3;
    sb_printf(sb, "        { let a_ = %s%d + %s%d; let b_ = %s%d - %s%d; "
              "let c_ = %s%d + %s%d; ", R(i0),R(i2), R(i0),R(i2), R(i1),R(i3));
    if (dir==1) sb_printf(sb, "let d_ = vec2<f32>(%s%d.y-%s%d.y, %s%d.x-%s%d.x); ",
                           R(i1),R(i3),R(i3),R(i1));
    else sb_printf(sb, "let d_ = vec2<f32>(%s%d.y-%s%d.y, %s%d.x-%s%d.x); ",
                   R(i3),R(i1),R(i1),R(i3));
    sb_printf(sb, "%s%d = a_+c_; %s%d = b_+d_; %s%d = a_-c_; %s%d = b_-d_; }\n",
              R(i0),R(i1),R(i2),R(i3));
  }
  /* Final permutation: swap (1,4),(2,8),(3,12),(6,9),(7,13),(11,14) */
  int swaps[][2] = {{1,4},{2,8},{3,12},{6,9},{7,13},{11,14}};
  for (int s = 0; s < 6; s++)
    sb_printf(sb, "        { let t_ = %s%d; %s%d = %s%d; %s%d = t_; }\n",
              R(swaps[s][0]),R(swaps[s][0]),R(swaps[s][1]),R(swaps[s][1]));
  sb_printf(sb, "      }\n");
  #undef R
}

/* ========================================================================== */
/* Rader butterfly — prime p via cyclic convolution                           */
/* ========================================================================== */

/* Emit in-register DFT of length M using O(M^2) with baked twiddles.
 * Operates on registers prefix[base..base+M-1].
 * Uses temp registers prefix[tmp..tmp+M-1]. */
static void emit_register_dft(StrBuf *sb, int M, int direction,
                               const char *pfx, int base, int tmp, int uid) {
  (void)uid;
  /* Direct DFT: O[k] = sum_{j=0}^{M-1} I[j] * W_M^{kj} */
  for (int k = 0; k < M; k++) {
    sb_printf(sb, "        %s%d = vec2<f32>(0.0, 0.0);\n", pfx, tmp + k);
    for (int j = 0; j < M; j++) {
      double angle = (double)direction * -2.0 * M_PI * k * j / M;
      double wr = cos(angle), wi = sin(angle);
      if (fabs(wr) < 1e-15) wr = 0;
      if (fabs(wi) < 1e-15) wi = 0;
      if (wr == 1.0 && wi == 0.0) {
        sb_printf(sb, "        %s%d = %s%d + %s%d;\n",
                  pfx, tmp+k, pfx, tmp+k, pfx, base+j);
      } else if (wr == -1.0 && wi == 0.0) {
        sb_printf(sb, "        %s%d = %s%d - %s%d;\n",
                  pfx, tmp+k, pfx, tmp+k, pfx, base+j);
      } else if (wr == 0.0 && wi == -1.0) {
        sb_printf(sb, "        %s%d = %s%d + vec2<f32>(%s%d.y, -%s%d.x);\n",
                  pfx,tmp+k,pfx,tmp+k,pfx,base+j,pfx,base+j);
      } else if (wr == 0.0 && wi == 1.0) {
        sb_printf(sb, "        %s%d = %s%d + vec2<f32>(-%s%d.y, %s%d.x);\n",
                  pfx,tmp+k,pfx,tmp+k,pfx,base+j,pfx,base+j);
      } else {
        sb_printf(sb, "        %s%d = %s%d + vec2<f32>(%s%d.x*",
                  pfx,tmp+k,pfx,tmp+k,pfx,base+j);
        sb_float(sb, wr);
        sb_printf(sb, " - %s%d.y*", pfx, base+j);
        sb_float(sb, wi);
        sb_printf(sb, ", %s%d.x*", pfx, base+j);
        sb_float(sb, wi);
        sb_printf(sb, " + %s%d.y*", pfx, base+j);
        sb_float(sb, wr);
        sb_printf(sb, ");\n");
      }
    }
  }
  /* Copy results back: base[k] = tmp[k] */
  for (int k = 0; k < M; k++)
    sb_printf(sb, "        %s%d = %s%d;\n", pfx, base+k, pfx, tmp+k);
}

static void emit_rader_butterfly(StrBuf *sb, int p, int direction,
                                  const char *pfx, int base, int uid,
                                  int rader_lut_offset) {
  (void)direction;
  int M = p - 1;
  int g = primitive_root(p);

  /* Compute g^j mod p for j=0..M-1 */
  int *g_pow = (int *)malloc(M * sizeof(int));
  g_pow[0] = 1;
  for (int j = 1; j < M; j++)
    g_pow[j] = (int)((long long)g_pow[j-1] * g % p);

  sb_printf(sb, "      { // Rader prime=%d, g=%d\n", p, g);

  /* Step 1: x0 = sum of all p inputs */
  sb_printf(sb, "        var x0_%d: vec2<f32> = %s%d;\n", uid, pfx, base);
  for (int j = 1; j < p; j++)
    sb_printf(sb, "        x0_%d = x0_%d + %s%d;\n", uid, uid, pfx, base+j);

  /* Step 2: Permute by g^j: a[j] = input[g^j mod p], j=0..M-1
   * Note: input[0] is the DC term, input[1..p-1] are the non-zero indices.
   * We store into the SAME registers starting at base. */
  /* First, save to temp vars */
  for (int j = 0; j < M; j++)
    sb_printf(sb, "        let ra_%d_%d = %s%d;\n", uid, j, pfx, base + g_pow[j]);

  /* Copy permuted values back into base registers */
  for (int j = 0; j < M; j++)
    sb_printf(sb, "        %s%d = ra_%d_%d;\n", pfx, base + j, uid, j);

  /* Step 3: Forward DFT of length M on registers base[0..M-1]
   * We use base[M] (= base[p-1]) and beyond as temp space.
   * Total temp needed: M registers starting at base+M */
  int tmp_base = base + M;
  sb_printf(sb, "        // Sub-FFT(%d) forward\n", M);
  emit_register_dft(sb, M, 1, pfx, base, tmp_base, uid * 1000);

  /* Step 4: Pointwise multiply with precomputed kernel from LUT */
  sb_printf(sb, "        // Pointwise multiply with Rader kernel\n");
  for (int k = 0; k < M; k++) {
    int li = rader_lut_offset + k;
    sb_printf(sb, "        { let kk: vec2<f32> = vec2<f32>("
              "lut.d[%du], lut.d[%du]);\n", li*2, li*2+1);
    sb_printf(sb, "          %s%d = vec2<f32>("
              "%s%d.x*kk.x - %s%d.y*kk.y, "
              "%s%d.x*kk.y + %s%d.y*kk.x); }\n",
              pfx, base+k, pfx, base+k, pfx, base+k,
              pfx, base+k, pfx, base+k);
  }

  /* Step 5: Inverse DFT of length M */
  sb_printf(sb, "        // Sub-FFT(%d) inverse\n", M);
  emit_register_dft(sb, M, -1, pfx, base, tmp_base, uid * 1000 + 500);

  /* Step 6: Scale by 1/M and add x0, then unpermute */
  double inv_M = 1.0 / M;
  sb_printf(sb, "        // Scale + unpermute\n");
  /* Save scaled results to temps */
  for (int j = 0; j < M; j++) {
    sb_printf(sb, "        let rb_%d_%d = x0_%d + %s%d * vec2<f32>(",
              uid, j, uid, pfx, base + j);
    sb_float(sb, inv_M);
    sb_printf(sb, ", ");
    sb_float(sb, inv_M);
    sb_printf(sb, ");\n");
  }

  /* Unpermute: output[g^j mod p] = result[j], output[0] = x0 */
  sb_printf(sb, "        %s%d = x0_%d;\n", pfx, base, uid);
  for (int j = 0; j < M; j++)
    sb_printf(sb, "        %s%d = rb_%d_%d;\n", pfx, base + g_pow[j], uid, j);

  sb_printf(sb, "      }\n");
  free(g_pow);
}

/* ========================================================================== */
/* Butterfly dispatch                                                         */
/* ========================================================================== */

static void emit_radix_butterfly(StrBuf *sb, int radix, int direction,
                                 const char *prefix, int base, int uid,
                                 int rader_lut_offset) {
  switch (radix) {
    case 2:  emit_radix2(sb, prefix, base, uid); break;
    case 3:  emit_radix3(sb, direction, prefix, base, uid); break;
    case 4:  emit_radix4(sb, direction, prefix, base, uid); break;
    case 5:  emit_radix5(sb, direction, prefix, base, uid); break;
    case 7:  emit_radix7(sb, direction, prefix, base, uid); break;
    case 8:  emit_radix8(sb, direction, prefix, base, uid); break;
    case 16: emit_radix16(sb, direction, prefix, base, uid); break;
    default:
      if (is_prime(radix) && rader_supported(radix))
        emit_rader_butterfly(sb, radix, direction, prefix, base, uid,
                             rader_lut_offset);
      break;
  }
}

/* ========================================================================== */
/* Main generator                                                             */
/* ========================================================================== */

static char *gen_fft_fused_mr(int n, int direction, int max_r) {
  int radices[MAX_STAGES];
  int n_stages;

  if (n < 2) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  n_stages = factorize_mr(n, radices, max_r);
  if (n_stages == 0) return NULL;

  int max_radix = 0;
  for (int i = 0; i < n_stages; i++)
    if (radices[i] > max_radix) max_radix = radices[i];

  /* ===== Direct register path: all stages in one thread, no shared mem ===== */
  if (uses_direct_path_mr(n, max_r)) {
    int B_dir = 256; /* 1 thread per FFT */
    StrBuf sb;
    sb_init(&sb);

    sb_printf(&sb, "struct SrcBuf { d: array<f32> };\n");
    sb_printf(&sb, "struct DstBuf { d: array<f32> };\n");
    sb_printf(&sb, "@group(0) @binding(0) var<storage, read> src: SrcBuf;\n");
    sb_printf(&sb, "@group(0) @binding(1) var<storage, read_write> dst: DstBuf;\n\n");
    sb_printf(&sb, "@compute @workgroup_size(%d)\n", B_dir);
    sb_printf(&sb, "fn main(\n");
    sb_printf(&sb, "  @builtin(local_invocation_id) lid: vec3<u32>,\n");
    sb_printf(&sb, "  @builtin(workgroup_id) wid: vec3<u32>\n");
    sb_printf(&sb, ") {\n");
    sb_printf(&sb, "  let t: u32 = lid.x;\n");
    sb_printf(&sb, "  let base: u32 = (wid.x * %uu + t) * %uu;\n", B_dir, n);

    if (n_stages == 1) {
      /* Single stage: just load → butterfly → store */
      for (int k = 0; k < n; k++)
        sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                  "src.d[(base + %uu) * 2u], src.d[(base + %uu) * 2u + 1u]);\n",
                  k, k, k);
      sb_printf(&sb, "\n");
      emit_radix_butterfly(&sb, n, direction, "v", 0, 0, 0);
      sb_printf(&sb, "\n");
      for (int k = 0; k < n; k++) {
        sb_printf(&sb, "  dst.d[(base + %uu) * 2u] = v%d.x;\n", k, k);
        sb_printf(&sb, "  dst.d[(base + %uu) * 2u + 1u] = v%d.y;\n", k, k);
      }
    } else {
      /* Multi-stage: two register arrays (v, w) + temp (r) for butterflies.
       * Stockham reorder happens via scatter-read/scatter-write between arrays. */
      for (int k = 0; k < n; k++)
        sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                  "src.d[(base + %uu) * 2u], src.d[(base + %uu) * 2u + 1u]);\n",
                  k, k, k);
      for (int k = 0; k < n; k++)
        sb_printf(&sb, "  var w%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", k);
      for (int k = 0; k < max_radix; k++)
        sb_printf(&sb, "  var r%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", k);

      int stride = 1;
      int src_is_v = 1; /* toggle: 1=read from v, write to w; 0=reverse */
      for (int s = 0; s < n_stages; s++) {
        int R = radices[s];
        int n_bf = n / R;
        const char *sp = src_is_v ? "v" : "w";
        const char *dp = src_is_v ? "w" : "v";

        sb_printf(&sb, "\n  // Stage %d: radix-%d, stride=%d\n", s, R, stride);

        for (int bf = 0; bf < n_bf; bf++) {
          int grp = bf / stride;
          int pos = bf % stride;

          /* Scatter-read into r0..r{R-1} */
          for (int k = 0; k < R; k++) {
            int si = grp * stride + pos + k * n_bf;
            sb_printf(&sb, "  r%d = %s%d;\n", k, sp, si);
          }

          /* Twiddle r1..r{R-1} (baked constants, no LUT) */
          if (stride > 1) {
            for (int k = 1; k < R; k++) {
              double angle = (double)direction * -2.0 * M_PI * k * pos
                             / (double)(stride * R);
              double wr = cos(angle), wi = sin(angle);
              if (fabs(wr) < 1e-15) wr = 0;
              if (fabs(wi) < 1e-15) wi = 0;
              if (fabs(wr - 1.0) < 1e-15 && fabs(wi) < 1e-15)
                continue; /* twiddle is 1+0i, skip */
              if (fabs(wr) < 1e-15 && fabs(wi - 1.0) < 1e-15) {
                sb_printf(&sb, "  r%d = vec2<f32>(-r%d.y, r%d.x);\n", k, k, k);
                continue;
              }
              if (fabs(wr) < 1e-15 && fabs(wi + 1.0) < 1e-15) {
                sb_printf(&sb, "  r%d = vec2<f32>(r%d.y, -r%d.x);\n", k, k, k);
                continue;
              }
              if (fabs(wr + 1.0) < 1e-15 && fabs(wi) < 1e-15) {
                sb_printf(&sb, "  r%d = -r%d;\n", k, k);
                continue;
              }
              sb_printf(&sb, "  { let tw = vec2<f32>(");
              sb_float(&sb, wr);
              sb_printf(&sb, ", ");
              sb_float(&sb, wi);
              sb_printf(&sb, "); r%d = vec2<f32>("
                        "r%d.x*tw.x - r%d.y*tw.y, "
                        "r%d.x*tw.y + r%d.y*tw.x); }\n",
                        k, k, k, k, k);
            }
          }

          /* Butterfly on r0..r{R-1} */
          emit_radix_butterfly(&sb, R, direction, "r", 0, s*1000+bf, 0);

          /* Scatter-write to destination */
          for (int k = 0; k < R; k++) {
            int di = grp * stride * R + pos + k * stride;
            sb_printf(&sb, "  %s%d = r%d;\n", dp, di, k);
          }
        }

        stride *= R;
        src_is_v = !src_is_v;
      }

      /* Store final result (whichever array is now "src") */
      const char *final_pfx = src_is_v ? "v" : "w";
      sb_printf(&sb, "\n");
      for (int k = 0; k < n; k++) {
        sb_printf(&sb, "  dst.d[(base + %uu) * 2u] = %s%d.x;\n",
                  k, final_pfx, k);
        sb_printf(&sb, "  dst.d[(base + %uu) * 2u + 1u] = %s%d.y;\n",
                  k, final_pfx, k);
      }
    }

    sb_printf(&sb, "}\n");
    return sb.buf;
  }

  /* ===== Shared-memory path: multi-stage ===== */
  int wpf = wg_per_fft_mr(n, max_r);
  if (wpf < 1) return NULL;
  int B = batch_per_wg_mr(n, max_r);
  int total_wg = wpf * B;

  int epr = n / wpf;
  int shr_stride = padded_stride(n);
  int total_shr = B * shr_stride;

  /* Compute LUT offsets for stage twiddles */
  int lut_offsets[MAX_STAGES];
  int lut_total = 0;
  {
    int stride = 1;
    for (int s = 0; s < n_stages; s++) {
      lut_offsets[s] = lut_total;
      int R = radices[s];
      if (stride > 1)
        lut_total += (R - 1) * stride;
      stride *= R;
    }
  }

  /* Compute Rader kernel LUT offsets per stage */
  int rader_offsets[MAX_STAGES];
  {
    int roff = lut_total;
    for (int s = 0; s < n_stages; s++) {
      rader_offsets[s] = roff;
      int R = radices[s];
      if (is_prime(R) && rader_supported(R))
        roff += R - 1;
    }
    lut_total = roff;
  }
  int has_lut = (lut_total > 0);

  /* Max registers needed: for Rader primes, need R + (R-1) temp registers */
  int max_regs_per_bf = max_radix;
  for (int s = 0; s < n_stages; s++) {
    int R = radices[s];
    if (is_prime(R) && rader_supported(R)) {
      int need = R + (R - 1); /* base + temp for DFT */
      if (need > max_regs_per_bf) max_regs_per_bf = need;
    }
  }

  StrBuf sb;
  sb_init(&sb);

  /* Bindings */
  sb_printf(&sb, "struct SrcBuf { d: array<f32> };\n");
  sb_printf(&sb, "struct DstBuf { d: array<f32> };\n");
  sb_printf(&sb, "@group(0) @binding(0) var<storage, read> src: SrcBuf;\n");
  sb_printf(&sb, "@group(0) @binding(1) var<storage, read_write> dst: DstBuf;\n");
  if (has_lut) {
    sb_printf(&sb, "struct LutBuf { d: array<f32> };\n");
    sb_printf(&sb, "@group(0) @binding(2) var<storage, read> lut: LutBuf;\n");
  }
  sb_printf(&sb, "\n");

  /* Shared memory */
  sb_printf(&sb, "var<workgroup> s_re: array<f32, %d>;\n", total_shr);
  sb_printf(&sb, "var<workgroup> s_im: array<f32, %d>;\n\n", total_shr);

  /* Entry point */
  sb_printf(&sb, "@compute @workgroup_size(%d)\n", total_wg);
  sb_printf(&sb, "fn main(\n");
  sb_printf(&sb, "  @builtin(local_invocation_id) lid: vec3<u32>,\n");
  sb_printf(&sb, "  @builtin(workgroup_id) wid: vec3<u32>\n");
  sb_printf(&sb, ") {\n");
  sb_printf(&sb, "  let t: u32 = lid.x;\n");

  if (B > 1) {
    sb_printf(&sb, "  let fft_t: u32 = t %% %uu;\n", wpf);
    sb_printf(&sb, "  let fft_id: u32 = t / %uu;\n", wpf);
    sb_printf(&sb, "  let shr_off: u32 = fft_id * %uu;\n", shr_stride);
    sb_printf(&sb, "  let base: u32 = (wid.x * %uu + fft_id) * %uu;\n\n", B, n);
  } else {
    sb_printf(&sb, "  let fft_t: u32 = t;\n");
    sb_printf(&sb, "  let shr_off: u32 = 0u;\n");
    sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n\n", n);
  }

  /* ---- Global load -> shared ---- */
  if (shr_stride == n) {
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  s_re[shr_off + fft_t + %uu] = "
                "src.d[(base + fft_t + %uu) * 2u];\n", e*wpf, e*wpf);
      sb_printf(&sb, "  s_im[shr_off + fft_t + %uu] = "
                "src.d[(base + fft_t + %uu) * 2u + 1u];\n", e*wpf, e*wpf);
    }
  } else {
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  { let li: u32 = fft_t + %uu; "
                "let pi: u32 = li + li / 16u;\n", e*wpf);
      sb_printf(&sb, "    s_re[shr_off + pi] = src.d[(base + li) * 2u];\n");
      sb_printf(&sb, "    s_im[shr_off + pi] = src.d[(base + li) * 2u + 1u]; }\n");
    }
  }
  sb_printf(&sb, "  workgroupBarrier();\n");

  /* ---- Stockham FFT stages ---- */
  int stride = 1;
  for (int stage = 0; stage < n_stages; stage++) {
    int R = radices[stage];
    int n_bf = n / R;
    int bpt = (n_bf + wpf - 1) / wpf;
    int read_stride = n / R;
    int total_regs = bpt * max_regs_per_bf;

    sb_printf(&sb, "\n  // Stage %d: radix-%d, stride=%d\n", stage, R, stride);
    sb_printf(&sb, "  {\n");
    for (int i = 0; i < total_regs; i++)
      sb_printf(&sb, "    var v%d: vec2<f32> = vec2<f32>(0.0, 0.0);\n", i);

    /* Phase 1: Read all butterfly elements from shared */
    for (int b = 0; b < bpt; b++) {
      int rb = b * max_regs_per_bf; /* register base for this butterfly */
      sb_printf(&sb, "    {\n");
      if (bpt == 1)
        sb_printf(&sb, "      let bf: u32 = fft_t;\n");
      else
        sb_printf(&sb, "      let bf: u32 = fft_t * %uu + %uu;\n", bpt, b);

      int needs_guard = (bpt * wpf > n_bf);
      if (needs_guard)
        sb_printf(&sb, "      if (bf < %uu) {\n", n_bf);
      const char *g = needs_guard ? "  " : "";

      if (shr_stride == n) {
        if (stride == 1) {
          for (int k = 0; k < R; k++)
            sb_printf(&sb, "      %sv%d = vec2<f32>("
                      "s_re[shr_off + bf + %uu], s_im[shr_off + bf + %uu]);\n",
                      g, rb+k, k*read_stride, k*read_stride);
        } else {
          sb_printf(&sb, "      %slet grp: u32 = bf / %uu;\n", g, stride);
          sb_printf(&sb, "      %slet pos: u32 = bf %% %uu;\n", g, stride);
          for (int k = 0; k < R; k++)
            sb_printf(&sb, "      %sv%d = vec2<f32>("
                      "s_re[shr_off + grp * %uu + pos + %uu], "
                      "s_im[shr_off + grp * %uu + pos + %uu]);\n",
                      g, rb+k, stride, k*read_stride, stride, k*read_stride);
        }
      } else {
        if (stride == 1) {
          for (int k = 0; k < R; k++) {
            sb_printf(&sb, "      %s{ let li: u32 = bf + %uu; "
                      "let pi: u32 = li + li / 16u;\n", g, k*read_stride);
            sb_printf(&sb, "      %s  v%d = vec2<f32>("
                      "s_re[shr_off + pi], s_im[shr_off + pi]); }\n",
                      g, rb+k);
          }
        } else {
          sb_printf(&sb, "      %slet grp: u32 = bf / %uu;\n", g, stride);
          sb_printf(&sb, "      %slet pos: u32 = bf %% %uu;\n", g, stride);
          for (int k = 0; k < R; k++) {
            sb_printf(&sb, "      %s{ let li: u32 = grp * %uu + pos + %uu; "
                      "let pi: u32 = li + li / 16u;\n", g, stride, k*read_stride);
            sb_printf(&sb, "      %s  v%d = vec2<f32>("
                      "s_re[shr_off + pi], s_im[shr_off + pi]); }\n",
                      g, rb+k);
          }
        }
      }
      if (needs_guard) sb_printf(&sb, "      }\n");
      sb_printf(&sb, "    }\n");
    }

    sb_printf(&sb, "    workgroupBarrier();\n");

    /* Phase 2: Twiddle + butterfly + write */
    for (int b = 0; b < bpt; b++) {
      int rb = b * max_regs_per_bf; /* register base for this butterfly */
      sb_printf(&sb, "    {\n");
      if (bpt == 1)
        sb_printf(&sb, "      let bf: u32 = fft_t;\n");
      else
        sb_printf(&sb, "      let bf: u32 = fft_t * %uu + %uu;\n", bpt, b);

      int needs_guard = (bpt * wpf > n_bf);
      if (needs_guard)
        sb_printf(&sb, "      if (bf < %uu) {\n", n_bf);
      const char *ind = needs_guard ? "        " : "      ";

      if (stride > 1) {
        sb_printf(&sb, "%slet grp: u32 = bf / %uu;\n", ind, stride);
        sb_printf(&sb, "%slet pos: u32 = bf %% %uu;\n", ind, stride);
      }

      /* Twiddles via LUT */
      if (stride > 1 && has_lut) {
        for (int k = 1; k < R; k++) {
          int lut_base = lut_offsets[stage] + (k-1) * stride;
          sb_printf(&sb, "%s{ let tw: vec2<f32> = vec2<f32>("
                    "lut.d[%uu + pos * 2u], lut.d[%uu + pos * 2u + 1u]);\n",
                    ind, lut_base*2, lut_base*2);
          int ri = rb + k;
          sb_printf(&sb, "%s  v%d = vec2<f32>("
                    "v%d.x*tw.x - v%d.y*tw.y, "
                    "v%d.x*tw.y + v%d.y*tw.x); }\n",
                    ind, ri, ri, ri, ri, ri);
        }
      } else if (stride > 1) {
        for (int k = 1; k < R; k++) {
          double tw_angle = (double)direction * -2.0 * M_PI * k / (double)(stride * R);
          sb_printf(&sb, "%s{ let tw_a: f32 = ", ind);
          sb_float(&sb, tw_angle);
          sb_printf(&sb, " * f32(pos);\n");
          sb_printf(&sb, "%s  let tw: vec2<f32> = vec2<f32>(cos(tw_a), sin(tw_a));\n", ind);
          int ri = rb + k;
          sb_printf(&sb, "%s  v%d = vec2<f32>("
                    "v%d.x*tw.x - v%d.y*tw.y, "
                    "v%d.x*tw.y + v%d.y*tw.x); }\n",
                    ind, ri, ri, ri, ri, ri);
        }
      }

      /* Butterfly */
      emit_radix_butterfly(&sb, R, direction, "v", rb,
                           stage*100+b, rader_offsets[stage]);

      /* Stockham write */
      if (shr_stride == n) {
        if (stride == 1) {
          for (int k = 0; k < R; k++) {
            sb_printf(&sb, "%ss_re[shr_off + bf * %uu + %uu] = v%d.x;\n",
                      ind, R, k, rb+k);
            sb_printf(&sb, "%ss_im[shr_off + bf * %uu + %uu] = v%d.y;\n",
                      ind, R, k, rb+k);
          }
        } else {
          for (int k = 0; k < R; k++) {
            sb_printf(&sb, "%ss_re[shr_off + grp * %uu + pos + %uu] = v%d.x;\n",
                      ind, stride*R, k*stride, rb+k);
            sb_printf(&sb, "%ss_im[shr_off + grp * %uu + pos + %uu] = v%d.y;\n",
                      ind, stride*R, k*stride, rb+k);
          }
        }
      } else {
        if (stride == 1) {
          for (int k = 0; k < R; k++) {
            sb_printf(&sb, "%s{ let li: u32 = bf * %uu + %uu; "
                      "let pi: u32 = li + li / 16u;\n", ind, R, k);
            sb_printf(&sb, "%s  s_re[shr_off + pi] = v%d.x; "
                      "s_im[shr_off + pi] = v%d.y; }\n", ind, rb+k, rb+k);
          }
        } else {
          for (int k = 0; k < R; k++) {
            sb_printf(&sb, "%s{ let li: u32 = grp * %uu + pos + %uu; "
                      "let pi: u32 = li + li / 16u;\n", ind, stride*R, k*stride);
            sb_printf(&sb, "%s  s_re[shr_off + pi] = v%d.x; "
                      "s_im[shr_off + pi] = v%d.y; }\n", ind, rb+k, rb+k);
          }
        }
      }

      if (needs_guard) sb_printf(&sb, "      }\n");
      sb_printf(&sb, "    }\n");
    }

    sb_printf(&sb, "    workgroupBarrier();\n");
    sb_printf(&sb, "  }\n");
    stride *= R;
  }

  /* ---- Shared -> global store ---- */
  sb_printf(&sb, "\n");
  if (shr_stride == n) {
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  dst.d[(base + fft_t + %uu) * 2u] = "
                "s_re[shr_off + fft_t + %uu];\n", e*wpf, e*wpf);
      sb_printf(&sb, "  dst.d[(base + fft_t + %uu) * 2u + 1u] = "
                "s_im[shr_off + fft_t + %uu];\n", e*wpf, e*wpf);
    }
  } else {
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  { let li: u32 = fft_t + %uu; "
                "let pi: u32 = li + li / 16u;\n", e*wpf);
      sb_printf(&sb, "    dst.d[(base + li) * 2u] = s_re[shr_off + pi];\n");
      sb_printf(&sb, "    dst.d[(base + li) * 2u + 1u] = s_im[shr_off + pi]; }\n");
    }
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* Public API — default (auto) */
char *gen_fft_fused(int n, int direction) { return gen_fft_fused_mr(n, direction, 0); }

/* Public API — explicit max_radix */
char *gen_fft_fused_ex(int n, int direction, int max_r) { return gen_fft_fused_mr(n, direction, max_r); }
