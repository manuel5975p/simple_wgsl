/*
 * fft_butterfly.h - Shared butterfly emitters and factorization helpers
 *                   for FFT WGSL shader generators.
 *
 * Provides primality helpers, mixed-radix factorization, specialized
 * radix butterflies (2,3,4,5,7,8,16), Rader's algorithm for supported
 * primes, and path/padding helpers.
 *
 * Include this header in exactly the .c files that need it — all functions
 * are static, so each translation unit gets its own copy with no link
 * conflicts.
 */

#ifndef FFT_BUTTERFLY_H
#define FFT_BUTTERFLY_H

#include "fft_strbuf.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX_STAGES
#define MAX_STAGES 20
#endif

#ifndef DIRECT_MAX_N
#define DIRECT_MAX_N 64
#endif

/* ========================================================================== */
/* Primality / primitive root                                                 */
/* ========================================================================== */

__attribute__((unused))
static int is_prime(int n) {
  if (n < 2) return 0;
  if (n < 4) return 1;
  if (n % 2 == 0 || n % 3 == 0) return 0;
  for (int i = 5; i * i <= n; i += 6)
    if (n % i == 0 || n % (i + 2) == 0) return 0;
  return 1;
}

/* modular exponentiation: base^exp mod m */
__attribute__((unused))
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
__attribute__((unused))
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
__attribute__((unused))
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
__attribute__((unused))
static int rader_supported(int p) {
  return is_prime(p) && p > 7 && factors_into_base(p - 1);
}

/* Factorize n into radices with optional radix cap.
 * Returns stage count, 0 on failure.
 * Tries base radices first (up to max_r), then Rader-supported primes. */
__attribute__((unused))
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

/* Radix cap: N<=32 gets direct register path (no shared memory, no barriers).
 * N>32 uses shared-memory path with large radices for fewer stages. */
__attribute__((unused))
static int radix_cap(int n) {
  if (n <= 32) return n;   /* direct register path: 1 thread/FFT, all in regs */
  return 16;               /* shared-memory: radix-16 max for fewest stages */
}

/* Resolve effective max radix: 0 = auto (use heuristic), else as given. */
__attribute__((unused))
static int effective_max_r(int n, int max_r) {
  return (max_r > 0) ? max_r : radix_cap(n);
}

__attribute__((unused))
static int factorize_mr(int n, int *radices, int max_r) {
  return factorize_ex(n, radices, effective_max_r(n, max_r));
}

/* ========================================================================== */
/* Path / padding helpers                                                     */
/* ========================================================================== */

__attribute__((unused))
static int uses_direct_path_mr(int n, int max_r) {
  int eff = effective_max_r(n, max_r);
  /* If radix cap < n, the shared-memory path gives multiple threads/FFT */
  if (eff < n) return 0;
  return n <= DIRECT_MAX_N && factors_into_base(n);
}

__attribute__((unused))
static int padded_stride(int n) {
  if (n < 32 || n > 4096) return n;
  if ((n & (n - 1)) == 0)
    return n + (n / 16);
  return n;
}

/* ========================================================================== */
/* Specialized radix butterflies                                              */
/* ========================================================================== */

__attribute__((unused))
static void emit_radix2(StrBuf *sb, const char *p, int b, int uid) {
  #define R(k) p, b + (k)
  sb_printf(sb, "      { let bt_%d = %s%d; %s%d = bt_%d + %s%d; "
                "%s%d = bt_%d - %s%d; }\n",
            uid, R(0), R(0), uid, R(1), R(1), uid, R(1));
  #undef R
}

__attribute__((unused))
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

__attribute__((unused))
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

__attribute__((unused))
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

__attribute__((unused))
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

__attribute__((unused))
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

__attribute__((unused))
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
  /* Intermediate twiddles: W_16^{r*c} for 4x4 decomposition.
   * c1 = cos(pi/8), s1 = sin(pi/8), sq = cos(pi/4) = sqrt(2)/2.
   * Twiddle map (r,c -> W_16^k):
   *   (1,1)->W^1  (1,2)->W^2  (1,3)->W^3
   *   (2,1)->W^2  (2,2)->W^4  (2,3)->W^6
   *   (3,1)->W^3  (3,2)->W^6  (3,3)->W^9
   */
  double c1 = 0.92387953251128675613;  /* cos(pi/8) */
  double s1 = 0.38268343236508977173;  /* sin(pi/8) */

  /* pos 5 (r=1,c=1): W_16^1 = c1 - i*s1  (fwd), c1 + i*s1 (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(5));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.x + ", R(5));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y, ", R(5));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y - ", R(5));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x);\n", R(5));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(5));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.x - ", R(5));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y, ", R(5));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x + ", R(5));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y);\n", R(5));
  }
  /* pos 6 (r=1,c=2): W_16^2 = sq - i*sq  (fwd), sq + i*sq (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(6));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.x+%s%d.y), ", R(6),R(6));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.y-%s%d.x));\n", R(6),R(6));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(6));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.x-%s%d.y), ", R(6),R(6));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.y+%s%d.x));\n", R(6),R(6));
  }
  /* pos 7 (r=1,c=3): W_16^3 = s1 - i*c1  (fwd), s1 + i*c1 (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(7));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x + ", R(7));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y, ", R(7));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y - ", R(7));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.x);\n", R(7));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(7));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x - ", R(7));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y, ", R(7));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.x + ", R(7));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y);\n", R(7));
  }
  /* pos 9 (r=2,c=1): W_16^2 = sq - i*sq  (fwd), sq + i*sq (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(9));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.x+%s%d.y), ", R(9),R(9));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.y-%s%d.x));\n", R(9),R(9));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(9));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.x-%s%d.y), ", R(9),R(9));
    sb_float(sb,sq); sb_printf(sb, "*(%s%d.y+%s%d.x));\n", R(9),R(9));
  }
  /* pos 10 (r=2,c=2): W_16^4 = -i (fwd), +i (inv) */
  if (dir==1) sb_printf(sb, "        %s%d = vec2<f32>(%s%d.y, -%s%d.x);\n", R(10),R(10),R(10));
  else sb_printf(sb, "        %s%d = vec2<f32>(-%s%d.y, %s%d.x);\n", R(10),R(10),R(10));
  /* pos 11 (r=2,c=3): W_16^6 = -sq - i*sq  (fwd), -sq + i*sq (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(11));
    sb_float(sb,-sq); sb_printf(sb, "*%s%d.x + ", R(11));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y, ", R(11));
    sb_float(sb,-sq); sb_printf(sb, "*%s%d.x - ", R(11));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y);\n", R(11));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(11));
    sb_float(sb,-sq); sb_printf(sb, "*%s%d.x - ", R(11));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y, ", R(11));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.x - ", R(11));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y);\n", R(11));
  }
  /* pos 13 (r=3,c=1): W_16^3 = s1 - i*c1  (fwd), s1 + i*c1 (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(13));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x + ", R(13));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y, ", R(13));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y - ", R(13));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.x);\n", R(13));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(13));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x - ", R(13));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y, ", R(13));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.x + ", R(13));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y);\n", R(13));
  }
  /* pos 14 (r=3,c=2): W_16^6 = -sq - i*sq  (fwd), -sq + i*sq (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(14));
    sb_float(sb,-sq); sb_printf(sb, "*%s%d.x + ", R(14));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y, ", R(14));
    sb_float(sb,-sq); sb_printf(sb, "*%s%d.x - ", R(14));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y);\n", R(14));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(14));
    sb_float(sb,-sq); sb_printf(sb, "*%s%d.x - ", R(14));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y, ", R(14));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.x - ", R(14));
    sb_float(sb,sq); sb_printf(sb, "*%s%d.y);\n", R(14));
  }
  /* pos 15 (r=3,c=3): W_16^9 = -c1 + i*s1  (fwd), -c1 - i*s1 (inv) */
  if (dir==1) {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(15));
    sb_float(sb,-c1); sb_printf(sb, "*%s%d.x - ", R(15));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y, ", R(15));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.x - ", R(15));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y);\n", R(15));
  } else {
    sb_printf(sb, "        %s%d = vec2<f32>(", R(15));
    sb_float(sb,-c1); sb_printf(sb, "*%s%d.x + ", R(15));
    sb_float(sb,s1); sb_printf(sb, "*%s%d.y, ", R(15));
    sb_float(sb,-s1); sb_printf(sb, "*%s%d.x - ", R(15));
    sb_float(sb,c1); sb_printf(sb, "*%s%d.y);\n", R(15));
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
__attribute__((unused))
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

__attribute__((unused))
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

__attribute__((unused))
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

#endif /* FFT_BUTTERFLY_H */
