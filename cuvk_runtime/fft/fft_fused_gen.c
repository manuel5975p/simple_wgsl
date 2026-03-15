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
#include "fft_strbuf.h"
#include "fft_bda.h"

#include <math.h>

#define MAX_SHARED_BYTES 32768 /* conservative shared memory limit */
#define DIRECT_MAX_N 64 /* sizes ≤ this use register-only path (no shared mem) */

#include "fft_butterfly.h"

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

static int batch_per_wg_impl(int n, int max_r, int wg_limit) {
  if (uses_direct_path_mr(n, max_r)) {
    /* The hybrid variant uses shared memory for coalesced I/O
     * (2 arrays of n floats per batch slot), so cap at the
     * shared memory limit.  The non-hybrid variant doesn't use
     * shared memory, but a conservative cap here is harmless. */
    int max_by_shmem = MAX_SHARED_BYTES / (n * 8);
    if (max_by_shmem < 1) max_by_shmem = 1;
    return wg_limit < max_by_shmem ? wg_limit : max_by_shmem;
  }

  int wpf = wg_per_fft_mr(n, max_r);
  if (wpf == 0) return 0;

  int shr = padded_stride(n);
  int max_by_threads = wg_limit / wpf;
  int max_by_shmem = MAX_SHARED_BYTES / (shr * 8);
  if (max_by_shmem < 1) max_by_shmem = 1;

  int B = max_by_threads;
  if (B > max_by_shmem) B = max_by_shmem;
  if (B < 1) B = 1;
  return B;
}

static int workgroup_size_impl(int n, int max_r, int wg_limit) {
  if (uses_direct_path_mr(n, max_r)) return batch_per_wg_impl(n, max_r, wg_limit);

  int wpf = wg_per_fft_mr(n, max_r);
  if (wpf == 0) return 0;
  return wpf * batch_per_wg_impl(n, max_r, wg_limit);
}

#define DEFAULT_WG_LIMIT 256

/* Public API — default (auto) */
int fft_fused_batch_per_wg(int n) { return batch_per_wg_impl(n, 0, DEFAULT_WG_LIMIT); }
int fft_fused_workgroup_size(int n) { return workgroup_size_impl(n, 0, DEFAULT_WG_LIMIT); }

/* Public API — explicit max_radix + wg_limit */
int fft_fused_batch_per_wg_ex(int n, int max_r, int wg_lim) {
  return batch_per_wg_impl(n, max_r, wg_lim > 0 ? wg_lim : DEFAULT_WG_LIMIT);
}
int fft_fused_workgroup_size_ex(int n, int max_r, int wg_lim) {
  return workgroup_size_impl(n, max_r, wg_lim > 0 ? wg_lim : DEFAULT_WG_LIMIT);
}

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
  if (uses_direct_path_mr(n, max_r)) return 0; /* direct path bakes twiddles inline */
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
/* Main generator                                                             */
/* ========================================================================== */

static char *gen_fft_fused_mr(int n, int direction, int max_r, int wg_limit, int total_batch,
                               int in_bs, int in_es, int out_bs, int out_es, int tw_n) {
  int radices[MAX_STAGES];
  int n_stages;

  if (n < 2) return NULL;
  if (direction != 1 && direction != -1) return NULL;

  n_stages = factorize_mr(n, radices, max_r);
  if (n_stages == 0) return NULL;

  int max_radix = 0;
  for (int i = 0; i < n_stages; i++)
    if (radices[i] > max_radix) max_radix = radices[i];

  /* ===== Direct register path: all stages in one thread ===== */
  if (uses_direct_path_mr(n, max_r)) {
    int B_dir = wg_limit; /* 1 thread per FFT */
    StrBuf sb;
    sb_init_cap(&sb, 65536);

    /* When batching (B_dir > 1) and data is contiguous, use shared memory
     * for coalesced global I/O but do the FFT entirely in registers.
     * This gives coalesced loads/stores (like shared-memory path) with
     * minimal barriers (2 total, vs 5+ for multi-stage shared-memory). */
    int use_hybrid = B_dir > 1 && in_es == 1 && in_bs == n &&
                     out_es == 1 && out_bs == n && total_batch == 0;
    int shr_stride_dir = n; /* no padding for small N */

    sb_emit_bda_src_dst(&sb, 0);
    sb_printf(&sb, "\n");

    if (use_hybrid) {
      sb_printf(&sb, "var<workgroup> s_re: array<f32, %d>;\n", B_dir * shr_stride_dir);
      sb_printf(&sb, "var<workgroup> s_im: array<f32, %d>;\n\n", B_dir * shr_stride_dir);
    }

    sb_printf(&sb, "@compute @workgroup_size(%d)\n", B_dir);
    sb_printf(&sb, "fn main(\n");
    sb_printf(&sb, "  @builtin(local_invocation_id) lid: vec3<u32>,\n");
    sb_printf(&sb, "  @builtin(workgroup_id) wid: vec3<u32>\n");
    sb_printf(&sb, ") {\n");
    sb_printf(&sb, "  let t: u32 = lid.x;\n");
    if (total_batch > 0) {
      sb_printf(&sb, "  let global_id: u32 = wid.x * %uu + t;\n", B_dir);
      sb_printf(&sb, "  if (global_id >= %uu) { return; }\n", total_batch);
    }

    if (use_hybrid) {
      /* ---- Coalesced load: global → shared memory ---- */
      int is_po2 = (n & (n - 1)) == 0;
      int log2n = 0;
      if (is_po2) { int tmp = n; while (tmp > 1) { log2n++; tmp >>= 1; } }
      sb_printf(&sb, "  { let wg_base: u32 = wid.x * %uu;\n", B_dir * n);
      for (int e = 0; e < n; e++) {
        int off = e * B_dir;
        sb_printf(&sb, "    { let gi: u32 = t");
        if (off > 0) sb_printf(&sb, " + %uu", off);
        sb_printf(&sb, ";\n");
        if (is_po2) {
          sb_printf(&sb, "      let f: u32 = gi >> %du; let k: u32 = gi & %uu;\n",
                    log2n, n - 1);
        } else {
          sb_printf(&sb, "      let f: u32 = gi / %uu; let k: u32 = gi %% %uu;\n", n, n);
        }
        sb_printf(&sb, "      s_re[f * %uu + k] = src.d[(wg_base + gi) * 2u];\n",
                  shr_stride_dir);
        sb_printf(&sb, "      s_im[f * %uu + k] = src.d[(wg_base + gi) * 2u + 1u]; }\n",
                  shr_stride_dir);
      }
      sb_printf(&sb, "  }\n");
      sb_printf(&sb, "  workgroupBarrier();\n\n");

      /* ---- Read own FFT from shared → registers ---- */
      for (int k = 0; k < n; k++)
        sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                  "s_re[t * %uu + %uu], s_im[t * %uu + %uu]);\n",
                  k, shr_stride_dir, k, shr_stride_dir, k);
    } else {
      /* ---- Direct global load → registers ---- */
      sb_printf(&sb, "  let base: u32 = (wid.x * %uu + t) * %uu;\n", B_dir, n);
      for (int k = 0; k < n; k++)
        sb_printf(&sb, "  var v%d: vec2<f32> = vec2<f32>("
                  "src.d[(base + %uu) * 2u], src.d[(base + %uu) * 2u + 1u]);\n",
                  k, k, k);
    }

    /* ---- In-register FFT ---- */
    if (n_stages == 1) {
      sb_printf(&sb, "\n");
      emit_radix_butterfly(&sb, n, direction, "v", 0, 0, 0);
    } else {
      /* Multi-stage: two register arrays (v, w) + temp (r) for butterflies.
       * Stockham reorder happens via scatter-read/scatter-write between arrays. */
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
    }

    /* ---- Store results ---- */
    /* For single-stage, result is in v. For multi-stage, track the toggle. */
    const char *final_pfx = "v";
    if (n_stages > 1) {
      int src_is_v_final = 1;
      for (int s = 0; s < n_stages; s++) src_is_v_final = !src_is_v_final;
      final_pfx = src_is_v_final ? "v" : "w";
    }

    if (use_hybrid) {
      /* ---- Write registers → shared memory ---- */
      sb_printf(&sb, "\n");
      for (int k = 0; k < n; k++) {
        sb_printf(&sb, "  s_re[t * %uu + %uu] = %s%d.x;\n",
                  shr_stride_dir, k, final_pfx, k);
        sb_printf(&sb, "  s_im[t * %uu + %uu] = %s%d.y;\n",
                  shr_stride_dir, k, final_pfx, k);
      }
      sb_printf(&sb, "  workgroupBarrier();\n\n");

      /* ---- Coalesced store: shared memory → global ---- */
      int is_po2 = (n & (n - 1)) == 0;
      int log2n = 0;
      if (is_po2) { int tmp = n; while (tmp > 1) { log2n++; tmp >>= 1; } }
      sb_printf(&sb, "  { let wg_base: u32 = wid.x * %uu;\n", B_dir * n);
      for (int e = 0; e < n; e++) {
        int off = e * B_dir;
        sb_printf(&sb, "    { let gi: u32 = t");
        if (off > 0) sb_printf(&sb, " + %uu", off);
        sb_printf(&sb, ";\n");
        if (is_po2) {
          sb_printf(&sb, "      let f: u32 = gi >> %du; let k: u32 = gi & %uu;\n",
                    log2n, n - 1);
        } else {
          sb_printf(&sb, "      let f: u32 = gi / %uu; let k: u32 = gi %% %uu;\n", n, n);
        }
        sb_printf(&sb, "      dst.d[(wg_base + gi) * 2u] = s_re[f * %uu + k];\n",
                  shr_stride_dir);
        sb_printf(&sb, "      dst.d[(wg_base + gi) * 2u + 1u] = s_im[f * %uu + k]; }\n",
                  shr_stride_dir);
      }
      sb_printf(&sb, "  }\n");
    } else {
      /* ---- Direct global store ---- */
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

  /* Note: a register-exchange path for 2-stage factorizations was attempted
   * and removed (see git history); it needs a shared-memory exchange step
   * between stages to be correct. */

  /* ===== Shared-memory path: multi-stage ===== */
  int wpf = wg_per_fft_mr(n, max_r);
  if (wpf < 1) return NULL;
  int B = batch_per_wg_impl(n, max_r, wg_limit);
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
  sb_init_cap(&sb, 131072);

  /* Bindings */
  sb_emit_bda_src_dst(&sb, has_lut);
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
    if (total_batch > 0) {
      sb_printf(&sb, "  let global_fft: u32 = wid.x * %uu + fft_id;\n", B);
      sb_printf(&sb, "  if (global_fft >= %uu) { return; }\n", total_batch);
    }
    sb_printf(&sb, "  let shr_off: u32 = fft_id * %uu;\n", shr_stride);
    sb_printf(&sb, "  let fft_idx: u32 = wid.x * %uu + fft_id;\n", B);
    sb_printf(&sb, "  let base_in: u32 = fft_idx * %uu;\n", in_bs);
    sb_printf(&sb, "  let base_out: u32 = fft_idx * %uu;\n\n", out_bs);
  } else {
    sb_printf(&sb, "  let fft_t: u32 = t;\n");
    sb_printf(&sb, "  let shr_off: u32 = 0u;\n");
    if (total_batch > 0) {
      sb_printf(&sb, "  if (wid.x >= %uu) { return; }\n", total_batch);
    }
    sb_printf(&sb, "  let fft_idx: u32 = wid.x;\n");
    sb_printf(&sb, "  let base_in: u32 = fft_idx * %uu;\n", in_bs);
    sb_printf(&sb, "  let base_out: u32 = fft_idx * %uu;\n\n", out_bs);
  }

  /* ---- Global load -> shared ---- */
  if (tw_n > 0) {
    /* Load with inline twiddle: W_{tw_n}^(fft_idx * k) */
    double tw_scale = direction * -2.0 * M_PI / tw_n;
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  {\n");
      sb_printf(&sb, "    let k: u32 = fft_t + %uu;\n", e*wpf);
      if (in_es == 1)
        sb_printf(&sb, "    let ga: u32 = (base_in + k) * 2u;\n");
      else
        sb_printf(&sb, "    let ga: u32 = (base_in + k * %uu) * 2u;\n", in_es);
      sb_printf(&sb, "    let re: f32 = src.d[ga];\n");
      sb_printf(&sb, "    let im: f32 = src.d[ga + 1u];\n");
      sb_printf(&sb, "    let a: f32 = ");
      sb_float(&sb, tw_scale);
      sb_printf(&sb, " * f32(fft_idx) * f32(k);\n");
      sb_printf(&sb, "    let tw_re: f32 = cos(a);\n");
      sb_printf(&sb, "    let tw_im: f32 = sin(a);\n");
      if (shr_stride == n) {
        sb_printf(&sb, "    s_re[shr_off + k] = re * tw_re - im * tw_im;\n");
        sb_printf(&sb, "    s_im[shr_off + k] = re * tw_im + im * tw_re;\n");
      } else {
        sb_printf(&sb, "    let pi: u32 = k + k / 16u;\n");
        sb_printf(&sb, "    s_re[shr_off + pi] = re * tw_re - im * tw_im;\n");
        sb_printf(&sb, "    s_im[shr_off + pi] = re * tw_im + im * tw_re;\n");
      }
      sb_printf(&sb, "  }\n");
    }
  } else if (in_es == 1) {
    /* Default contiguous load */
    if (B > 1 && in_bs == n && total_batch == 0) {
      /* Coalesced load: consecutive threads access consecutive global elements,
       * then scatter to per-FFT shared memory positions.
       * gi = t + e*wg_size indexes linearly into B*N contiguous elements.
       * f = gi/N (which FFT), k = gi%N (which element within FFT). */
      int is_po2 = (n & (n - 1)) == 0;
      int log2n = 0;
      if (is_po2) { int tmp = n; while (tmp > 1) { log2n++; tmp >>= 1; } }
      sb_printf(&sb, "  { let wg_base: u32 = wid.x * %uu;\n", B * n);
      for (int e = 0; e < epr; e++) {
        int off = e * total_wg;
        if (off == 0)
          sb_printf(&sb, "    let gi%d: u32 = t;\n", e);
        else
          sb_printf(&sb, "    let gi%d: u32 = t + %uu;\n", e, off);
        if (is_po2) {
          sb_printf(&sb, "    let f%d: u32 = gi%d >> %du;\n", e, e, log2n);
          sb_printf(&sb, "    let k%d: u32 = gi%d & %uu;\n", e, e, n - 1);
        } else {
          sb_printf(&sb, "    let f%d: u32 = gi%d / %uu;\n", e, e, n);
          sb_printf(&sb, "    let k%d: u32 = gi%d %% %uu;\n", e, e, n);
        }
        if (shr_stride == n) {
          sb_printf(&sb, "    s_re[f%d * %uu + k%d] = src.d[(wg_base + gi%d) * 2u];\n",
                    e, shr_stride, e, e);
          sb_printf(&sb, "    s_im[f%d * %uu + k%d] = src.d[(wg_base + gi%d) * 2u + 1u];\n",
                    e, shr_stride, e, e);
        } else {
          sb_printf(&sb, "    { let pi%d: u32 = k%d + k%d / 16u;\n", e, e, e);
          sb_printf(&sb, "      s_re[f%d * %uu + pi%d] = src.d[(wg_base + gi%d) * 2u];\n",
                    e, shr_stride, e, e);
          sb_printf(&sb, "      s_im[f%d * %uu + pi%d] = src.d[(wg_base + gi%d) * 2u + 1u]; }\n",
                    e, shr_stride, e, e);
        }
      }
      sb_printf(&sb, "  }\n");
    } else if (shr_stride == n) {
      for (int e = 0; e < epr; e++) {
        sb_printf(&sb, "  s_re[shr_off + fft_t + %uu] = "
                  "src.d[(base_in + fft_t + %uu) * 2u];\n", e*wpf, e*wpf);
        sb_printf(&sb, "  s_im[shr_off + fft_t + %uu] = "
                  "src.d[(base_in + fft_t + %uu) * 2u + 1u];\n", e*wpf, e*wpf);
      }
    } else {
      for (int e = 0; e < epr; e++) {
        sb_printf(&sb, "  { let li: u32 = fft_t + %uu; "
                  "let pi: u32 = li + li / 16u;\n", e*wpf);
        sb_printf(&sb, "    s_re[shr_off + pi] = src.d[(base_in + li) * 2u];\n");
        sb_printf(&sb, "    s_im[shr_off + pi] = src.d[(base_in + li) * 2u + 1u]; }\n");
      }
    }
  } else if (B > 1) {
    /* Tiled coalesced load: all threads cooperatively scan B*N input
     * elements in flat order for B-wide coalescing. */
    int bn = B * n;
    int iters = (bn + total_wg - 1) / total_wg;
    int is_B_po2 = (B & (B - 1)) == 0;
    int log2B = 0;
    if (is_B_po2) { int tmp = B; while (tmp > 1) { log2B++; tmp >>= 1; } }

    sb_printf(&sb, "  { let wg_base: u32 = wid.x * %uu;\n", B);
    for (int e = 0; e < iters; e++) {
      int off = e * total_wg;
      sb_printf(&sb, "    {\n");
      if (off == 0)
        sb_printf(&sb, "      let flat: u32 = t;\n");
      else
        sb_printf(&sb, "      let flat: u32 = t + %uu;\n", off);

      int need_guard = (off + total_wg > bn);
      if (need_guard)
        sb_printf(&sb, "      if (flat < %uu) {\n", bn);
      const char *g = need_guard ? "  " : "";

      if (is_B_po2) {
        sb_printf(&sb, "      %slet f: u32 = flat & %uu;\n", g, B - 1);
        sb_printf(&sb, "      %slet k: u32 = flat >> %du;\n", g, log2B);
      } else {
        sb_printf(&sb, "      %slet f: u32 = flat %% %uu;\n", g, B);
        sb_printf(&sb, "      %slet k: u32 = flat / %uu;\n", g, B);
      }

      sb_printf(&sb, "      %slet addr: u32 = (wg_base + f) * %uu + k * %uu;\n",
                g, in_bs, in_es);

      if (shr_stride == n) {
        sb_printf(&sb, "      %ss_re[f * %uu + k] = src.d[addr * 2u];\n", g, shr_stride);
        sb_printf(&sb, "      %ss_im[f * %uu + k] = src.d[addr * 2u + 1u];\n", g, shr_stride);
      } else {
        sb_printf(&sb, "      %s{ let pi: u32 = k + k / 16u;\n", g);
        sb_printf(&sb, "      %ss_re[f * %uu + pi] = src.d[addr * 2u];\n", g, shr_stride);
        sb_printf(&sb, "      %ss_im[f * %uu + pi] = src.d[addr * 2u + 1u]; }\n", g, shr_stride);
      }

      if (need_guard)
        sb_printf(&sb, "      }\n");
      sb_printf(&sb, "    }\n");
    }
    sb_printf(&sb, "  }\n");
  } else {
    /* Strided load (no twiddle), B==1 */
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  {\n");
      sb_printf(&sb, "    let k: u32 = fft_t + %uu;\n", e*wpf);
      if (shr_stride == n) {
        sb_printf(&sb, "    s_re[shr_off + k] = "
                  "src.d[(base_in + k * %uu) * 2u];\n", in_es);
        sb_printf(&sb, "    s_im[shr_off + k] = "
                  "src.d[(base_in + k * %uu) * 2u + 1u];\n", in_es);
      } else {
        sb_printf(&sb, "    let pi: u32 = k + k / 16u;\n");
        sb_printf(&sb, "    s_re[shr_off + pi] = "
                  "src.d[(base_in + k * %uu) * 2u];\n", in_es);
        sb_printf(&sb, "    s_im[shr_off + pi] = "
                  "src.d[(base_in + k * %uu) * 2u + 1u];\n", in_es);
      }
      sb_printf(&sb, "  }\n");
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
  if (out_es == 1) {
    if (B > 1 && out_bs == n && total_batch == 0) {
      /* Coalesced store: consecutive threads write consecutive global elements */
      int is_po2 = (n & (n - 1)) == 0;
      int log2n = 0;
      if (is_po2) { int tmp = n; while (tmp > 1) { log2n++; tmp >>= 1; } }
      sb_printf(&sb, "  { let wg_base: u32 = wid.x * %uu;\n", B * n);
      for (int e = 0; e < epr; e++) {
        int off = e * total_wg;
        if (off == 0)
          sb_printf(&sb, "    let gi%d: u32 = t;\n", e);
        else
          sb_printf(&sb, "    let gi%d: u32 = t + %uu;\n", e, off);
        if (is_po2) {
          sb_printf(&sb, "    let f%d: u32 = gi%d >> %du;\n", e, e, log2n);
          sb_printf(&sb, "    let k%d: u32 = gi%d & %uu;\n", e, e, n - 1);
        } else {
          sb_printf(&sb, "    let f%d: u32 = gi%d / %uu;\n", e, e, n);
          sb_printf(&sb, "    let k%d: u32 = gi%d %% %uu;\n", e, e, n);
        }
        if (shr_stride == n) {
          sb_printf(&sb, "    dst.d[(wg_base + gi%d) * 2u] = s_re[f%d * %uu + k%d];\n",
                    e, e, shr_stride, e);
          sb_printf(&sb, "    dst.d[(wg_base + gi%d) * 2u + 1u] = s_im[f%d * %uu + k%d];\n",
                    e, e, shr_stride, e);
        } else {
          sb_printf(&sb, "    { let pi%d: u32 = k%d + k%d / 16u;\n", e, e, e);
          sb_printf(&sb, "      dst.d[(wg_base + gi%d) * 2u] = s_re[f%d * %uu + pi%d];\n",
                    e, e, shr_stride, e);
          sb_printf(&sb, "      dst.d[(wg_base + gi%d) * 2u + 1u] = s_im[f%d * %uu + pi%d]; }\n",
                    e, e, shr_stride, e);
        }
      }
      sb_printf(&sb, "  }\n");
    } else if (shr_stride == n) {
      for (int e = 0; e < epr; e++) {
        sb_printf(&sb, "  dst.d[(base_out + fft_t + %uu) * 2u] = "
                  "s_re[shr_off + fft_t + %uu];\n", e*wpf, e*wpf);
        sb_printf(&sb, "  dst.d[(base_out + fft_t + %uu) * 2u + 1u] = "
                  "s_im[shr_off + fft_t + %uu];\n", e*wpf, e*wpf);
      }
    } else {
      for (int e = 0; e < epr; e++) {
        sb_printf(&sb, "  { let li: u32 = fft_t + %uu; "
                  "let pi: u32 = li + li / 16u;\n", e*wpf);
        sb_printf(&sb, "    dst.d[(base_out + li) * 2u] = s_re[shr_off + pi];\n");
        sb_printf(&sb, "    dst.d[(base_out + li) * 2u + 1u] = s_im[shr_off + pi]; }\n");
      }
    }
  } else if (B > 1) {
    /* Tiled coalesced store: all threads cooperatively scan B*N output
     * elements in flat order.  Consecutive threads write consecutive
     * batch indices (f, f+1) whose addresses differ by out_bs,
     * achieving B-wide coalescing instead of stride-out_es. */
    int bn = B * n;
    int iters = (bn + total_wg - 1) / total_wg;
    int is_B_po2 = (B & (B - 1)) == 0;
    int log2B = 0;
    if (is_B_po2) { int tmp = B; while (tmp > 1) { log2B++; tmp >>= 1; } }

    sb_printf(&sb, "  { let wg_base: u32 = wid.x * %uu;\n", B);
    for (int e = 0; e < iters; e++) {
      int off = e * total_wg;
      sb_printf(&sb, "    {\n");
      if (off == 0)
        sb_printf(&sb, "      let flat: u32 = t;\n");
      else
        sb_printf(&sb, "      let flat: u32 = t + %uu;\n", off);

      int need_guard = (off + total_wg > bn);
      if (need_guard)
        sb_printf(&sb, "      if (flat < %uu) {\n", bn);
      const char *g = need_guard ? "  " : "";

      if (is_B_po2) {
        sb_printf(&sb, "      %slet f: u32 = flat & %uu;\n", g, B - 1);
        sb_printf(&sb, "      %slet k: u32 = flat >> %du;\n", g, log2B);
      } else {
        sb_printf(&sb, "      %slet f: u32 = flat %% %uu;\n", g, B);
        sb_printf(&sb, "      %slet k: u32 = flat / %uu;\n", g, B);
      }

      if (shr_stride == n) {
        sb_printf(&sb, "      %slet val_re: f32 = s_re[f * %uu + k];\n", g, shr_stride);
        sb_printf(&sb, "      %slet val_im: f32 = s_im[f * %uu + k];\n", g, shr_stride);
      } else {
        sb_printf(&sb, "      %s{ let pi: u32 = k + k / 16u;\n", g);
        sb_printf(&sb, "      %slet val_re: f32 = s_re[f * %uu + pi];\n", g, shr_stride);
        sb_printf(&sb, "      %slet val_im: f32 = s_im[f * %uu + pi];\n", g, shr_stride);
      }

      sb_printf(&sb, "      %slet addr: u32 = (wg_base + f) * %uu + k * %uu;\n",
                g, out_bs, out_es);
      sb_printf(&sb, "      %sdst.d[addr * 2u] = val_re;\n", g);
      sb_printf(&sb, "      %sdst.d[addr * 2u + 1u] = val_im;\n", g);

      if (shr_stride != n)
        sb_printf(&sb, "      %s}\n", g);  /* close the pi block */

      if (need_guard)
        sb_printf(&sb, "      }\n");
      sb_printf(&sb, "    }\n");
    }
    sb_printf(&sb, "  }\n");
  } else {
    /* Strided store, B==1: single FFT per workgroup, no tiling benefit */
    for (int e = 0; e < epr; e++) {
      sb_printf(&sb, "  {\n");
      sb_printf(&sb, "    let k: u32 = fft_t + %uu;\n", e*wpf);
      if (shr_stride == n) {
        sb_printf(&sb, "    dst.d[(base_out + k * %uu) * 2u] = "
                  "s_re[shr_off + k];\n", out_es);
        sb_printf(&sb, "    dst.d[(base_out + k * %uu) * 2u + 1u] = "
                  "s_im[shr_off + k];\n", out_es);
      } else {
        sb_printf(&sb, "    let pi: u32 = k + k / 16u;\n");
        sb_printf(&sb, "    dst.d[(base_out + k * %uu) * 2u] = "
                  "s_re[shr_off + pi];\n", out_es);
        sb_printf(&sb, "    dst.d[(base_out + k * %uu) * 2u + 1u] = "
                  "s_im[shr_off + pi];\n", out_es);
      }
      sb_printf(&sb, "  }\n");
    }
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* Public API — default (auto) */
char *gen_fft_fused(int n, int direction) {
  return gen_fft_fused_mr(n, direction, 0, DEFAULT_WG_LIMIT, 0, n, 1, n, 1, 0);
}

/* Public API — explicit max_radix + wg_limit */
char *gen_fft_fused_ex(int n, int direction, int max_r, int wg_lim) {
  return gen_fft_fused_mr(n, direction, max_r, wg_lim > 0 ? wg_lim : DEFAULT_WG_LIMIT, 0,
                           n, 1, n, 1, 0);
}

/* Public API — with bounds guard for sub-batch dispatches (four-step FFT) */
char *gen_fft_fused_bounded(int n, int direction, int total_batch) {
  return gen_fft_fused_mr(n, direction, 0, DEFAULT_WG_LIMIT, total_batch, n, 1, n, 1, 0);
}

/*
 * Public API — strided I/O for two-pass four-step FFT.
 *
 * Generates a fused FFT kernel with strided global memory access and
 * optional inline twiddle application.  This eliminates transpose stages
 * in the four-step FFT decomposition (5 stages → 2 stages).
 *
 * For N = N1 × N2 four-step FFT:
 *   Pass 1 (column DFTs): gen_fft_fused_strided(N1, dir, 0, wg, N2, 1, N2, N1, 1, 0)
 *   Pass 2 (twiddle+row): gen_fft_fused_strided(N2, dir, 0, wg, N1, 1, N1, 1, N1, N)
 *
 * Parameters:
 *   n            - sub-FFT size (N1 or N2)
 *   direction    - 1=forward, -1=inverse
 *   max_radix    - 0=auto, 2..16=cap
 *   wg_limit     - workgroup size limit (0=default)
 *   total_batch  - number of sub-FFTs (for bounds guard; 0=no guard)
 *   in_bs        - input batch stride (spacing between consecutive sub-FFT inputs)
 *   in_es        - input element stride (spacing between elements within a sub-FFT)
 *   out_bs       - output batch stride
 *   out_es       - output element stride
 *   tw_n         - twiddle modulus (0=no twiddle; >0: apply W_{tw_n}^(fft_index*k))
 */
char *gen_fft_fused_strided(int n, int direction, int max_radix, int wg_limit,
                             int total_batch,
                             int in_bs, int in_es, int out_bs, int out_es,
                             int tw_n) {
  return gen_fft_fused_mr(n, direction, max_radix,
                           wg_limit > 0 ? wg_limit : DEFAULT_WG_LIMIT,
                           total_batch, in_bs, in_es, out_bs, out_es, tw_n);
}
