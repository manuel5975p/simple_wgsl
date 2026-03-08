#ifndef FFT_OPTIMAL_GEN_H
#define FFT_OPTIMAL_GEN_H

#ifdef __cplusplus
extern "C" {
#endif

#define FFT_OPT_DIRECT   0
#define FFT_OPT_CT_SPLIT 1
#define FFT_OPT_MAX_LOG2 10

typedef struct {
  int type;   /* FFT_OPT_DIRECT or FFT_OPT_CT_SPLIT */
  int radix;  /* for CT_SPLIT: R in N = R * M */
} FftOptPlan;

/*
 * Generate a self-contained WGSL compute shader for a complete FFT of size n
 * (n must be a power of 2, 2 <= n <= 256).
 *
 * plan_table[ilog2(k)] defines the decomposition for sub-FFT size k.
 * direction: 1 = forward, -1 = inverse.
 * Returns malloc'd WGSL string (caller frees), or NULL on error.
 */
char *gen_fft_optimal(int n, const FftOptPlan *plan_table, int direction);

/*
 * Generate a self-contained WGSL compute shader for a direct (naive) DFT of
 * any size n (2 <= n <= 256). O(n^2) operations, fully inlined.
 * Returns malloc'd WGSL string (caller frees), or NULL on error.
 */
char *gen_fft_direct(int n, int direction);

/*
 * Generate a self-contained WGSL compute shader for an FFT of arbitrary size n
 * (2 <= n <= 256) using Bluestein's algorithm (chirp-z transform).
 *
 * Internally uses a power-of-2 FFT of size M = next_pow2(2*n-1).
 * pow2_plan_table[ilog2(k)] defines the decomposition for internal sub-FFTs.
 * direction: 1 = forward, -1 = inverse.
 * Returns malloc'd WGSL string (caller frees), or NULL on error.
 */
char *gen_fft_bluestein(int n, const FftOptPlan *pow2_plan_table, int direction);

/* ========================================================================== */
/* General FFT plan table — indexed by N directly, supports any size          */
/* ========================================================================== */

#define FFT_PLAN_MAX_N    512

#define FFT_PLAN_DFT       0  /* Direct DFT, O(N^2) */
#define FFT_PLAN_CT        1  /* Cooley-Tukey: N = radix * (N/radix) */
#define FFT_PLAN_BLUESTEIN 2  /* Bluestein chirp-z transform */
#define FFT_PLAN_RADER     3  /* Rader's algorithm for primes */

typedef struct {
  int type;    /* FFT_PLAN_DFT, FFT_PLAN_CT, FFT_PLAN_BLUESTEIN, or FFT_PLAN_RADER */
  int radix;   /* For FFT_PLAN_CT: R in N = R * M */
} FftPlan;

/*
 * Generate a self-contained WGSL compute shader for an FFT of any size n
 * (2 <= n <= FFT_PLAN_MAX_N).
 *
 * plans[k] defines the strategy for sub-FFT of size k. All sizes appearing
 * in the decomposition tree must have valid entries.
 *
 * direction: 1 = forward, -1 = inverse.
 * Returns malloc'd WGSL string (caller frees), or NULL on error.
 */
char *gen_fft(int n, const FftPlan *plans, int direction);

#ifdef __cplusplus
}
#endif

#endif
