/*
 * cufft_test.cpp - FFT correctness tests using FFTW3 as reference
 *
 * Tests the cuFFT shim (Vulkan Stockham FFT) against FFTW3f for:
 *   - 1D C2C forward/inverse
 *   - 1D R2C
 *   - 1D C2R
 *   - 1D R2C→C2R roundtrip
 *   - 2D C2C forward/inverse
 *   - Mixed-radix sizes
 */

#include <gtest/gtest.h>
#include <fftw3.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "cuda.h"
#include "cufft.h"
}

/* ========================================================================== */
/* Helpers                                                                     */
/* ========================================================================== */

static float max_abs_error(const float *a, const float *b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/* Fill buffer with deterministic pseudo-random floats in [-1, 1] */
static void fill_random(float *buf, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        buf[i] = (float)(rand() % 10000) / 5000.0f - 1.0f;
}

/* ========================================================================== */
/* Test fixture                                                                */
/* ========================================================================== */

class CufftTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) {
            skip_ = true;
            return;
        }
        int count = 0;
        cuDeviceGetCount(&count);
        if (count == 0) {
            skip_ = true;
            return;
        }
        int dev_id = 0;
        const char *env = getenv("CUVK_DEVICE");
        if (env) dev_id = atoi(env);
        CUdevice dev;
        cuDeviceGet(&dev, dev_id);
        cuCtxCreate(&ctx_, NULL, 0, dev);
    }

    static void TearDownTestSuite() {
        if (ctx_) cuCtxDestroy(ctx_);
    }

    void SetUp() override {
        if (skip_) GTEST_SKIP() << "No Vulkan/CUDA device";
    }

    /* GPU alloc + upload */
    CUdeviceptr gpu_alloc(size_t bytes) {
        CUdeviceptr ptr = 0;
        EXPECT_EQ(cuMemAlloc(&ptr, bytes), CUDA_SUCCESS);
        allocs_.push_back(ptr);
        return ptr;
    }

    CUdeviceptr gpu_upload(const void *data, size_t bytes) {
        CUdeviceptr ptr = gpu_alloc(bytes);
        cuMemcpyHtoD(ptr, data, bytes);
        return ptr;
    }

    void gpu_download(CUdeviceptr ptr, void *dst, size_t bytes) {
        cuMemcpyDtoH(dst, ptr, bytes);
    }

    void TearDown() override {
        for (auto p : allocs_) cuMemFree(p);
        allocs_.clear();
    }

    static bool skip_;
    static CUcontext ctx_;
    std::vector<CUdeviceptr> allocs_;
};

bool CufftTest::skip_ = false;
CUcontext CufftTest::ctx_ = nullptr;

/* ========================================================================== */
/* 1D C2C forward                                                              */
/* ========================================================================== */

class CufftC2CForwardTest : public CufftTest,
                             public ::testing::WithParamInterface<int> {};

TEST_P(CufftC2CForwardTest, MatchesFftw) {
    int n = GetParam();

    /* Random input */
    std::vector<float> h_in(n * 2);
    fill_random(h_in.data(), n * 2, 42 + n);

    /* FFTW reference */
    std::vector<float> ref_out(n * 2);
    {
        fftwf_complex *in  = (fftwf_complex *)h_in.data();
        fftwf_complex *out = (fftwf_complex *)ref_out.data();
        fftwf_plan p = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    /* GPU cuFFT */
    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_C2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(n * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), n * 2);
    float tol = 1e-3f * sqrtf((float)n);
    EXPECT_LT(err, tol) << "N=" << n << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftC2CForwardTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      2048, 4096, 8192, 16384,
                      /* four-step (> 4096) */
                      32768, 65536,
                      /* mixed radix */
                      6, 10, 12, 14, 15, 18, 20, 24, 30, 48, 60,
                      96, 120, 210, 480, 720, 1536, 2310));

/* ========================================================================== */
/* 1D C2C inverse                                                              */
/* ========================================================================== */

class CufftC2CInverseTest : public CufftTest,
                             public ::testing::WithParamInterface<int> {};

TEST_P(CufftC2CInverseTest, MatchesFftw) {
    int n = GetParam();

    std::vector<float> h_in(n * 2);
    fill_random(h_in.data(), n * 2, 99 + n);

    /* FFTW reference (backward = inverse, unnormalized) */
    std::vector<float> ref_out(n * 2);
    {
        fftwf_complex *in  = (fftwf_complex *)h_in.data();
        fftwf_complex *out = (fftwf_complex *)ref_out.data();
        fftwf_plan p = fftwf_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_C2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_INVERSE),
              CUFFT_SUCCESS);

    std::vector<float> h_out(n * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), n * 2);
    float tol = 1e-3f * sqrtf((float)n);
    EXPECT_LT(err, tol) << "N=" << n << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftC2CInverseTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      4096, 16384,
                      6, 12, 24, 30, 120, 480, 1536));

/* ========================================================================== */
/* 1D C2C roundtrip                                                            */
/* ========================================================================== */

class CufftC2CRoundtripTest : public CufftTest,
                               public ::testing::WithParamInterface<int> {};

TEST_P(CufftC2CRoundtripTest, FwdThenInvGivesNx) {
    int n = GetParam();

    std::vector<float> h_in(n * 2);
    fill_random(h_in.data(), n * 2, 77 + n);

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_C2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_INVERSE),
              CUFFT_SUCCESS);

    std::vector<float> h_out(n * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    /* Expected: h_in * N */
    float max_err = 0.0f;
    for (int i = 0; i < n * 2; i++) {
        float expected = h_in[i] * (float)n;
        float e = fabsf(h_out[i] - expected);
        if (e > max_err) max_err = e;
    }
    float tol = 1e-2f * (float)n;
    EXPECT_LT(max_err, tol) << "N=" << n << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftC2CRoundtripTest,
    ::testing::Values(2, 4, 8, 16, 64, 256, 1024, 4096,
                      6, 12, 30, 120, 480, 1536));

/* ========================================================================== */
/* 1D C2C impulse (sanity check)                                               */
/* ========================================================================== */

class CufftC2CImpulseTest : public CufftTest,
                             public ::testing::WithParamInterface<int> {};

TEST_P(CufftC2CImpulseTest, DeltaGivesFlat) {
    int n = GetParam();

    std::vector<float> h_in(n * 2, 0.0f);
    h_in[0] = 1.0f;

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_C2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(n * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e_re = fabsf(h_out[2 * i] - 1.0f);
        float e_im = fabsf(h_out[2 * i + 1]);
        if (e_re > max_err) max_err = e_re;
        if (e_im > max_err) max_err = e_im;
    }
    EXPECT_LT(max_err, 1e-5f) << "N=" << n;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftC2CImpulseTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      4096, 16384, 6, 12, 24, 30, 120, 480));

/* ========================================================================== */
/* 1D R2C                                                                      */
/* ========================================================================== */

class CufftR2CTest : public CufftTest,
                      public ::testing::WithParamInterface<int> {};

TEST_P(CufftR2CTest, MatchesFftw) {
    int n = GetParam();
    int out_n = n / 2 + 1;

    std::vector<float> h_in(n);
    fill_random(h_in.data(), n, 200 + n);

    /* FFTW reference */
    std::vector<float> ref_out(out_n * 2);
    {
        /* FFTW r2c needs aligned input; use in-place safe copy */
        std::vector<float> tmp_in(h_in);
        fftwf_plan p = fftwf_plan_dft_r2c_1d(n, tmp_in.data(),
                           (fftwf_complex *)ref_out.data(), FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    CUdeviceptr d_in = gpu_upload(h_in.data(), n * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(out_n * sizeof(cufftComplex));

    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_R2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecR2C(plan, (cufftReal *)d_in, (cufftComplex *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(out_n * 2);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), out_n * 2);
    float tol = 1e-3f * sqrtf((float)n);
    EXPECT_LT(err, tol) << "N=" << n << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftR2CTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      2048, 4096, 8192, 16384));

/* ========================================================================== */
/* 1D R2C impulse                                                              */
/* ========================================================================== */

class CufftR2CImpulseTest : public CufftTest,
                             public ::testing::WithParamInterface<int> {};

TEST_P(CufftR2CImpulseTest, DeltaGivesFlat) {
    int n = GetParam();
    int out_n = n / 2 + 1;

    std::vector<float> h_in(n, 0.0f);
    h_in[0] = 1.0f;

    CUdeviceptr d_in = gpu_upload(h_in.data(), n * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(out_n * sizeof(cufftComplex));

    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_R2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecR2C(plan, (cufftReal *)d_in, (cufftComplex *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(out_n * 2);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float max_err = 0.0f;
    for (int i = 0; i < out_n; i++) {
        float e_re = fabsf(h_out[2 * i] - 1.0f);
        float e_im = fabsf(h_out[2 * i + 1]);
        if (e_re > max_err) max_err = e_re;
        if (e_im > max_err) max_err = e_im;
    }
    EXPECT_LT(max_err, 1e-5f) << "N=" << n;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftR2CImpulseTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      2048, 4096, 8192, 16384));

/* ========================================================================== */
/* 1D C2R                                                                      */
/* ========================================================================== */

class CufftC2RTest : public CufftTest,
                      public ::testing::WithParamInterface<int> {};

TEST_P(CufftC2RTest, MatchesFftw) {
    int n = GetParam();
    int in_n = n / 2 + 1;

    /* Generate a valid frequency-domain signal: R2C of random reals */
    std::vector<float> h_real(n);
    fill_random(h_real.data(), n, 300 + n);

    std::vector<float> h_freq(in_n * 2);
    {
        std::vector<float> tmp(h_real);
        fftwf_plan p = fftwf_plan_dft_r2c_1d(n, tmp.data(),
                           (fftwf_complex *)h_freq.data(), FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    /* FFTW C2R reference */
    std::vector<float> ref_out(n);
    {
        std::vector<float> tmp_freq(h_freq);
        fftwf_plan p = fftwf_plan_dft_c2r_1d(n, (fftwf_complex *)tmp_freq.data(),
                           ref_out.data(), FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    /* GPU cuFFT C2R */
    CUdeviceptr d_in = gpu_upload(h_freq.data(), in_n * sizeof(cufftComplex));
    CUdeviceptr d_out = gpu_alloc(n * sizeof(float));

    cufftHandle plan;
    ASSERT_EQ(cufftPlan1d(&plan, n, CUFFT_C2R, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2R(plan, (cufftComplex *)d_in, (cufftReal *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(n);
    gpu_download(d_out, h_out.data(), n * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), n);
    float tol = 1e-2f * (float)n;
    EXPECT_LT(err, tol) << "N=" << n << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftC2RTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      2048, 4096, 8192, 16384));

/* ========================================================================== */
/* 1D R2C → C2R roundtrip                                                     */
/* ========================================================================== */

class CufftR2CC2RRoundtripTest : public CufftTest,
                                  public ::testing::WithParamInterface<int> {};

TEST_P(CufftR2CC2RRoundtripTest, GivesNx) {
    int n = GetParam();
    int half = n / 2 + 1;

    std::vector<float> h_in(n);
    fill_random(h_in.data(), n, 400 + n);

    CUdeviceptr d_real = gpu_upload(h_in.data(), n * sizeof(float));
    CUdeviceptr d_freq = gpu_alloc(half * sizeof(cufftComplex));
    CUdeviceptr d_real_out = gpu_alloc(n * sizeof(float));

    cufftHandle plan_r2c, plan_c2r;
    ASSERT_EQ(cufftPlan1d(&plan_r2c, n, CUFFT_R2C, 1), CUFFT_SUCCESS);
    ASSERT_EQ(cufftPlan1d(&plan_c2r, n, CUFFT_C2R, 1), CUFFT_SUCCESS);

    ASSERT_EQ(cufftExecR2C(plan_r2c, (cufftReal *)d_real,
                            (cufftComplex *)d_freq), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2R(plan_c2r, (cufftComplex *)d_freq,
                            (cufftReal *)d_real_out), CUFFT_SUCCESS);

    std::vector<float> h_out(n);
    gpu_download(d_real_out, h_out.data(), n * sizeof(float));
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float expected = h_in[i] * (float)n;
        float e = fabsf(h_out[i] - expected);
        if (e > max_err) max_err = e;
    }
    float tol = 1e-2f * (float)n;
    EXPECT_LT(max_err, tol) << "N=" << n << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftR2CC2RRoundtripTest,
    ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      2048, 4096, 8192, 16384));

/* ========================================================================== */
/* 2D C2C forward                                                              */
/* ========================================================================== */

class Cufft2DC2CForwardTest : public CufftTest,
    public ::testing::WithParamInterface<std::pair<int,int>> {};

TEST_P(Cufft2DC2CForwardTest, MatchesFftw) {
    auto [nx, ny] = GetParam();
    int total = nx * ny;

    std::vector<float> h_in(total * 2);
    fill_random(h_in.data(), total * 2, 500 + nx * 100 + ny);

    /* FFTW 2D reference */
    std::vector<float> ref_out(total * 2);
    {
        fftwf_complex *in  = (fftwf_complex *)h_in.data();
        fftwf_complex *out = (fftwf_complex *)ref_out.data();
        fftwf_plan p = fftwf_plan_dft_2d(nx, ny, in, out,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan2d(&plan, nx, ny, CUFFT_C2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total * 2);
    float tol = 1e-2f * sqrtf((float)total);
    EXPECT_LT(err, tol) << nx << "x" << ny << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft2DC2CForwardTest,
    ::testing::Values(
        std::make_pair(4, 4),
        std::make_pair(8, 8),
        std::make_pair(16, 16),
        std::make_pair(32, 32),
        std::make_pair(64, 64),
        std::make_pair(8, 16),
        std::make_pair(16, 8),
        std::make_pair(4, 32),
        std::make_pair(32, 4),
        std::make_pair(16, 64),
        std::make_pair(64, 16),
        std::make_pair(128, 128),
        std::make_pair(256, 256),
        std::make_pair(256, 128),
        std::make_pair(128, 256),
        /* Four-step on innermost axis (ny > 4096) */
        std::make_pair(4, 8192),
        std::make_pair(8, 8192),
        std::make_pair(2, 16384)
    ));

/* ========================================================================== */
/* 2D C2C roundtrip                                                            */
/* ========================================================================== */

class Cufft2DC2CRoundtripTest : public CufftTest,
    public ::testing::WithParamInterface<std::pair<int,int>> {};

TEST_P(Cufft2DC2CRoundtripTest, FwdThenInvGivesNxNy) {
    auto [nx, ny] = GetParam();
    int total = nx * ny;

    std::vector<float> h_in(total * 2);
    fill_random(h_in.data(), total * 2, 600 + nx * 100 + ny);

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan2d(&plan, nx, ny, CUFFT_C2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_INVERSE),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float max_err = 0.0f;
    for (int i = 0; i < total * 2; i++) {
        float expected = h_in[i] * (float)total;
        float e = fabsf(h_out[i] - expected);
        if (e > max_err) max_err = e;
    }
    float tol = 1e-1f * (float)total;
    EXPECT_LT(max_err, tol) << nx << "x" << ny << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft2DC2CRoundtripTest,
    ::testing::Values(
        std::make_pair(4, 4),
        std::make_pair(8, 8),
        std::make_pair(16, 16),
        std::make_pair(32, 32),
        std::make_pair(64, 64),
        std::make_pair(8, 16),
        std::make_pair(16, 8),
        std::make_pair(128, 128),
        std::make_pair(256, 256),
        std::make_pair(256, 128),
        std::make_pair(128, 256),
        /* Four-step on innermost axis */
        std::make_pair(4, 8192),
        std::make_pair(8, 8192)
    ));

/* ========================================================================== */
/* 2D C2C impulse                                                              */
/* ========================================================================== */

class Cufft2DImpulseTest : public CufftTest,
    public ::testing::WithParamInterface<std::pair<int,int>> {};

TEST_P(Cufft2DImpulseTest, DeltaGivesFlat) {
    auto [nx, ny] = GetParam();
    int total = nx * ny;

    std::vector<float> h_in(total * 2, 0.0f);
    h_in[0] = 1.0f;

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan2d(&plan, nx, ny, CUFFT_C2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float e_re = fabsf(h_out[2 * i] - 1.0f);
        float e_im = fabsf(h_out[2 * i + 1]);
        if (e_re > max_err) max_err = e_re;
        if (e_im > max_err) max_err = e_im;
    }
    EXPECT_LT(max_err, 1e-5f) << nx << "x" << ny;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft2DImpulseTest,
    ::testing::Values(
        std::make_pair(4, 4),
        std::make_pair(8, 8),
        std::make_pair(16, 16),
        std::make_pair(32, 32),
        std::make_pair(64, 64),
        std::make_pair(8, 16),
        std::make_pair(16, 8),
        std::make_pair(4, 32),
        std::make_pair(32, 4),
        std::make_pair(128, 128),
        std::make_pair(256, 256),
        std::make_pair(256, 128),
        std::make_pair(128, 256),
        /* Four-step on innermost axis */
        std::make_pair(4, 8192)
    ));

/* ========================================================================== */
/* 2D R2C                                                                      */
/* ========================================================================== */

class Cufft2DR2CTest : public CufftTest,
    public ::testing::WithParamInterface<std::pair<int,int>> {};

TEST_P(Cufft2DR2CTest, MatchesFftw) {
    auto [nx, ny] = GetParam();
    int total_r = nx * ny;
    int padded_y = ny / 2 + 1;
    int total_c = nx * padded_y;

    std::vector<float> h_in(total_r);
    fill_random(h_in.data(), total_r, 7000 + nx * 100 + ny);

    /* FFTW 2D R2C reference */
    std::vector<float> ref_out(total_c * 2);
    {
        float *in = (float *)fftwf_malloc(sizeof(float) * total_r);
        memcpy(in, h_in.data(), sizeof(float) * total_r);
        fftwf_complex *out = (fftwf_complex *)ref_out.data();
        fftwf_plan p = fftwf_plan_dft_r2c_2d(nx, ny, in, out, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        fftwf_free(in);
    }

    CUdeviceptr d_in = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(total_c * 2 * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan2d(&plan, nx, ny, CUFFT_R2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecR2C(plan, (cufftReal *)d_in, (cufftComplex *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total_c * 2);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total_c * 2);
    float tol = 1e-2f * sqrtf((float)total_r);
    EXPECT_LT(err, tol) << nx << "x" << ny << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft2DR2CTest,
    ::testing::Values(
        std::make_pair(4, 4),
        std::make_pair(8, 8),
        std::make_pair(16, 16),
        std::make_pair(32, 32),
        std::make_pair(64, 64),
        std::make_pair(128, 128),
        std::make_pair(8, 16),
        std::make_pair(16, 8),
        std::make_pair(64, 128),
        std::make_pair(128, 64),
        std::make_pair(256, 128),
        std::make_pair(128, 256),
        std::make_pair(512, 128),
        std::make_pair(128, 512)
    ));

/* ========================================================================== */
/* 2D R2C→C2R roundtrip                                                        */
/* ========================================================================== */

class Cufft2DR2CC2RRoundtripTest : public CufftTest,
    public ::testing::WithParamInterface<std::pair<int,int>> {};

TEST_P(Cufft2DR2CC2RRoundtripTest, RoundtripGivesNxNy) {
    auto [nx, ny] = GetParam();
    int total_r = nx * ny;
    int padded_y = ny / 2 + 1;
    int total_c = nx * padded_y;

    std::vector<float> h_in(total_r);
    fill_random(h_in.data(), total_r, 8000 + nx * 100 + ny);

    CUdeviceptr d_real = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    CUdeviceptr d_complex = gpu_alloc(total_c * 2 * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(total_r * sizeof(float));

    cufftHandle plan_r2c, plan_c2r;
    ASSERT_EQ(cufftPlan2d(&plan_r2c, nx, ny, CUFFT_R2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftPlan2d(&plan_c2r, nx, ny, CUFFT_C2R), CUFFT_SUCCESS);

    ASSERT_EQ(cufftExecR2C(plan_r2c, (cufftReal *)d_real,
                            (cufftComplex *)d_complex), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2R(plan_c2r, (cufftComplex *)d_complex,
                            (cufftReal *)d_out), CUFFT_SUCCESS);

    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    std::vector<float> h_out(total_r);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));

    float max_err = 0.0f;
    float scale = 1.0f / (float)total_r;
    for (int i = 0; i < total_r; i++) {
        float e = fabsf(h_out[i] * scale - h_in[i]);
        if (e > max_err) max_err = e;
    }
    EXPECT_LT(max_err, 1e-4f) << nx << "x" << ny << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft2DR2CC2RRoundtripTest,
    ::testing::Values(
        std::make_pair(4, 4),
        std::make_pair(8, 8),
        std::make_pair(16, 16),
        std::make_pair(32, 32),
        std::make_pair(64, 64),
        std::make_pair(128, 128),
        std::make_pair(64, 128),
        std::make_pair(128, 64),
        std::make_pair(256, 128),
        std::make_pair(512, 128)
    ));

/* ========================================================================== */
/* Batched 2D R2C via cufftPlanMany                                            */
/* ========================================================================== */

struct BatchR2CParams { int nx, ny, batch; };

class CufftPlanMany2DR2CTest : public CufftTest,
    public ::testing::WithParamInterface<BatchR2CParams> {};

TEST_P(CufftPlanMany2DR2CTest, MatchesFftw) {
    auto [nx, ny, batch] = GetParam();
    int total_r = nx * ny;
    int padded_y = ny / 2 + 1;
    int total_c = nx * padded_y;

    std::vector<float> h_in(total_r * batch);
    fill_random(h_in.data(), total_r * batch, 9000 + nx * 100 + ny * 10 + batch);

    /* FFTW reference: batch independent 2D R2C transforms */
    std::vector<float> ref_out(total_c * 2 * batch);
    for (int b = 0; b < batch; b++) {
        float *in = (float *)fftwf_malloc(sizeof(float) * total_r);
        memcpy(in, h_in.data() + b * total_r, sizeof(float) * total_r);
        fftwf_complex *out = (fftwf_complex *)(ref_out.data() + b * total_c * 2);
        fftwf_plan p = fftwf_plan_dft_r2c_2d(nx, ny, in, out, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        fftwf_free(in);
    }

    CUdeviceptr d_in = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(total_c * 2 * batch * sizeof(float));

    cufftHandle plan;
    int dims[2] = {nx, ny};
    ASSERT_EQ(cufftPlanMany(&plan, 2, dims,
                             NULL, 1, total_r,
                             NULL, 1, total_c,
                             CUFFT_R2C, batch), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecR2C(plan, (cufftReal *)d_in, (cufftComplex *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total_c * 2 * batch);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total_c * 2 * batch);
    float tol = 1e-2f * sqrtf((float)total_r);
    EXPECT_LT(err, tol) << nx << "x" << ny << " batch=" << batch
                         << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftPlanMany2DR2CTest,
    ::testing::Values(
        BatchR2CParams{4, 4, 2},
        BatchR2CParams{8, 8, 3},
        BatchR2CParams{16, 16, 4},
        BatchR2CParams{32, 32, 2},
        BatchR2CParams{64, 64, 2},
        BatchR2CParams{64, 128, 2},
        BatchR2CParams{128, 64, 3},
        BatchR2CParams{128, 128, 2},
        BatchR2CParams{512, 128, 2}
    ));

/* ========================================================================== */
/* Batched 2D C2C via cufftPlanMany                                            */
/* ========================================================================== */

class CufftPlanMany2DC2CTest : public CufftTest,
    public ::testing::WithParamInterface<BatchR2CParams> {};

TEST_P(CufftPlanMany2DC2CTest, MatchesFftw) {
    auto [nx, ny, batch] = GetParam();
    int total = nx * ny;

    std::vector<float> h_in(total * 2 * batch);
    fill_random(h_in.data(), total * 2 * batch, 9500 + nx * 100 + ny * 10 + batch);

    /* FFTW reference */
    std::vector<float> ref_out(total * 2 * batch);
    for (int b = 0; b < batch; b++) {
        fftwf_complex *in  = (fftwf_complex *)(h_in.data() + b * total * 2);
        fftwf_complex *out = (fftwf_complex *)(ref_out.data() + b * total * 2);
        fftwf_plan p = fftwf_plan_dft_2d(nx, ny, in, out,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));

    cufftHandle plan;
    int dims[2] = {nx, ny};
    ASSERT_EQ(cufftPlanMany(&plan, 2, dims,
                             NULL, 1, total,
                             NULL, 1, total,
                             CUFFT_C2C, batch), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2 * batch);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total * 2 * batch);
    float tol = 1e-2f * sqrtf((float)total);
    EXPECT_LT(err, tol) << nx << "x" << ny << " batch=" << batch
                         << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CufftPlanMany2DC2CTest,
    ::testing::Values(
        BatchR2CParams{4, 4, 2},
        BatchR2CParams{8, 8, 3},
        BatchR2CParams{16, 16, 4},
        BatchR2CParams{32, 32, 2},
        BatchR2CParams{64, 64, 2},
        BatchR2CParams{128, 128, 2}
    ));

/* ========================================================================== */
/* 3D C2C forward                                                              */
/* ========================================================================== */

using Dim3 = std::tuple<int,int,int>;

class Cufft3DC2CForwardTest : public CufftTest,
    public ::testing::WithParamInterface<Dim3> {};

TEST_P(Cufft3DC2CForwardTest, MatchesFftw) {
    auto [nx, ny, nz] = GetParam();
    int total = nx * ny * nz;

    std::vector<float> h_in(total * 2);
    fill_random(h_in.data(), total * 2, 1000 + nx * 100 + ny * 10 + nz);

    /* FFTW 3D reference */
    std::vector<float> ref_out(total * 2);
    {
        fftwf_complex *in  = (fftwf_complex *)h_in.data();
        fftwf_complex *out = (fftwf_complex *)ref_out.data();
        fftwf_plan p = fftwf_plan_dft_3d(nx, ny, nz, in, out,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total * 2);
    float tol = 1e-2f * sqrtf((float)total);
    EXPECT_LT(err, tol) << nx << "x" << ny << "x" << nz << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft3DC2CForwardTest,
    ::testing::Values(
        Dim3{4, 4, 4},
        Dim3{8, 8, 8},
        Dim3{16, 16, 16},
        Dim3{4, 4, 8},
        Dim3{4, 8, 4},
        Dim3{8, 4, 4},
        Dim3{4, 8, 16},
        Dim3{16, 4, 8},
        Dim3{8, 16, 4},
        Dim3{32, 32, 32},
        /* Four-step on innermost axis (nz > 4096) */
        Dim3{2, 2, 8192},
        Dim3{4, 2, 8192}
    ));

/* ========================================================================== */
/* 3D C2C roundtrip                                                            */
/* ========================================================================== */

class Cufft3DC2CRoundtripTest : public CufftTest,
    public ::testing::WithParamInterface<Dim3> {};

TEST_P(Cufft3DC2CRoundtripTest, FwdThenInvGivesNxNyNz) {
    auto [nx, ny, nz] = GetParam();
    int total = nx * ny * nz;

    std::vector<float> h_in(total * 2);
    fill_random(h_in.data(), total * 2, 1100 + nx * 100 + ny * 10 + nz);

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_INVERSE),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float max_err = 0.0f;
    for (int i = 0; i < total * 2; i++) {
        float expected = h_in[i] * (float)total;
        float e = fabsf(h_out[i] - expected);
        if (e > max_err) max_err = e;
    }
    float tol = 1e-1f * (float)total;
    EXPECT_LT(max_err, tol) << nx << "x" << ny << "x" << nz
                              << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft3DC2CRoundtripTest,
    ::testing::Values(
        Dim3{4, 4, 4},
        Dim3{8, 8, 8},
        Dim3{16, 16, 16},
        Dim3{4, 8, 16},
        Dim3{16, 4, 8},
        Dim3{32, 32, 32},
        /* Four-step on innermost axis */
        Dim3{2, 2, 8192}
    ));

/* ========================================================================== */
/* 3D C2C impulse                                                              */
/* ========================================================================== */

class Cufft3DImpulseTest : public CufftTest,
    public ::testing::WithParamInterface<Dim3> {};

TEST_P(Cufft3DImpulseTest, DeltaGivesFlat) {
    auto [nx, ny, nz] = GetParam();
    int total = nx * ny * nz;

    std::vector<float> h_in(total * 2, 0.0f);
    h_in[0] = 1.0f;

    CUdeviceptr d_data = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2C(plan, (cufftComplex *)d_data,
                            (cufftComplex *)d_data, CUFFT_FORWARD),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total * 2);
    gpu_download(d_data, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float e_re = fabsf(h_out[2 * i] - 1.0f);
        float e_im = fabsf(h_out[2 * i + 1]);
        if (e_re > max_err) max_err = e_re;
        if (e_im > max_err) max_err = e_im;
    }
    EXPECT_LT(max_err, 1e-5f) << nx << "x" << ny << "x" << nz;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft3DImpulseTest,
    ::testing::Values(
        Dim3{4, 4, 4},
        Dim3{8, 8, 8},
        Dim3{16, 16, 16},
        Dim3{4, 8, 16},
        Dim3{16, 4, 8},
        Dim3{8, 16, 4},
        Dim3{32, 32, 32}
    ));

/* ========================================================================== */
/* 3D R2C                                                                      */
/* ========================================================================== */

class Cufft3DR2CTest : public CufftTest,
    public ::testing::WithParamInterface<Dim3> {};

TEST_P(Cufft3DR2CTest, MatchesFftw) {
    auto [nx, ny, nz] = GetParam();
    int total_r = nx * ny * nz;
    int padded_z = nz / 2 + 1;
    int total_c = nx * ny * padded_z;

    std::vector<float> h_in(total_r);
    fill_random(h_in.data(), total_r, 2000 + nx * 100 + ny * 10 + nz);

    /* FFTW 3D R2C reference */
    std::vector<float> ref_out(total_c * 2);
    {
        /* FFTW needs aligned input; copy to FFTW-allocated buffer */
        float *in = (float *)fftwf_malloc(sizeof(float) * total_r);
        memcpy(in, h_in.data(), sizeof(float) * total_r);
        fftwf_complex *out = (fftwf_complex *)ref_out.data();
        fftwf_plan p = fftwf_plan_dft_r2c_3d(nx, ny, nz, in, out,
                                               FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        fftwf_free(in);
    }

    CUdeviceptr d_in = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(total_c * 2 * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan3d(&plan, nx, ny, nz, CUFFT_R2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecR2C(plan, (cufftReal *)d_in, (cufftComplex *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total_c * 2);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total_c * 2);
    float tol = 1e-2f * sqrtf((float)total_r);
    EXPECT_LT(err, tol) << nx << "x" << ny << "x" << nz << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft3DR2CTest,
    ::testing::Values(
        Dim3{4, 4, 4},
        Dim3{8, 8, 8},
        Dim3{16, 16, 16},
        Dim3{4, 4, 8},
        Dim3{4, 8, 4},
        Dim3{8, 4, 4},
        Dim3{4, 8, 16},
        Dim3{16, 4, 8},
        Dim3{32, 32, 32}
    ));

/* ========================================================================== */
/* 3D C2R                                                                      */
/* ========================================================================== */

class Cufft3DC2RTest : public CufftTest,
    public ::testing::WithParamInterface<Dim3> {};

TEST_P(Cufft3DC2RTest, MatchesFftw) {
    auto [nx, ny, nz] = GetParam();
    int total_r = nx * ny * nz;
    int padded_z = nz / 2 + 1;
    int total_c = nx * ny * padded_z;

    /* Start with valid frequency-domain data: R2C of random reals */
    std::vector<float> h_real(total_r);
    fill_random(h_real.data(), total_r, 3000 + nx * 100 + ny * 10 + nz);

    std::vector<float> h_freq(total_c * 2);
    {
        float *in = (float *)fftwf_malloc(sizeof(float) * total_r);
        memcpy(in, h_real.data(), sizeof(float) * total_r);
        fftwf_complex *out = (fftwf_complex *)h_freq.data();
        fftwf_plan p = fftwf_plan_dft_r2c_3d(nx, ny, nz, in, out,
                                               FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        fftwf_free(in);
    }

    /* FFTW C2R reference */
    std::vector<float> ref_out(total_r);
    {
        fftwf_complex *in = (fftwf_complex *)fftwf_malloc(
            sizeof(fftwf_complex) * total_c);
        memcpy(in, h_freq.data(), sizeof(float) * total_c * 2);
        float *out = (float *)fftwf_malloc(sizeof(float) * total_r);
        fftwf_plan p = fftwf_plan_dft_c2r_3d(nx, ny, nz, in, out,
                                               FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        memcpy(ref_out.data(), out, sizeof(float) * total_r);
        fftwf_free(in);
        fftwf_free(out);
    }

    CUdeviceptr d_in = gpu_upload(h_freq.data(), h_freq.size() * sizeof(float));
    CUdeviceptr d_out = gpu_alloc(total_r * sizeof(float));
    cufftHandle plan;
    ASSERT_EQ(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2R), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2R(plan, (cufftComplex *)d_in, (cufftReal *)d_out),
              CUFFT_SUCCESS);

    std::vector<float> h_out(total_r);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));
    cufftDestroy(plan);

    float err = max_abs_error(h_out.data(), ref_out.data(), total_r);
    float tol = 1e-2f * sqrtf((float)total_r);
    EXPECT_LT(err, tol) << nx << "x" << ny << "x" << nz << " max_err=" << err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft3DC2RTest,
    ::testing::Values(
        Dim3{4, 4, 4},
        Dim3{8, 8, 8},
        Dim3{16, 16, 16},
        Dim3{4, 4, 8},
        Dim3{4, 8, 4},
        Dim3{8, 4, 4},
        Dim3{4, 8, 16},
        Dim3{16, 4, 8},
        Dim3{32, 32, 32}
    ));

/* ========================================================================== */
/* 3D R2C→C2R roundtrip                                                        */
/* ========================================================================== */

class Cufft3DR2CC2RRoundtripTest : public CufftTest,
    public ::testing::WithParamInterface<Dim3> {};

TEST_P(Cufft3DR2CC2RRoundtripTest, RoundtripGivesNxNyNz) {
    auto [nx, ny, nz] = GetParam();
    int total_r = nx * ny * nz;
    int padded_z = nz / 2 + 1;
    int total_c = nx * ny * padded_z;

    std::vector<float> h_in(total_r);
    fill_random(h_in.data(), total_r, 4000 + nx * 100 + ny * 10 + nz);

    CUdeviceptr d_real = gpu_upload(h_in.data(), h_in.size() * sizeof(float));
    CUdeviceptr d_complex = gpu_alloc(total_c * 2 * sizeof(float));

    /* R2C */
    cufftHandle plan_r2c;
    ASSERT_EQ(cufftPlan3d(&plan_r2c, nx, ny, nz, CUFFT_R2C), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecR2C(plan_r2c, (cufftReal *)d_real,
                            (cufftComplex *)d_complex), CUFFT_SUCCESS);
    cufftDestroy(plan_r2c);

    /* C2R */
    CUdeviceptr d_out = gpu_alloc(total_r * sizeof(float));
    cufftHandle plan_c2r;
    ASSERT_EQ(cufftPlan3d(&plan_c2r, nx, ny, nz, CUFFT_C2R), CUFFT_SUCCESS);
    ASSERT_EQ(cufftExecC2R(plan_c2r, (cufftComplex *)d_complex,
                            (cufftReal *)d_out), CUFFT_SUCCESS);
    cufftDestroy(plan_c2r);

    std::vector<float> h_out(total_r);
    gpu_download(d_out, h_out.data(), h_out.size() * sizeof(float));

    float max_err = 0.0f;
    for (int i = 0; i < total_r; i++) {
        float expected = h_in[i] * (float)total_r;
        float e = fabsf(h_out[i] - expected);
        if (e > max_err) max_err = e;
    }
    float tol = 1e-1f * (float)total_r;
    EXPECT_LT(max_err, tol) << nx << "x" << ny << "x" << nz
                              << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(Sizes, Cufft3DR2CC2RRoundtripTest,
    ::testing::Values(
        Dim3{4, 4, 4},
        Dim3{8, 8, 8},
        Dim3{16, 16, 16},
        Dim3{4, 8, 16},
        Dim3{16, 4, 8},
        Dim3{32, 32, 32}
    ));
