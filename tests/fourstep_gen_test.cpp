#include <gtest/gtest.h>
extern "C" {
#include "fft_fourstep_gen.h"
}
#include "test_utils.h"
using namespace wgsl_test;

/* ========================================================================== */
/* Twiddle + Transpose tests                                                  */
/* ========================================================================== */

TEST(FourStepGenTest, TwiddleTranspose_Forward) {
    char *wgsl = gen_fft_twiddle_transpose(8, 16, 1, 64);
    ASSERT_NE(wgsl, nullptr);
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << result.error;
    free(wgsl);
}

TEST(FourStepGenTest, TwiddleTranspose_Inverse) {
    char *wgsl = gen_fft_twiddle_transpose(8, 16, -1, 64);
    ASSERT_NE(wgsl, nullptr);
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << result.error;
    free(wgsl);
}

TEST(FourStepGenTest, TwiddleTranspose_Square) {
    char *wgsl = gen_fft_twiddle_transpose(32, 32, 1, 256);
    ASSERT_NE(wgsl, nullptr);
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << result.error;
    free(wgsl);
}

/* ========================================================================== */
/* Pure Transpose tests                                                       */
/* ========================================================================== */

TEST(FourStepGenTest, Transpose_Basic) {
    char *wgsl = gen_fft_transpose(8, 16, 64);
    ASSERT_NE(wgsl, nullptr);
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << result.error;
    free(wgsl);
}

TEST(FourStepGenTest, Transpose_Square) {
    char *wgsl = gen_fft_transpose(32, 32, 256);
    ASSERT_NE(wgsl, nullptr);
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << result.error;
    free(wgsl);
}

/* ========================================================================== */
/* Decomposition tests                                                        */
/* ========================================================================== */

TEST(FourStepGenTest, Decompose_Small) {
    int n1, n2;
    int ok = fft_fourstep_decompose(128, 4096, &n1, &n2);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(n1 * n2, 128);
}

TEST(FourStepGenTest, Decompose_Balanced) {
    int n1, n2;
    int ok = fft_fourstep_decompose(65536, 4096, &n1, &n2);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(n1 * n2, 65536);
    EXPECT_LE(n1, 4096);
    EXPECT_LE(n2, 4096);
    EXPECT_GE(n1, 16);
    EXPECT_GE(n2, 16);
}

TEST(FourStepGenTest, Decompose_Large) {
    int n1, n2;
    int ok = fft_fourstep_decompose(1048576, 4096, &n1, &n2);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(n1 * n2, 1048576);
}

TEST(FourStepGenTest, Decompose_Fails) {
    int n1, n2;
    int ok = fft_fourstep_decompose(17000003, 4096, &n1, &n2);
    EXPECT_EQ(ok, 0);
}

TEST(FourStepGenTest, Decompose_NoSplit) {
    int n1, n2;
    int ok = fft_fourstep_decompose(256, 4096, &n1, &n2);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(n1, 1);
    EXPECT_EQ(n2, 256);
}
