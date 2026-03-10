#include <gtest/gtest.h>
extern "C" {
#include "fft_2d_gen.h"
}
#include "test_utils.h"
using namespace wgsl_test;

/* ========================================================================== */
/* Tiled transpose tests                                                      */
/* ========================================================================== */

class TransposeTiledTest : public ::testing::TestWithParam<
    std::tuple<int, int, int>> {};

TEST_P(TransposeTiledTest, Compiles) {
    auto [nx, ny, tile_dim] = GetParam();
    char *wgsl = gen_transpose_tiled(nx, ny, tile_dim);
    ASSERT_NE(wgsl, nullptr) << nx << "x" << ny << " tile=" << tile_dim;
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << wgsl;
    free(wgsl);
}

INSTANTIATE_TEST_SUITE_P(Sizes, TransposeTiledTest, ::testing::Values(
    std::make_tuple(4, 4, 4),
    std::make_tuple(8, 8, 4),
    std::make_tuple(8, 8, 8),
    std::make_tuple(16, 16, 8),
    std::make_tuple(16, 16, 16),
    std::make_tuple(32, 32, 16),
    std::make_tuple(32, 32, 32),
    std::make_tuple(64, 64, 16),
    std::make_tuple(64, 64, 32),
    std::make_tuple(128, 128, 32),
    std::make_tuple(256, 256, 32),
    // Rectangular
    std::make_tuple(8, 16, 8),
    std::make_tuple(16, 8, 8),
    std::make_tuple(64, 128, 32),
    std::make_tuple(128, 64, 32)
));

TEST(TransposeTiledTest, RejectsNonPo2Tile) {
    EXPECT_EQ(gen_transpose_tiled(8, 8, 3), nullptr);
    EXPECT_EQ(gen_transpose_tiled(8, 8, 6), nullptr);
}

TEST(TransposeTiledTest, RejectsInvalidDims) {
    EXPECT_EQ(gen_transpose_tiled(0, 8, 4), nullptr);
    EXPECT_EQ(gen_transpose_tiled(8, 0, 4), nullptr);
    EXPECT_EQ(gen_transpose_tiled(8, 8, 1), nullptr);
}

TEST(TransposeTiledTest, WorkgroupSize) {
    EXPECT_EQ(transpose_tiled_workgroup_size(4), 16);
    EXPECT_EQ(transpose_tiled_workgroup_size(8), 64);
    EXPECT_EQ(transpose_tiled_workgroup_size(16), 256);
    EXPECT_EQ(transpose_tiled_workgroup_size(32), 1024);
}

/* ========================================================================== */
/* 2D fused FFT tests                                                         */
/* ========================================================================== */

class Fft2dFusedTest : public ::testing::TestWithParam<
    std::tuple<int, int, int>> {};

TEST_P(Fft2dFusedTest, CompilesForward) {
    auto [nx, ny, max_radix] = GetParam();
    char *wgsl = gen_fft_2d_fused(nx, ny, 1, max_radix);
    ASSERT_NE(wgsl, nullptr) << nx << "x" << ny << " mr=" << max_radix;
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << wgsl;
    free(wgsl);
}

TEST_P(Fft2dFusedTest, CompilesInverse) {
    auto [nx, ny, max_radix] = GetParam();
    char *wgsl = gen_fft_2d_fused(nx, ny, -1, max_radix);
    ASSERT_NE(wgsl, nullptr) << nx << "x" << ny << " mr=" << max_radix;
    auto result = CompileWgsl(wgsl);
    EXPECT_TRUE(result.success) << wgsl;
    free(wgsl);
}

INSTANTIATE_TEST_SUITE_P(Sizes, Fft2dFusedTest, ::testing::Values(
    std::make_tuple(2, 2, 0),
    std::make_tuple(4, 4, 0),
    std::make_tuple(8, 8, 0),
    std::make_tuple(16, 16, 0),
    std::make_tuple(32, 32, 0),
    // Rectangular
    std::make_tuple(4, 8, 0),
    std::make_tuple(8, 4, 0),
    std::make_tuple(8, 16, 0),
    std::make_tuple(16, 8, 0),
    // Explicit max_radix
    std::make_tuple(16, 16, 4),
    std::make_tuple(16, 16, 8),
    std::make_tuple(32, 32, 8),
    std::make_tuple(32, 32, 16)
));

TEST(Fft2dFusedTest, WorkgroupSize) {
    int wg = fft_2d_fused_workgroup_size(4, 4, 0);
    EXPECT_GT(wg, 0);
}

TEST(Fft2dFusedTest, LutSizeSmall) {
    // 4x4 with auto radix should use direct path (baked twiddles)
    int lut = fft_2d_fused_lut_size(4, 4, 1, 0);
    // May or may not need LUT depending on radix cap
    EXPECT_GE(lut, 0);
}

TEST(Fft2dFusedTest, RejectsInvalid) {
    EXPECT_EQ(gen_fft_2d_fused(1, 4, 1, 0), nullptr);
    EXPECT_EQ(gen_fft_2d_fused(4, 1, 1, 0), nullptr);
    EXPECT_EQ(gen_fft_2d_fused(4, 4, 0, 0), nullptr);  // direction=0
}
