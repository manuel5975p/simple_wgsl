/*
 * cuvk_init_test.cpp - Tests for CUDA-on-Vulkan init, device, and context APIs
 */

#include <gtest/gtest.h>

extern "C" {
#include "cuda.h"
}

class CuvkInitTest : public ::testing::Test {
protected:
    void SetUp() override {
        /* Ensure cuInit is called before each test */
        CUresult res = cuInit(0);
        ASSERT_EQ(res, CUDA_SUCCESS) << "cuInit failed - no Vulkan device?";
    }
};

/* cuInit succeeds */
TEST_F(CuvkInitTest, InitSucceeds) {
    /* SetUp already called cuInit, just verify it returned success */
    CUresult res = cuInit(0);
    EXPECT_EQ(res, CUDA_SUCCESS);
}

/* Double-init is OK */
TEST_F(CuvkInitTest, DoubleInitOk) {
    CUresult res = cuInit(0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuInit(0);
    EXPECT_EQ(res, CUDA_SUCCESS);
}

/* cuDeviceGetCount returns at least 1 device */
TEST_F(CuvkInitTest, DeviceCount) {
    int count = 0;
    CUresult res = cuDeviceGetCount(&count);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GE(count, 1);
}

/* cuDeviceGet with ordinal 0 */
TEST_F(CuvkInitTest, DeviceGet) {
    CUdevice dev = -1;
    CUresult res = cuDeviceGet(&dev, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(dev, 0);
}

/* cuDeviceGet with invalid ordinal */
TEST_F(CuvkInitTest, DeviceGetInvalid) {
    CUdevice dev = -1;
    CUresult res = cuDeviceGet(&dev, 9999);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_DEVICE);
}

/* cuDeviceGetName returns a non-empty name */
TEST_F(CuvkInitTest, DeviceGetName) {
    char name[256] = {0};
    CUresult res = cuDeviceGetName(name, sizeof(name), 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GT(strlen(name), 0u);
}

/* cuDeviceTotalMem returns non-zero memory */
TEST_F(CuvkInitTest, DeviceTotalMem) {
    size_t bytes = 0;
    CUresult res = cuDeviceTotalMem(&bytes, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GT(bytes, 0u);
}

/* cuDeviceGetAttribute returns sane values for key attributes */
TEST_F(CuvkInitTest, DeviceGetAttribute) {
    int val = 0;

    /* MAX_THREADS_PER_BLOCK should be positive */
    CUresult res = cuDeviceGetAttribute(
        &val, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GT(val, 0);

    /* MAX_BLOCK_DIM_X should be positive */
    res = cuDeviceGetAttribute(
        &val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GT(val, 0);

    /* WARP_SIZE should be a power of 2, typically 32 or 64 */
    res = cuDeviceGetAttribute(
        &val, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GT(val, 0);
    /* Check it's a power of 2 */
    EXPECT_EQ(val & (val - 1), 0);

    /* MAX_SHARED_MEMORY_PER_BLOCK should be positive */
    res = cuDeviceGetAttribute(
        &val, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_GT(val, 0);

    /* Unknown attribute returns SUCCESS with zero (for libcudart compat) */
    res = cuDeviceGetAttribute(
        &val, (CUdevice_attribute)99999, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(val, 0);
}

/* cuCtxCreate + cuCtxDestroy lifecycle */
TEST_F(CuvkInitTest, CtxCreateDestroy) {
    CUcontext ctx = nullptr;
    CUresult res = cuCtxCreate(&ctx, NULL, 0, 0);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    /* Should be set as current */
    CUcontext current = nullptr;
    res = cuCtxGetCurrent(&current);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(current, ctx);

    /* Synchronize should work */
    res = cuCtxSynchronize();
    EXPECT_EQ(res, CUDA_SUCCESS);

    /* Destroy */
    res = cuCtxDestroy(ctx);
    EXPECT_EQ(res, CUDA_SUCCESS);

    /* Current context should be cleared */
    res = cuCtxGetCurrent(&current);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(current, nullptr);
}

/* cuDriverGetVersion returns 12000 */
TEST_F(CuvkInitTest, DriverVersion) {
    int version = 0;
    CUresult res = cuDriverGetVersion(&version);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(version, 13020);
}

/* cuCtxSetCurrent / cuCtxGetCurrent */
TEST_F(CuvkInitTest, CtxSetGetCurrent) {
    CUcontext ctx = nullptr;
    CUresult res = cuCtxCreate(&ctx, NULL, 0, 0);
    ASSERT_EQ(res, CUDA_SUCCESS);

    /* Set to NULL */
    res = cuCtxSetCurrent(NULL);
    EXPECT_EQ(res, CUDA_SUCCESS);

    CUcontext current = nullptr;
    res = cuCtxGetCurrent(&current);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(current, nullptr);

    /* Set back */
    res = cuCtxSetCurrent(ctx);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current);
    EXPECT_EQ(res, CUDA_SUCCESS);
    EXPECT_EQ(current, ctx);

    /* Cleanup */
    cuCtxDestroy(ctx);
}
