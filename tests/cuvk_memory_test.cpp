#include <gtest/gtest.h>
#include <vector>
#include <cstring>
extern "C" {
#include "cuda.h"
}

class CuvkMemoryTest : public ::testing::Test {
protected:
    CUcontext ctx = NULL;
    void SetUp() override {
        cuInit(0);
        CUdevice dev;
        cuDeviceGet(&dev, 0);
        cuCtxCreate(&ctx, NULL, 0, dev);
    }
    void TearDown() override {
        if (ctx) cuCtxDestroy(ctx);
    }
};

TEST_F(CuvkMemoryTest, AllocFree) {
    CUdeviceptr ptr = 0;
    EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, 1024));
    EXPECT_NE(0u, ptr);
    EXPECT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
}

TEST_F(CuvkMemoryTest, MultipleAllocs) {
    CUdeviceptr a = 0, b = 0, c = 0;
    EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&a, 1024));
    EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&b, 2048));
    EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&c, 512));
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    cuMemFree(a);
    cuMemFree(b);
    cuMemFree(c);
}

TEST_F(CuvkMemoryTest, HtoDtoH) {
    CUdeviceptr ptr = 0;
    cuMemAlloc(&ptr, 256);
    float src[64];
    for (int i = 0; i < 64; i++) src[i] = (float)i;
    EXPECT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(ptr, src, sizeof(src)));
    float dst[64] = {};
    EXPECT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(dst, ptr, sizeof(dst)));
    for (int i = 0; i < 64; i++) EXPECT_FLOAT_EQ(src[i], dst[i]);
    cuMemFree(ptr);
}

TEST_F(CuvkMemoryTest, DtoD) {
    CUdeviceptr a = 0, b = 0;
    cuMemAlloc(&a, 256);
    cuMemAlloc(&b, 256);
    float src[64];
    for (int i = 0; i < 64; i++) src[i] = (float)(i * 2);
    cuMemcpyHtoD(a, src, sizeof(src));
    EXPECT_EQ(CUDA_SUCCESS, cuMemcpyDtoD(b, a, sizeof(src)));
    float dst[64] = {};
    cuMemcpyDtoH(dst, b, sizeof(dst));
    for (int i = 0; i < 64; i++) EXPECT_FLOAT_EQ(src[i], dst[i]);
    cuMemFree(a);
    cuMemFree(b);
}

TEST_F(CuvkMemoryTest, MemsetD32) {
    CUdeviceptr ptr = 0;
    cuMemAlloc(&ptr, 256);
    EXPECT_EQ(CUDA_SUCCESS, cuMemsetD32(ptr, 0xDEADBEEF, 64));
    uint32_t dst[64] = {};
    cuMemcpyDtoH(dst, ptr, 256);
    for (int i = 0; i < 64; i++) EXPECT_EQ(0xDEADBEEFu, dst[i]);
    cuMemFree(ptr);
}

TEST_F(CuvkMemoryTest, MemsetD8) {
    CUdeviceptr ptr = 0;
    cuMemAlloc(&ptr, 64);
    EXPECT_EQ(CUDA_SUCCESS, cuMemsetD8(ptr, 0xAB, 64));
    uint8_t dst[64] = {};
    cuMemcpyDtoH(dst, ptr, 64);
    for (int i = 0; i < 64; i++) EXPECT_EQ(0xAB, dst[i]);
    cuMemFree(ptr);
}

TEST_F(CuvkMemoryTest, MemsetD16) {
    CUdeviceptr ptr = 0;
    cuMemAlloc(&ptr, 128);
    EXPECT_EQ(CUDA_SUCCESS, cuMemsetD16(ptr, 0x1234, 64));
    uint16_t dst[64] = {};
    cuMemcpyDtoH(dst, ptr, 128);
    for (int i = 0; i < 64; i++) EXPECT_EQ(0x1234, dst[i]);
    cuMemFree(ptr);
}
