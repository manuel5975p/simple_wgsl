/*
 * cuvk_stream_test.cpp - Tests for CUDA-on-Vulkan stream and event APIs
 */

#include <gtest/gtest.h>

extern "C" {
#include "cuda.h"
}

class CuvkStreamTest : public ::testing::Test {
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

TEST_F(CuvkStreamTest, CreateDestroy) {
    CUstream stream = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuStreamCreate(&stream, 0));
    EXPECT_NE(nullptr, stream);
    EXPECT_EQ(CUDA_SUCCESS, cuStreamDestroy(stream));
}

TEST_F(CuvkStreamTest, SyncEmpty) {
    CUstream stream = NULL;
    cuStreamCreate(&stream, 0);
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(stream));
    cuStreamDestroy(stream);
}

TEST_F(CuvkStreamTest, QueryIdle) {
    CUstream stream = NULL;
    cuStreamCreate(&stream, 0);
    EXPECT_EQ(CUDA_SUCCESS, cuStreamQuery(stream));
    cuStreamDestroy(stream);
}

TEST_F(CuvkStreamTest, NullStreamSync) {
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(NULL));
}

TEST_F(CuvkStreamTest, EventCreateDestroy) {
    CUevent event = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuEventCreate(&event, 0));
    EXPECT_NE(nullptr, event);
    EXPECT_EQ(CUDA_SUCCESS, cuEventDestroy(event));
}

TEST_F(CuvkStreamTest, EventRecordSync) {
    CUevent event = NULL;
    cuEventCreate(&event, 0);
    EXPECT_EQ(CUDA_SUCCESS, cuEventRecord(event, NULL));
    EXPECT_EQ(CUDA_SUCCESS, cuEventSynchronize(event));
    cuEventDestroy(event);
}

TEST_F(CuvkStreamTest, EventElapsedTime) {
    CUevent start = NULL, end = NULL;
    cuEventCreate(&start, 0);
    cuEventCreate(&end, 0);
    cuEventRecord(start, NULL);
    cuEventRecord(end, NULL);
    cuEventSynchronize(end);
    float ms = -1;
    EXPECT_EQ(CUDA_SUCCESS, cuEventElapsedTime(&ms, start, end));
    EXPECT_GE(ms, 0.0f);
    cuEventDestroy(start);
    cuEventDestroy(end);
}
