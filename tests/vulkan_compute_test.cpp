#include <gtest/gtest.h>
#include <cmath>
#include "test_utils.h"

#ifdef WGSL_HAS_VULKAN
#include "vulkan_compute_harness.h"

class VulkanComputeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            ctx_ = std::make_unique<vk_compute::VulkanContext>();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Vulkan not available: " << e.what();
        }
    }

    static void TearDownTestSuite() {
        ctx_.reset();
    }

    void SetUp() override {
        if (!ctx_) {
            GTEST_SKIP() << "Vulkan context not initialized";
        }
    }

    static std::unique_ptr<vk_compute::VulkanContext> ctx_;
};

std::unique_ptr<vk_compute::VulkanContext> VulkanComputeTest::ctx_;

TEST_F(VulkanComputeTest, BufferCopy) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = input.data[id.x];
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], input_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, ScalarAdd) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> a: Buffer;
        @group(0) @binding(1) var<storage, read> b: Buffer;
        @group(0) @binding(2) var<storage, read_write> result: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            result.data[id.x] = a.data[id.x] + b.data[id.x];
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] + b_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, ScalarMultiply) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> a: Buffer;
        @group(0) @binding(1) var<storage, read> b: Buffer;
        @group(0) @binding(2) var<storage, read_write> result: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            result.data[id.x] = a.data[id.x] * b.data[id.x];
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b_data = {10.0f, 10.0f, 10.0f, 10.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] * b_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, IntegerArithmetic) {
    const char* source = R"(
        struct IntBuffer {
            data: array<i32>,
        };

        @group(0) @binding(0) var<storage, read> a: IntBuffer;
        @group(0) @binding(1) var<storage, read> b: IntBuffer;
        @group(0) @binding(2) var<storage, read_write> result: IntBuffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            result.data[id.x] = a.data[id.x] + b.data[id.x] * 2;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<int32_t> a_data = {1, 2, 3, 4};
    std::vector<int32_t> b_data = {10, 20, 30, 40};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(int32_t));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<int32_t>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_EQ(output_data[i], a_data[i] + b_data[i] * 2) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, ConditionalSelect) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> a: Buffer;
        @group(0) @binding(1) var<storage, read> b: Buffer;
        @group(0) @binding(2) var<storage, read_write> result: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            if (a.data[id.x] > b.data[id.x]) {
                result.data[id.x] = a.data[id.x];
            } else {
                result.data[id.x] = b.data[id.x];
            }
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {5.0f, 2.0f, 8.0f, 1.0f};
    std::vector<float> b_data = {3.0f, 7.0f, 4.0f, 9.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        float expected = a_data[i] > b_data[i] ? a_data[i] : b_data[i];
        EXPECT_FLOAT_EQ(output_data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, LoopSum) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                sum = sum + input.data[i];
            }
            output.data[id.x] = sum;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, 1);

    auto output_data = output.download<float>(1);
    float expected = 1.0f + 2.0f + 3.0f + 4.0f;
    EXPECT_FLOAT_EQ(output_data[0], expected);
}

TEST_F(VulkanComputeTest, MathAbs) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = abs(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], std::abs(input_data[i])) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, MathMinMax) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> a: Buffer;
        @group(0) @binding(1) var<storage, read> b: Buffer;
        @group(0) @binding(2) var<storage, read_write> min_out: Buffer;
        @group(0) @binding(3) var<storage, read_write> max_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            min_out.data[id.x] = min(a.data[id.x], b.data[id.x]);
            max_out.data[id.x] = max(a.data[id.x], b.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> b_data = {4.0f, 2.0f, 6.0f, 7.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto min_out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));
    auto max_out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, &min_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {3, &max_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(a_data.size()));

    auto min_data = min_out.download<float>(a_data.size());
    auto max_data = max_out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(min_data[i], std::min(a_data[i], b_data[i])) << "min mismatch at " << i;
        EXPECT_FLOAT_EQ(max_data[i], std::max(a_data[i], b_data[i])) << "max mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, MathClamp) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = clamp(input.data[id.x], 0.0, 1.0);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {-0.5f, 0.5f, 1.5f, 0.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    std::vector<float> expected = {0.0f, 0.5f, 1.0f, 0.0f};
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, MathFloorCeil) {
    const char* source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> floor_out: Buffer;
        @group(0) @binding(2) var<storage, read_write> ceil_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            floor_out.data[id.x] = floor(input.data[id.x]);
            ceil_out.data[id.x] = ceil(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.3f, 2.7f, -1.3f, -2.7f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto floor_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto ceil_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {
        {0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, &floor_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, &ceil_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    }, static_cast<uint32_t>(input_data.size()));

    auto floor_data = floor_out.download<float>(input_data.size());
    auto ceil_data = ceil_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(floor_data[i], std::floor(input_data[i])) << "floor mismatch at " << i;
        EXPECT_FLOAT_EQ(ceil_data[i], std::ceil(input_data[i])) << "ceil mismatch at " << i;
    }
}

#endif // WGSL_HAS_VULKAN
