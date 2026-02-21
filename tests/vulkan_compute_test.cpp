#include <gtest/gtest.h>
#include <cmath>
#include "test_utils.h"

#ifdef WGSL_HAS_VULKAN
#include "vulkan_compute_harness.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
extern "C" {
#include "stb_image_write.h"
}

class VulkanComputeTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        try {
            ctx_ = std::make_unique<vk_compute::VulkanContext>();
        } catch (const std::exception &e) {
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
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], input_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, ScalarAdd) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] + b_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, ScalarMultiply) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] * b_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, IntegerArithmetic) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<int32_t>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_EQ(output_data[i], a_data[i] + b_data[i] * 2) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, ConditionalSelect) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        float expected = a_data[i] > b_data[i] ? a_data[i] : b_data[i];
        EXPECT_FLOAT_EQ(output_data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, LoopSum) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto output_data = output.download<float>(1);
    float expected = 1.0f + 2.0f + 3.0f + 4.0f;
    EXPECT_FLOAT_EQ(output_data[0], expected);
}

TEST_F(VulkanComputeTest, MathAbs) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], std::abs(input_data[i])) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, MathMinMax) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &min_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, &max_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto min_data = min_out.download<float>(a_data.size());
    auto max_data = max_out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(min_data[i], std::min(a_data[i], b_data[i])) << "min mismatch at " << i;
        EXPECT_FLOAT_EQ(max_data[i], std::max(a_data[i], b_data[i])) << "max mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, MathClamp) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    std::vector<float> expected = {0.0f, 0.5f, 1.0f, 0.0f};
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, MathFloorCeil) {
    const char *source = R"(
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
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &floor_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &ceil_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto floor_data = floor_out.download<float>(input_data.size());
    auto ceil_data = ceil_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(floor_data[i], std::floor(input_data[i])) << "floor mismatch at " << i;
        EXPECT_FLOAT_EQ(ceil_data[i], std::ceil(input_data[i])) << "ceil mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, RoundtripSpirvIdentity) {
    const char *source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> a: Buffer;
        @group(0) @binding(1) var<storage, read> b: Buffer;
        @group(0) @binding(2) var<storage, read_write> result: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var x: f32 = a.data[id.x];
            var y: f32 = b.data[id.x];
            if (x > y) {
                result.data[id.x] = x * 2.0 + y;
            } else {
                result.data[id.x] = y * 2.0 - x;
            }
        }
    )";

    auto first_compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(first_compile.success) << "First compile failed: " << first_compile.error;

    auto raised = wgsl_test::RaiseSpirvToWgsl(first_compile.spirv);
    ASSERT_TRUE(raised.success) << "Raise failed: " << raised.error;

    auto second_compile = wgsl_test::CompileWgsl(raised.wgsl.c_str());
    ASSERT_TRUE(second_compile.success) << "Second compile failed: " << second_compile.error
                                        << "\nRaised WGSL:\n"
                                        << raised.wgsl;

    std::vector<float> a_data = {5.0f, 2.0f, 8.0f, 1.0f, 3.0f, 7.0f, 4.0f, 6.0f};
    std::vector<float> b_data = {3.0f, 7.0f, 4.0f, 9.0f, 6.0f, 2.0f, 8.0f, 5.0f};

    auto a1 = ctx_->createStorageBuffer(a_data);
    auto b1 = ctx_->createStorageBuffer(b_data);
    auto out1 = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline1 = ctx_->createPipeline(first_compile.spirv);
    ctx_->dispatch(pipeline1, {{0, &a1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto a2 = ctx_->createStorageBuffer(a_data);
    auto b2 = ctx_->createStorageBuffer(b_data);
    auto out2 = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline2 = ctx_->createPipeline(second_compile.spirv);
    ctx_->dispatch(pipeline2, {{0, &a2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto result1 = out1.download<float>(a_data.size());
    auto result2 = out2.download<float>(a_data.size());

    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(result1[i], result2[i])
            << "Mismatch at index " << i
            << "\nDirect SPIR-V: " << result1[i]
            << "\nRoundtrip SPIR-V: " << result2[i];
    }
}

TEST_F(VulkanComputeTest, RoundtripWithLoop) {
    const char *source = R"(
        struct Buffer {
            data: array<f32>,
        };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                sum = sum + input.data[i] * f32(i + 1u);
            }
            output.data[id.x] = sum;
        }
    )";

    auto first_compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(first_compile.success) << "First compile failed: " << first_compile.error;

    auto raised = wgsl_test::RaiseSpirvToWgsl(first_compile.spirv);
    ASSERT_TRUE(raised.success) << "Raise failed: " << raised.error;

    auto second_compile = wgsl_test::CompileWgsl(raised.wgsl.c_str());
    ASSERT_TRUE(second_compile.success) << "Second compile failed: " << second_compile.error
                                        << "\nRaised WGSL:\n"
                                        << raised.wgsl;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto in1 = ctx_->createStorageBuffer(input_data);
    auto out1 = ctx_->createStorageBuffer(sizeof(float));
    auto pipeline1 = ctx_->createPipeline(first_compile.spirv);
    ctx_->dispatch(pipeline1, {{0, &in1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &out1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto in2 = ctx_->createStorageBuffer(input_data);
    auto out2 = ctx_->createStorageBuffer(sizeof(float));
    auto pipeline2 = ctx_->createPipeline(second_compile.spirv);
    ctx_->dispatch(pipeline2, {{0, &in2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &out2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto result1 = out1.download<float>(1);
    auto result2 = out2.download<float>(1);

    EXPECT_FLOAT_EQ(result1[0], result2[0])
        << "Direct SPIR-V: " << result1[0]
        << "\nRoundtrip SPIR-V: " << result2[0];
}

TEST_F(VulkanComputeTest, TrigSinCos) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> sin_out: Buffer;
        @group(0) @binding(2) var<storage, read_write> cos_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            sin_out.data[id.x] = sin(input.data[id.x]);
            cos_out.data[id.x] = cos(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    const float PI = 3.14159265359f;
    std::vector<float> input_data = {0.0f, PI / 6.0f, PI / 4.0f, PI / 3.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f, 2.0f * PI};
    auto input = ctx_->createStorageBuffer(input_data);
    auto sin_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto cos_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &sin_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &cos_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto sin_data = sin_out.download<float>(input_data.size());
    auto cos_data = cos_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(sin_data[i], std::sin(input_data[i]), 1e-5f) << "sin mismatch at " << i;
        EXPECT_NEAR(cos_data[i], std::cos(input_data[i]), 1e-5f) << "cos mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, TrigTan) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = tan(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    const float PI = 3.14159265359f;
    std::vector<float> input_data = {0.0f, PI / 6.0f, PI / 4.0f, PI / 3.0f, -PI / 4.0f, -PI / 6.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(output_data[i], std::tan(input_data[i]), 1e-5f) << "tan mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, TrigInverse) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> asin_out: Buffer;
        @group(0) @binding(2) var<storage, read_write> acos_out: Buffer;
        @group(0) @binding(3) var<storage, read_write> atan_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            asin_out.data[id.x] = asin(input.data[id.x]);
            acos_out.data[id.x] = acos(input.data[id.x]);
            atan_out.data[id.x] = atan(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto asin_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto acos_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto atan_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &asin_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &acos_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, &atan_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto asin_data = asin_out.download<float>(input_data.size());
    auto acos_data = acos_out.download<float>(input_data.size());
    auto atan_data = atan_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(asin_data[i], std::asin(input_data[i]), 1e-3f) << "asin mismatch at " << i;
        EXPECT_NEAR(acos_data[i], std::acos(input_data[i]), 1e-3f) << "acos mismatch at " << i;
        EXPECT_NEAR(atan_data[i], std::atan(input_data[i]), 1e-5f) << "atan mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, TrigAtan2) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> y_in: Buffer;
        @group(0) @binding(1) var<storage, read> x_in: Buffer;
        @group(0) @binding(2) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = atan2(y_in.data[id.x], x_in.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> y_data = {1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 1.0f};
    std::vector<float> x_data = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.0f};
    auto y_in = ctx_->createStorageBuffer(y_data);
    auto x_in = ctx_->createStorageBuffer(x_data);
    auto output = ctx_->createStorageBuffer(y_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &y_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &x_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(y_data.size()));

    auto output_data = output.download<float>(y_data.size());
    for (size_t i = 0; i < y_data.size(); i++) {
        EXPECT_NEAR(output_data[i], std::atan2(y_data[i], x_data[i]), 1e-5f) << "atan2 mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, TrigHyperbolic) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> sinh_out: Buffer;
        @group(0) @binding(2) var<storage, read_write> cosh_out: Buffer;
        @group(0) @binding(3) var<storage, read_write> tanh_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            sinh_out.data[id.x] = sinh(input.data[id.x]);
            cosh_out.data[id.x] = cosh(input.data[id.x]);
            tanh_out.data[id.x] = tanh(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto sinh_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto cosh_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto tanh_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &sinh_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &cosh_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, &tanh_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto sinh_data = sinh_out.download<float>(input_data.size());
    auto cosh_data = cosh_out.download<float>(input_data.size());
    auto tanh_data = tanh_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(sinh_data[i], std::sinh(input_data[i]), 1e-5f) << "sinh mismatch at " << i;
        EXPECT_NEAR(cosh_data[i], std::cosh(input_data[i]), 1e-5f) << "cosh mismatch at " << i;
        EXPECT_NEAR(tanh_data[i], std::tanh(input_data[i]), 1e-5f) << "tanh mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, ExpLogPow) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> exp_out: Buffer;
        @group(0) @binding(2) var<storage, read_write> log_out: Buffer;
        @group(0) @binding(3) var<storage, read_write> exp2_out: Buffer;
        @group(0) @binding(4) var<storage, read_write> log2_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            exp_out.data[id.x] = exp(input.data[id.x]);
            log_out.data[id.x] = log(input.data[id.x]);
            exp2_out.data[id.x] = exp2(input.data[id.x]);
            log2_out.data[id.x] = log2(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto exp_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto log_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto exp2_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto log2_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &exp_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &log_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, &exp2_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {4, &log2_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto exp_data = exp_out.download<float>(input_data.size());
    auto log_data = log_out.download<float>(input_data.size());
    auto exp2_data = exp2_out.download<float>(input_data.size());
    auto log2_data = log2_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        float exp_tol = std::abs(std::exp(input_data[i])) * 1e-5f;
        float exp2_tol = std::abs(std::exp2(input_data[i])) * 1e-5f;
        EXPECT_NEAR(exp_data[i], std::exp(input_data[i]), std::max(exp_tol, 1e-4f)) << "exp mismatch at " << i;
        EXPECT_NEAR(log_data[i], std::log(input_data[i]), 1e-5f) << "log mismatch at " << i;
        EXPECT_NEAR(exp2_data[i], std::exp2(input_data[i]), std::max(exp2_tol, 1e-4f)) << "exp2 mismatch at " << i;
        EXPECT_NEAR(log2_data[i], std::log2(input_data[i]), 1e-5f) << "log2 mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, SqrtInverseSqrt) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> sqrt_out: Buffer;
        @group(0) @binding(2) var<storage, read_write> inverseSqrt_out: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            sqrt_out.data[id.x] = sqrt(input.data[id.x]);
            inverseSqrt_out.data[id.x] = inverseSqrt(input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 0.25f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto sqrt_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto inverseSqrt_out = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &sqrt_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &inverseSqrt_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto sqrt_data = sqrt_out.download<float>(input_data.size());
    auto inverseSqrt_data = inverseSqrt_out.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(sqrt_data[i], std::sqrt(input_data[i]), 1e-5f) << "sqrt mismatch at " << i;
        EXPECT_NEAR(inverseSqrt_data[i], 1.0f / std::sqrt(input_data[i]), 1e-5f) << "inverseSqrt mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, PowFunction) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> base_in: Buffer;
        @group(0) @binding(1) var<storage, read> exp_in: Buffer;
        @group(0) @binding(2) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = pow(base_in.data[id.x], exp_in.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> base_data = {2.0f, 3.0f, 4.0f, 2.0f, 10.0f};
    std::vector<float> exp_data = {3.0f, 2.0f, 0.5f, 10.0f, 2.0f};
    auto base_in = ctx_->createStorageBuffer(base_data);
    auto exp_in = ctx_->createStorageBuffer(exp_data);
    auto output = ctx_->createStorageBuffer(base_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &base_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &exp_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(base_data.size()));

    auto output_data = output.download<float>(base_data.size());
    for (size_t i = 0; i < base_data.size(); i++) {
        EXPECT_NEAR(output_data[i], std::pow(base_data[i], exp_data[i]), 1e-3f) << "pow mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, CompileTimeUnrolledDotProduct) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> a: Buffer;
        @group(0) @binding(1) var<storage, read> b: Buffer;
        @group(0) @binding(2) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let base: u32 = id.x * 4u;
            var sum: f32 = 0.0;
            sum = sum + a.data[base + 0u] * b.data[base + 0u];
            sum = sum + a.data[base + 1u] * b.data[base + 1u];
            sum = sum + a.data[base + 2u] * b.data[base + 2u];
            sum = sum + a.data[base + 3u] * b.data[base + 3u];
            output.data[id.x] = sum;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b_data = {2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto output = ctx_->createStorageBuffer(2 * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 2);

    auto output_data = output.download<float>(2);
    float expected0 = 1.0f * 2.0f + 2.0f * 3.0f + 3.0f * 4.0f + 4.0f * 5.0f;
    float expected1 = 5.0f * 1.0f + 6.0f * 1.0f + 7.0f * 1.0f + 8.0f * 1.0f;
    EXPECT_NEAR(output_data[0], expected0, 1e-5f);
    EXPECT_NEAR(output_data[1], expected1, 1e-5f);
}

TEST_F(VulkanComputeTest, CompileTimeUnrolledPolynomial) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> x_in: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let x: f32 = x_in.data[id.x];
            let c0: f32 = 1.0;
            let c1: f32 = 2.0;
            let c2: f32 = 3.0;
            let c3: f32 = 4.0;
            output.data[id.x] = c0 + c1*x + c2*x*x + c3*x*x*x;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> x_data = {0.0f, 1.0f, 2.0f, -1.0f, 0.5f};
    auto x_in = ctx_->createStorageBuffer(x_data);
    auto output = ctx_->createStorageBuffer(x_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &x_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(x_data.size()));

    auto output_data = output.download<float>(x_data.size());
    for (size_t i = 0; i < x_data.size(); i++) {
        float x = x_data[i];
        float expected = 1.0f + 2.0f * x + 3.0f * x * x + 4.0f * x * x * x;
        EXPECT_NEAR(output_data[i], expected, 1e-5f) << "polynomial mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, TrigIdentitySinCosSquared) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let angle: f32 = input.data[id.x];
            let s: f32 = sin(angle);
            let c: f32 = cos(angle);
            output.data[id.x] = s*s + c*c;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    const float PI = 3.14159265359f;
    std::vector<float> input_data = {0.0f, PI / 4.0f, PI / 2.0f, PI, 1.5f, 2.7f, -0.5f, 3.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(output_data[i], 1.0f, 1e-5f) << "sin^2 + cos^2 should be 1 at " << i;
    }
}

TEST_F(VulkanComputeTest, SmoothstepFunction) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = smoothstep(0.0, 1.0, input.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {-0.5f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto smoothstep_cpu = [](float edge0, float edge1, float x) {
        float t = std::max(0.0f, std::min((x - edge0) / (edge1 - edge0), 1.0f));
        return t * t * (3.0f - 2.0f * t);
    };

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(output_data[i], smoothstep_cpu(0.0f, 1.0f, input_data[i]), 1e-5f) << "smoothstep mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, MixLerp) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> a_in: Buffer;
        @group(0) @binding(1) var<storage, read> b_in: Buffer;
        @group(0) @binding(2) var<storage, read> t_in: Buffer;
        @group(0) @binding(3) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = mix(a_in.data[id.x], b_in.data[id.x], t_in.data[id.x]);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {0.0f, 10.0f, -5.0f, 100.0f};
    std::vector<float> b_data = {10.0f, 20.0f, 5.0f, 200.0f};
    std::vector<float> t_data = {0.0f, 0.5f, 0.5f, 0.25f};
    auto a_in = ctx_->createStorageBuffer(a_data);
    auto b_in = ctx_->createStorageBuffer(b_data);
    auto t_in = ctx_->createStorageBuffer(t_data);
    auto output = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &a_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &t_in, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = output.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        float expected = a_data[i] * (1.0f - t_data[i]) + b_data[i] * t_data[i];
        EXPECT_NEAR(output_data[i], expected, 1e-5f) << "mix mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, ComplexTrigExpression) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let x: f32 = input.data[id.x];
            let sinx: f32 = sin(x);
            let cosx: f32 = cos(x);
            let tanx: f32 = tan(x);
            output.data[id.x] = sinx * cosx + tanx * exp(-x * x);
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {0.0f, 0.5f, 1.0f, -0.5f, 0.25f, 0.75f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        float x = input_data[i];
        float expected = std::sin(x) * std::cos(x) + std::tan(x) * std::exp(-x * x);
        EXPECT_NEAR(output_data[i], expected, 1e-4f) << "complex trig mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, CompileTimeUnrolledMatrixMultiply2x2) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> mat_a: Buffer;
        @group(0) @binding(1) var<storage, read> mat_b: Buffer;
        @group(0) @binding(2) var<storage, read_write> mat_c: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let a00: f32 = mat_a.data[0]; let a01: f32 = mat_a.data[1];
            let a10: f32 = mat_a.data[2]; let a11: f32 = mat_a.data[3];
            let b00: f32 = mat_b.data[0]; let b01: f32 = mat_b.data[1];
            let b10: f32 = mat_b.data[2]; let b11: f32 = mat_b.data[3];
            mat_c.data[0] = a00*b00 + a01*b10;
            mat_c.data[1] = a00*b01 + a01*b11;
            mat_c.data[2] = a10*b00 + a11*b10;
            mat_c.data[3] = a10*b01 + a11*b11;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
    auto mat_a = ctx_->createStorageBuffer(a_data);
    auto mat_b = ctx_->createStorageBuffer(b_data);
    auto mat_c = ctx_->createStorageBuffer(4 * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &mat_a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &mat_b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &mat_c, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto output_data = mat_c.download<float>(4);
    EXPECT_NEAR(output_data[0], 1.0f * 5.0f + 2.0f * 7.0f, 1e-5f);
    EXPECT_NEAR(output_data[1], 1.0f * 6.0f + 2.0f * 8.0f, 1e-5f);
    EXPECT_NEAR(output_data[2], 3.0f * 5.0f + 4.0f * 7.0f, 1e-5f);
    EXPECT_NEAR(output_data[3], 3.0f * 6.0f + 4.0f * 8.0f, 1e-5f);
}

TEST_F(VulkanComputeTest, RoundtripTrigFunctions) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };

        @group(0) @binding(0) var<storage, read> input: Buffer;
        @group(0) @binding(1) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let x: f32 = input.data[id.x];
            output.data[id.x] = sin(x) * cos(x) + tan(x * 0.5);
        }
    )";

    auto first_compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(first_compile.success) << "First compile failed: " << first_compile.error;

    auto raised = wgsl_test::RaiseSpirvToWgsl(first_compile.spirv);
    ASSERT_TRUE(raised.success) << "Raise failed: " << raised.error;

    auto second_compile = wgsl_test::CompileWgsl(raised.wgsl.c_str());
    ASSERT_TRUE(second_compile.success) << "Second compile failed: " << second_compile.error
                                        << "\nRaised WGSL:\n"
                                        << raised.wgsl;

    std::vector<float> input_data = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f};

    auto in1 = ctx_->createStorageBuffer(input_data);
    auto out1 = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto pipeline1 = ctx_->createPipeline(first_compile.spirv);
    ctx_->dispatch(pipeline1, {{0, &in1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &out1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto in2 = ctx_->createStorageBuffer(input_data);
    auto out2 = ctx_->createStorageBuffer(input_data.size() * sizeof(float));
    auto pipeline2 = ctx_->createPipeline(second_compile.spirv);
    ctx_->dispatch(pipeline2, {{0, &in2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &out2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto result1 = out1.download<float>(input_data.size());
    auto result2 = out2.download<float>(input_data.size());

    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_NEAR(result1[i], result2[i], 1e-5f)
            << "Mismatch at index " << i
            << "\nDirect SPIR-V: " << result1[i]
            << "\nRoundtrip SPIR-V: " << result2[i];
    }
}

// ============================================================================
// GLSL Compute Tests
// ============================================================================

TEST_F(VulkanComputeTest, GlslBufferCopy) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer InputBuf {
            float data[];
        } input_buf;

        layout(std430, set = 0, binding = 1) buffer OutputBuf {
            float data[];
        } output_buf;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            output_buf.data[id] = input_buf.data[id];
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], input_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, GlslScalarAdd) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer ABuf {
            float data[];
        } a;

        layout(std430, set = 0, binding = 1) buffer BBuf {
            float data[];
        } b;

        layout(std430, set = 0, binding = 2) buffer ResultBuf {
            float data[];
        } result;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            result.data[id] = a.data[id] + b.data[id];
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] + b_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, GlslScalarMultiply) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer ABuf {
            float data[];
        } a;

        layout(std430, set = 0, binding = 1) buffer BBuf {
            float data[];
        } b;

        layout(std430, set = 0, binding = 2) buffer ResultBuf {
            float data[];
        } result;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            result.data[id] = a.data[id] * b.data[id];
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b_data = {10.0f, 10.0f, 10.0f, 10.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], a_data[i] * b_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, GlslConditionalSelect) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer ABuf {
            float data[];
        } a;

        layout(std430, set = 0, binding = 1) buffer BBuf {
            float data[];
        } b;

        layout(std430, set = 0, binding = 2) buffer ResultBuf {
            float data[];
        } result;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            if (a.data[id] > b.data[id]) {
                result.data[id] = a.data[id];
            } else {
                result.data[id] = b.data[id];
            }
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {5.0f, 2.0f, 8.0f, 1.0f};
    std::vector<float> b_data = {3.0f, 7.0f, 4.0f, 9.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto output_data = out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        float expected = a_data[i] > b_data[i] ? a_data[i] : b_data[i];
        EXPECT_FLOAT_EQ(output_data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, GlslLoopSum) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer InputBuf {
            float data[];
        } input_buf;

        layout(std430, set = 0, binding = 1) buffer OutputBuf {
            float data[];
        } output_buf;

        void main() {
            float sum = 0.0;
            for (uint i = 0u; i < 4u; i = i + 1u) {
                sum = sum + input_buf.data[i];
            }
            output_buf.data[gl_GlobalInvocationID.x] = sum;
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto output_data = output.download<float>(1);
    float expected = 1.0f + 2.0f + 3.0f + 4.0f;
    EXPECT_FLOAT_EQ(output_data[0], expected);
}

TEST_F(VulkanComputeTest, GlslMathAbs) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer InputBuf {
            float data[];
        } input_buf;

        layout(std430, set = 0, binding = 1) buffer OutputBuf {
            float data[];
        } output_buf;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            output_buf.data[id] = abs(input_buf.data[id]);
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(input_data.size()));

    auto output_data = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(output_data[i], std::abs(input_data[i])) << "Mismatch at index " << i;
    }
}

TEST_F(VulkanComputeTest, GlslMathMinMax) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer ABuf {
            float data[];
        } a;

        layout(std430, set = 0, binding = 1) buffer BBuf {
            float data[];
        } b;

        layout(std430, set = 0, binding = 2) buffer MinBuf {
            float data[];
        } min_out;

        layout(std430, set = 0, binding = 3) buffer MaxBuf {
            float data[];
        } max_out;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            min_out.data[id] = min(a.data[id], b.data[id]);
            max_out.data[id] = max(a.data[id], b.data[id]);
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> a_data = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> b_data = {4.0f, 2.0f, 6.0f, 7.0f};
    auto a = ctx_->createStorageBuffer(a_data);
    auto b = ctx_->createStorageBuffer(b_data);
    auto min_out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));
    auto max_out = ctx_->createStorageBuffer(a_data.size() * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, &min_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, &max_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, static_cast<uint32_t>(a_data.size()));

    auto min_data = min_out.download<float>(a_data.size());
    auto max_data = max_out.download<float>(a_data.size());
    for (size_t i = 0; i < a_data.size(); i++) {
        EXPECT_FLOAT_EQ(min_data[i], std::min(a_data[i], b_data[i])) << "min mismatch at " << i;
        EXPECT_FLOAT_EQ(max_data[i], std::max(a_data[i], b_data[i])) << "max mismatch at " << i;
    }
}

// ============================================================================
// Mandelbrot Image Tests
// ============================================================================

static void mandelbrot_colorize(const std::vector<float> &data, uint32_t width, uint32_t height,
    std::vector<uint8_t> &image) {
    image.resize(width * height * 4);
    for (uint32_t i = 0; i < width * height; i++) {
        float t = data[i];
        uint8_t r, g, b;
        if (t >= 1.0f) {
            r = g = b = 0;
        } else {
            r = (uint8_t)(9.0f * (1.0f - t) * t * t * t * 255.0f);
            g = (uint8_t)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
            b = (uint8_t)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);
        }
        image[i * 4 + 0] = r;
        image[i * 4 + 1] = g;
        image[i * 4 + 2] = b;
        image[i * 4 + 3] = 255;
    }
}

TEST_F(VulkanComputeTest, WgslMandelbrotImage) {
    const char *source = R"(
        struct Buffer { data: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Buffer;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var px: u32 = id.x;
            var py: u32 = id.y;
            var scale: f32 = 3.5 / 256.0;
            var cx: f32 = f32(px) * scale - 2.5;
            var cy: f32 = f32(py) * scale - 1.75;
            var zr: f32 = cx;
            var zi: f32 = cy;
            var zr2: f32 = 0.0;
            var zi2: f32 = 0.0;
            var iters: i32 = 0;
            var mag: f32 = 0.0;
            var i: i32 = 0;

            for (i = 0; i < 150; i = i + 1) {
                zr2 = zr * zr - zi * zi + cx;
                zi2 = 2.0 * zr * zi + cy;
                zr = zr2;
                zi = zi2;
                mag = zr * zr + zi * zi;
                if (mag > 4.0) {
                    i = 150;
                }
                if (mag <= 4.0) {
                    iters = iters + 1;
                }
            }
            output.data[py * 256u + px] = f32(iters) / 150.0;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    const uint32_t W = 256, H = 256;
    auto output = ctx_->createStorageBuffer(W * H * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, W, H);

    auto data = output.download<float>(W * H);

    std::vector<uint8_t> image;
    mandelbrot_colorize(data, W, H, image);
    stbi_write_png("mandelbrot_wgsl.png", W, H, 4, image.data(), W * 4);

    int black = 0, nonblack = 0;
    for (uint32_t i = 0; i < W * H; i++) {
        if (data[i] >= 1.0f) black++;
        else nonblack++;
    }
    EXPECT_GT(black, 1000) << "Should have substantial set interior";
    EXPECT_GT(nonblack, 1000) << "Should have substantial set exterior";
}

TEST_F(VulkanComputeTest, GlslMandelbrotImage) {
    const char *source = R"(
        #version 450
        layout(local_size_x = 1) in;

        layout(std430, set = 0, binding = 0) buffer OutputBuf {
            float data[];
        } output_buf;

        void main() {
            uint px = gl_GlobalInvocationID.x;
            uint py = gl_GlobalInvocationID.y;
            float scale = 3.5 / 256.0;
            float cx = float(px) * scale - 2.5;
            float cy = float(py) * scale - 1.75;
            float zr = cx;
            float zi = cy;
            float zr2 = 0.0;
            float zi2 = 0.0;
            int iters = 0;
            float mag = 0.0;
            int i = 0;

            for (i = 0; i < 150; i = i + 1) {
                zr2 = zr * zr - zi * zi + cx;
                zi2 = 2.0 * zr * zi + cy;
                zr = zr2;
                zi = zi2;
                mag = zr * zr + zi * zi;
                if (mag > 4.0) {
                    i = 150;
                }
                if (mag <= 4.0) {
                    iters = iters + 1;
                }
            }
            output_buf.data[py * 256u + px] = float(iters) / 150.0;
        }
    )";

    auto result = wgsl_test::CompileGlsl(source, WGSL_STAGE_COMPUTE);
    ASSERT_TRUE(result.success) << result.error;

    const uint32_t W = 256, H = 256;
    auto output = ctx_->createStorageBuffer(W * H * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, W, H);

    auto data = output.download<float>(W * H);

    std::vector<uint8_t> image;
    mandelbrot_colorize(data, W, H, image);
    stbi_write_png("mandelbrot_glsl.png", W, H, 4, image.data(), W * 4);

    int black = 0, nonblack = 0;
    for (uint32_t i = 0; i < W * H; i++) {
        if (data[i] >= 1.0f) black++;
        else nonblack++;
    }
    EXPECT_GT(black, 1000) << "Should have substantial set interior";
    EXPECT_GT(nonblack, 1000) << "Should have substantial set exterior";
}

// ============================================================================
// Matrix operation GPU verification tests
// ============================================================================

TEST_F(VulkanComputeTest, MatrixTimesVector_Identity3x3) {
    const char *source = R"(
        struct In  { v: vec3<f32>, };
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read> input: In;
        @group(0) @binding(1) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let r = m * input.v;
            output.v[0] = r.x;
            output.v[1] = r.y;
            output.v[2] = r.z;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    // Identity * (1,2,3) = (1,2,3)
    std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 0.0f}; // padded to 16 bytes
    auto input = ctx_->createStorageBuffer(in_data);
    auto output = ctx_->createStorageBuffer(3 * sizeof(float));

    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(3);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST_F(VulkanComputeTest, MatrixTimesVector_Scale3x3) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            let m = mat3x3<f32>(
                vec3<f32>(2.,0.,0.),
                vec3<f32>(0.,3.,0.),
                vec3<f32>(0.,0.,4.));
            let v = vec3<f32>(1., 1., 1.);
            let r = m * v;
            output.v[0] = r.x;
            output.v[1] = r.y;
            output.v[2] = r.z;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(3 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(3);
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    EXPECT_FLOAT_EQ(out[2], 4.0f);
}

TEST_F(VulkanComputeTest, MatrixTimesVector_4x4_Translation) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            // Translation matrix: translate by (10, 20, 30)
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(10.,20.,30.,1.));
            let v = vec4<f32>(1., 2., 3., 1.);
            let r = m * v;
            output.v[0] = r.x;
            output.v[1] = r.y;
            output.v[2] = r.z;
            output.v[3] = r.w;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(4 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(4);
    EXPECT_FLOAT_EQ(out[0], 11.0f); // 1 + 10*1
    EXPECT_FLOAT_EQ(out[1], 22.0f); // 2 + 20*1
    EXPECT_FLOAT_EQ(out[2], 33.0f); // 3 + 30*1
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_F(VulkanComputeTest, VectorTimesMatrix_3x3) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            let v = vec3<f32>(1., 2., 3.);
            let m = mat3x3<f32>(
                vec3<f32>(2.,0.,0.),
                vec3<f32>(0.,3.,0.),
                vec3<f32>(0.,0.,4.));
            let r = v * m;
            output.v[0] = r.x;
            output.v[1] = r.y;
            output.v[2] = r.z;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(3 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(3);
    EXPECT_FLOAT_EQ(out[0], 2.0f);  // (1,2,3) dot col0=(2,0,0)
    EXPECT_FLOAT_EQ(out[1], 6.0f);  // (1,2,3) dot col1=(0,3,0)
    EXPECT_FLOAT_EQ(out[2], 12.0f); // (1,2,3) dot col2=(0,0,4)
}

TEST_F(VulkanComputeTest, MatrixTimesMatrix_3x3_Identity) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            let a = mat3x3<f32>(
                vec3<f32>(1.,2.,3.),
                vec3<f32>(4.,5.,6.),
                vec3<f32>(7.,8.,9.));
            let id = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let r = a * id;
            // Extract first column of result
            let c0 = r * vec3<f32>(1.,0.,0.);
            output.v[0] = c0.x;
            output.v[1] = c0.y;
            output.v[2] = c0.z;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(3 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(3);
    // a * identity = a, so first column should be (1,2,3)
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST_F(VulkanComputeTest, MatrixTimesScalar_3x3) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,2.,3.),
                vec3<f32>(4.,5.,6.),
                vec3<f32>(7.,8.,9.));
            let scaled = m * 2.0;
            // Extract first column
            let c0 = scaled * vec3<f32>(1.,0.,0.);
            output.v[0] = c0.x;
            output.v[1] = c0.y;
            output.v[2] = c0.z;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(3 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(3);
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 6.0f);
}

TEST_F(VulkanComputeTest, MatVecChain_ModelViewProjection) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            // Scale by 2
            let model = mat4x4<f32>(
                vec4<f32>(2.,0.,0.,0.),
                vec4<f32>(0.,2.,0.,0.),
                vec4<f32>(0.,0.,2.,0.),
                vec4<f32>(0.,0.,0.,1.));
            // Translate by (1,0,0)
            let view = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(1.,0.,0.,1.));
            let pos = vec4<f32>(1., 0., 0., 1.);
            let mvp = view * model;
            let r = mvp * pos;
            output.v[0] = r.x;
            output.v[1] = r.y;
            output.v[2] = r.z;
            output.v[3] = r.w;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(4 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(4);
    // model * (1,0,0,1) = (2,0,0,1)
    // view * (2,0,0,1) = (2+1, 0, 0, 1) = (3,0,0,1)
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_F(VulkanComputeTest, MatrixTimesVector_GeneralMultiply) {
    const char *source = R"(
        struct Out { v: array<f32>, };
        @group(0) @binding(0) var<storage, read_write> output: Out;

        @compute @workgroup_size(1)
        fn main() {
            // Non-trivial matrix
            let m = mat3x3<f32>(
                vec3<f32>(1., 4., 7.),
                vec3<f32>(2., 5., 8.),
                vec3<f32>(3., 6., 9.));
            let v = vec3<f32>(1., 1., 1.);
            let r = m * v;
            output.v[0] = r.x;
            output.v[1] = r.y;
            output.v[2] = r.z;
        }
    )";

    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(3 * sizeof(float));
    auto pipeline = ctx_->createPipeline(result.spirv);
    ctx_->dispatch(pipeline, {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, 1);

    auto out = output.download<float>(3);
    // m * (1,1,1) = col0 + col1 + col2 = (1+2+3, 4+5+6, 7+8+9) = (6, 15, 24)
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 15.0f);
    EXPECT_FLOAT_EQ(out[2], 24.0f);
}

// ============================================================================
// Push Constants via var<immediate>
// ============================================================================

using vk_compute::VulkanError;

// Helper: compile WGSL with per-entry-point immediate lowering
static wgsl_test::CompileResult CompileImmediate(const char *source,
                                                  const char *entry_point,
                                                  SsirLayoutRule layout = SSIR_LAYOUT_STD430) {
    wgsl_test::CompileResult result;
    result.success = false;

    WgslAstNode *ast = wgsl_parse(source);
    if (!ast) { result.error = "Parse failed"; return result; }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) { wgsl_free_ast(ast); result.error = "Resolve failed"; return result; }

    uint32_t *spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    opts.entry_point = entry_point;
    opts.immediate_layout = layout;

    WgslLowerResult lower_result =
        wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &spirv_size);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lower_result != WGSL_LOWER_OK) { result.error = "Lower failed"; return result; }

    result.spirv.assign(spirv, spirv + spirv_size);
    wgsl_lower_free(spirv);

    if (!wgsl_test::ValidateSpirv(result.spirv.data(), result.spirv.size(), &result.error))
        return result;

    result.success = true;
    return result;
}

// Helper: create a compute pipeline with push constant range
struct PushConstantPipeline {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    ~PushConstantPipeline() {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
    }
};

static PushConstantPipeline createComputePipelineWithPush(
    vk_compute::VulkanContext &ctx,
    const std::vector<uint32_t> &spirv,
    uint32_t push_size,
    const char *entry = "main")
{
    PushConstantPipeline p;
    p.device = ctx.device();

    VkShaderModuleCreateInfo shader_info = {};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size() * sizeof(uint32_t);
    shader_info.pCode = spirv.data();
    VK_CHECK(vkCreateShaderModule(ctx.device(), &shader_info, nullptr, &p.shader));

    // Descriptor set layout: up to 16 storage buffers
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < 16; i++) {
        VkDescriptorSetLayoutBinding b = {};
        b.binding = i;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }
    VkDescriptorSetLayoutCreateInfo dl_info = {};
    dl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl_info.bindingCount = static_cast<uint32_t>(bindings.size());
    dl_info.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device(), &dl_info, nullptr, &p.desc_layout));

    // Push constant range
    VkPushConstantRange push_range = {};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_size;

    VkPipelineLayoutCreateInfo pl_info = {};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &p.desc_layout;
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &push_range;
    VK_CHECK(vkCreatePipelineLayout(ctx.device(), &pl_info, nullptr, &p.layout));

    VkComputePipelineCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = p.shader;
    ci.stage.pName = entry;
    ci.layout = p.layout;
    VK_CHECK(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline));

    return p;
}

// Helper: dispatch with push constants, descriptors, command recording
static void dispatchWithPush(
    vk_compute::VulkanContext &ctx,
    PushConstantPipeline &pipeline,
    const std::vector<vk_compute::DescriptorBinding> &bindings,
    const void *push_data, uint32_t push_size,
    uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1)
{
    // Create temporary descriptor pool
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 16;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool desc_pool;
    VK_CHECK(vkCreateDescriptorPool(ctx.device(), &pool_info, nullptr, &desc_pool));

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = desc_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline.desc_layout;

    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device(), &alloc_info, &desc_set));

    // Update descriptor set
    std::vector<VkDescriptorBufferInfo> buf_infos(bindings.size());
    std::vector<VkWriteDescriptorSet> writes(bindings.size());
    for (size_t i = 0; i < bindings.size(); i++) {
        buf_infos[i].buffer = bindings[i].buffer->handle();
        buf_infos[i].offset = 0;
        buf_infos[i].range = bindings[i].buffer->size();
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = bindings[i].binding;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].type;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx.device(), static_cast<uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    ctx.executeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, push_size, push_data);
        vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    });

    vkDestroyDescriptorPool(ctx.device(), desc_pool, nullptr);
}

// --- Compute push constant tests ---

TEST_F(VulkanComputeTest, Immediate_ScaleBuffer) {
    // Push a scalar scale factor, multiply each buffer element
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = input.data[id.x] * scale;
        }
    )";

    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    float scale = 2.5f;
    auto pipeline = createComputePipelineWithPush(*ctx_, result.spirv, sizeof(float));
    dispatchWithPush(*ctx_, pipeline,
        {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        &scale, sizeof(float),
        static_cast<uint32_t>(input_data.size()));

    auto out = output.download<float>(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        EXPECT_FLOAT_EQ(out[i], input_data[i] * 2.5f) << "Mismatch at " << i;
    }
}

TEST_F(VulkanComputeTest, Immediate_MultipleFields) {
    // Push multiple values: scale (f32) + offset (u32)
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        var<immediate> offset: u32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = input.data[id.x + offset] * scale;
        }
    )";

    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(4 * sizeof(float));

    struct { float scale; uint32_t offset; } pc = {0.5f, 2};
    auto pipeline = createComputePipelineWithPush(*ctx_, result.spirv, sizeof(pc));
    dispatchWithPush(*ctx_, pipeline,
        {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        &pc, sizeof(pc),
        4);

    auto out = output.download<float>(4);
    // output[i] = input[i+2] * 0.5
    EXPECT_FLOAT_EQ(out[0], 15.0f);  // 30 * 0.5
    EXPECT_FLOAT_EQ(out[1], 20.0f);  // 40 * 0.5
    EXPECT_FLOAT_EQ(out[2], 25.0f);  // 50 * 0.5
    EXPECT_FLOAT_EQ(out[3], 30.0f);  // 60 * 0.5
}

TEST_F(VulkanComputeTest, Immediate_Vec4Push) {
    // Push a vec4f color, write it into the output buffer
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> color: vec4f;

        struct Buf { data: array<vec4f> };
        @group(0) @binding(0) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = color;
        }
    )";

    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(4 * 4 * sizeof(float)); // 4 vec4s

    float color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    auto pipeline = createComputePipelineWithPush(*ctx_, result.spirv, sizeof(color));
    dispatchWithPush(*ctx_, pipeline,
        {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        color, sizeof(color),
        4);

    auto out = output.download<float>(16);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(out[i * 4 + 0], 0.25f) << "element " << i << " .x";
        EXPECT_FLOAT_EQ(out[i * 4 + 1], 0.5f) << "element " << i << " .y";
        EXPECT_FLOAT_EQ(out[i * 4 + 2], 0.75f) << "element " << i << " .z";
        EXPECT_FLOAT_EQ(out[i * 4 + 3], 1.0f) << "element " << i << " .w";
    }
}

TEST_F(VulkanComputeTest, Immediate_ScaleAndBias) {
    // Two push constants used together: output = input * scale + bias
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        var<immediate> bias: f32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = input.data[id.x] * scale + bias;
        }
    )";

    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {2.0f, 4.0f, 6.0f, 8.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    struct { float scale; float bias; } push = {3.0f, 10.0f};
    auto pipeline = createComputePipelineWithPush(*ctx_, result.spirv, sizeof(push));
    dispatchWithPush(*ctx_, pipeline,
        {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        &push, sizeof(push),
        static_cast<uint32_t>(input_data.size()));

    auto out = output.download<float>(input_data.size());
    // 2*3+10=16, 4*3+10=22, 6*3+10=28, 8*3+10=34
    EXPECT_FLOAT_EQ(out[0], 16.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 28.0f);
    EXPECT_FLOAT_EQ(out[3], 34.0f);
}

TEST_F(VulkanComputeTest, Immediate_PerEntrypoint) {
    // Two entry points using different subsets of immediates
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: f32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn add_a(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = output.data[id.x] + a;
        }

        @compute @workgroup_size(1)
        fn mul_b(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = output.data[id.x] * b;
        }
    )";

    // Compile each entry point separately
    auto r_add = CompileImmediate(source, "add_a");
    ASSERT_TRUE(r_add.success) << "add_a: " << r_add.error;

    auto r_mul = CompileImmediate(source, "mul_b");
    ASSERT_TRUE(r_mul.success) << "mul_b: " << r_mul.error;

    // Run add_a with a=10
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto buf = ctx_->createStorageBuffer(data);
        float a = 10.0f;
        auto pipeline = createComputePipelineWithPush(*ctx_, r_add.spirv, sizeof(float), "add_a");
        dispatchWithPush(*ctx_, pipeline,
            {{0, &buf, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
            &a, sizeof(float), 4);

        auto out = buf.download<float>(4);
        EXPECT_FLOAT_EQ(out[0], 11.0f);
        EXPECT_FLOAT_EQ(out[1], 12.0f);
        EXPECT_FLOAT_EQ(out[2], 13.0f);
        EXPECT_FLOAT_EQ(out[3], 14.0f);
    }

    // Run mul_b with b=3
    {
        std::vector<float> data = {2.0f, 3.0f, 4.0f, 5.0f};
        auto buf = ctx_->createStorageBuffer(data);
        float b = 3.0f;
        auto pipeline = createComputePipelineWithPush(*ctx_, r_mul.spirv, sizeof(float), "mul_b");
        dispatchWithPush(*ctx_, pipeline,
            {{0, &buf, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
            &b, sizeof(float), 4);

        auto out = buf.download<float>(4);
        EXPECT_FLOAT_EQ(out[0], 6.0f);
        EXPECT_FLOAT_EQ(out[1], 9.0f);
        EXPECT_FLOAT_EQ(out[2], 12.0f);
        EXPECT_FLOAT_EQ(out[3], 15.0f);
    }
}

TEST_F(VulkanComputeTest, Immediate_Conditional) {
    // Push constant used in conditional logic: output = (flag > 0) ? input * scale : input
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        var<immediate> flag: u32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var v = input.data[id.x];
            if (flag > 0u) {
                v = v * scale;
            }
            output.data[id.x] = v;
        }
    )";

    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << result.error;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = ctx_->createStorageBuffer(input_data);
    auto output = ctx_->createStorageBuffer(input_data.size() * sizeof(float));

    // flag=1, scale=5: should multiply
    struct { float scale; uint32_t flag; } push = {5.0f, 1u};
    auto pipeline = createComputePipelineWithPush(*ctx_, result.spirv, sizeof(push));
    dispatchWithPush(*ctx_, pipeline,
        {{0, &input, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        &push, sizeof(push),
        static_cast<uint32_t>(input_data.size()));

    auto out = output.download<float>(input_data.size());
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 10.0f);
    EXPECT_FLOAT_EQ(out[2], 15.0f);
    EXPECT_FLOAT_EQ(out[3], 20.0f);
}

TEST_F(VulkanComputeTest, Immediate_DynamicUpdate) {
    // Same pipeline, different push constant values per dispatch
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> multiplier: f32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read_write> output: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = f32(id.x) * multiplier;
        }
    )";

    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << result.error;

    auto output = ctx_->createStorageBuffer(4 * sizeof(float));
    auto pipeline = createComputePipelineWithPush(*ctx_, result.spirv, sizeof(float));

    // First dispatch: multiplier = 2
    {
        float m = 2.0f;
        dispatchWithPush(*ctx_, pipeline,
            {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
            &m, sizeof(float), 4);

        auto out = output.download<float>(4);
        EXPECT_FLOAT_EQ(out[0], 0.0f);
        EXPECT_FLOAT_EQ(out[1], 2.0f);
        EXPECT_FLOAT_EQ(out[2], 4.0f);
        EXPECT_FLOAT_EQ(out[3], 6.0f);
    }

    // Second dispatch: multiplier = 5
    {
        float m = 5.0f;
        dispatchWithPush(*ctx_, pipeline,
            {{0, &output, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
            &m, sizeof(float), 4);

        auto out = output.download<float>(4);
        EXPECT_FLOAT_EQ(out[0], 0.0f);
        EXPECT_FLOAT_EQ(out[1], 5.0f);
        EXPECT_FLOAT_EQ(out[2], 10.0f);
        EXPECT_FLOAT_EQ(out[3], 15.0f);
    }
}

#endif // WGSL_HAS_VULKAN
