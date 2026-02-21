#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef WGSL_HAS_VULKAN
#include "vulkan_graphics_harness.h"

class VulkanGraphicsTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        try {
            ctx_ = std::make_unique<vk_graphics::GraphicsContext>();
        } catch (const std::exception &e) {
            GTEST_SKIP() << "Vulkan graphics not available: " << e.what();
        }
    }

    static void TearDownTestSuite() {
        ctx_.reset();
    }

    void SetUp() override {
        if (!ctx_) {
            GTEST_SKIP() << "Vulkan graphics context not initialized";
        }
    }

    static std::unique_ptr<vk_graphics::GraphicsContext> ctx_;
};

std::unique_ptr<vk_graphics::GraphicsContext> VulkanGraphicsTest::ctx_;

// Helper to extract RGBA from packed uint32_t (assuming RGBA8_Unorm layout)
inline void unpackRGBA(uint32_t pixel, uint8_t &r, uint8_t &g, uint8_t &b, uint8_t &a) {
    r = (pixel >> 0) & 0xFF;
    g = (pixel >> 8) & 0xFF;
    b = (pixel >> 16) & 0xFF;
    a = (pixel >> 24) & 0xFF;
}

// Full-screen triangle vertices (covers entire viewport)
struct SimpleVertex {
    float x, y;
};

static const std::vector<SimpleVertex> kFullScreenTriangle = {
    {-1.0f, -1.0f},
    {3.0f, -1.0f},
    {-1.0f, 3.0f},
};

// Test: Solid color fill - fragment shader outputs constant color
TEST_F(VulkanGraphicsTest, SolidColorFill) {
    // Compile vertex and fragment shaders separately to avoid duplicate type errors
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), width * height);

    // Check center pixel is red
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 250) << "Red channel should be ~255";
    EXPECT_LE(g, 5) << "Green channel should be ~0";
    EXPECT_LE(b, 5) << "Blue channel should be ~0";
    EXPECT_GE(a, 250) << "Alpha channel should be ~255";
}

// Test: Clear color verification
TEST_F(VulkanGraphicsTest, ClearColor) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Small triangle that doesn't cover the corners
    std::vector<SimpleVertex> small_tri = {
        {0.0f, 0.0f},
        {0.1f, 0.0f},
        {0.0f, 0.1f},
    };
    auto vb = ctx_->createVertexBuffer(small_tri);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);

    // Clear to green
    vk_graphics::ClearColor green = {0.0f, 1.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3}, {}, green);

    auto pixels = target.downloadAs<uint32_t>();

    // Check corner pixel (should be clear color = green)
    uint32_t corner = pixels[0];
    uint8_t r, g, b, a;
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0 (clear color)";
    EXPECT_GE(g, 250) << "Green should be ~255 (clear color)";
    EXPECT_LE(b, 5) << "Blue should be ~0 (clear color)";
}

// Test: Vertex attribute passing with color
TEST_F(VulkanGraphicsTest, VertexAttributes) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
            @location(1) color: vec3f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec3f,
        };

        @vertex fn main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.position = vec4f(in.position, 0.0, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char *fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Vertex data: position (2 floats) + color (3 floats)
    struct ColorVertex {
        float x, y;
        float r, g, b;
    };

    // Full-screen triangle with blue color
    std::vector<ColorVertex> vertices = {
        {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        {3.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 3.0f, 0.0f, 0.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(ColorVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColorVertex, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColorVertex, r)},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    // Check center pixel is blue
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_LE(g, 5) << "Green should be ~0";
    EXPECT_GE(b, 250) << "Blue should be ~255";
}

// Test: Color interpolation across triangle
TEST_F(VulkanGraphicsTest, ColorInterpolation) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
            @location(1) color: vec3f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec3f,
        };

        @vertex fn main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.position = vec4f(in.position, 0.0, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char *fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    struct ColorVertex {
        float x, y;
        float r, g, b;
    };

    // Triangle with RGB corners - full screen coverage
    std::vector<ColorVertex> vertices = {
        {-1.0f, -1.0f, 1.0f, 0.0f, 0.0f}, // Red at bottom-left
        {3.0f, -1.0f, 0.0f, 1.0f, 0.0f},  // Green at right
        {-1.0f, 3.0f, 0.0f, 0.0f, 1.0f},  // Blue at top
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(ColorVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColorVertex, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColorVertex, r)},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    // Check bottom-left corner is reddish
    uint32_t bl = pixels[(height - 2) * width + 1];
    uint8_t r, g, b, a;
    unpackRGBA(bl, r, g, b, a);
    EXPECT_GT(r, 100) << "Bottom-left should have significant red";

    // Check that center has mixed colors (none should be 0 or 255)
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GT(r, 20) << "Center should have some red";
    EXPECT_GT(g, 20) << "Center should have some green";
}

// Test: Indexed drawing
TEST_F(VulkanGraphicsTest, IndexedDrawing) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 1.0, 0.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Quad as 4 vertices
    std::vector<SimpleVertex> vertices = {
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {1.0f, 1.0f},
        {-1.0f, 1.0f},
    };

    // Two triangles forming the quad
    std::vector<uint16_t> indices = {0, 1, 2, 0, 2, 3};

    auto vb = ctx_->createVertexBuffer(vertices);
    auto ib = ctx_->createIndexBuffer(indices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->drawIndexed(pipeline, target, &vb, &ib, VK_INDEX_TYPE_UINT16,
        {.index_count = 6});

    auto pixels = target.downloadAs<uint32_t>();

    // Check center pixel is yellow
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 250) << "Red should be ~255";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_LE(b, 5) << "Blue should be ~0";
}

// Test: Fragment shader math operations
TEST_F(VulkanGraphicsTest, FragmentMathOps) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            let a = 0.5;
            let b = abs(-0.3);
            let c = clamp(1.5, 0.0, 1.0);
            let d = max(0.2, 0.1);
            return vec4f(a, b, c, d);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);

    // r = 0.5 * 255 = 127.5
    EXPECT_NEAR(r, 128, 2) << "Red should be ~0.5";
    // g = 0.3 * 255 = 76.5
    EXPECT_NEAR(g, 77, 2) << "Green should be ~0.3";
    // b = 1.0 * 255 = 255
    EXPECT_GE(b, 250) << "Blue should be ~1.0";
    // a = 0.2 * 255 = 51
    EXPECT_NEAR(a, 51, 2) << "Alpha should be ~0.2";
}

// Test: Vertex position transformation
TEST_F(VulkanGraphicsTest, VertexTransform) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            let scaled = in.position * 0.5;
            return vec4f(scaled, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(0.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Full quad vertices, will be scaled to half
    std::vector<SimpleVertex> vertices = {
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {-1.0f, 1.0f},
        {1.0f, -1.0f},
        {1.0f, 1.0f},
        {-1.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);

    // Clear to black
    vk_graphics::ClearColor black = {0.0f, 0.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 6}, {}, black);

    auto pixels = target.downloadAs<uint32_t>();

    // Center should be cyan (inside scaled quad)
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_GE(b, 250) << "Blue should be ~255";

    // Corner should be black (outside the scaled quad)
    uint32_t corner = pixels[0];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Corner red should be ~0 (clear)";
    EXPECT_LE(g, 5) << "Corner green should be ~0 (clear)";
    EXPECT_LE(b, 5) << "Corner blue should be ~0 (clear)";
}

// Test: Vector operations in shaders
TEST_F(VulkanGraphicsTest, VectorOperations) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            let a = vec3f(0.5, 0.5, 0.5);
            let b = vec3f(0.5, 0.0, 0.5);
            let sum = a + b;
            let clamped = clamp(sum, vec3f(0.0), vec3f(1.0));
            return vec4f(clamped, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);

    EXPECT_GE(r, 250) << "Red should be 1.0";
    EXPECT_NEAR(g, 128, 2) << "Green should be 0.5";
    EXPECT_GE(b, 250) << "Blue should be 1.0";
}

// Test: Simple conditional in fragment shader
TEST_F(VulkanGraphicsTest, FragmentConditional) {
    const char *vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) ndc: vec2f,
        };

        @vertex fn main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.position = vec4f(in.position, 0.0, 1.0);
            out.ndc = in.position;
            return out;
        }
    )";

    const char *fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            if (ndc.x > 0.0) {
                return vec4f(1.0, 0.0, 0.0, 1.0);
            } else {
                return vec4f(0.0, 0.0, 1.0, 1.0);
            }
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint8_t r, g, b, a;

    // Left side (x < 0) should be blue
    uint32_t left = pixels[(height / 2) * width + (width / 4)];
    unpackRGBA(left, r, g, b, a);
    EXPECT_LE(r, 5) << "Left side red should be ~0";
    EXPECT_GE(b, 250) << "Left side blue should be ~255";

    // Right side (x > 0) should be red
    uint32_t right = pixels[(height / 2) * width + (width * 3 / 4)];
    unpackRGBA(right, r, g, b, a);
    EXPECT_GE(r, 250) << "Right side red should be ~255";
    EXPECT_LE(b, 5) << "Right side blue should be ~0";
}

// ============================================================================
// GLSL Graphics Tests
// ============================================================================

TEST_F(VulkanGraphicsTest, GlslSolidColorFill) {
    const char *vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        #version 450
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), width * height);

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 250) << "Red channel should be ~255";
    EXPECT_LE(g, 5) << "Green channel should be ~0";
    EXPECT_LE(b, 5) << "Blue channel should be ~0";
    EXPECT_GE(a, 250) << "Alpha channel should be ~255";
}

TEST_F(VulkanGraphicsTest, GlslVertexAttributes) {
    const char *vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 color;

        layout(location = 0) out vec3 fragColor;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragColor = color;
        }
    )";

    const char *fs_source = R"(
        #version 450
        layout(location = 0) in vec3 fragColor;
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(fragColor, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    struct ColorVertex {
        float x, y;
        float r, g, b;
    };

    std::vector<ColorVertex> vertices = {
        {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        {3.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 3.0f, 0.0f, 0.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(ColorVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColorVertex, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColorVertex, r)},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_LE(g, 5) << "Green should be ~0";
    EXPECT_GE(b, 250) << "Blue should be ~255";
}

TEST_F(VulkanGraphicsTest, GlslFragmentMathOps) {
    const char *vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        #version 450
        layout(location = 0) out vec4 outColor;

        void main() {
            float a = 0.5;
            float b = abs(-0.3);
            float c = clamp(1.5, 0.0, 1.0);
            float d = max(0.2, 0.1);
            outColor = vec4(a, b, c, d);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);

    EXPECT_NEAR(r, 128, 2) << "Red should be ~0.5";
    EXPECT_NEAR(g, 77, 2) << "Green should be ~0.3";
    EXPECT_GE(b, 250) << "Blue should be ~1.0";
    EXPECT_NEAR(a, 51, 2) << "Alpha should be ~0.2";
}

TEST_F(VulkanGraphicsTest, GlslVertexTransform) {
    const char *vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;

        void main() {
            vec2 scaled = position * 0.5;
            gl_Position = vec4(scaled, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        #version 450
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(0.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    std::vector<SimpleVertex> vertices = {
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {-1.0f, 1.0f},
        {1.0f, -1.0f},
        {1.0f, 1.0f},
        {-1.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    vk_graphics::ClearColor black = {0.0f, 0.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 6}, {}, black);

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_GE(b, 250) << "Blue should be ~255";

    uint32_t corner = pixels[0];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Corner red should be ~0 (clear)";
    EXPECT_LE(g, 5) << "Corner green should be ~0 (clear)";
    EXPECT_LE(b, 5) << "Corner blue should be ~0 (clear)";
}

// ============================================================================
// Push Constants via var<immediate> — Graphics Pipeline
// ============================================================================

using vk_graphics::VulkanError;

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

// Helper: create a graphics pipeline with push constant range
struct GfxPushPipeline {
    VkShaderModule vs_module = VK_NULL_HANDLE;
    VkShaderModule fs_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    ~GfxPushPipeline() {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        if (vs_module) vkDestroyShaderModule(device, vs_module, nullptr);
        if (fs_module) vkDestroyShaderModule(device, fs_module, nullptr);
    }
};

static GfxPushPipeline createGfxPipelineWithPush(
    vk_graphics::GraphicsContext &ctx,
    const std::vector<uint32_t> &vs_spirv, const char *vs_entry,
    const std::vector<uint32_t> &fs_spirv, const char *fs_entry,
    uint32_t push_size, VkShaderStageFlags push_stages,
    uint32_t vertex_stride,
    const std::vector<vk_graphics::VertexAttribute> &vertex_attrs,
    VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM)
{
    GfxPushPipeline p;
    p.device = ctx.device();

    // Shader modules
    VkShaderModuleCreateInfo vs_ci = {};
    vs_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vs_ci.codeSize = vs_spirv.size() * sizeof(uint32_t);
    vs_ci.pCode = vs_spirv.data();
    VK_CHECK(vkCreateShaderModule(ctx.device(), &vs_ci, nullptr, &p.vs_module));

    VkShaderModuleCreateInfo fs_ci = {};
    fs_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fs_ci.codeSize = fs_spirv.size() * sizeof(uint32_t);
    fs_ci.pCode = fs_spirv.data();
    VK_CHECK(vkCreateShaderModule(ctx.device(), &fs_ci, nullptr, &p.fs_module));

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = p.vs_module;
    stages[0].pName = vs_entry;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = p.fs_module;
    stages[1].pName = fs_entry;

    // Empty descriptor set layout
    VkDescriptorSetLayoutCreateInfo dl_info = {};
    dl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device(), &dl_info, nullptr, &p.desc_layout));

    // Push constant range
    VkPushConstantRange push_range = {};
    push_range.stageFlags = push_stages;
    push_range.offset = 0;
    push_range.size = push_size;

    VkPipelineLayoutCreateInfo pl_info = {};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 0;
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &push_range;
    VK_CHECK(vkCreatePipelineLayout(ctx.device(), &pl_info, nullptr, &p.layout));

    // Vertex input
    VkVertexInputBindingDescription binding_desc = {};
    binding_desc.binding = 0;
    binding_desc.stride = vertex_stride;
    binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> attr_descs;
    for (const auto &a : vertex_attrs) {
        VkVertexInputAttributeDescription d = {};
        d.location = a.location;
        d.binding = 0;
        d.format = a.format;
        d.offset = a.offset;
        attr_descs.push_back(d);
    }

    VkPipelineVertexInputStateCreateInfo vertex_input = {};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (vertex_stride > 0 && !attr_descs.empty()) {
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding_desc;
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attr_descs.size());
        vertex_input.pVertexAttributeDescriptions = attr_descs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp = {};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms_state = {};
    ms_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineColorBlendAttachmentState blend_att = {};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &blend_att;

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyn_states;

    VkPipelineRenderingCreateInfo rendering = {};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format;

    VkGraphicsPipelineCreateInfo gfx_ci = {};
    gfx_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_ci.pNext = &rendering;
    gfx_ci.stageCount = 2;
    gfx_ci.pStages = stages;
    gfx_ci.pVertexInputState = &vertex_input;
    gfx_ci.pInputAssemblyState = &ia;
    gfx_ci.pViewportState = &vp;
    gfx_ci.pRasterizationState = &rs;
    gfx_ci.pMultisampleState = &ms_state;
    gfx_ci.pDepthStencilState = &ds;
    gfx_ci.pColorBlendState = &cb;
    gfx_ci.pDynamicState = &dyn;
    gfx_ci.layout = p.layout;

    VK_CHECK(vkCreateGraphicsPipelines(ctx.device(), VK_NULL_HANDLE, 1, &gfx_ci, nullptr, &p.pipeline));
    return p;
}

// Helper: draw a full-screen triangle with push constants
static void drawWithPush(
    vk_graphics::GraphicsContext &ctx,
    GfxPushPipeline &pipeline,
    vk_graphics::Image &target,
    vk_graphics::Buffer *vb,
    uint32_t vertex_count,
    const void *push_data, uint32_t push_size, VkShaderStageFlags push_stages,
    vk_graphics::ClearColor clear = {})
{
    ctx.executeCommands([&](VkCommandBuffer cmd) {
        ctx.transitionImageLayout(cmd, target.handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color_att = {};
        color_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_att.imageView = target.view();
        color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};

        VkRenderingInfo ri = {};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = {{0, 0}, {target.width(), target.height()}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color_att;

        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

        VkViewport viewport = {0, 0,
            static_cast<float>(target.width()),
            static_cast<float>(target.height()),
            0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor = {{0, 0}, {target.width(), target.height()}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdPushConstants(cmd, pipeline.layout, push_stages,
            0, push_size, push_data);

        if (vb) {
            VkBuffer buf = vb->handle();
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &buf, &offset);
        }

        vkCmdDraw(cmd, vertex_count, 1, 0, 0);
        vkCmdEndRendering(cmd);

        ctx.transitionImageLayout(cmd, target.handle(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });
}

// --- Graphics push constant tests ---

TEST_F(VulkanGraphicsTest, Immediate_FragmentColor) {
    // Push a color via immediate to fragment shader, fill entire screen
    const char *vs_source = R"(
        struct VertexInput { @location(0) position: vec2f };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        enable immediate_address_space;
        var<immediate> color: vec4f;

        @fragment fn main() -> @location(0) vec4f {
            return color;
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;

    auto fs = CompileImmediate(fs_source, "main");
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);
    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    float color[4] = {0.0f, 1.0f, 0.0f, 1.0f}; // green
    auto pipeline = createGfxPipelineWithPush(*ctx_,
        vs.spirv, "main", fs.spirv, "main",
        sizeof(color), VK_SHADER_STAGE_FRAGMENT_BIT,
        sizeof(SimpleVertex),
        {{0, VK_FORMAT_R32G32_SFLOAT, 0}});

    drawWithPush(*ctx_, pipeline, target, &vb, 3,
        color, sizeof(color), VK_SHADER_STAGE_FRAGMENT_BIT);

    auto pixels = target.downloadAs<uint32_t>();
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_LE(b, 5) << "Blue should be ~0";
    EXPECT_GE(a, 250) << "Alpha should be ~255";
}

TEST_F(VulkanGraphicsTest, Immediate_FragmentColorDynamic) {
    // Same pipeline, different push constant colors per draw
    const char *vs_source = R"(
        struct VertexInput { @location(0) position: vec2f };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        enable immediate_address_space;
        var<immediate> color: vec4f;

        @fragment fn main() -> @location(0) vec4f {
            return color;
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = CompileImmediate(fs_source, "main");
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);
    const uint32_t W = 32, H = 32;

    auto pipeline = createGfxPipelineWithPush(*ctx_,
        vs.spirv, "main", fs.spirv, "main",
        16, VK_SHADER_STAGE_FRAGMENT_BIT,
        sizeof(SimpleVertex),
        {{0, VK_FORMAT_R32G32_SFLOAT, 0}});

    // Draw red
    {
        auto target = ctx_->createColorTarget(W, H);
        float color[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        drawWithPush(*ctx_, pipeline, target, &vb, 3,
            color, 16, VK_SHADER_STAGE_FRAGMENT_BIT);

        auto pixels = target.downloadAs<uint32_t>();
        uint8_t r, g, b, a;
        unpackRGBA(pixels[(H / 2) * W + (W / 2)], r, g, b, a);
        EXPECT_GE(r, 250);
        EXPECT_LE(g, 5);
    }

    // Draw blue
    {
        auto target = ctx_->createColorTarget(W, H);
        float color[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        drawWithPush(*ctx_, pipeline, target, &vb, 3,
            color, 16, VK_SHADER_STAGE_FRAGMENT_BIT);

        auto pixels = target.downloadAs<uint32_t>();
        uint8_t r, g, b, a;
        unpackRGBA(pixels[(H / 2) * W + (W / 2)], r, g, b, a);
        EXPECT_LE(r, 5);
        EXPECT_GE(b, 250);
    }
}

TEST_F(VulkanGraphicsTest, Immediate_VertexScale) {
    // Push a scale factor to the vertex shader to shrink the triangle
    const char *vs_source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;

        struct VertexInput { @location(0) position: vec2f };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position * scale, 0.0, 1.0);
        }
    )";

    const char *fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs = CompileImmediate(vs_source, "main");
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);
    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    // Scale = 0.5: triangle covers less area, corners should be clear (black)
    float scale = 0.5f;
    auto pipeline = createGfxPipelineWithPush(*ctx_,
        vs.spirv, "main", fs.spirv, "main",
        sizeof(float), VK_SHADER_STAGE_VERTEX_BIT,
        sizeof(SimpleVertex),
        {{0, VK_FORMAT_R32G32_SFLOAT, 0}});

    drawWithPush(*ctx_, pipeline, target, &vb, 3,
        &scale, sizeof(float), VK_SHADER_STAGE_VERTEX_BIT);

    auto pixels = target.downloadAs<uint32_t>();

    // Center-ish should be white (inside triangle)
    uint8_t r, g, b, a;
    uint32_t center = pixels[(H / 4) * W + (W / 4)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 200) << "Center should be lit (white triangle)";

    // Far corner should be black (clear color, outside scaled triangle)
    uint32_t corner = pixels[(H - 1) * W + (W - 1)];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Far corner should be clear (black)";
    EXPECT_LE(g, 5);
    EXPECT_LE(b, 5);
}

#endif // WGSL_HAS_VULKAN
