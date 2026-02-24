/*
 * cuda_to_vulkan_test.cpp — End-to-end CUDA → Vulkan compute dispatch tests
 *
 * Two test fixtures:
 *   PtxVulkanTest    — hand-crafted PTX → SSIR → SPIR-V → Vulkan dispatch
 *   CudaToVulkanTest — .cu → nvcc --ptx → PTX → SSIR → SPIR-V → Vulkan dispatch
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef WGSL_HAS_VULKAN
#include "vulkan_compute_harness.h"

extern "C" {
#include "simple_wgsl.h"
}

/* ============================================================================
 * Helpers
 * ============================================================================ */

struct SpirvResult {
    bool success;
    std::string error;
    std::vector<uint32_t> spirv;
};

// PTX source → SPIR-V binary
static SpirvResult CompilePtxToSpirv(const std::string &ptx) {
    SpirvResult out;
    out.success = false;

    SsirModule *mod = nullptr;
    char *err = nullptr;
    PtxToSsirOptions ptx_opts = {};
    ptx_opts.preserve_names = 1;

    PtxToSsirResult pr = ptx_to_ssir(ptx.c_str(), &ptx_opts, &mod, &err);
    if (pr != PTX_TO_SSIR_OK) {
        out.error = "PTX parse failed: " + std::string(err ? err : "unknown");
        ptx_to_ssir_free(err);
        return out;
    }
    ptx_to_ssir_free(err);

    uint32_t *words = nullptr;
    size_t count = 0;
    SsirToSpirvOptions spirv_opts = {};
    spirv_opts.spirv_version = 0x00010300; // SPIR-V 1.3 for Vulkan 1.1
    spirv_opts.enable_debug_names = 1;

    SsirToSpirvResult sr = ssir_to_spirv(mod, &spirv_opts, &words, &count);
    ssir_module_destroy(mod);

    if (sr != SSIR_TO_SPIRV_OK) {
        out.error = "SPIR-V emit failed: " +
                    std::string(ssir_to_spirv_result_string(sr));
        return out;
    }

    out.spirv.assign(words, words + count);
    ssir_to_spirv_free(words);
    out.success = true;
    return out;
}

// Check if nvcc is available on PATH
static bool NvccAvailable() {
    int rc = system("nvcc --version > /dev/null 2>&1");
    return rc == 0;
}

struct CudaCompileResult {
    bool success;
    std::string error;
    std::string ptx;
};

// Compile CUDA source to PTX via nvcc
static CudaCompileResult CompileCudaToPtx(const std::string &cuda_source) {
    CudaCompileResult out;
    out.success = false;

    // Write .cu to temp file
    char cu_path[] = "/tmp/cuda_vk_test_XXXXXX.cu";
    int fd = mkstemps(cu_path, 3); // 3 = strlen(".cu")
    if (fd < 0) {
        out.error = "Failed to create temp .cu file";
        return out;
    }
    write(fd, cuda_source.data(), cuda_source.size());
    close(fd);

    // Derive .ptx path
    std::string ptx_path(cu_path);
    ptx_path.replace(ptx_path.size() - 3, 3, ".ptx");

    // Compile with nvcc
    std::string cmd = "nvcc --ptx -o " + ptx_path + " " + cu_path + " 2>&1";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        out.error = "Failed to run nvcc";
        unlink(cu_path);
        return out;
    }

    std::string nvcc_output;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) {
        nvcc_output += buf;
    }
    int rc = pclose(pipe);
    unlink(cu_path);

    if (rc != 0) {
        out.error = "nvcc failed (rc=" + std::to_string(rc) + "): " + nvcc_output;
        unlink(ptx_path.c_str());
        return out;
    }

    // Read PTX output
    std::ifstream ifs(ptx_path);
    if (!ifs.is_open()) {
        out.error = "Failed to read PTX output: " + ptx_path;
        return out;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out.ptx = ss.str();
    ifs.close();
    unlink(ptx_path.c_str());

    if (out.ptx.empty()) {
        out.error = "nvcc produced empty PTX output";
        return out;
    }

    out.success = true;
    return out;
}

/* ============================================================================
 * Fixture: PtxVulkanTest — hand-crafted PTX dispatched on Vulkan
 * ============================================================================ */

class PtxVulkanTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        try {
            ctx_ = std::make_unique<vk_compute::VulkanContext>();
        } catch (const std::exception &e) {
            skip_reason_ = std::string("Vulkan not available: ") + e.what();
        }
    }

    static void TearDownTestSuite() { ctx_.reset(); }

    void SetUp() override {
        if (!ctx_) GTEST_SKIP() << skip_reason_;
    }

    // Helper: compile PTX and create pipeline
    vk_compute::ComputePipeline PtxPipeline(const std::string &ptx,
                                             const char *entry) {
        auto result = CompilePtxToSpirv(ptx);
        EXPECT_TRUE(result.success) << result.error;
        if (!result.success)
            throw std::runtime_error(result.error);
        return ctx_->createPipeline(result.spirv, entry);
    }

    static std::unique_ptr<vk_compute::VulkanContext> ctx_;
    static std::string skip_reason_;
};

std::unique_ptr<vk_compute::VulkanContext> PtxVulkanTest::ctx_;
std::string PtxVulkanTest::skip_reason_;

/* --------------------------------------------------------------------------
 * PTX: Vector Add (a[i] + b[i] → c[i])
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, VectorAdd) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry vec_add(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<7>;

            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [c_ptr];

            // global_id = ctaid.x * ntid.x + tid.x
            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd3, %r0;
            mul.lo.u64 %rd3, %rd3, 4;

            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];

            add.u64 %rd5, %rd1, %rd3;
            ld.global.f32 %f1, [%rd5];

            add.f32 %f2, %f0, %f1;
            add.u64 %rd6, %rd2, %rd3;
            st.global.f32 [%rd6], %f2;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "vec_add");

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    auto buf_a = ctx_->createStorageBuffer(a);
    auto buf_b = ctx_->createStorageBuffer(b);
    auto buf_c = ctx_->createStorageBuffer(a.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_c, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(a.size()));

    auto c = buf_c.download<float>(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: Vector Scale (a[i] * 2.5 → b[i])
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, VectorScale) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry vec_scale(
            .param .u64 in_ptr,
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [in_ptr];
            ld.param.u64 %rd1, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd2, %r0;
            mul.lo.u64 %rd2, %rd2, 4;

            add.u64 %rd3, %rd0, %rd2;
            ld.global.f32 %f0, [%rd3];

            // scale by 2.5
            mul.f32 %f1, %f0, 0f40200000;

            add.u64 %rd4, %rd1, %rd2;
            st.global.f32 [%rd4], %f1;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "vec_scale");

    std::vector<float> input = {2.0f, 4.0f, 6.0f, 8.0f};
    auto buf_in = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto output = buf_out.download<float>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], input[i] * 2.5f) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: SAXPY (a*x[i] + y[i] → out[i])
 * Uses fma instruction.
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, Saxpy) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry saxpy(
            .param .u64 x_ptr,
            .param .u64 y_ptr,
            .param .u64 out_ptr,
            .param .u64 a_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<5>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<8>;

            ld.param.u64 %rd0, [x_ptr];
            ld.param.u64 %rd1, [y_ptr];
            ld.param.u64 %rd2, [out_ptr];
            ld.param.u64 %rd3, [a_ptr];

            // Load scalar 'a' from first element of a_ptr buffer
            ld.global.f32 %f3, [%rd3];

            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd4, %r0;
            mul.lo.u64 %rd4, %rd4, 4;

            add.u64 %rd5, %rd0, %rd4;
            ld.global.f32 %f0, [%rd5];

            add.u64 %rd6, %rd1, %rd4;
            ld.global.f32 %f1, [%rd6];

            // out = a*x + y
            fma.rn.f32 %f2, %f3, %f0, %f1;

            add.u64 %rd7, %rd2, %rd4;
            st.global.f32 [%rd7], %f2;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "saxpy");

    const float a_val = 3.0f;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> a_buf = {a_val};

    auto buf_x   = ctx_->createStorageBuffer(x);
    auto buf_y   = ctx_->createStorageBuffer(y);
    auto buf_out = ctx_->createStorageBuffer(x.size() * sizeof(float));
    auto buf_a   = ctx_->createStorageBuffer(a_buf);

    ctx_->dispatch(pipeline,
        {{0, &buf_x,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_y,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {3, &buf_a,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(x.size()));

    auto out = buf_out.download<float>(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        EXPECT_FLOAT_EQ(out[i], a_val * x[i] + y[i]) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: Conditional Write (out[i] = a[i] > 0 ? a[i] : 0)
 * Tests predicated branching.
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, ConditionalWrite) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry cond_write(
            .param .u64 in_ptr,
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .pred %p0;
            .reg .f32 %f<2>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [in_ptr];
            ld.param.u64 %rd1, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd2, %r0;
            mul.lo.u64 %rd2, %rd2, 4;

            add.u64 %rd3, %rd0, %rd2;
            ld.global.f32 %f0, [%rd3];

            // if a[i] > 0, write a[i]; else write 0
            setp.gt.f32 %p0, %f0, 0f00000000;
            selp.f32 %f1, %f0, 0f00000000, %p0;

            add.u64 %rd4, %rd1, %rd2;
            st.global.f32 [%rd4], %f1;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "cond_write");

    std::vector<float> input = {5.0f, -3.0f, 0.0f, 7.0f, -1.0f, 10.0f};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto output = buf_out.download<float>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        float expected = input[i] > 0.0f ? input[i] : 0.0f;
        EXPECT_FLOAT_EQ(output[i], expected) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: Bitwise Ops (out[i] = (a[i] & 0xFF) | (b[i] << 8))
 * Tests AND, OR, SHL with u32.
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, BitwiseOps) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry bitwise(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<6>;
            .reg .u64 %rd<7>;

            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd3, %r0;
            mul.lo.u64 %rd3, %rd3, 4;

            add.u64 %rd4, %rd0, %rd3;
            ld.global.u32 %r1, [%rd4];

            add.u64 %rd5, %rd1, %rd3;
            ld.global.u32 %r2, [%rd5];

            and.b32 %r3, %r1, 255;
            shl.b32 %r4, %r2, 8;
            or.b32 %r5, %r3, %r4;

            add.u64 %rd6, %rd2, %rd3;
            st.global.u32 [%rd6], %r5;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "bitwise");

    std::vector<uint32_t> a = {0x1234, 0xABCD, 0x00FF, 0xFFFF};
    std::vector<uint32_t> b = {0x56,   0x78,   0x9A,   0x00};
    auto buf_a   = ctx_->createStorageBuffer(a);
    auto buf_b   = ctx_->createStorageBuffer(b);
    auto buf_out = ctx_->createStorageBuffer(a.size() * sizeof(uint32_t));

    ctx_->dispatch(pipeline,
        {{0, &buf_a,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_b,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(a.size()));

    auto out = buf_out.download<uint32_t>(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        uint32_t expected = (a[i] & 0xFF) | (b[i] << 8);
        EXPECT_EQ(out[i], expected) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: Type Conversion (int → float → int roundtrip)
 * Tests cvt instructions.
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, TypeConversion) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry type_cvt(
            .param .u64 in_ptr,
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .s32 %r<2>;
            .reg .f32 %f<2>;
            .reg .u32 %r_tid;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [in_ptr];
            ld.param.u64 %rd1, [out_ptr];

            mov.u32 %r_tid, %ctaid.x;
            cvt.u64.u32 %rd2, %r_tid;
            mul.lo.u64 %rd2, %rd2, 4;

            // Load int
            add.u64 %rd3, %rd0, %rd2;
            ld.global.s32 %r0, [%rd3];

            // int → float
            cvt.rn.f32.s32 %f0, %r0;

            // float * 2.0
            mul.f32 %f1, %f0, 0f40000000;

            // float → int (truncate)
            cvt.rzi.s32.f32 %r1, %f1;

            // Store int
            add.u64 %rd4, %rd1, %rd2;
            st.global.s32 [%rd4], %r1;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "type_cvt");

    std::vector<int32_t> input = {5, -3, 100, 0, -42, 7};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(int32_t));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto out = buf_out.download<int32_t>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        int32_t expected = static_cast<int32_t>(static_cast<float>(input[i]) * 2.0f);
        EXPECT_EQ(out[i], expected) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: Vector Multiply-Add (mad: out[i] = a[i] * b[i] + c[i])
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, MulAdd) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry mul_add(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr,
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<4>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<9>;

            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [c_ptr];
            ld.param.u64 %rd3, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd4, %r0;
            mul.lo.u64 %rd4, %rd4, 4;

            add.u64 %rd5, %rd0, %rd4;
            ld.global.f32 %f0, [%rd5];

            add.u64 %rd6, %rd1, %rd4;
            ld.global.f32 %f1, [%rd6];

            add.u64 %rd7, %rd2, %rd4;
            ld.global.f32 %f2, [%rd7];

            // out = a * b + c
            fma.rn.f32 %f3, %f0, %f1, %f2;

            add.u64 %rd8, %rd3, %rd4;
            st.global.f32 [%rd8], %f3;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "mul_add");

    std::vector<float> a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b = {10.0f, 10.0f, 10.0f, 10.0f};
    std::vector<float> c = {1.0f, 2.0f, 3.0f, 4.0f};

    auto buf_a   = ctx_->createStorageBuffer(a);
    auto buf_b   = ctx_->createStorageBuffer(b);
    auto buf_c   = ctx_->createStorageBuffer(c);
    auto buf_out = ctx_->createStorageBuffer(a.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_a,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_b,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_c,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {3, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(a.size()));

    auto out = buf_out.download<float>(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        float expected = a[i] * b[i] + c[i];
        EXPECT_FLOAT_EQ(out[i], expected) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * PTX: Min/Max (out[i] = max(min(a[i], 100.0), 0.0)) — clamp to [0, 100]
 * -------------------------------------------------------------------------- */
TEST_F(PtxVulkanTest, MinMax) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry clamp(
            .param .u64 in_ptr,
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [in_ptr];
            ld.param.u64 %rd1, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            cvt.u64.u32 %rd2, %r0;
            mul.lo.u64 %rd2, %rd2, 4;

            add.u64 %rd3, %rd0, %rd2;
            ld.global.f32 %f0, [%rd3];

            // clamp: max(min(x, 100.0), 0.0)
            min.f32 %f1, %f0, 0f42C80000;
            max.f32 %f2, %f1, 0f00000000;

            add.u64 %rd4, %rd1, %rd2;
            st.global.f32 [%rd4], %f2;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "clamp");

    std::vector<float> input = {-50.0f, 0.0f, 50.0f, 100.0f, 150.0f, -0.1f};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto out = buf_out.download<float>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        float expected = std::fmax(std::fmin(input[i], 100.0f), 0.0f);
        EXPECT_FLOAT_EQ(out[i], expected) << "index " << i;
    }
}

/* ============================================================================
 * Fixture: CudaToVulkanTest — CUDA .cu → nvcc → PTX → Vulkan dispatch
 * ============================================================================ */

class CudaToVulkanTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        if (!NvccAvailable()) {
            skip_reason_ = "nvcc not found in PATH";
            return;
        }
        try {
            ctx_ = std::make_unique<vk_compute::VulkanContext>();
        } catch (const std::exception &e) {
            skip_reason_ = std::string("Vulkan not available: ") + e.what();
        }
    }

    static void TearDownTestSuite() { ctx_.reset(); }

    void SetUp() override {
        if (!ctx_) GTEST_SKIP() << skip_reason_;
    }

    // Helper: CUDA source → nvcc → PTX → SPIR-V → pipeline
    vk_compute::ComputePipeline CudaPipeline(const std::string &cuda_source,
                                              const char *kernel_name) {
        auto cuda_result = CompileCudaToPtx(cuda_source);
        EXPECT_TRUE(cuda_result.success) << cuda_result.error;
        if (!cuda_result.success)
            throw std::runtime_error(cuda_result.error);

        // Debug: print PTX if test debugging
        if (::testing::GTEST_FLAG(print_time)) {
            fprintf(stderr, "--- PTX for %s ---\n%s\n--- end PTX ---\n",
                    kernel_name, cuda_result.ptx.c_str());
        }

        auto spirv_result = CompilePtxToSpirv(cuda_result.ptx);
        EXPECT_TRUE(spirv_result.success) << spirv_result.error;
        if (!spirv_result.success)
            throw std::runtime_error(spirv_result.error);

        return ctx_->createPipeline(spirv_result.spirv, kernel_name);
    }

    static std::unique_ptr<vk_compute::VulkanContext> ctx_;
    static std::string skip_reason_;
};

std::unique_ptr<vk_compute::VulkanContext> CudaToVulkanTest::ctx_;
std::string CudaToVulkanTest::skip_reason_;

/* --------------------------------------------------------------------------
 * CUDA: Vector Add
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, VectorAdd) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void vector_add(const float *a, const float *b, float *c) {
            int i = blockIdx.x;
            c[i] = a[i] + b[i];
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "vector_add");

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    auto buf_a = ctx_->createStorageBuffer(a);
    auto buf_b = ctx_->createStorageBuffer(b);
    auto buf_c = ctx_->createStorageBuffer(a.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_c, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(a.size()));

    auto c = buf_c.download<float>(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * CUDA: Vector Scale (multiply by constant)
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, VectorScale) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void vec_scale(const float *input, float *output) {
            int i = blockIdx.x;
            output[i] = input[i] * 3.0f;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "vec_scale");

    std::vector<float> input = {2.0f, 4.0f, 6.0f, 8.0f};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto output = buf_out.download<float>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], input[i] * 3.0f) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * CUDA: SAXPY (a*x + y) — scalar passed via buffer
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, Saxpy) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void saxpy(const float *x, const float *y, float *out,
                              const float *a_buf) {
            int i = blockIdx.x;
            float a = a_buf[0];
            out[i] = a * x[i] + y[i];
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "saxpy");

    const float a_val = 2.5f;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> a_buf = {a_val};

    auto buf_x   = ctx_->createStorageBuffer(x);
    auto buf_y   = ctx_->createStorageBuffer(y);
    auto buf_out = ctx_->createStorageBuffer(x.size() * sizeof(float));
    auto buf_a   = ctx_->createStorageBuffer(a_buf);

    ctx_->dispatch(pipeline,
        {{0, &buf_x,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_y,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {3, &buf_a,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(x.size()));

    auto out = buf_out.download<float>(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        EXPECT_NEAR(out[i], a_val * x[i] + y[i], 1e-5f) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * CUDA: Negate (out[i] = -in[i])
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, Negate) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void negate(const float *input, float *output) {
            int i = blockIdx.x;
            output[i] = -input[i];
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "negate");

    std::vector<float> input = {1.0f, -2.0f, 3.5f, 0.0f, -100.0f};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto output = buf_out.download<float>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], -input[i]) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * CUDA: Integer Bitwise (out[i] = (a[i] ^ b[i]) & 0xFFFF)
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, IntegerBitwise) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void int_bitwise(const int *a, const int *b, int *out) {
            int i = blockIdx.x;
            out[i] = (a[i] ^ b[i]) & 0xFFFF;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "int_bitwise");

    std::vector<int32_t> a = {0x12345678, static_cast<int32_t>(0xAABBCCDD),
                              0x00000000, static_cast<int32_t>(0xFFFFFFFF)};
    std::vector<int32_t> b = {static_cast<int32_t>(0x87654321), 0x11223344,
                              static_cast<int32_t>(0xDEADBEEF), 0x00000000};
    auto buf_a   = ctx_->createStorageBuffer(a);
    auto buf_b   = ctx_->createStorageBuffer(b);
    auto buf_out = ctx_->createStorageBuffer(a.size() * sizeof(int32_t));

    ctx_->dispatch(pipeline,
        {{0, &buf_a,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_b,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(a.size()));

    auto out = buf_out.download<int32_t>(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        int32_t expected = (a[i] ^ b[i]) & 0xFFFF;
        EXPECT_EQ(out[i], expected) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * CUDA: Type Conversion (int → float, multiply, float → int)
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, TypeConversion) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void type_cvt(const int *input, int *output) {
            int i = blockIdx.x;
            float f = (float)input[i];
            f = f * 2.0f;
            output[i] = (int)f;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "type_cvt");

    std::vector<int32_t> input = {5, -3, 100, 0, 42};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(int32_t));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto out = buf_out.download<int32_t>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        int32_t expected = static_cast<int32_t>(static_cast<float>(input[i]) * 2.0f);
        EXPECT_EQ(out[i], expected) << "index " << i;
    }
}

/* --------------------------------------------------------------------------
 * CUDA: Square Root (out[i] = sqrtf(in[i]))
 * -------------------------------------------------------------------------- */
TEST_F(CudaToVulkanTest, Sqrt) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void sqrt_kernel(const float *input, float *output) {
            int i = blockIdx.x;
            output[i] = sqrtf(input[i]);
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "sqrt_kernel");

    std::vector<float> input = {4.0f, 9.0f, 16.0f, 25.0f, 100.0f, 1.0f};
    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(input.size() * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        static_cast<uint32_t>(input.size()));

    auto output = buf_out.download<float>(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(output[i], std::sqrt(input[i]), 1e-4f) << "index " << i;
    }
}

/* ============================================================================
 * PTX: 2D Dispatch — Matrix Transpose (rows x cols workgroups, each size 1)
 *   out[col * rows + row] = in[row * cols + col]
 * ============================================================================ */
TEST_F(PtxVulkanTest, MatrixTranspose2D) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry transpose(
            .param .u64 in_ptr,
            .param .u64 out_ptr,
            .param .u64 dims_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;
            .reg .u64 %rd<8>;
            .reg .f32 %f<1>;

            ld.param.u64 %rd0, [in_ptr];
            ld.param.u64 %rd1, [out_ptr];
            ld.param.u64 %rd2, [dims_ptr];

            // row = ctaid.x, col = ctaid.y
            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %ctaid.y;

            // Load dims: rows = dims[0], cols = dims[1]
            ld.global.u32 %r2, [%rd2];        // rows
            ld.global.u32 %r3, [%rd2+4];      // cols

            // src_idx = row * cols + col
            mul.lo.u32 %r4, %r0, %r3;
            add.u32 %r4, %r4, %r1;
            cvt.u64.u32 %rd3, %r4;
            mul.lo.u64 %rd3, %rd3, 4;
            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];

            // dst_idx = col * rows + row
            mul.lo.u32 %r5, %r1, %r2;
            add.u32 %r5, %r5, %r0;
            cvt.u64.u32 %rd5, %r5;
            mul.lo.u64 %rd5, %rd5, 4;
            add.u64 %rd6, %rd1, %rd5;
            st.global.f32 [%rd6], %f0;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "transpose");

    const uint32_t rows = 3, cols = 4;
    // Input: 3x4 matrix (row-major)
    std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    std::vector<uint32_t> dims = {rows, cols};

    auto buf_in   = ctx_->createStorageBuffer(input);
    auto buf_out  = ctx_->createStorageBuffer(static_cast<size_t>(rows) * cols * sizeof(float));
    auto buf_dims = ctx_->createStorageBuffer(dims);

    // Dispatch rows x cols workgroups
    ctx_->dispatch(pipeline,
        {{0, &buf_in,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_dims, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        rows, cols);

    auto output = buf_out.download<float>(static_cast<size_t>(rows) * cols);
    // Verify transpose: out[col][row] == in[row][col]
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            float expected = input[r * cols + c];
            float actual = output[c * rows + r];
            EXPECT_FLOAT_EQ(actual, expected)
                << "transpose[" << c << "][" << r << "]";
        }
    }
}

/* ============================================================================
 * PTX: 3D Dispatch — Volume fill
 *   out[x + y*X + z*X*Y] = x*100 + y*10 + z
 * ============================================================================ */
TEST_F(PtxVulkanTest, VolumeFill3D) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry volume_fill(
            .param .u64 out_ptr,
            .param .u64 dims_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<12>;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [out_ptr];
            ld.param.u64 %rd1, [dims_ptr];

            // x = ctaid.x, y = ctaid.y, z = ctaid.z
            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %ctaid.y;
            mov.u32 %r2, %ctaid.z;

            // Load X dimension from dims[0]
            ld.global.u32 %r3, [%rd1];        // X
            ld.global.u32 %r4, [%rd1+4];      // Y

            // linear = x + y*X + z*X*Y
            mul.lo.u32 %r5, %r1, %r3;         // y*X
            mul.lo.u32 %r6, %r2, %r3;         // z*X
            mul.lo.u32 %r6, %r6, %r4;         // z*X*Y
            add.u32 %r7, %r0, %r5;            // x + y*X
            add.u32 %r7, %r7, %r6;            // x + y*X + z*X*Y

            // value = x*100 + y*10 + z
            mul.lo.u32 %r8, %r0, 100;
            mul.lo.u32 %r9, %r1, 10;
            add.u32 %r10, %r8, %r9;
            add.u32 %r10, %r10, %r2;

            // Store
            cvt.u64.u32 %rd2, %r7;
            mul.lo.u64 %rd2, %rd2, 4;
            add.u64 %rd3, %rd0, %rd2;
            st.global.u32 [%rd3], %r10;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "volume_fill");

    const uint32_t X = 4, Y = 3, Z = 2;
    std::vector<uint32_t> dims = {X, Y};

    auto buf_out  = ctx_->createStorageBuffer(static_cast<size_t>(X) * Y * Z * sizeof(uint32_t));
    auto buf_dims = ctx_->createStorageBuffer(dims);

    ctx_->dispatch(pipeline,
        {{0, &buf_out,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_dims, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        X, Y, Z);

    auto output = buf_out.download<uint32_t>(static_cast<size_t>(X) * Y * Z);
    for (uint32_t z = 0; z < Z; z++) {
        for (uint32_t y = 0; y < Y; y++) {
            for (uint32_t x = 0; x < X; x++) {
                uint32_t idx = x + y * X + z * X * Y;
                uint32_t expected = x * 100 + y * 10 + z;
                EXPECT_EQ(output[idx], expected)
                    << "at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

/* ============================================================================
 * PTX: Non-trivial workgroup size — global_id = ctaid.x * ntid.x + tid.x
 * Tests .reqntid with block size > 1.
 * ============================================================================ */
TEST_F(PtxVulkanTest, BlockSize64) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry fill_gid(
            .param .u64 out_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [out_ptr];

            // global_id = ctaid.x * ntid.x + tid.x
            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ntid.x;
            mul.lo.u32 %r3, %r0, %r2;   // ctaid.x * ntid.x
            add.u32 %r3, %r3, %r1;      // + tid.x

            cvt.u64.u32 %rd1, %r3;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.u32 [%rd2], %r3;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "fill_gid");

    const uint32_t N = 256;
    const uint32_t BLOCK = 64;
    auto buf_out = ctx_->createStorageBuffer(N * sizeof(uint32_t));

    // Dispatch N/BLOCK workgroups, each with BLOCK threads
    ctx_->dispatch(pipeline,
        {{0, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        N / BLOCK);

    auto output = buf_out.download<uint32_t>(N);
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(output[i], i) << "global_id " << i;
    }
}

/* ============================================================================
 * PTX: 2D blocks — global_id_x/y = ctaid.x * ntid.x + tid.x, etc.
 * Fills a 2D grid with (gid_x, gid_y) packed as gid_y * width + gid_x.
 * ============================================================================ */
TEST_F(PtxVulkanTest, Block2D_8x8) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry fill_2d(
            .param .u64 out_ptr,
            .param .u64 width_ptr
        )
        .reqntid 8, 8, 1
        {
            .reg .u32 %r<10>;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [out_ptr];
            ld.param.u64 %rd1, [width_ptr];

            // gid_x = ctaid.x * ntid.x + tid.x
            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ntid.x;
            mul.lo.u32 %r3, %r0, %r2;
            add.u32 %r3, %r3, %r1;

            // gid_y = ctaid.y * ntid.y + tid.y
            mov.u32 %r4, %ctaid.y;
            mov.u32 %r5, %tid.y;
            mov.u32 %r6, %ntid.y;
            mul.lo.u32 %r7, %r4, %r6;
            add.u32 %r7, %r7, %r5;

            // linear = gid_y * width + gid_x
            ld.global.u32 %r8, [%rd1];     // width
            mul.lo.u32 %r9, %r7, %r8;
            add.u32 %r9, %r9, %r3;

            // Store global ID at linear position
            cvt.u64.u32 %rd2, %r9;
            mul.lo.u64 %rd2, %rd2, 4;
            add.u64 %rd3, %rd0, %rd2;
            st.global.u32 [%rd3], %r9;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "fill_2d");

    const uint32_t WIDTH = 32, HEIGHT = 16;
    const uint32_t BLOCK_X = 8, BLOCK_Y = 8;
    std::vector<uint32_t> width_buf = {WIDTH};

    auto buf_out   = ctx_->createStorageBuffer(static_cast<size_t>(WIDTH) * HEIGHT * sizeof(uint32_t));
    auto buf_width = ctx_->createStorageBuffer(width_buf);

    ctx_->dispatch(pipeline,
        {{0, &buf_out,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_width, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        WIDTH / BLOCK_X, HEIGHT / BLOCK_Y);

    auto output = buf_out.download<uint32_t>(static_cast<size_t>(WIDTH) * HEIGHT);
    for (uint32_t y = 0; y < HEIGHT; y++) {
        for (uint32_t x = 0; x < WIDTH; x++) {
            uint32_t idx = y * WIDTH + x;
            EXPECT_EQ(output[idx], idx)
                << "at (" << x << "," << y << ")";
        }
    }
}

/* ============================================================================
 * PTX: Large 1D dispatch (4096 elements, block size 256)
 * ============================================================================ */
TEST_F(PtxVulkanTest, LargeDispatch4K) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry scale4k(
            .param .u64 in_ptr,
            .param .u64 out_ptr
        )
        .reqntid 256, 1, 1
        {
            .reg .f32 %f<2>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<5>;

            ld.param.u64 %rd0, [in_ptr];
            ld.param.u64 %rd1, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ntid.x;
            mul.lo.u32 %r3, %r0, %r2;
            add.u32 %r3, %r3, %r1;

            cvt.u64.u32 %rd2, %r3;
            mul.lo.u64 %rd2, %rd2, 4;

            add.u64 %rd3, %rd0, %rd2;
            ld.global.f32 %f0, [%rd3];

            mul.f32 %f1, %f0, 0f40000000;   // * 2.0

            add.u64 %rd4, %rd1, %rd2;
            st.global.f32 [%rd4], %f1;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "scale4k");

    const uint32_t N = 4096;
    const uint32_t BLOCK = 256;
    std::vector<float> input(N);
    for (uint32_t i = 0; i < N; i++) input[i] = static_cast<float>(i) * 0.1f;

    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(N * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        N / BLOCK);

    auto output = buf_out.download<float>(N);
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_NEAR(output[i], input[i] * 2.0f, 1e-4f) << "index " << i;
    }
}

/* ============================================================================
 * PTX: nctaid (num_workgroups) usage — normalize by grid size
 *   out[i] = float(ctaid.x) / float(nctaid.x)  (produces 0..1 range)
 * ============================================================================ */
TEST_F(PtxVulkanTest, NumWorkgroups) {
    const char *ptx = R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry normalize(
            .param .u64 out_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<2>;
            .reg .f32 %f<3>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [out_ptr];

            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %nctaid.x;

            cvt.rn.f32.u32 %f0, %r0;
            cvt.rn.f32.u32 %f1, %r1;
            div.rn.f32 %f2, %f0, %f1;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f2;

            ret;
        }
    )";

    auto pipeline = PtxPipeline(ptx, "normalize");

    const uint32_t N = 16;
    auto buf_out = ctx_->createStorageBuffer(N * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        N);

    auto output = buf_out.download<float>(N);
    for (uint32_t i = 0; i < N; i++) {
        float expected = static_cast<float>(i) / static_cast<float>(N);
        EXPECT_NEAR(output[i], expected, 1e-5f) << "index " << i;
    }
}

/* ============================================================================
 * CUDA: 2D dispatch — matrix element-wise add
 * ============================================================================ */
TEST_F(CudaToVulkanTest, MatrixAdd2D) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void matrix_add(const float *a, const float *b, float *c,
                                    const int *dims) {
            int row = blockIdx.x;
            int col = blockIdx.y;
            int cols = dims[1];
            int idx = row * cols + col;
            c[idx] = a[idx] + b[idx];
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "matrix_add");

    const uint32_t ROWS = 4, COLS = 5;
    const uint32_t N = ROWS * COLS;
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 10.0f;
    }
    std::vector<int32_t> dims = {static_cast<int32_t>(ROWS),
                                  static_cast<int32_t>(COLS)};

    auto buf_a    = ctx_->createStorageBuffer(a);
    auto buf_b    = ctx_->createStorageBuffer(b);
    auto buf_c    = ctx_->createStorageBuffer(N * sizeof(float));
    auto buf_dims = ctx_->createStorageBuffer(dims);

    ctx_->dispatch(pipeline,
        {{0, &buf_a,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_b,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {2, &buf_c,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {3, &buf_dims, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        ROWS, COLS);

    auto c = buf_c.download<float>(N);
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << "index " << i;
    }
}

/* ============================================================================
 * CUDA: Squared index (out[i] = i*i) — uses blockIdx.x only
 * ============================================================================ */
TEST_F(CudaToVulkanTest, SquaredIndex) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void fill_squared(int *output) {
            int i = blockIdx.x;
            output[i] = i * i;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "fill_squared");

    const uint32_t N = 128;
    auto buf_out = ctx_->createStorageBuffer(N * sizeof(int32_t));

    ctx_->dispatch(pipeline,
        {{0, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        N);

    auto output = buf_out.download<int32_t>(N);
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(output[i], static_cast<int32_t>(i * i)) << "index " << i;
    }
}

/* ============================================================================
 * CUDA: Large dispatch (8192 elements)
 * ============================================================================ */
TEST_F(CudaToVulkanTest, LargeDispatch8K) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void add_const(const float *input, float *output) {
            int i = blockIdx.x;
            output[i] = input[i] + 42.0f;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "add_const");

    const uint32_t N = 8192;
    std::vector<float> input(N);
    for (uint32_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

    auto buf_in  = ctx_->createStorageBuffer(input);
    auto buf_out = ctx_->createStorageBuffer(N * sizeof(float));

    ctx_->dispatch(pipeline,
        {{0, &buf_in,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        N);

    auto output = buf_out.download<float>(N);
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(output[i], input[i] + 42.0f) << "index " << i;
    }
}

/* ============================================================================
 * CUDA: 3D dispatch — volume indexing
 * ============================================================================ */
TEST_F(CudaToVulkanTest, Volume3D) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void volume_idx(int *output, const int *dims) {
            int x = blockIdx.x;
            int y = blockIdx.y;
            int z = blockIdx.z;
            int X = dims[0];
            int Y = dims[1];
            int idx = x + y * X + z * X * Y;
            output[idx] = x + y * 10 + z * 100;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "volume_idx");

    const uint32_t X = 5, Y = 4, Z = 3;
    std::vector<int32_t> dims = {static_cast<int32_t>(X),
                                  static_cast<int32_t>(Y),
                                  static_cast<int32_t>(Z)};

    auto buf_out  = ctx_->createStorageBuffer(static_cast<size_t>(X) * Y * Z * sizeof(int32_t));
    auto buf_dims = ctx_->createStorageBuffer(dims);

    ctx_->dispatch(pipeline,
        {{0, &buf_out,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_dims, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        X, Y, Z);

    auto output = buf_out.download<int32_t>(static_cast<size_t>(X) * Y * Z);
    for (uint32_t z = 0; z < Z; z++) {
        for (uint32_t y = 0; y < Y; y++) {
            for (uint32_t x = 0; x < X; x++) {
                uint32_t idx = x + y * X + z * X * Y;
                int32_t expected = static_cast<int32_t>(x + y * 10 + z * 100);
                EXPECT_EQ(output[idx], expected)
                    << "at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

/* ============================================================================
 * CUDA: 2D grid — matrix fill using blockIdx.x/y
 * ============================================================================ */
TEST_F(CudaToVulkanTest, MatrixFill2D) {
    const char *cuda_source = R"(
        extern "C"
        __global__ void matrix_fill(int *output, const int *dims) {
            int x = blockIdx.x;
            int y = blockIdx.y;
            int width = dims[0];
            int idx = y * width + x;
            output[idx] = x * 1000 + y;
        }
    )";

    auto pipeline = CudaPipeline(cuda_source, "matrix_fill");

    const uint32_t WIDTH = 16, HEIGHT = 8;
    std::vector<int32_t> dims = {static_cast<int32_t>(WIDTH),
                                  static_cast<int32_t>(HEIGHT)};

    auto buf_out  = ctx_->createStorageBuffer(static_cast<size_t>(WIDTH) * HEIGHT * sizeof(int32_t));
    auto buf_dims = ctx_->createStorageBuffer(dims);

    ctx_->dispatch(pipeline,
        {{0, &buf_out,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
         {1, &buf_dims, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}},
        WIDTH, HEIGHT);

    auto output = buf_out.download<int32_t>(static_cast<size_t>(WIDTH) * HEIGHT);
    for (uint32_t y = 0; y < HEIGHT; y++) {
        for (uint32_t x = 0; x < WIDTH; x++) {
            uint32_t idx = y * WIDTH + x;
            int32_t expected = static_cast<int32_t>(x * 1000 + y);
            EXPECT_EQ(output[idx], expected)
                << "at (" << x << "," << y << ")";
        }
    }
}

#endif /* WGSL_HAS_VULKAN */
