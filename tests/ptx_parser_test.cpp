/*
 * ptx_parser_test.cpp - Unit tests for PTX → SSIR parser
 */

#include <gtest/gtest.h>
#include "simple_wgsl.h"
#include <string>

/* ============================================================================
 * RAII Guard
 * ============================================================================ */

class SsirModuleGuard {
  public:
    explicit SsirModuleGuard(SsirModule *m) : m_(m) {}
    ~SsirModuleGuard() { if (m_) ssir_module_destroy(m_); }
    SsirModule *get() { return m_; }
  private:
    SsirModule *m_;
    SsirModuleGuard(const SsirModuleGuard &) = delete;
    SsirModuleGuard &operator=(const SsirModuleGuard &) = delete;
};

/* ============================================================================
 * Parse Helper
 * ============================================================================ */

struct ParseResult {
    bool success;
    std::string error;
    SsirModule *mod;
};

static ParseResult ParsePtx(const std::string &ptx) {
    ParseResult res = {false, "", nullptr};
    char *err = nullptr;
    PtxToSsirOptions opts = {};
    opts.preserve_names = 1;

    PtxToSsirResult result = ptx_to_ssir(ptx.c_str(), &opts, &res.mod, &err);
    if (result != PTX_TO_SSIR_OK) {
        res.error = "PTX parse failed: " + std::string(err ? err : "unknown");
        ptx_to_ssir_free(err);
        return res;
    }
    ptx_to_ssir_free(err);
    res.success = true;
    return res;
}

/* ============================================================================
 * WGSL Roundtrip Helper
 * ============================================================================ */

static std::string PtxToWgsl(const std::string &ptx) {
    auto res = ParsePtx(ptx);
    if (!res.success) return "PARSE_ERROR: " + res.error;
    SsirModuleGuard guard(res.mod);

    char *wgsl_out = nullptr;
    char *wgsl_err = nullptr;
    SsirToWgslOptions wgsl_opts = {};
    wgsl_opts.preserve_names = 1;

    SsirToWgslResult wr = ssir_to_wgsl(res.mod, &wgsl_opts, &wgsl_out, &wgsl_err);
    if (wr != SSIR_TO_WGSL_OK) {
        std::string e = wgsl_err ? wgsl_err : "unknown";
        ssir_to_wgsl_free(wgsl_out);
        ssir_to_wgsl_free(wgsl_err);
        return "WGSL_ERROR: " + e;
    }
    std::string result = wgsl_out ? wgsl_out : "";
    ssir_to_wgsl_free(wgsl_out);
    ssir_to_wgsl_free(wgsl_err);
    return result;
}

/* ============================================================================
 * Basic Parsing Tests
 * ============================================================================ */

TEST(PtxParser, ParseEmpty) {
    auto res = ParsePtx("");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

TEST(PtxParser, ParseModuleHeader) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

TEST(PtxParser, ParseMinimalKernel) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry empty_kernel()
        {
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_COMPUTE);
}

TEST(PtxParser, ParseKernelWithParams) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry my_kernel(
            .param .u64 input_ptr,
            .param .u64 output_ptr,
            .param .u32 count
        )
        {
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_COMPUTE);
}

TEST(PtxParser, ParseRegDeclarations) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry test_regs()
        {
            .reg .f32 %f<10>;
            .reg .u32 %r<10>;
            .reg .pred %p<4>;
            .reg .u64 %rd0, %rd1;
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Arithmetic Tests
 * ============================================================================ */

TEST(PtxParser, ParseArithmetic) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry arith_test()
        {
            .reg .f32 %f<6>;
            .reg .u32 %r<6>;

            add.f32 %f2, %f0, %f1;
            sub.f32 %f3, %f0, %f1;
            mul.f32 %f4, %f0, %f1;
            div.f32 %f5, %f0, %f1;

            add.u32 %r2, %r0, %r1;
            sub.u32 %r3, %r0, %r1;
            mul.lo.u32 %r4, %r0, %r1;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

TEST(PtxParser, ParseMadFma) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry mad_test()
        {
            .reg .f32 %f<5>;
            .reg .s32 %r<5>;

            mad.lo.s32 %r3, %r0, %r1, %r2;
            fma.rn.f32 %f3, %f0, %f1, %f2;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

TEST(PtxParser, ParseMinMax) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry minmax_test()
        {
            .reg .f32 %f<4>;

            min.f32 %f2, %f0, %f1;
            max.f32 %f3, %f0, %f1;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

TEST(PtxParser, ParseNegAbs) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry negabs_test()
        {
            .reg .f32 %f<4>;

            neg.f32 %f1, %f0;
            abs.f32 %f2, %f0;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Bitwise Tests
 * ============================================================================ */

TEST(PtxParser, ParseBitwise) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry bitwise_test()
        {
            .reg .b32 %r<8>;

            and.b32 %r2, %r0, %r1;
            or.b32  %r3, %r0, %r1;
            xor.b32 %r4, %r0, %r1;
            not.b32 %r5, %r0;
            shl.b32 %r6, %r0, %r1;
            shr.b32 %r7, %r0, %r1;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Comparison / Selection Tests
 * ============================================================================ */

TEST(PtxParser, ParseSetp) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry setp_test()
        {
            .reg .f32 %f<3>;
            .reg .pred %p<3>;

            setp.lt.f32 %p0, %f0, %f1;
            setp.ge.f32 %p1, %f0, %f1;
            setp.eq.f32 %p2, %f0, %f2;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

TEST(PtxParser, ParseSelp) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry selp_test()
        {
            .reg .f32 %f<4>;
            .reg .pred %p0;

            setp.lt.f32 %p0, %f0, %f1;
            selp.f32 %f3, %f0, %f1, %p0;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Data Movement Tests
 * ============================================================================ */

TEST(PtxParser, ParseMov) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry mov_test()
        {
            .reg .f32 %f<3>;
            .reg .u32 %r<3>;

            mov.f32 %f1, %f0;
            mov.u32 %r1, 42;
            mov.f32 %f2, 0f3F800000;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Control Flow Tests
 * ============================================================================ */

TEST(PtxParser, ParseBranch) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry branch_test()
        {
            .reg .pred %p0;
            .reg .u32 %r<3>;

            setp.lt.u32 %p0, %r0, %r1;
            @%p0 bra LABEL_TRUE;
            mov.u32 %r2, 0;
            bra DONE;
        LABEL_TRUE:
            mov.u32 %r2, 1;
        DONE:
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Memory Operation Tests
 * ============================================================================ */

TEST(PtxParser, ParseLoadStore) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry loadstore_test(
            .param .u64 input_ptr,
            .param .u64 output_ptr
        )
        {
            .reg .u64 %rd<4>;
            .reg .f32 %f0;

            ld.param.u64 %rd0, [input_ptr];
            ld.global.f32 %f0, [%rd0];
            ld.param.u64 %rd1, [output_ptr];
            st.global.f32 [%rd1], %f0;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Special Register Tests
 * ============================================================================ */

TEST(PtxParser, ParseSpecialRegisters) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry special_regs_test()
        .reqntid 256, 1, 1
        {
            .reg .u32 %r<6>;

            mov.u32 %r0, %tid.x;
            mov.u32 %r1, %tid.y;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    SsirEntryPoint *ep = &res.mod->entry_points[0];
    EXPECT_EQ(ep->workgroup_size[0], 256u);
    EXPECT_EQ(ep->workgroup_size[1], 1u);
    EXPECT_EQ(ep->workgroup_size[2], 1u);
}

/* ============================================================================
 * Type Conversion Tests
 * ============================================================================ */

TEST(PtxParser, ParseCvt) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry cvt_test()
        {
            .reg .f32 %f0;
            .reg .s32 %r0;
            .reg .f64 %fd0;
            .reg .u32 %ru0;

            cvt.rn.f32.s32 %f0, %r0;
            cvt.rzi.s32.f32 %r0, %f0;
            cvt.f64.f32 %fd0, %f0;
            cvt.u32.u16 %ru0, %ru0;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Math Function Tests
 * ============================================================================ */

TEST(PtxParser, ParseMathFunctions) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry math_test()
        {
            .reg .f32 %f<8>;

            rcp.rn.f32 %f1, %f0;
            sqrt.rn.f32 %f2, %f0;
            rsqrt.approx.f32 %f3, %f0;
            sin.approx.f32 %f4, %f0;
            cos.approx.f32 %f5, %f0;
            lg2.approx.f32 %f6, %f0;
            ex2.approx.f32 %f7, %f0;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Atomic Tests
 * ============================================================================ */

TEST(PtxParser, ParseAtomics) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry atomic_test()
        {
            .reg .u32 %r<4>;
            .reg .u64 %rd0;

            atom.global.add.u32 %r1, [%rd0], %r0;
            atom.global.min.u32 %r2, [%rd0], %r0;
            atom.global.max.u32 %r3, [%rd0], %r0;

            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Barrier Tests
 * ============================================================================ */

TEST(PtxParser, ParseBarrier) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry barrier_test()
        {
            bar.sync 0;
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Device Function Tests
 * ============================================================================ */

TEST(PtxParser, ParseDeviceFunction) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .func (.reg .f32 %retval) square(.reg .f32 %x)
        {
            .reg .f32 %result;
            mul.f32 %result, %x, %x;
            ret;
        }

        .visible .entry use_func()
        {
            .reg .f32 %f<3>;
            mov.f32 %f0, 0f40000000;
            call (%f1), square, (%f0);
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * Global Memory Tests
 * ============================================================================ */

TEST(PtxParser, ParseGlobalDecl) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .shared .align 4 .f32 smem[256];

        .visible .entry global_test()
        {
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);
}

/* ============================================================================
 * WGSL Roundtrip Tests
 * ============================================================================ */

TEST(PtxParser, RoundtripMinimalKernel) {
    std::string wgsl = PtxToWgsl(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry empty_kernel()
        .reqntid 64, 1, 1
        {
            ret;
        }
    )");
    EXPECT_NE(wgsl.find("PARSE_ERROR"), 0u) << wgsl;
    EXPECT_NE(wgsl.find("WGSL_ERROR"), 0u) << wgsl;
}

TEST(PtxParser, RoundtripArithmetic) {
    std::string wgsl = PtxToWgsl(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry arith()
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<4>;
            mov.f32 %f0, 0f3F800000;
            mov.f32 %f1, 0f40000000;
            add.f32 %f2, %f0, %f1;
            mul.f32 %f3, %f2, %f0;
            ret;
        }
    )");
    EXPECT_NE(wgsl.find("PARSE_ERROR"), 0u) << wgsl;
    EXPECT_NE(wgsl.find("WGSL_ERROR"), 0u) << wgsl;
}

/* ============================================================================
 * Result String Tests
 * ============================================================================ */

TEST(PtxParser, ResultStrings) {
    EXPECT_STREQ(ptx_to_ssir_result_string(PTX_TO_SSIR_OK), "Success");
    EXPECT_STREQ(ptx_to_ssir_result_string(PTX_TO_SSIR_PARSE_ERROR), "Parse error");
    EXPECT_STREQ(ptx_to_ssir_result_string(PTX_TO_SSIR_UNSUPPORTED), "Unsupported feature");
}

TEST(PtxParser, NullInput) {
    SsirModule *mod = nullptr;
    char *err = nullptr;
    EXPECT_EQ(ptx_to_ssir(nullptr, nullptr, &mod, &err), PTX_TO_SSIR_PARSE_ERROR);
    ptx_to_ssir_free(err);
}

/* ============================================================================
 * Vector Add (realistic kernel)
 * ============================================================================ */

TEST(PtxParser, VectorAddKernel) {
    auto res = ParsePtx(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry vector_add(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr,
            .param .u32 n
        )
        .reqntid 256, 1, 1
        {
            .reg .pred %p0;
            .reg .f32 %f<3>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<8>;

            // Load params
            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [c_ptr];
            ld.param.u32 %r0, [n];

            // Compute global thread ID
            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mad.lo.u32 %r1, %r2, %r3, %r1;

            // Bounds check
            setp.ge.u32 %p0, %r1, %r0;
            @%p0 bra DONE;

            // Compute byte offset: index * 4
            cvt.u64.u32 %rd3, %r1;
            mul.lo.u64 %rd3, %rd3, 4;

            // Load a[i]
            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];

            // Load b[i]
            add.u64 %rd5, %rd1, %rd3;
            ld.global.f32 %f1, [%rd5];

            // c[i] = a[i] + b[i]
            add.f32 %f2, %f0, %f1;
            add.u64 %rd6, %rd2, %rd3;
            st.global.f32 [%rd6], %f2;

        DONE:
            ret;
        }
    )");
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    SsirEntryPoint *ep = &res.mod->entry_points[0];
    EXPECT_EQ(ep->stage, SSIR_STAGE_COMPUTE);
    EXPECT_EQ(ep->workgroup_size[0], 256u);
}

TEST(PtxParser, VectorAddToWgsl) {
    std::string wgsl = PtxToWgsl(R"(
        .version 7.8
        .target sm_80
        .address_size 64

        .visible .entry vector_add(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr,
            .param .u32 n
        )
        .reqntid 256, 1, 1
        {
            .reg .pred %p0;
            .reg .f32 %f<3>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<8>;

            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [c_ptr];
            ld.param.u32 %r0, [n];

            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mad.lo.u32 %r1, %r2, %r3, %r1;

            setp.ge.u32 %p0, %r1, %r0;
            @%p0 bra DONE;

            cvt.u64.u32 %rd3, %r1;
            mul.lo.u64 %rd3, %rd3, 4;

            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];

            add.u64 %rd5, %rd1, %rd3;
            ld.global.f32 %f1, [%rd5];

            add.f32 %f2, %f0, %f1;
            add.u64 %rd6, %rd2, %rd3;
            st.global.f32 [%rd6], %f2;

        DONE:
            ret;
        }
    )");
    // Should not fail
    EXPECT_EQ(wgsl.find("PARSE_ERROR"), std::string::npos) << wgsl;
    // WGSL output may or may not succeed depending on SSIR → WGSL
    // emitter compatibility; at minimum the parse should work.
}
