#include <gtest/gtest.h>
#include <cstdio>
extern "C" {
#include "simple_wgsl.h"
}

TEST(PtxSpirvDebug, DumpVecAdd) {
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

            mov.u32 %r0, %tid.x;
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

    SsirModule *mod = NULL;
    char *err = NULL;
    PtxToSsirOptions opts = {.preserve_names = 1};
    PtxToSsirResult pr = ptx_to_ssir(ptx, &opts, &mod, &err);
    ASSERT_EQ(pr, PTX_TO_SSIR_OK) << (err ? err : "unknown");
    ptx_to_ssir_free(err);

    // Print SSIR info
    printf("Functions: %u\n", mod->function_count);
    for (uint32_t i = 0; i < mod->function_count; i++) {
        SsirFunction *f = &mod->functions[i];
        printf("  Func[%u]: id=%u params=%u locals=%u blocks=%u\n",
               i, f->id, f->param_count, f->local_count, f->block_count);
    }
    printf("Entry points: %u\n", mod->entry_point_count);
    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        SsirEntryPoint *ep = &mod->entry_points[i];
        printf("  EP[%u]: func=%u stage=%d wg=(%u,%u,%u) iface=%u\n",
               i, ep->function, ep->stage,
               ep->workgroup_size[0], ep->workgroup_size[1], ep->workgroup_size[2],
               ep->interface_count);
    }
    printf("Globals: %u\n", mod->global_count);
    for (uint32_t i = 0; i < mod->global_count; i++) {
        SsirGlobalVar *g = &mod->globals[i];
        printf("  Global[%u]: id=%u group=%d binding=%d\n",
               i, g->id, g->group, g->binding);
    }

    // Emit SPIR-V
    uint32_t *words = NULL;
    size_t count = 0;
    SsirToSpirvOptions spirv_opts = {.spirv_version = 0x00010300, .enable_debug_names = 1};
    SsirToSpirvResult sr = ssir_to_spirv(mod, &spirv_opts, &words, &count);
    ASSERT_EQ(sr, SSIR_TO_SPIRV_OK);
    printf("\nSPIR-V: %zu words\n", count);

    // Write to file
    FILE *fp = fopen("/tmp/ptx_debug.spv", "wb");
    fwrite(words, sizeof(uint32_t), count, fp);
    fclose(fp);

    // Also try WGSL
    char *wgsl_out = NULL, *wgsl_err = NULL;
    SsirToWgslOptions wgsl_opts = {.preserve_names = 1};
    ssir_to_wgsl(mod, &wgsl_opts, &wgsl_out, &wgsl_err);
    if (wgsl_out) printf("\nWGSL:\n%s\n", wgsl_out);
    else printf("\nWGSL error: %s\n", wgsl_err ? wgsl_err : "unknown");
    ssir_to_wgsl_free(wgsl_out);
    ssir_to_wgsl_free(wgsl_err);

    ssir_to_spirv_free(words);
    ssir_module_destroy(mod);
}
