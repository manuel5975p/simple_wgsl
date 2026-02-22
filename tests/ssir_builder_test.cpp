// Tests that directly exercise the SSIR builder API (ssir.c) to increase
// coverage of paths not reached through the WGSL->SPIR-V->SSIR pipeline.
#include <gtest/gtest.h>

extern "C" {
#include "simple_wgsl.h"
}

namespace {

// ============================================================================
// RAII helpers
// ============================================================================

struct ModGuard {
    SsirModule *m;
    explicit ModGuard(SsirModule *mod) : m(mod) {}
    ~ModGuard() { if (m) ssir_module_destroy(m); }
    SsirModule *get() { return m; }
};

// Build a minimal void->void function with a single entry block and a
// return_void terminator.  Returns func_id and block_id via out-params.
static void build_minimal_func(SsirModule *mod,
                                uint32_t *out_func, uint32_t *out_block) {
    uint32_t void_t = ssir_type_void(mod);
    uint32_t func_id = ssir_function_create(mod, "f", void_t);
    uint32_t blk_id  = ssir_block_create(mod, func_id, "entry");
    ssir_build_return_void(mod, func_id, blk_id);
    if (out_func)  *out_func  = func_id;
    if (out_block) *out_block = blk_id;
}

// ============================================================================
// Module-level API
// ============================================================================

TEST(SsirModule, CreateDestroy) {
    SsirModule *mod = ssir_module_create();
    ASSERT_NE(mod, nullptr);
    ssir_module_destroy(mod);
}

TEST(SsirModule, AllocId) {
    ModGuard g(ssir_module_create());
    uint32_t a = ssir_module_alloc_id(g.get());
    uint32_t b = ssir_module_alloc_id(g.get());
    EXPECT_NE(a, 0u);
    EXPECT_NE(b, 0u);
    EXPECT_NE(a, b);
}

TEST(SsirModule, SetName) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_module_alloc_id(g.get());
    // Should not crash
    ssir_set_name(g.get(), id, "my_var");
}

TEST(SsirModule, SetClipSpace) {
    ModGuard g(ssir_module_create());
    ssir_module_set_clip_space(g.get(), SSIR_CLIP_SPACE_DIRECTX);
    EXPECT_EQ(g.get()->clip_space, SSIR_CLIP_SPACE_DIRECTX);
    ssir_module_set_clip_space(g.get(), SSIR_CLIP_SPACE_OPENGL);
    EXPECT_EQ(g.get()->clip_space, SSIR_CLIP_SPACE_OPENGL);
}

TEST(SsirModule, BuildLookup) {
    ModGuard g(ssir_module_create());
    build_minimal_func(g.get(), nullptr, nullptr);
    // Should not crash
    ssir_module_build_lookup(g.get());
}

// ============================================================================
// Extended scalar types
// ============================================================================

TEST(SsirType, Bool) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_bool(g.get());
    ASSERT_NE(t, 0u);
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_BOOL);
    EXPECT_TRUE(ssir_type_is_bool(ty));
    EXPECT_FALSE(ssir_type_is_integer(ty));

    // Second call returns cached id
    EXPECT_EQ(ssir_type_bool(g.get()), t);
}

TEST(SsirType, F16) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_f16(g.get());
    ASSERT_NE(t, 0u);
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_F16);
    EXPECT_TRUE(ssir_type_is_float(ty));
    EXPECT_TRUE(ssir_type_is_scalar(ty));
}

TEST(SsirType, F64) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_f64(g.get());
    ASSERT_NE(t, 0u);
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_F64);
    EXPECT_TRUE(ssir_type_is_float(ty));
}

TEST(SsirType, I8) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_i8(g.get());
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_I8);
    EXPECT_TRUE(ssir_type_is_integer(ty));
    EXPECT_TRUE(ssir_type_is_signed(ty));
}

TEST(SsirType, U8) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_u8(g.get());
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_U8);
    EXPECT_TRUE(ssir_type_is_unsigned(ty));
}

TEST(SsirType, I16) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_i16(g.get());
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_I16);
    EXPECT_TRUE(ssir_type_is_signed(ty));
}

TEST(SsirType, U16) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_u16(g.get());
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_U16);
    EXPECT_TRUE(ssir_type_is_unsigned(ty));
}

TEST(SsirType, I64) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_i64(g.get());
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_I64);
    EXPECT_TRUE(ssir_type_is_signed(ty));
}

TEST(SsirType, U64) {
    ModGuard g(ssir_module_create());
    uint32_t t = ssir_type_u64(g.get());
    SsirType *ty = ssir_get_type(g.get(), t);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_U64);
    EXPECT_TRUE(ssir_type_is_unsigned(ty));
}

TEST(SsirType, MatrixType) {
    ModGuard g(ssir_module_create());
    uint32_t f32 = ssir_type_f32(g.get());
    uint32_t vec3 = ssir_type_vec(g.get(), f32, 3);
    uint32_t mat = ssir_type_mat(g.get(), vec3, 4, 3);
    ASSERT_NE(mat, 0u);
    SsirType *ty = ssir_get_type(g.get(), mat);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_MAT);
    EXPECT_TRUE(ssir_type_is_matrix(ty));
}

TEST(SsirType, ArrayStride) {
    ModGuard g(ssir_module_create());
    uint32_t f32 = ssir_type_f32(g.get());
    uint32_t arr = ssir_type_array_stride(g.get(), f32, 16, 16);
    ASSERT_NE(arr, 0u);
    SsirType *ty = ssir_get_type(g.get(), arr);
    ASSERT_NE(ty, nullptr);
    EXPECT_EQ(ty->kind, SSIR_TYPE_ARRAY);
}

TEST(SsirType, ScalarOf_Vec) {
    ModGuard g(ssir_module_create());
    uint32_t f32   = ssir_type_f32(g.get());
    uint32_t vec4  = ssir_type_vec(g.get(), f32, 4);
    uint32_t elem  = ssir_type_scalar_of(g.get(), vec4);
    EXPECT_EQ(elem, f32);
}

TEST(SsirType, ScalarOf_Matrix) {
    ModGuard g(ssir_module_create());
    uint32_t f32  = ssir_type_f32(g.get());
    uint32_t col  = ssir_type_vec(g.get(), f32, 3);
    uint32_t mat  = ssir_type_mat(g.get(), col, 3, 3);
    uint32_t elem = ssir_type_scalar_of(g.get(), mat);
    EXPECT_EQ(elem, f32);
}

TEST(SsirType, ScalarOf_Scalar) {
    ModGuard g(ssir_module_create());
    uint32_t f32  = ssir_type_f32(g.get());
    // Scalar of a scalar is itself
    uint32_t elem = ssir_type_scalar_of(g.get(), f32);
    EXPECT_EQ(elem, f32);
}

TEST(SsirType, TypeKindName) {
    const char *name = ssir_type_kind_name(SSIR_TYPE_F32);
    ASSERT_NE(name, nullptr);
    EXPECT_NE(name[0], '\0');
}

// ============================================================================
// Extended constants
// ============================================================================

TEST(SsirConst, F16) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_f16(g.get(), 0x3C00 /* 1.0f in f16 */);
    ASSERT_NE(id, 0u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->f16_val, (uint16_t)0x3C00);
}

TEST(SsirConst, F64) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_f64(g.get(), 3.14159265358979);
    ASSERT_NE(id, 0u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->f64_val, 3.14159265358979);
}

TEST(SsirConst, I8) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_i8(g.get(), -42);
    ASSERT_NE(id, 0u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->i8_val, (int8_t)-42);
}

TEST(SsirConst, U8) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_u8(g.get(), 200u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->u8_val, (uint8_t)200u);
}

TEST(SsirConst, I16) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_i16(g.get(), -1000);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->i16_val, (int16_t)-1000);
}

TEST(SsirConst, U16) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_u16(g.get(), 60000u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->u16_val, (uint16_t)60000u);
}

TEST(SsirConst, I64) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_i64(g.get(), (int64_t)-1234567890123LL);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->i64_val, (int64_t)-1234567890123LL);
}

TEST(SsirConst, U64) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_u64(g.get(), (uint64_t)9876543210ULL);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->u64_val, (uint64_t)9876543210ULL);
}

TEST(SsirConst, SpecBool) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_spec_bool(g.get(), true, 7u);
    ASSERT_NE(id, 0u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->is_specialization);
    EXPECT_EQ(c->spec_id, 7u);
    EXPECT_TRUE(c->bool_val);
}

TEST(SsirConst, SpecI32) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_spec_i32(g.get(), -99, 1u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->is_specialization);
    EXPECT_EQ(c->i32_val, -99);
    EXPECT_EQ(c->spec_id, 1u);
}

TEST(SsirConst, SpecU32) {
    ModGuard g(ssir_module_create());
    uint32_t id = ssir_const_spec_u32(g.get(), 42u, 2u);
    SsirConstant *c = ssir_get_constant(g.get(), id);
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->is_specialization);
    EXPECT_EQ(c->u32_val, 42u);
}

// ============================================================================
// Global variables
// ============================================================================

TEST(SsirGlobal, CreateAndMetadata) {
    ModGuard g(ssir_module_create());
    uint32_t f32    = ssir_type_f32(g.get());
    uint32_t ptr_t  = ssir_type_ptr(g.get(), f32, SSIR_ADDR_UNIFORM);
    uint32_t gid    = ssir_global_var(g.get(), "u_val", ptr_t);
    ASSERT_NE(gid, 0u);

    ssir_global_set_group(g.get(), gid, 0u);
    ssir_global_set_binding(g.get(), gid, 2u);

    SsirGlobalVar *gv = ssir_get_global(g.get(), gid);
    ASSERT_NE(gv, nullptr);
    EXPECT_TRUE(gv->has_group);
    EXPECT_EQ(gv->group, 0u);
    EXPECT_TRUE(gv->has_binding);
    EXPECT_EQ(gv->binding, 2u);
}

TEST(SsirGlobal, NonWritable) {
    ModGuard g(ssir_module_create());
    uint32_t f32   = ssir_type_f32(g.get());
    uint32_t ptr_t = ssir_type_ptr(g.get(), f32, SSIR_ADDR_STORAGE);
    uint32_t gid   = ssir_global_var(g.get(), "buf", ptr_t);
    ssir_global_set_non_writable(g.get(), gid, true);
    SsirGlobalVar *gv = ssir_get_global(g.get(), gid);
    ASSERT_NE(gv, nullptr);
    EXPECT_TRUE(gv->non_writable);
}

TEST(SsirGlobal, Invariant) {
    ModGuard g(ssir_module_create());
    uint32_t f32   = ssir_type_f32(g.get());
    uint32_t ptr_t = ssir_type_ptr(g.get(), f32, SSIR_ADDR_OUTPUT);
    uint32_t gid   = ssir_global_var(g.get(), "pos", ptr_t);
    ssir_global_set_invariant(g.get(), gid, true);
    SsirGlobalVar *gv = ssir_get_global(g.get(), gid);
    ASSERT_NE(gv, nullptr);
    EXPECT_TRUE(gv->invariant);
}

TEST(SsirGlobal, InterpSampling) {
    ModGuard g(ssir_module_create());
    uint32_t f32   = ssir_type_f32(g.get());
    uint32_t ptr_t = ssir_type_ptr(g.get(), f32, SSIR_ADDR_INPUT);
    uint32_t gid   = ssir_global_var(g.get(), "vary", ptr_t);
    ssir_global_set_interp_sampling(g.get(), gid, SSIR_INTERP_SAMPLING_CENTROID);
    SsirGlobalVar *gv = ssir_get_global(g.get(), gid);
    ASSERT_NE(gv, nullptr);
    EXPECT_EQ(gv->interp_sampling, SSIR_INTERP_SAMPLING_CENTROID);
}

TEST(SsirGlobal, Initializer) {
    ModGuard g(ssir_module_create());
    uint32_t f32   = ssir_type_f32(g.get());
    uint32_t ptr_t = ssir_type_ptr(g.get(), f32, SSIR_ADDR_PRIVATE);
    uint32_t gid   = ssir_global_var(g.get(), "priv", ptr_t);
    uint32_t cid   = ssir_const_f32(g.get(), 0.0f);
    ssir_global_set_initializer(g.get(), gid, cid);
    SsirGlobalVar *gv = ssir_get_global(g.get(), gid);
    ASSERT_NE(gv, nullptr);
    EXPECT_TRUE(gv->has_initializer);
    EXPECT_EQ(gv->initializer, cid);
}

// ============================================================================
// Entry points
// ============================================================================

TEST(SsirEntryPoint, CreateVertex) {
    ModGuard g(ssir_module_create());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "vs_main", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");
    ssir_build_return_void(g.get(), func_id, blk);

    uint32_t ep = ssir_entry_point_create(g.get(), SSIR_STAGE_VERTEX,
                                          func_id, "vs_main");
    EXPECT_EQ(ep, 0u); // returns index 0
    SsirEntryPoint *ep_ptr = ssir_get_entry_point(g.get(), 0u);
    ASSERT_NE(ep_ptr, nullptr);
    EXPECT_EQ(ep_ptr->stage, SSIR_STAGE_VERTEX);
}

TEST(SsirEntryPoint, EarlyFragmentTests) {
    ModGuard g(ssir_module_create());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "fs", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");
    ssir_build_return_void(g.get(), func_id, blk);

    uint32_t ep = ssir_entry_point_create(g.get(), SSIR_STAGE_FRAGMENT,
                                          func_id, "fs");
    ssir_entry_point_set_early_fragment_tests(g.get(), ep, true);
    SsirEntryPoint *ep_ptr = ssir_get_entry_point(g.get(), ep);
    ASSERT_NE(ep_ptr, nullptr);
    EXPECT_TRUE(ep_ptr->early_fragment_tests);
}

TEST(SsirEntryPoint, DepthReplacing) {
    ModGuard g(ssir_module_create());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "fs2", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");
    ssir_build_return_void(g.get(), func_id, blk);

    uint32_t ep = ssir_entry_point_create(g.get(), SSIR_STAGE_FRAGMENT,
                                          func_id, "fs2");
    ssir_entry_point_set_depth_replacing(g.get(), ep, true);
    SsirEntryPoint *ep_ptr = ssir_get_entry_point(g.get(), ep);
    ASSERT_NE(ep_ptr, nullptr);
    EXPECT_TRUE(ep_ptr->depth_replacing);
}

// ============================================================================
// Function / block / parameter building
// ============================================================================

TEST(SsirFunction, AddParam) {
    ModGuard g(ssir_module_create());
    uint32_t f32 = ssir_type_f32(g.get());
    uint32_t void_t = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "add_f", void_t);
    uint32_t p = ssir_function_add_param(g.get(), func_id, "x", f32);
    EXPECT_NE(p, 0u);
}

TEST(SsirFunction, BlockCreateWithId) {
    ModGuard g(ssir_module_create());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "g", void_t);
    uint32_t bid     = ssir_module_alloc_id(g.get());
    uint32_t blk     = ssir_block_create_with_id(g.get(), func_id, bid, "named");
    EXPECT_EQ(blk, bid);
    SsirBlock *bp = ssir_get_block(g.get(), func_id, blk);
    ASSERT_NE(bp, nullptr);
    ssir_build_return_void(g.get(), func_id, blk);
}

// ============================================================================
// Instruction builder - control flow
// ============================================================================

TEST(SsirBuild, Unreachable) {
    ModGuard g(ssir_module_create());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "unreach_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");
    // unreachable terminates the block
    ssir_build_unreachable(g.get(), func_id, blk);
    SsirBlock *bp = ssir_get_block(g.get(), func_id, blk);
    ASSERT_NE(bp, nullptr);
    ASSERT_GT(bp->inst_count, 0u);
    EXPECT_EQ(bp->insts[bp->inst_count - 1].op, SSIR_OP_UNREACHABLE);
}

TEST(SsirBuild, BranchCondMerge) {
    ModGuard g(ssir_module_create());
    uint32_t bool_t  = ssir_type_bool(g.get());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "bcm_fn", void_t);

    uint32_t entry   = ssir_block_create(g.get(), func_id, "entry");
    uint32_t t_blk   = ssir_block_create(g.get(), func_id, "true_blk");
    uint32_t f_blk   = ssir_block_create(g.get(), func_id, "false_blk");
    uint32_t merge   = ssir_block_create(g.get(), func_id, "merge");

    uint32_t cond = ssir_const_bool(g.get(), true);
    ssir_build_branch_cond_merge(g.get(), func_id, entry,
                                 cond, t_blk, f_blk, merge);
    ssir_build_return_void(g.get(), func_id, t_blk);
    ssir_build_return_void(g.get(), func_id, f_blk);
    ssir_build_return_void(g.get(), func_id, merge);

    SsirBlock *ep = ssir_get_block(g.get(), func_id, entry);
    ASSERT_NE(ep, nullptr);
    ASSERT_GT(ep->inst_count, 0u);
    EXPECT_EQ(ep->insts[ep->inst_count - 1].op, SSIR_OP_BRANCH_COND);
    (void)bool_t;
}

TEST(SsirBuild, Phi) {
    ModGuard g(ssir_module_create());
    uint32_t i32_t   = ssir_type_i32(g.get());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "phi_fn", void_t);

    uint32_t pred_a = ssir_block_create(g.get(), func_id, "pred_a");
    uint32_t pred_b = ssir_block_create(g.get(), func_id, "pred_b");
    uint32_t join   = ssir_block_create(g.get(), func_id, "join");

    uint32_t v_a = ssir_const_i32(g.get(), 1);
    uint32_t v_b = ssir_const_i32(g.get(), 2);

    // Each incoming pair: (value_id, block_id)
    uint32_t incoming[] = { v_a, pred_a, v_b, pred_b };
    uint32_t phi_id = ssir_build_phi(g.get(), func_id, join,
                                     i32_t, incoming, 2);
    EXPECT_NE(phi_id, 0u);

    ssir_build_return_void(g.get(), func_id, pred_a);
    ssir_build_return_void(g.get(), func_id, pred_b);
    uint32_t ret_v = ssir_build_add(g.get(), func_id, join, i32_t,
                                    phi_id, phi_id);
    ssir_build_return(g.get(), func_id, join, ret_v);
}

TEST(SsirBuild, Call) {
    ModGuard g(ssir_module_create());
    uint32_t i32_t   = ssir_type_i32(g.get());
    uint32_t void_t  = ssir_type_void(g.get());

    // callee: returns i32
    uint32_t callee_id = ssir_function_create(g.get(), "callee", i32_t);
    uint32_t cb        = ssir_block_create(g.get(), callee_id, "entry");
    uint32_t cv        = ssir_const_i32(g.get(), 42);
    ssir_build_return(g.get(), callee_id, cb, cv);

    // caller: calls callee with no args
    uint32_t caller_id = ssir_function_create(g.get(), "caller", void_t);
    uint32_t blk       = ssir_block_create(g.get(), caller_id, "entry");
    uint32_t result    = ssir_build_call(g.get(), caller_id, blk,
                                         i32_t, callee_id, nullptr, 0);
    EXPECT_NE(result, 0u);
    ssir_build_return_void(g.get(), caller_id, blk);
}

// ============================================================================
// Instruction builder - arithmetic/logic
// ============================================================================

TEST(SsirBuild, GeInstruction) {
    ModGuard g(ssir_module_create());
    uint32_t i32_t   = ssir_type_i32(g.get());
    uint32_t bool_t  = ssir_type_bool(g.get());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "ge_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");

    uint32_t a  = ssir_const_i32(g.get(), 3);
    uint32_t b  = ssir_const_i32(g.get(), 2);
    uint32_t r  = ssir_build_ge(g.get(), func_id, blk, bool_t, a, b);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
    (void)i32_t;
}

TEST(SsirBuild, MatTranspose) {
    ModGuard g(ssir_module_create());
    uint32_t f32   = ssir_type_f32(g.get());
    uint32_t col   = ssir_type_vec(g.get(), f32, 4);
    uint32_t mat44 = ssir_type_mat(g.get(), col, 4, 4);
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "transpose_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");

    // Build a composite matrix constant to transpose
    uint32_t fzero = ssir_const_f32(g.get(), 0.0f);
    uint32_t vec_c = ssir_const_composite(g.get(), col,
                                          (uint32_t[]){fzero, fzero, fzero, fzero}, 4);
    uint32_t mat_c = ssir_const_composite(g.get(), mat44,
                                          (uint32_t[]){vec_c, vec_c, vec_c, vec_c}, 4);

    uint32_t r = ssir_build_mat_transpose(g.get(), func_id, blk, mat44, mat_c);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
}

TEST(SsirBuild, Rem) {
    ModGuard g(ssir_module_create());
    uint32_t i32_t   = ssir_type_i32(g.get());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "rem_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");
    uint32_t a = ssir_const_i32(g.get(), 10);
    uint32_t b = ssir_const_i32(g.get(), 3);
    uint32_t r = ssir_build_rem(g.get(), func_id, blk, i32_t, a, b);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
}

// ============================================================================
// Instruction builder - composites
// ============================================================================

TEST(SsirBuild, Insert) {
    ModGuard g(ssir_module_create());
    uint32_t f32     = ssir_type_f32(g.get());
    uint32_t vec4    = ssir_type_vec(g.get(), f32, 4);
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "insert_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");

    uint32_t fzero = ssir_const_f32(g.get(), 0.0f);
    uint32_t fone  = ssir_const_f32(g.get(), 1.0f);
    uint32_t base  = ssir_const_composite(g.get(), vec4,
                                          (uint32_t[]){fzero, fzero, fzero, fzero}, 4);
    uint32_t r = ssir_build_insert(g.get(), func_id, blk, vec4, base, fone, 2);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
}

TEST(SsirBuild, ExtractDyn) {
    ModGuard g(ssir_module_create());
    uint32_t f32     = ssir_type_f32(g.get());
    uint32_t vec4    = ssir_type_vec(g.get(), f32, 4);
    uint32_t u32_t   = ssir_type_u32(g.get());
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "extdyn_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");

    uint32_t fzero = ssir_const_f32(g.get(), 0.0f);
    uint32_t vec   = ssir_const_composite(g.get(), vec4,
                                          (uint32_t[]){fzero, fzero, fzero, fzero}, 4);
    uint32_t idx   = ssir_const_u32(g.get(), 1u);
    uint32_t r = ssir_build_extract_dyn(g.get(), func_id, blk, f32, vec, idx);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
    (void)u32_t;
}

TEST(SsirBuild, InsertDyn) {
    ModGuard g(ssir_module_create());
    uint32_t f32     = ssir_type_f32(g.get());
    uint32_t vec4    = ssir_type_vec(g.get(), f32, 4);
    uint32_t void_t  = ssir_type_void(g.get());
    uint32_t func_id = ssir_function_create(g.get(), "insdyn_fn", void_t);
    uint32_t blk     = ssir_block_create(g.get(), func_id, "entry");

    uint32_t fzero = ssir_const_f32(g.get(), 0.0f);
    uint32_t fone  = ssir_const_f32(g.get(), 1.0f);
    uint32_t vec   = ssir_const_composite(g.get(), vec4,
                                          (uint32_t[]){fzero, fzero, fzero, fzero}, 4);
    uint32_t idx   = ssir_const_u32(g.get(), 2u);
    uint32_t r = ssir_build_insert_dyn(g.get(), func_id, blk, vec4, vec, fone, idx);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
}

TEST(SsirBuild, ArrayLen) {
    ModGuard g(ssir_module_create());
    uint32_t f32      = ssir_type_f32(g.get());
    uint32_t rt_arr   = ssir_type_runtime_array(g.get(), f32);
    uint32_t ptr_t    = ssir_type_ptr(g.get(), rt_arr, SSIR_ADDR_STORAGE);
    uint32_t void_t   = ssir_type_void(g.get());
    uint32_t func_id  = ssir_function_create(g.get(), "arrlen_fn", void_t);
    uint32_t blk      = ssir_block_create(g.get(), func_id, "entry");

    uint32_t gid = ssir_global_var(g.get(), "buf", ptr_t);
    uint32_t r   = ssir_build_array_len(g.get(), func_id, blk, gid);
    EXPECT_NE(r, 0u);
    ssir_build_return_void(g.get(), func_id, blk);
}

} // namespace
