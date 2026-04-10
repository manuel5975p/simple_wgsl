/*
 * ssir_to_v3d.c -- SSIR-to-V3D QPU backend for BCM2712 (Pi 5, VideoCore VII)
 *
 * Compiler backend: SSIR (Simple Shader IR) -> V3D 7.1 QPU 64-bit machine code.
 * Targets bare-metal compute shaders with workgroup_size=16.
 *
 * TMU protocol: TMUC+TMUA only. TMUAU (waddr 13) is broken in bare-metal.
 * Float opcodes: FADD=5, FSUB=69, FMUL=21 (all with F32_UNPACK_NONE=+5).
 *
 * Steps:
 *   1. API skeleton + validation
 *   2. Context setup + prologue/epilogue
 *   3. Integer arithmetic + memory access
 *   4. Float arithmetic + builtins + conversions
 *   5. Comparisons + if/else predication
 *   6. Loop unrolling
 *   7. Hardening (BITCAST, RETURN_VOID, UNREACHABLE)
 */

#define _POSIX_C_SOURCE 200809L

#include "simple_wgsl_internal.h"
#include "../src/qpu_asm.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Types (SsirToV3dResult, SsirToV3dOptions, SsirToV3dOutput) in simple_wgsl.h */

/* ========================================================================== */
/* Internal context                                                           */
/* ========================================================================== */

#define V3D_MAX_RF       62
#define V3D_RF_UNASSIGNED 0xFF
#define V3D_MAX_BINDINGS 8

typedef struct {
    const SsirModule    *mod;
    SsirToV3dOptions     opts;
    const SsirEntryPoint *ep;
    const SsirFunction   *fn;

    struct qpu_prog prog;

    uint32_t *uniforms;
    uint32_t  uniform_count;
    uint32_t  uniform_cap;

    /* Register allocation: ssir_id -> RF index */
    uint8_t  *ssir_to_rf;
    uint8_t   rf_in_use[64];
    uint8_t   next_free_rf;
    uint8_t   max_rf;

    /* Instruction lookup */
    SsirInst **inst_map;
    uint32_t   inst_map_cap;

    /* Use-count tracking for register lifetime */
    uint32_t *use_counts;
    uint32_t *remaining_uses;

    /* Binding table: storage buffers */
    struct {
        uint32_t global_id;
        uint8_t  base_rf;
        uint32_t group;
        uint32_t binding;
    } bindings[V3D_MAX_BINDINGS];
    uint32_t binding_count;

    /* Reserved RFs */
    uint8_t rf_eidx;
    uint8_t rf_global_id;
    uint8_t rf_zero;
    uint8_t general_pool_start;

    char error[512];
} V3dCtx;

/* ========================================================================== */
/* Forward declarations                                                       */
/* ========================================================================== */

static SsirToV3dResult v3d_validate(V3dCtx *ctx);
static SsirToV3dResult v3d_setup(V3dCtx *ctx);
static SsirToV3dResult v3d_emit_prologue(V3dCtx *ctx);
static SsirToV3dResult v3d_emit_body(V3dCtx *ctx);
static SsirToV3dResult v3d_emit_epilogue(V3dCtx *ctx);

static SsirToV3dResult v3d_emit_block(V3dCtx *ctx, const SsirBlock *blk);
static SsirToV3dResult v3d_emit_inst(V3dCtx *ctx, const SsirInst *inst);

/* ========================================================================== */
/* Step 1: API skeleton                                                       */
/* ========================================================================== */

const char *ssir_to_v3d_result_string(SsirToV3dResult result)
{
    switch (result) {
    case SSIR_TO_V3D_OK:              return "ok";
    case SSIR_TO_V3D_ERR_INVALID_INPUT: return "invalid input";
    case SSIR_TO_V3D_ERR_UNSUPPORTED:   return "unsupported feature";
    case SSIR_TO_V3D_ERR_INTERNAL:      return "internal error";
    case SSIR_TO_V3D_ERR_OOM:           return "out of memory";
    case SSIR_TO_V3D_ERR_REG_PRESSURE:  return "register pressure exceeded";
    case SSIR_TO_V3D_ERR_CONTROL_FLOW:  return "unsupported control flow";
    }
    return "unknown error";
}

void ssir_to_v3d_free(SsirToV3dOutput *out)
{
    if (!out) return;
    free(out->instructions);
    free(out->uniforms);
    memset(out, 0, sizeof(*out));
}

/* ========================================================================== */
/* Uniform stream helpers                                                     */
/* ========================================================================== */

static int v3d_add_uniform(V3dCtx *ctx, uint32_t val)
{
    if (ctx->uniform_count >= ctx->uniform_cap) {
        uint32_t nc = ctx->uniform_cap ? ctx->uniform_cap * 2 : 32;
        uint32_t *nu = (uint32_t *)realloc(ctx->uniforms, nc * sizeof(uint32_t));
        if (!nu) return -1;
        ctx->uniforms = nu;
        ctx->uniform_cap = nc;
    }
    ctx->uniforms[ctx->uniform_count++] = val;
    return 0;
}

/* Load a 32-bit constant into an RF via ldunifrf + uniform stream. */
static int v3d_load_const_rf(V3dCtx *ctx, uint8_t rf, uint32_t val)
{
    if (v3d_add_uniform(ctx, val) < 0) return -1;
    return qpu_emit(&ctx->prog, qpu_ldunifrf(rf));
}

/* ========================================================================== */
/* Register allocator                                                         */
/* ========================================================================== */

static uint8_t alloc_rf(V3dCtx *ctx)
{
    for (uint8_t i = ctx->general_pool_start; i < ctx->max_rf; i++) {
        if (!ctx->rf_in_use[i]) {
            ctx->rf_in_use[i] = 1;
            return i;
        }
    }
    return V3D_RF_UNASSIGNED;
}

static void free_rf(V3dCtx *ctx, uint8_t rf)
{
    if (rf >= ctx->general_pool_start && rf < ctx->max_rf)
        ctx->rf_in_use[rf] = 0;
}

/*
 * Pending-free list: operand RFs that reached remaining_uses==0 but must
 * stay live until the consuming instruction is fully emitted.
 */
#define V3D_PENDING_FREE_MAX 8
static uint32_t pending_free_ids[V3D_PENDING_FREE_MAX];
static int pending_free_count = 0;

static void flush_pending_frees(V3dCtx *ctx)
{
    for (int i = 0; i < pending_free_count; i++) {
        uint32_t id = pending_free_ids[i];
        if (id < ctx->inst_map_cap && ctx->ssir_to_rf[id] != V3D_RF_UNASSIGNED) {
            uint8_t rf = ctx->ssir_to_rf[id];
            if (rf >= ctx->general_pool_start) {
                free_rf(ctx, rf);
                ctx->ssir_to_rf[id] = V3D_RF_UNASSIGNED;
            }
        }
    }
    pending_free_count = 0;
}

/* Get the RF holding an SSIR value. If constant, load via ldunifrf.
 * Frees are deferred to flush_pending_frees() after the instruction emits. */
static uint8_t get_operand_rf(V3dCtx *ctx, uint32_t ssir_id)
{
    /* Already in RF? */
    if (ssir_id < ctx->inst_map_cap && ctx->ssir_to_rf[ssir_id] != V3D_RF_UNASSIGNED) {
        uint8_t rf = ctx->ssir_to_rf[ssir_id];
        /* Decrement remaining uses; defer free if exhausted */
        if (ssir_id < ctx->inst_map_cap && ctx->remaining_uses[ssir_id] > 0) {
            ctx->remaining_uses[ssir_id]--;
            if (ctx->remaining_uses[ssir_id] == 0 && rf >= ctx->general_pool_start) {
                if (pending_free_count < V3D_PENDING_FREE_MAX)
                    pending_free_ids[pending_free_count++] = ssir_id;
            }
        }
        return rf;
    }

    /* Check if it's a constant */
    SsirConstant *c = ssir_get_constant((SsirModule *)ctx->mod, ssir_id);
    if (c) {
        uint32_t bits = 0;
        switch (c->kind) {
        case SSIR_CONST_BOOL:  bits = c->bool_val ? 0xFFFFFFFF : 0; break;
        case SSIR_CONST_I32:   bits = (uint32_t)c->i32_val; break;
        case SSIR_CONST_U32:   bits = c->u32_val; break;
        case SSIR_CONST_F32:   memcpy(&bits, &c->f32_val, 4); break;
        default:
            /* Unsupported constant kind -- load zero */
            bits = 0;
            break;
        }
        uint8_t rf = alloc_rf(ctx);
        if (rf == V3D_RF_UNASSIGNED) return rf;
        v3d_load_const_rf(ctx, rf, bits);
        ctx->ssir_to_rf[ssir_id] = rf;

        /* Decrement remaining uses; defer free */
        if (ssir_id < ctx->inst_map_cap && ctx->remaining_uses[ssir_id] > 0) {
            ctx->remaining_uses[ssir_id]--;
            if (ctx->remaining_uses[ssir_id] == 0 && rf >= ctx->general_pool_start) {
                if (pending_free_count < V3D_PENDING_FREE_MAX)
                    pending_free_ids[pending_free_count++] = ssir_id;
            }
        }
        return rf;
    }

    /* Check if it's a global variable (buffer base pointer) */
    for (uint32_t i = 0; i < ctx->binding_count; i++) {
        if (ctx->bindings[i].global_id == ssir_id)
            return ctx->bindings[i].base_rf;
    }

    /* Not found -- should not happen in well-formed SSIR */
    return V3D_RF_UNASSIGNED;
}

/* Allocate result RF for an instruction. */
static uint8_t alloc_result_rf(V3dCtx *ctx, uint32_t ssir_id)
{
    uint8_t rf = alloc_rf(ctx);
    if (rf != V3D_RF_UNASSIGNED && ssir_id < ctx->inst_map_cap)
        ctx->ssir_to_rf[ssir_id] = rf;
    return rf;
}

/* Resolve SSIR type kind for an instruction's result type. */
static SsirTypeKind v3d_result_type_kind(V3dCtx *ctx, const SsirInst *inst)
{
    SsirType *t = ssir_get_type((SsirModule *)ctx->mod, inst->type);
    if (!t) return SSIR_TYPE_VOID;
    return t->kind;
}

/* Resolve SSIR type kind for an operand. */
static SsirTypeKind v3d_operand_type_kind(V3dCtx *ctx, uint32_t ssir_id)
{
    /* Check instruction result type */
    SsirInst *inst = sw_find_inst(ctx->inst_map, ctx->inst_map_cap, ssir_id);
    if (inst) {
        SsirType *t = ssir_get_type((SsirModule *)ctx->mod, inst->type);
        if (t) return t->kind;
    }
    /* Check constant type */
    SsirConstant *c = ssir_get_constant((SsirModule *)ctx->mod, ssir_id);
    if (c) {
        SsirType *t = ssir_get_type((SsirModule *)ctx->mod, c->type);
        if (t) return t->kind;
    }
    return SSIR_TYPE_VOID;
}

/* ========================================================================== */
/* Local QPU helpers (not in qpu_asm.h)                                       */
/* ========================================================================== */

static inline uint64_t qpu_a_min(uint8_t d, uint8_t a, uint8_t b) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_MIN, 0, 63, a, b);
}
static inline uint64_t qpu_a_max(uint8_t d, uint8_t a, uint8_t b) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_MAX, 0, 63, a, b);
}
static inline uint64_t qpu_a_umin(uint8_t d, uint8_t a, uint8_t b) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_UMIN, 0, 63, a, b);
}
static inline uint64_t qpu_a_umax(uint8_t d, uint8_t a, uint8_t b) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_UMAX, 0, 63, a, b);
}
static inline uint64_t qpu_m_umul24(uint8_t d, uint8_t c, uint8_t dd) {
    return pack_alu(OP_M_UMUL24, SIG_NONE, COND_NONE, false, true,
                    d, WADDR_NOP, OP_A_NOP_BASE, c, dd, 0, 0);
}
static inline uint64_t qpu_a_ftouz(uint8_t d, uint8_t a) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_FTOUZ_BASE, 0, 63, a, OP_A_FTOUZ_RB);
}
static inline uint64_t qpu_a_utof(uint8_t d, uint8_t a) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_UTOF_BASE, 0, 63, a, 36);
}
static inline uint64_t qpu_a_ffloor(uint8_t d, uint8_t a) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_FFLOOR_BASE, 0, 63, a, OP_A_FFLOOR_RB);
}
static inline uint64_t qpu_a_fceil(uint8_t d, uint8_t a) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_FCEIL_BASE, 0, 63, a, OP_A_FCEIL_RB);
}
static inline uint64_t qpu_a_ftrunc(uint8_t d, uint8_t a) {
    return pack_alu(OP_M_NOP, SIG_NONE, COND_NONE, true, false,
                    WADDR_NOP, d, OP_A_FTRUNC_BASE, 0, 63, a, OP_A_FTRUNC_RB);
}

/* Emit 2 NOP delay slots after SFU instruction */
static int v3d_sfu_delay(V3dCtx *ctx)
{
    if (qpu_emit(&ctx->prog, qpu_nop()) < 0) return -1;
    if (qpu_emit(&ctx->prog, qpu_nop()) < 0) return -1;
    return 0;
}

/* ========================================================================== */
/* Error formatting                                                           */
/* ========================================================================== */

#define V3D_ERR(ctx, result, ...) do { \
    snprintf((ctx)->error, sizeof((ctx)->error), __VA_ARGS__); \
    return (result); \
} while (0)

#define V3D_CHECK_RF(ctx, rf) do { \
    if ((rf) == V3D_RF_UNASSIGNED) \
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_REG_PRESSURE, \
                "register pressure exceeded"); \
} while (0)

#define V3D_CHECK_EMIT(ctx, r) do { \
    if ((r) < 0) \
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "qpu_emit failed (OOM)"); \
} while (0)

/* ========================================================================== */
/* Step 1: Validation                                                         */
/* ========================================================================== */

static bool v3d_is_supported_opcode(SsirOpcode op)
{
    switch (op) {
    case SSIR_OP_ADD: case SSIR_OP_SUB: case SSIR_OP_MUL: case SSIR_OP_DIV:
    case SSIR_OP_NEG:
    case SSIR_OP_BIT_AND: case SSIR_OP_BIT_OR: case SSIR_OP_BIT_XOR:
    case SSIR_OP_BIT_NOT:
    case SSIR_OP_SHL: case SSIR_OP_SHR: case SSIR_OP_SHR_LOGICAL:
    case SSIR_OP_EQ: case SSIR_OP_NE: case SSIR_OP_LT: case SSIR_OP_LE:
    case SSIR_OP_GT: case SSIR_OP_GE:
    case SSIR_OP_AND: case SSIR_OP_OR: case SSIR_OP_NOT:
    case SSIR_OP_LOAD: case SSIR_OP_STORE: case SSIR_OP_ACCESS:
    case SSIR_OP_EXTRACT:
    case SSIR_OP_BRANCH: case SSIR_OP_BRANCH_COND:
    case SSIR_OP_PHI:
    case SSIR_OP_RETURN_VOID: case SSIR_OP_UNREACHABLE:
    case SSIR_OP_LOOP_MERGE: case SSIR_OP_SELECTION_MERGE:
    case SSIR_OP_BUILTIN: case SSIR_OP_CONVERT: case SSIR_OP_BITCAST:
        return true;
    default:
        return false;
    }
}

static bool v3d_is_supported_builtin(SsirBuiltinId id)
{
    switch (id) {
    case SSIR_BUILTIN_SIN: case SSIR_BUILTIN_COS:
    case SSIR_BUILTIN_EXP: case SSIR_BUILTIN_EXP2:
    case SSIR_BUILTIN_LOG: case SSIR_BUILTIN_LOG2:
    case SSIR_BUILTIN_POW: case SSIR_BUILTIN_SQRT:
    case SSIR_BUILTIN_INVERSESQRT:
    case SSIR_BUILTIN_ABS: case SSIR_BUILTIN_SIGN:
    case SSIR_BUILTIN_FLOOR: case SSIR_BUILTIN_CEIL: case SSIR_BUILTIN_TRUNC:
    case SSIR_BUILTIN_FRACT:
    case SSIR_BUILTIN_MIN: case SSIR_BUILTIN_MAX: case SSIR_BUILTIN_CLAMP:
    case SSIR_BUILTIN_SATURATE:
    case SSIR_BUILTIN_FMA:
        return true;
    default:
        return false;
    }
}

static bool v3d_is_supported_type(V3dCtx *ctx, uint32_t type_id)
{
    SsirType *t = ssir_get_type((SsirModule *)ctx->mod, type_id);
    if (!t) return false;
    switch (t->kind) {
    case SSIR_TYPE_I32: case SSIR_TYPE_U32: case SSIR_TYPE_F32:
    case SSIR_TYPE_BOOL: case SSIR_TYPE_VOID:
    case SSIR_TYPE_VEC:  /* allowed for builtin loads (e.g. global_invocation_id) */
        return true;
    case SSIR_TYPE_PTR:
        return t->ptr.space == SSIR_ADDR_STORAGE ||
               t->ptr.space == SSIR_ADDR_INPUT;
    case SSIR_TYPE_RUNTIME_ARRAY:
        return true;
    case SSIR_TYPE_STRUCT: {
        /* Only support structs wrapping a runtime array */
        if (t->struc.member_count == 1) {
            SsirType *mt = ssir_get_type((SsirModule *)ctx->mod, t->struc.members[0]);
            if (mt && mt->kind == SSIR_TYPE_RUNTIME_ARRAY) return true;
        }
        return false;
    }
    default:
        return false;
    }
}

static SsirToV3dResult v3d_validate(V3dCtx *ctx)
{
    /* Find compute entry point */
    const SsirModule *mod = ctx->mod;
    const SsirEntryPoint *ep = NULL;

    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        SsirEntryPoint *e = ssir_get_entry_point((SsirModule *)mod, i);
        if (e && e->stage == SSIR_STAGE_COMPUTE) {
            ep = e;
            break;
        }
    }
    if (!ep)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_INVALID_INPUT,
                "no compute entry point found");

    ctx->ep = ep;

    /* Validate workgroup size = [16, 1, 1] */
    if (ep->workgroup_size[0] != 16 ||
        ep->workgroup_size[1] != 1 ||
        ep->workgroup_size[2] != 1)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                "workgroup_size must be [16,1,1], got [%u,%u,%u]",
                ep->workgroup_size[0], ep->workgroup_size[1],
                ep->workgroup_size[2]);

    /* Get function */
    SsirFunction *fn = ssir_get_function((SsirModule *)mod, ep->function);
    if (!fn)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_INVALID_INPUT,
                "entry point function %u not found", ep->function);
    ctx->fn = fn;

    /* Count and validate bindings */
    uint32_t binding_count = 0;
    for (uint32_t i = 0; i < ep->interface_count; i++) {
        SsirGlobalVar *g = ssir_get_global((SsirModule *)mod, ep->interface[i]);
        if (!g) continue;
        if (g->builtin != SSIR_BUILTIN_NONE) continue;
        if (!g->has_group || !g->has_binding) continue;

        /* Builtin globals (e.g. global_invocation_id) have ADDR_INPUT -- skip */
        SsirType *pt = ssir_get_type((SsirModule *)mod, g->type);
        if (!pt || pt->kind != SSIR_TYPE_PTR) continue;
        if (g->builtin != SSIR_BUILTIN_NONE) continue;  /* skip builtins */

        if (pt->ptr.space != SSIR_ADDR_STORAGE)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                    "global %u: only storage buffers supported", g->id);

        binding_count++;
    }
    if (binding_count > V3D_MAX_BINDINGS)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                "too many bindings (%u > %u)", binding_count, V3D_MAX_BINDINGS);

    /* Walk all instructions: check opcodes and types */
    for (uint32_t bi = 0; bi < fn->block_count; bi++) {
        const SsirBlock *blk = &fn->blocks[bi];
        for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
            const SsirInst *inst = &blk->insts[ii];

            if (!v3d_is_supported_opcode(inst->op))
                V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                        "unsupported opcode %d (%s)",
                        inst->op, ssir_opcode_name(inst->op));

            if (inst->op == SSIR_OP_BUILTIN && inst->extra_count > 0) {
                SsirBuiltinId bid = (SsirBuiltinId)inst->extra[0];
                if (!v3d_is_supported_builtin(bid))
                    V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                            "unsupported builtin %d (%s)",
                            bid, ssir_builtin_name(bid));
            }

            if (inst->type && inst->op != SSIR_OP_BRANCH &&
                inst->op != SSIR_OP_BRANCH_COND &&
                inst->op != SSIR_OP_RETURN_VOID &&
                inst->op != SSIR_OP_UNREACHABLE &&
                inst->op != SSIR_OP_STORE &&
                inst->op != SSIR_OP_LOOP_MERGE &&
                inst->op != SSIR_OP_SELECTION_MERGE) {
                if (!v3d_is_supported_type(ctx, inst->type))
                    V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                            "unsupported result type %u for opcode %s",
                            inst->type, ssir_opcode_name(inst->op));
            }
        }
    }

    return SSIR_TO_V3D_OK;
}

/* ========================================================================== */
/* Step 2: Context setup                                                      */
/* ========================================================================== */

static int binding_cmp(const void *a, const void *b)
{
    const struct { uint32_t global_id; uint8_t base_rf; uint32_t group; uint32_t binding; }
        *ba = a, *bb = b;
    if (ba->group != bb->group) return (int)ba->group - (int)bb->group;
    return (int)ba->binding - (int)bb->binding;
}

static SsirToV3dResult v3d_setup(V3dCtx *ctx)
{
    const SsirModule *mod = ctx->mod;
    const SsirFunction *fn = ctx->fn;
    const SsirEntryPoint *ep = ctx->ep;

    /* Build inst_map */
    ctx->inst_map_cap = mod->next_id;
    ctx->inst_map = (SsirInst **)calloc(ctx->inst_map_cap, sizeof(SsirInst *));
    if (!ctx->inst_map)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "inst_map alloc failed");

    for (uint32_t bi = 0; bi < fn->block_count; bi++) {
        SsirBlock *blk = &fn->blocks[bi];
        for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
            uint32_t rid = blk->insts[ii].result;
            if (rid && rid < ctx->inst_map_cap)
                ctx->inst_map[rid] = &blk->insts[ii];
        }
    }

    /* Build use counts */
    ctx->use_counts = (uint32_t *)calloc(mod->next_id, sizeof(uint32_t));
    ctx->remaining_uses = (uint32_t *)calloc(mod->next_id, sizeof(uint32_t));
    if (!ctx->use_counts || !ctx->remaining_uses)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "use_counts alloc failed");

    ssir_count_uses((SsirFunction *)fn, ctx->use_counts, mod->next_id);

    /* Fixup: ssir_count_uses treats ACCESS extra[] as literal indices,
     * but they may be SSA values (runtime array indices). Count them. */
    for (uint32_t bi = 0; bi < fn->block_count; bi++) {
        const SsirBlock *blk = &fn->blocks[bi];
        for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
            const SsirInst *inst = &blk->insts[ii];
            if (inst->op == SSIR_OP_ACCESS && inst->extra_count > 0) {
                for (uint16_t ei = 0; ei < inst->extra_count; ei++) {
                    uint32_t eid = inst->extra[ei];
                    if (eid > 0 && eid < mod->next_id &&
                        !ssir_get_constant((SsirModule *)mod, eid))
                        ctx->use_counts[eid]++;
                }
            }
        }
    }

    memcpy(ctx->remaining_uses, ctx->use_counts,
           mod->next_id * sizeof(uint32_t));

    /* Allocate ssir_to_rf */
    ctx->ssir_to_rf = (uint8_t *)malloc(mod->next_id);
    if (!ctx->ssir_to_rf)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "ssir_to_rf alloc failed");
    memset(ctx->ssir_to_rf, V3D_RF_UNASSIGNED, mod->next_id);

    /* Scan bindings */
    ctx->binding_count = 0;
    for (uint32_t i = 0; i < ep->interface_count; i++) {
        SsirGlobalVar *g = ssir_get_global((SsirModule *)mod, ep->interface[i]);
        if (!g) continue;
        if (g->builtin != SSIR_BUILTIN_NONE) continue;
        if (!g->has_group || !g->has_binding) continue;

        SsirType *pt = ssir_get_type((SsirModule *)mod, g->type);
        if (!pt || pt->kind != SSIR_TYPE_PTR ||
            pt->ptr.space != SSIR_ADDR_STORAGE) continue;

        uint32_t bi = ctx->binding_count;
        ctx->bindings[bi].global_id = g->id;
        ctx->bindings[bi].group = g->group;
        ctx->bindings[bi].binding = g->binding;
        ctx->bindings[bi].base_rf = 0; /* assigned below */
        ctx->binding_count++;
    }

    /* Sort bindings by (group, binding) */
    if (ctx->binding_count > 1)
        qsort(ctx->bindings, ctx->binding_count,
              sizeof(ctx->bindings[0]), binding_cmp);

    /* Reserve RFs:
     *   rf[0..B-1]    = buffer base pointers
     *   rf[B]         = EIDX
     *   rf[B+1]       = global_id
     *   rf[B+2]       = zero register
     *   rf[B+3..]     = general pool
     */
    uint8_t B = (uint8_t)ctx->binding_count;
    for (uint8_t i = 0; i < B; i++) {
        ctx->bindings[i].base_rf = i;
        ctx->rf_in_use[i] = 1;
        /* Map the global's SSIR ID to this RF */
        if (ctx->bindings[i].global_id < ctx->inst_map_cap)
            ctx->ssir_to_rf[ctx->bindings[i].global_id] = i;
    }
    ctx->rf_eidx = B;
    ctx->rf_global_id = B + 1;
    ctx->rf_zero = B + 2;
    ctx->general_pool_start = B + 3;
    ctx->max_rf = V3D_MAX_RF;

    /* Mark reserved RFs in use */
    ctx->rf_in_use[ctx->rf_eidx] = 1;
    ctx->rf_in_use[ctx->rf_global_id] = 1;
    ctx->rf_in_use[ctx->rf_zero] = 1;

    return SSIR_TO_V3D_OK;
}

/* ========================================================================== */
/* Step 2: Prologue                                                           */
/* ========================================================================== */

static SsirToV3dResult v3d_emit_prologue(V3dCtx *ctx)
{
    int r;

    /* 1) ldunifrf for each buffer base address */
    for (uint32_t i = 0; i < ctx->binding_count; i++) {
        r = v3d_load_const_rf(ctx, ctx->bindings[i].base_rf, 0/*placeholder*/);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "prologue ldunifrf failed");
    }

    /* 2) TMUC init: WRTMUC signal consumes next uniform as TMU config.
     *    Bare-metal TMU protocol requires writing 0xFFFFFFFF to TMUC
     *    before any TMU operations. The WRTMUC signal automatically
     *    consumes the next uniform from the stream as the config word.
     */
    if (v3d_add_uniform(ctx, 0xFFFFFFFF) < 0)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "TMUC uniform failed");
    r = qpu_emit(&ctx->prog, qpu_nop_sig(SIG_WRTMUC));
    V3D_CHECK_EMIT(ctx, r);

    /* 3) EIDX -> rf_eidx */
    r = qpu_emit(&ctx->prog, qpu_a_eidx(ctx->rf_eidx));
    V3D_CHECK_EMIT(ctx, r);

    /* 4) global_id = TIDX * 16 + EIDX
     *    TIDX returns the thread index (dispatch_id * workgroup_size).
     *    For wg_size=16, global_id = tidx * 16 + eidx.
     *    But wait -- on V3D 7.1, TIDX already gives the QPU thread index
     *    within the workgroup. Since we have 16 lanes per QPU, and
     *    wg_size=16 means 1 QPU thread: global_id = wg_id * 16 + eidx.
     *    The TIDX instruction gives the thread index within the dispatch.
     *
     *    Actually, for bare-metal CSD: TIDX = flat thread index.
     *    global_id = TIDX * 16 + EIDX gives the per-lane global ID.
     */
    {
        uint8_t tmp_tidx = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, tmp_tidx);
        r = qpu_emit(&ctx->prog, qpu_a_tidx(tmp_tidx));
        V3D_CHECK_EMIT(ctx, r);

        /* shl tmp_tidx, tmp_tidx, 4  (multiply by 16) */
        r = qpu_emit(&ctx->prog, qpu_a_shl_imm(tmp_tidx, tmp_tidx, 4));
        V3D_CHECK_EMIT(ctx, r);

        /* add global_id, tmp_tidx, eidx */
        r = qpu_emit(&ctx->prog, qpu_a_add(ctx->rf_global_id, tmp_tidx, ctx->rf_eidx));
        V3D_CHECK_EMIT(ctx, r);

        free_rf(ctx, tmp_tidx);
    }

    /* 5) Zero register: xor rf_zero, rf_eidx, rf_eidx */
    r = qpu_emit(&ctx->prog, qpu_a_xor(ctx->rf_zero, ctx->rf_eidx, ctx->rf_eidx));
    V3D_CHECK_EMIT(ctx, r);

    return SSIR_TO_V3D_OK;
}

/* ========================================================================== */
/* Step 2: Epilogue                                                           */
/* ========================================================================== */

static SsirToV3dResult v3d_emit_epilogue(V3dCtx *ctx)
{
    int r = qpu_thrsw_end(&ctx->prog);
    if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "epilogue failed");
    return SSIR_TO_V3D_OK;
}

/* ========================================================================== */
/* Steps 3-7: Instruction emission                                            */
/* ========================================================================== */

/* ----------- Step 3: Integer arithmetic ----------- */

static SsirToV3dResult v3d_emit_add(V3dCtx *ctx, const SsirInst *inst)
{
    SsirTypeKind tk = v3d_result_type_kind(ctx, inst);
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;
    if (tk == SSIR_TYPE_F32)
        r = qpu_emit(&ctx->prog, qpu_a_fadd(d, a, b));
    else
        r = qpu_emit(&ctx->prog, qpu_a_add(d, a, b));
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

static SsirToV3dResult v3d_emit_sub(V3dCtx *ctx, const SsirInst *inst)
{
    SsirTypeKind tk = v3d_result_type_kind(ctx, inst);
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;
    if (tk == SSIR_TYPE_F32)
        r = qpu_emit(&ctx->prog, qpu_a_fsub(d, a, b));
    else
        r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

static SsirToV3dResult v3d_emit_mul(V3dCtx *ctx, const SsirInst *inst)
{
    SsirTypeKind tk = v3d_result_type_kind(ctx, inst);
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;
    if (tk == SSIR_TYPE_F32)
        r = qpu_emit(&ctx->prog, qpu_m_fmul(d, a, b));
    else
        r = qpu_emit(&ctx->prog, qpu_m_umul24(d, a, b));
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

static SsirToV3dResult v3d_emit_div(V3dCtx *ctx, const SsirInst *inst)
{
    SsirTypeKind tk = v3d_result_type_kind(ctx, inst);
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    if (tk == SSIR_TYPE_F32) {
        /* recip(b), 2 NOPs, fmul(a, recip_result) */
        int r;
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        r = qpu_emit(&ctx->prog, qpu_a_recip(t, b));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");

        r = qpu_emit(&ctx->prog, qpu_m_fmul(d, a, t));
        V3D_CHECK_EMIT(ctx, r);

        free_rf(ctx, t);
    } else {
        /* Integer division not directly supported; approximate via f32:
         * itof(a), itof(b), recip(b_f), 2 NOP, fmul, ftoiz */
        int r;
        uint8_t af = alloc_rf(ctx);
        uint8_t bf = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, af); V3D_CHECK_RF(ctx, bf);

        if (tk == SSIR_TYPE_U32) {
            r = qpu_emit(&ctx->prog, qpu_a_utof(af, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_utof(bf, b));
            V3D_CHECK_EMIT(ctx, r);
        } else {
            r = qpu_emit(&ctx->prog, qpu_a_itof(af, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_itof(bf, b));
            V3D_CHECK_EMIT(ctx, r);
        }

        r = qpu_emit(&ctx->prog, qpu_a_recip(bf, bf));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");

        r = qpu_emit(&ctx->prog, qpu_m_fmul(af, af, bf));
        V3D_CHECK_EMIT(ctx, r);

        if (tk == SSIR_TYPE_U32)
            r = qpu_emit(&ctx->prog, qpu_a_ftouz(d, af));
        else
            r = qpu_emit(&ctx->prog, qpu_a_ftoiz(d, af));
        V3D_CHECK_EMIT(ctx, r);

        free_rf(ctx, af);
        free_rf(ctx, bf);
    }

    return SSIR_TO_V3D_OK;
}

static SsirToV3dResult v3d_emit_neg(V3dCtx *ctx, const SsirInst *inst)
{
    SsirTypeKind tk = v3d_result_type_kind(ctx, inst);
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    V3D_CHECK_RF(ctx, a);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;
    if (tk == SSIR_TYPE_F32)
        r = qpu_emit(&ctx->prog, qpu_a_fsub(d, ctx->rf_zero, a));
    else
        r = qpu_emit(&ctx->prog, qpu_a_neg(d, a));
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

/* Bitwise ops */
static SsirToV3dResult v3d_emit_bitwise(V3dCtx *ctx, const SsirInst *inst)
{
    int r;
    if (inst->op == SSIR_OP_BIT_NOT) {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_not(d, a));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }

    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);
    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    switch (inst->op) {
    case SSIR_OP_BIT_AND: r = qpu_emit(&ctx->prog, qpu_a_and(d, a, b)); break;
    case SSIR_OP_BIT_OR:  r = qpu_emit(&ctx->prog, qpu_a_or(d, a, b));  break;
    case SSIR_OP_BIT_XOR: r = qpu_emit(&ctx->prog, qpu_a_xor(d, a, b)); break;
    default: V3D_ERR(ctx, SSIR_TO_V3D_ERR_INTERNAL, "bad bitwise op");
    }
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

/* Shifts */
static SsirToV3dResult v3d_emit_shift(V3dCtx *ctx, const SsirInst *inst)
{
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);
    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;
    switch (inst->op) {
    case SSIR_OP_SHL:         r = qpu_emit(&ctx->prog, qpu_a_shl(d, a, b)); break;
    case SSIR_OP_SHR:         r = qpu_emit(&ctx->prog, qpu_a_asr(d, a, b)); break;
    case SSIR_OP_SHR_LOGICAL: r = qpu_emit(&ctx->prog, qpu_a_shr(d, a, b)); break;
    default: V3D_ERR(ctx, SSIR_TO_V3D_ERR_INTERNAL, "bad shift op");
    }
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

/* ----------- Step 3: Memory access ----------- */

/* ACCESS: compute &base[indices...].
 * SSIR ACCESS: operands[0]=base, extra[]=indices.
 * For struct{runtime_array<T>}: extra[0]=member_idx (const 0, no-op for offset 0),
 *                                extra[1]=array_idx (runtime).
 * We only need the last index for address computation: base + idx * 4. */
static SsirToV3dResult v3d_emit_access(V3dCtx *ctx, const SsirInst *inst)
{
    uint8_t base_rf = get_operand_rf(ctx, inst->operands[0]);
    V3D_CHECK_RF(ctx, base_rf);

    /* No indices -> just alias the base pointer */
    if (inst->extra_count == 0) {
        if (inst->result < ctx->inst_map_cap)
            ctx->ssir_to_rf[inst->result] = base_rf;
        return SSIR_TO_V3D_OK;
    }

    /* Find the dynamic array index.
     * For struct wrappers: extra[0] = struct member index (const 0 -> offset 0, skip).
     *                      extra[1] = array element index (runtime).
     * For plain arrays:    extra[0] = array element index. */
    uint32_t idx_id = 0;
    int found_dynamic = 0;
    for (uint16_t i = 0; i < inst->extra_count; i++) {
        uint32_t eid = inst->extra[i];
        SsirConstant *c = ssir_get_constant((SsirModule *)ctx->mod, eid);
        if (c) {
            /* Static index -- for struct member offset 0, skip.
             * For nonzero offsets, would need to add static offset. */
            continue;
        }
        /* Dynamic index */
        idx_id = eid;
        found_dynamic = 1;
    }

    if (!found_dynamic) {
        /* All indices static (e.g. struct member 0 only) -> alias base */
        if (inst->result < ctx->inst_map_cap)
            ctx->ssir_to_rf[inst->result] = base_rf;
        return SSIR_TO_V3D_OK;
    }

    uint8_t idx_rf = get_operand_rf(ctx, idx_id);
    V3D_CHECK_RF(ctx, idx_rf);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    /* shl(t, idx, 2) -> idx * 4 */
    uint8_t t = alloc_rf(ctx);
    V3D_CHECK_RF(ctx, t);

    int r;
    r = qpu_emit(&ctx->prog, qpu_a_shl_imm(t, idx_rf, 2));
    V3D_CHECK_EMIT(ctx, r);

    /* add(d, base, t) */
    r = qpu_emit(&ctx->prog, qpu_a_add(d, base_rf, t));
    V3D_CHECK_EMIT(ctx, r);

    free_rf(ctx, t);
    return SSIR_TO_V3D_OK;
}

/* Check if an SSIR ID is a builtin global (global_invocation_id etc.) */
static SsirBuiltinVar v3d_get_builtin_var(V3dCtx *ctx, uint32_t id)
{
    SsirGlobalVar *g = ssir_get_global((SsirModule *)ctx->mod, id);
    if (g) return g->builtin;
    return SSIR_BUILTIN_NONE;
}

/* LOAD: TMU read or builtin load.
 * Builtin globals (global_invocation_id) -> use pre-computed RF.
 * Storage buffer pointers -> TMU read: mov tmua, addr -> nop -> ldtmu dst */
static SsirToV3dResult v3d_emit_load(V3dCtx *ctx, const SsirInst *inst)
{
    uint32_t ptr_id = inst->operands[0];

    /* Check if this is loading a builtin global */
    SsirBuiltinVar bv = v3d_get_builtin_var(ctx, ptr_id);
    if (bv == SSIR_BUILTIN_GLOBAL_INVOCATION_ID) {
        /* The result is a vec3<u32>. We'll map it so that EXTRACT .x
         * picks up rf_global_id. For now, just assign rf_global_id. */
        if (inst->result < ctx->inst_map_cap)
            ctx->ssir_to_rf[inst->result] = ctx->rf_global_id;
        return SSIR_TO_V3D_OK;
    }

    /* Regular TMU load */
    uint8_t addr = get_operand_rf(ctx, ptr_id);
    V3D_CHECK_RF(ctx, addr);

    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;
    r = qpu_emit(&ctx->prog, qpu_tmu_read_addr(addr));
    V3D_CHECK_EMIT(ctx, r);
    r = qpu_emit(&ctx->prog, qpu_nop());
    V3D_CHECK_EMIT(ctx, r);
    r = qpu_emit(&ctx->prog, qpu_ldtmu(d));
    V3D_CHECK_EMIT(ctx, r);

    return SSIR_TO_V3D_OK;
}

/* STORE: TMU write. mov tmud, data -> mov tmua, addr -> tmuwt */
static SsirToV3dResult v3d_emit_store(V3dCtx *ctx, const SsirInst *inst)
{
    /* operands[0] = pointer, operands[1] = value */
    uint8_t addr = get_operand_rf(ctx, inst->operands[0]);
    uint8_t data = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, addr); V3D_CHECK_RF(ctx, data);

    int r;
    r = qpu_emit(&ctx->prog, qpu_tmu_write_data(data));
    V3D_CHECK_EMIT(ctx, r);
    r = qpu_emit(&ctx->prog, qpu_tmu_write_addr(addr));
    V3D_CHECK_EMIT(ctx, r);
    r = qpu_emit(&ctx->prog, qpu_a_tmuwt());
    V3D_CHECK_EMIT(ctx, r);

    return SSIR_TO_V3D_OK;
}

/* ----------- Step 4: Builtins ----------- */

static SsirToV3dResult v3d_emit_builtin(V3dCtx *ctx, const SsirInst *inst)
{
    if (inst->extra_count < 1)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_INTERNAL, "builtin missing extra[0]");

    SsirBuiltinId bid = (SsirBuiltinId)inst->extra[0];
    SsirTypeKind tk = v3d_result_type_kind(ctx, inst);
    int r;

    switch (bid) {

    /* ------- SFU operations ------- */
    case SSIR_BUILTIN_INVERSESQRT: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_rsqrt(d, a));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_SQRT: {
        /* rsqrt(t, a) + 2 NOP + fmul(d, a, t) */
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        r = qpu_emit(&ctx->prog, qpu_a_rsqrt(t, a));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");
        /* recip of rsqrt: 1/rsqrt(a) = sqrt(a). But we have rsqrt.
         * sqrt(a) = a * rsqrt(a) */
        r = qpu_emit(&ctx->prog, qpu_m_fmul(d, a, t));
        V3D_CHECK_EMIT(ctx, r);
        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_EXP2: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_exp(d, a));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_LOG2: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_log(d, a));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");
        return SSIR_TO_V3D_OK;
    }

    /* ------- Trig: V3D SIN computes sin(x * pi) ------- */
    case SSIR_BUILTIN_SIN: {
        /* sin(x) = v3d_sin(x / pi) = v3d_sin(x * (1/pi)) */
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        /* 1/pi = 0x3EA2F983 */
        r = v3d_load_const_rf(ctx, t, 0x3EA2F983);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "sin const failed");

        r = qpu_emit(&ctx->prog, qpu_m_fmul(t, a, t));
        V3D_CHECK_EMIT(ctx, r);
        r = qpu_emit(&ctx->prog, qpu_a_sin(d, t));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");
        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_COS: {
        /* cos(x) = sin(x + pi/2) = v3d_sin((x + pi/2) / pi)
         *        = v3d_sin(x/pi + 0.5) */
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t1 = alloc_rf(ctx);
        uint8_t t2 = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t1); V3D_CHECK_RF(ctx, t2);

        /* 1/pi */
        r = v3d_load_const_rf(ctx, t1, 0x3EA2F983);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "cos 1/pi failed");
        r = qpu_emit(&ctx->prog, qpu_m_fmul(t1, a, t1));
        V3D_CHECK_EMIT(ctx, r);

        /* 0.5f */
        r = v3d_load_const_rf(ctx, t2, 0x3F000000);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "cos 0.5 failed");
        r = qpu_emit(&ctx->prog, qpu_a_fadd(t1, t1, t2));
        V3D_CHECK_EMIT(ctx, r);

        r = qpu_emit(&ctx->prog, qpu_a_sin(d, t1));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");

        free_rf(ctx, t1);
        free_rf(ctx, t2);
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_EXP: {
        /* exp(x) = exp2(x * log2(e))
         * log2(e) = 1.4426950408.. = 0x3FB8AA3B */
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        r = v3d_load_const_rf(ctx, t, 0x3FB8AA3B);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "exp log2e failed");
        r = qpu_emit(&ctx->prog, qpu_m_fmul(t, a, t));
        V3D_CHECK_EMIT(ctx, r);
        r = qpu_emit(&ctx->prog, qpu_a_exp(d, t));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");
        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_LOG: {
        /* log(x) = log2(x) * ln(2)
         * ln(2) = 0.693147.. = 0x3F317218 */
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        r = qpu_emit(&ctx->prog, qpu_a_log(t, a));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");

        uint8_t c = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, c);
        r = v3d_load_const_rf(ctx, c, 0x3F317218);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "log ln2 failed");
        r = qpu_emit(&ctx->prog, qpu_m_fmul(d, t, c));
        V3D_CHECK_EMIT(ctx, r);
        free_rf(ctx, t);
        free_rf(ctx, c);
        return SSIR_TO_V3D_OK;
    }

    case SSIR_BUILTIN_POW: {
        /* pow(a,b) = exp2(b * log2(a)) */
        uint8_t va = get_operand_rf(ctx, inst->operands[0]);
        uint8_t vb = get_operand_rf(ctx, inst->operands[1]);
        V3D_CHECK_RF(ctx, va); V3D_CHECK_RF(ctx, vb);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        /* log2(a) */
        r = qpu_emit(&ctx->prog, qpu_a_log(t, va));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");

        /* t = b * log2(a) */
        r = qpu_emit(&ctx->prog, qpu_m_fmul(t, vb, t));
        V3D_CHECK_EMIT(ctx, r);

        /* exp2(t) */
        r = qpu_emit(&ctx->prog, qpu_a_exp(d, t));
        V3D_CHECK_EMIT(ctx, r);
        if (v3d_sfu_delay(ctx) < 0)
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "SFU delay failed");

        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    /* ------- ABS ------- */
    case SSIR_BUILTIN_ABS: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);

        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        if (tk == SSIR_TYPE_F32) {
            /* abs(x) = fmax(x, -x) */
            r = qpu_emit(&ctx->prog, qpu_a_fsub(t, ctx->rf_zero, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_fmax(d, a, t));
            V3D_CHECK_EMIT(ctx, r);
        } else {
            /* abs(x) = max(x, -x) for signed */
            r = qpu_emit(&ctx->prog, qpu_a_neg(t, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_max(d, a, t));
            V3D_CHECK_EMIT(ctx, r);
        }
        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    /* ------- SIGN ------- */
    case SSIR_BUILTIN_SIGN: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);

        if (tk == SSIR_TYPE_F32) {
            /*
             * sign(x):
             *   neg_mask = asr(x, 31)           -- ~0 if x<0 (IEEE754 sign bit)
             *   neg_x    = fsub(zero, x)
             *   pos_mask = asr(neg_x, 31)       -- ~0 if x>0
             *   Load 1.0 (0x3F800000), -1.0 (0xBF800000)
             *   pos_val  = and(pos_mask, 0x3F800000)  -- 1.0 bits if positive
             *   neg_val  = and(neg_mask, 0xBF800000)  -- -1.0 bits if negative
             *   result   = or(pos_val, neg_val)
             */
            uint8_t neg_mask = alloc_rf(ctx);
            uint8_t neg_x = alloc_rf(ctx);
            uint8_t pos_mask = alloc_rf(ctx);
            uint8_t c1 = alloc_rf(ctx);
            uint8_t c2 = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, neg_mask); V3D_CHECK_RF(ctx, neg_x);
            V3D_CHECK_RF(ctx, pos_mask); V3D_CHECK_RF(ctx, c1);
            V3D_CHECK_RF(ctx, c2);

            /* Shift count 31 -- load via uniform */
            uint8_t s31 = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, s31);
            r = v3d_load_const_rf(ctx, s31, 31);
            if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "sign const failed");

            r = qpu_emit(&ctx->prog, qpu_a_asr(neg_mask, a, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_fsub(neg_x, ctx->rf_zero, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(pos_mask, neg_x, s31));
            V3D_CHECK_EMIT(ctx, r);

            r = v3d_load_const_rf(ctx, c1, 0x3F800000); /* 1.0 bits */
            if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "sign 1.0 failed");
            r = v3d_load_const_rf(ctx, c2, 0xBF800000); /* -1.0 bits */
            if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "sign -1.0 failed");

            r = qpu_emit(&ctx->prog, qpu_a_and(c1, pos_mask, c1));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_and(c2, neg_mask, c2));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, c1, c2));
            V3D_CHECK_EMIT(ctx, r);

            free_rf(ctx, neg_mask); free_rf(ctx, neg_x);
            free_rf(ctx, pos_mask); free_rf(ctx, c1);
            free_rf(ctx, c2); free_rf(ctx, s31);
        } else {
            /* Integer sign: asr(x, 31) gives -1 or 0.
             * neg(-x, 31) gives -1 or 0 for the positive side.
             * sign = asr(neg_x, 31) + asr(x, 31)
             * Wait: asr(x,31) = -1 if x<0, 0 if x>=0
             *       asr(-x,31) = -1 if x>0, 0 if x<=0
             * asr(-x,31) - asr(x,31):
             *   x>0: -1 - 0 = -1  -- wrong, want +1
             * Use: -asr(-x,31) + asr(x,31)? no...
             * Correct: sign(x) = (x>0) - (x<0)
             *   neg_part = asr(x, 31)       -> -1 or 0
             *   neg_x    = neg(x)
             *   pos_part = asr(neg_x, 31)   -> -1 or 0
             *   result   = sub(pos_part, neg_part) -- negate both: -(-1)-(-1)=0? No.
             *   x>0: neg_part=0, pos_part=-1. sub(-1, 0) = -1 -- still wrong.
             *
             * Fix: negate pos_part: sign = neg(pos_part) + neg_part
             *   x>0: neg(-1) + 0 = 1. Correct.
             *   x<0: neg(0) + (-1) = -1. Correct.
             *   x=0: neg(0) + 0 = 0. Correct.
             * So: sign = sub(neg_part, pos_part) -- sub(asr(x,31), asr(neg_x,31))
             * Wait: sub(0, -1) = 1 for x>0. sub(-1, 0) = -1 for x<0. sub(0,0) = 0. Yes!
             */
            uint8_t t1 = alloc_rf(ctx);
            uint8_t t2 = alloc_rf(ctx);
            uint8_t s31 = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, t1); V3D_CHECK_RF(ctx, t2); V3D_CHECK_RF(ctx, s31);

            r = v3d_load_const_rf(ctx, s31, 31);
            if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "sign s31 failed");

            r = qpu_emit(&ctx->prog, qpu_a_asr(t1, a, s31));  /* neg_part */
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(t2, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(t2, t2, s31)); /* pos_part */
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, t1, t2));   /* neg_part - pos_part */
            V3D_CHECK_EMIT(ctx, r);

            free_rf(ctx, t1); free_rf(ctx, t2); free_rf(ctx, s31);
        }
        return SSIR_TO_V3D_OK;
    }

    /* ------- Floor / Ceil / Trunc ------- */
    case SSIR_BUILTIN_FLOOR: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_ffloor(d, a));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }
    case SSIR_BUILTIN_CEIL: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_fceil(d, a));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }
    case SSIR_BUILTIN_TRUNC: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_ftrunc(d, a));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }

    /* ------- FRACT(x) = x - floor(x) ------- */
    case SSIR_BUILTIN_FRACT: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        r = qpu_emit(&ctx->prog, qpu_a_ffloor(t, a));
        V3D_CHECK_EMIT(ctx, r);
        r = qpu_emit(&ctx->prog, qpu_a_fsub(d, a, t));
        V3D_CHECK_EMIT(ctx, r);
        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    /* ------- MIN / MAX ------- */
    case SSIR_BUILTIN_MIN: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        uint8_t b = get_operand_rf(ctx, inst->operands[1]);
        V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);

        if (tk == SSIR_TYPE_F32)
            r = qpu_emit(&ctx->prog, qpu_a_fmin(d, a, b));
        else if (tk == SSIR_TYPE_U32)
            r = qpu_emit(&ctx->prog, qpu_a_umin(d, a, b));
        else
            r = qpu_emit(&ctx->prog, qpu_a_min(d, a, b));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }
    case SSIR_BUILTIN_MAX: {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        uint8_t b = get_operand_rf(ctx, inst->operands[1]);
        V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);

        if (tk == SSIR_TYPE_F32)
            r = qpu_emit(&ctx->prog, qpu_a_fmax(d, a, b));
        else if (tk == SSIR_TYPE_U32)
            r = qpu_emit(&ctx->prog, qpu_a_umax(d, a, b));
        else
            r = qpu_emit(&ctx->prog, qpu_a_max(d, a, b));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }

    /* ------- CLAMP(x, lo, hi) = max(lo, min(x, hi)) ------- */
    case SSIR_BUILTIN_CLAMP: {
        uint8_t x  = get_operand_rf(ctx, inst->operands[0]);
        uint8_t lo = get_operand_rf(ctx, inst->operands[1]);
        uint8_t hi = get_operand_rf(ctx, inst->operands[2]);
        V3D_CHECK_RF(ctx, x); V3D_CHECK_RF(ctx, lo); V3D_CHECK_RF(ctx, hi);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        if (tk == SSIR_TYPE_F32) {
            r = qpu_emit(&ctx->prog, qpu_a_fmax(t, x, lo));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_fmin(d, t, hi));
            V3D_CHECK_EMIT(ctx, r);
        } else if (tk == SSIR_TYPE_U32) {
            r = qpu_emit(&ctx->prog, qpu_a_umax(t, x, lo));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_umin(d, t, hi));
            V3D_CHECK_EMIT(ctx, r);
        } else {
            r = qpu_emit(&ctx->prog, qpu_a_max(t, x, lo));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_min(d, t, hi));
            V3D_CHECK_EMIT(ctx, r);
        }
        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    /* ------- SATURATE(x) = clamp(x, 0.0, 1.0) ------- */
    case SSIR_BUILTIN_SATURATE: {
        uint8_t x = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, x);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        uint8_t c1 = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t); V3D_CHECK_RF(ctx, c1);

        /* fmax(t, x, 0.0) -- rf_zero holds 0 which is also float 0.0 */
        r = qpu_emit(&ctx->prog, qpu_a_fmax(t, x, ctx->rf_zero));
        V3D_CHECK_EMIT(ctx, r);

        /* Load 1.0 */
        r = v3d_load_const_rf(ctx, c1, 0x3F800000);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "saturate 1.0 failed");

        r = qpu_emit(&ctx->prog, qpu_a_fmin(d, t, c1));
        V3D_CHECK_EMIT(ctx, r);

        free_rf(ctx, t); free_rf(ctx, c1);
        return SSIR_TO_V3D_OK;
    }

    /* ------- FMA(a, b, c) = a*b + c ------- */
    case SSIR_BUILTIN_FMA: {
        uint8_t va = get_operand_rf(ctx, inst->operands[0]);
        uint8_t vb = get_operand_rf(ctx, inst->operands[1]);
        uint8_t vc = get_operand_rf(ctx, inst->operands[2]);
        V3D_CHECK_RF(ctx, va); V3D_CHECK_RF(ctx, vb); V3D_CHECK_RF(ctx, vc);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        uint8_t t = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, t);

        r = qpu_emit(&ctx->prog, qpu_m_fmul(t, va, vb));
        V3D_CHECK_EMIT(ctx, r);
        r = qpu_emit(&ctx->prog, qpu_a_fadd(d, t, vc));
        V3D_CHECK_EMIT(ctx, r);

        free_rf(ctx, t);
        return SSIR_TO_V3D_OK;
    }

    default:
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                "unsupported builtin %d in codegen", bid);
    }
}

/* ----------- Step 4: Conversions ----------- */

static SsirToV3dResult v3d_emit_convert(V3dCtx *ctx, const SsirInst *inst)
{
    SsirTypeKind dst_tk = v3d_result_type_kind(ctx, inst);
    SsirTypeKind src_tk = v3d_operand_type_kind(ctx, inst->operands[0]);

    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    V3D_CHECK_RF(ctx, a);
    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    int r;

    if (src_tk == SSIR_TYPE_F32 && dst_tk == SSIR_TYPE_I32) {
        r = qpu_emit(&ctx->prog, qpu_a_ftoiz(d, a));
    } else if (src_tk == SSIR_TYPE_F32 && dst_tk == SSIR_TYPE_U32) {
        r = qpu_emit(&ctx->prog, qpu_a_ftouz(d, a));
    } else if (src_tk == SSIR_TYPE_I32 && dst_tk == SSIR_TYPE_F32) {
        r = qpu_emit(&ctx->prog, qpu_a_itof(d, a));
    } else if (src_tk == SSIR_TYPE_U32 && dst_tk == SSIR_TYPE_F32) {
        r = qpu_emit(&ctx->prog, qpu_a_utof(d, a));
    } else if (src_tk == SSIR_TYPE_BOOL && dst_tk == SSIR_TYPE_I32) {
        /* bool (~0 or 0) -> i32 (1 or 0): and with 1 */
        uint8_t c1 = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, c1);
        r = v3d_load_const_rf(ctx, c1, 1);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "bool2int failed");
        r = qpu_emit(&ctx->prog, qpu_a_and(d, a, c1));
        free_rf(ctx, c1);
    } else if (src_tk == SSIR_TYPE_BOOL && dst_tk == SSIR_TYPE_U32) {
        uint8_t c1 = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, c1);
        r = v3d_load_const_rf(ctx, c1, 1);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "bool2uint failed");
        r = qpu_emit(&ctx->prog, qpu_a_and(d, a, c1));
        free_rf(ctx, c1);
    } else if (src_tk == SSIR_TYPE_I32 && dst_tk == SSIR_TYPE_U32) {
        /* Reinterpret -- just mov */
        r = qpu_emit(&ctx->prog, qpu_a_mov(d, a));
    } else if (src_tk == SSIR_TYPE_U32 && dst_tk == SSIR_TYPE_I32) {
        r = qpu_emit(&ctx->prog, qpu_a_mov(d, a));
    } else if (src_tk == SSIR_TYPE_BOOL && dst_tk == SSIR_TYPE_F32) {
        /* bool -> f32: select 1.0 or 0.0 */
        uint8_t c1 = alloc_rf(ctx);
        V3D_CHECK_RF(ctx, c1);
        r = v3d_load_const_rf(ctx, c1, 0x3F800000);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "bool2f32 failed");
        r = qpu_emit(&ctx->prog, qpu_a_and(d, a, c1));
        free_rf(ctx, c1);
    } else {
        /* Fallback: mov */
        r = qpu_emit(&ctx->prog, qpu_a_mov(d, a));
    }
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

/* ----------- Step 5: Comparisons ----------- */

/*
 * All comparisons produce ~0 (true) or 0 (false) bitmask.
 *
 * Signed i32 comparisons:
 *   LT(a,b): sub(d,a,b), asr(d,d,31)
 *   GT(a,b): sub(d,b,a), asr(d,d,31)
 *   LE(a,b): GT then NOT
 *   GE(a,b): LT then NOT
 *   EQ(a,b): sub(d,a,b), neg(n,d), or(d,d,n), asr(d,d,31), not(d,d)
 *   NE(a,b): sub(d,a,b), neg(n,d), or(d,d,n), asr(d,d,31)
 *
 * Float f32:
 *   FLT: fsub then asr
 *   FGT: fsub(b,a) then asr
 *   FEQ, FNE: similar to integer but with fsub
 *
 * Unsigned u32:
 *   ULT(a,b): sub(t,b,a), and negate/or trick -- but must handle wrap.
 *   Use: umax(t,a,b), sub(d,t,a), neg(n,d), or(d,d,n), asr(d,d,31)
 *   This checks if a < b by seeing if umax(a,b) != a.
 */

static SsirToV3dResult v3d_emit_cmp(V3dCtx *ctx, const SsirInst *inst)
{
    /* Determine operand type (look at operand, not result which is bool) */
    SsirTypeKind op_tk = v3d_operand_type_kind(ctx, inst->operands[0]);

    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);
    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);
    int r;

    /* We need a shift-by-31 register */
    uint8_t s31 = alloc_rf(ctx);
    V3D_CHECK_RF(ctx, s31);
    r = v3d_load_const_rf(ctx, s31, 31);
    if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "cmp s31 failed");

    if (op_tk == SSIR_TYPE_F32) {
        /* Float comparisons */
        switch (inst->op) {
        case SSIR_OP_LT: {
            r = qpu_emit(&ctx->prog, qpu_a_fsub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_GT: {
            r = qpu_emit(&ctx->prog, qpu_a_fsub(d, b, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_LE: {
            r = qpu_emit(&ctx->prog, qpu_a_fsub(d, b, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_GE: {
            r = qpu_emit(&ctx->prog, qpu_a_fsub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_EQ: {
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_fsub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, n);
            break;
        }
        case SSIR_OP_NE: {
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_fsub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, n);
            break;
        }
        default:
            free_rf(ctx, s31);
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_INTERNAL, "bad float cmp op");
        }
    } else if (op_tk == SSIR_TYPE_U32) {
        /* Unsigned comparisons using UMAX trick */
        switch (inst->op) {
        case SSIR_OP_LT: {
            /* ULT(a,b): t=umax(a,b), d=sub(t,a), n=neg(d), d=or(d,n), d=asr(d,31) */
            uint8_t t = alloc_rf(ctx);
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, t); V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_umax(t, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, t, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, t); free_rf(ctx, n);
            break;
        }
        case SSIR_OP_GT: {
            /* UGT(a,b) = ULT(b,a) */
            uint8_t t = alloc_rf(ctx);
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, t); V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_umax(t, b, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, t, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, t); free_rf(ctx, n);
            break;
        }
        case SSIR_OP_LE: {
            /* ULE(a,b) = NOT(UGT(a,b)) */
            uint8_t t = alloc_rf(ctx);
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, t); V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_umax(t, b, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, t, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, t); free_rf(ctx, n);
            break;
        }
        case SSIR_OP_GE: {
            /* UGE(a,b) = NOT(ULT(a,b)) */
            uint8_t t = alloc_rf(ctx);
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, t); V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_umax(t, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, t, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, t); free_rf(ctx, n);
            break;
        }
        case SSIR_OP_EQ: {
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, n);
            break;
        }
        case SSIR_OP_NE: {
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, n);
            break;
        }
        default:
            free_rf(ctx, s31);
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_INTERNAL, "bad u32 cmp op");
        }
    } else {
        /* Signed i32 comparisons */
        switch (inst->op) {
        case SSIR_OP_LT: {
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_GT: {
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, b, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_LE: {
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, b, a));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_GE: {
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            break;
        }
        case SSIR_OP_EQ: {
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(d, d));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, n);
            break;
        }
        case SSIR_OP_NE: {
            uint8_t n = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, n);
            r = qpu_emit(&ctx->prog, qpu_a_sub(d, a, b));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_neg(n, d));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, d, n));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_asr(d, d, s31));
            V3D_CHECK_EMIT(ctx, r);
            free_rf(ctx, n);
            break;
        }
        default:
            free_rf(ctx, s31);
            V3D_ERR(ctx, SSIR_TO_V3D_ERR_INTERNAL, "bad i32 cmp op");
        }
    }

    free_rf(ctx, s31);
    return SSIR_TO_V3D_OK;
}

/* Logical AND / OR / NOT */
static SsirToV3dResult v3d_emit_logical(V3dCtx *ctx, const SsirInst *inst)
{
    int r;
    if (inst->op == SSIR_OP_NOT) {
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        r = qpu_emit(&ctx->prog, qpu_a_not(d, a));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }

    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    uint8_t b = get_operand_rf(ctx, inst->operands[1]);
    V3D_CHECK_RF(ctx, a); V3D_CHECK_RF(ctx, b);
    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);

    if (inst->op == SSIR_OP_AND)
        r = qpu_emit(&ctx->prog, qpu_a_and(d, a, b));
    else
        r = qpu_emit(&ctx->prog, qpu_a_or(d, a, b));
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

/* ----------- Step 5: If/else predication ----------- */

/*
 * Predication model:
 *   SELECTION_MERGE marks the merge block ID.
 *   BRANCH_COND(cond, true_block, false_block) follows.
 *   We emit both branches sequentially. At PHI nodes in the merge block,
 *   we select: result = (cond & true_val) | (~cond & false_val).
 *
 * For straightline code (no structured if/else in the block sequence),
 * we handle SELECTION_MERGE and BRANCH_COND as no-ops (the blocks get
 * emitted in order anyway), and handle PHI at merge block entry.
 */

static SsirToV3dResult v3d_emit_phi(V3dCtx *ctx, const SsirInst *inst)
{
    /*
     * PHI node: operands are [value0, block0, value1, block1, ...]
     * For if/else predication, we need the condition mask.
     *
     * Simplified approach: since we emit blocks sequentially and the last
     * branch was predicated, we just pick the most recently computed value.
     * This works for trivially scheduled code.
     *
     * For full correctness with structured control flow:
     * Walk operands in pairs (value, block). Find which operand corresponds
     * to the true-branch and which to the false-branch by checking which
     * block IDs they reference. Then use the predication mask to select.
     *
     * If only one value is already in an RF, just use that (this covers
     * the case where one branch was taken and its result is the live one).
     *
     * General approach: for each PHI with 2 sources, emit a select.
     */
    if (inst->operand_count < 2) {
        /* Degenerate PHI -- just copy the single value */
        uint8_t a = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, a);
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        int r = qpu_emit(&ctx->prog, qpu_a_mov(d, a));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }

    if (inst->operand_count >= 4) {
        /* Standard 2-source PHI: operands = [val0, blk0, val1, blk1]
         * We need a condition mask. Look for the BRANCH_COND that produced
         * the divergence. For predicated execution, the condition RF should
         * still be live.
         *
         * Practical approach: try to find the condition RF. If we can't,
         * fall back to using the first value that has an RF assigned.
         */
        uint32_t val0_id = inst->operands[0];
        uint32_t val1_id = inst->operands[2];

        uint8_t v0 = get_operand_rf(ctx, val0_id);
        uint8_t v1 = get_operand_rf(ctx, val1_id);

        if (v0 == V3D_RF_UNASSIGNED && v1 != V3D_RF_UNASSIGNED) {
            /* Only v1 is available */
            uint8_t d = alloc_result_rf(ctx, inst->result);
            V3D_CHECK_RF(ctx, d);
            int r = qpu_emit(&ctx->prog, qpu_a_mov(d, v1));
            V3D_CHECK_EMIT(ctx, r);
            return SSIR_TO_V3D_OK;
        }
        if (v0 != V3D_RF_UNASSIGNED && v1 == V3D_RF_UNASSIGNED) {
            uint8_t d = alloc_result_rf(ctx, inst->result);
            V3D_CHECK_RF(ctx, d);
            int r = qpu_emit(&ctx->prog, qpu_a_mov(d, v0));
            V3D_CHECK_EMIT(ctx, r);
            return SSIR_TO_V3D_OK;
        }

        V3D_CHECK_RF(ctx, v0);
        V3D_CHECK_RF(ctx, v1);

        /* Find the condition: scan backwards for the BRANCH_COND that
         * targeted the blocks referenced by this PHI */
        uint8_t cond_rf = V3D_RF_UNASSIGNED;
        for (uint32_t bi = 0; bi < ctx->fn->block_count; bi++) {
            const SsirBlock *blk = &ctx->fn->blocks[bi];
            for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
                const SsirInst *bi_inst = &blk->insts[ii];
                if (bi_inst->op == SSIR_OP_BRANCH_COND) {
                    uint32_t cond_id = bi_inst->operands[0];
                    if (cond_id < ctx->inst_map_cap &&
                        ctx->ssir_to_rf[cond_id] != V3D_RF_UNASSIGNED) {
                        cond_rf = ctx->ssir_to_rf[cond_id];
                    }
                }
            }
        }

        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);

        if (cond_rf != V3D_RF_UNASSIGNED) {
            /* select: d = (cond & v0) | (~cond & v1) */
            uint8_t t1 = alloc_rf(ctx);
            uint8_t t2 = alloc_rf(ctx);
            uint8_t nc = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, t1); V3D_CHECK_RF(ctx, t2); V3D_CHECK_RF(ctx, nc);
            int r;

            r = qpu_emit(&ctx->prog, qpu_a_and(t1, cond_rf, v0));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_not(nc, cond_rf));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_and(t2, nc, v1));
            V3D_CHECK_EMIT(ctx, r);
            r = qpu_emit(&ctx->prog, qpu_a_or(d, t1, t2));
            V3D_CHECK_EMIT(ctx, r);

            free_rf(ctx, t1); free_rf(ctx, t2); free_rf(ctx, nc);
        } else {
            /* No condition found -- use first value */
            int r = qpu_emit(&ctx->prog, qpu_a_mov(d, v0));
            V3D_CHECK_EMIT(ctx, r);
        }
        return SSIR_TO_V3D_OK;
    }

    /* Fallback for odd operand counts */
    uint8_t a = get_operand_rf(ctx, inst->operands[0]);
    V3D_CHECK_RF(ctx, a);
    uint8_t d = alloc_result_rf(ctx, inst->result);
    V3D_CHECK_RF(ctx, d);
    int r = qpu_emit(&ctx->prog, qpu_a_mov(d, a));
    V3D_CHECK_EMIT(ctx, r);
    return SSIR_TO_V3D_OK;
}

/* ----------- Step 6: Loop unrolling ----------- */

/*
 * Loop detection and unrolling.
 *
 * Structure:
 *   header block: LOOP_MERGE(merge_block, continue_block)
 *   body blocks...
 *   continue block: increment induction var, BRANCH(header) or BRANCH_COND
 *   merge block: code after loop
 *
 * Algorithm:
 *   1. Find LOOP_MERGE in header block
 *   2. Find PHI in header: the induction variable (one source from outside loop,
 *      one from continue block)
 *   3. Find the BRANCH_COND comparing induction var to bound
 *   4. Compute iteration count
 *   5. For each iteration, substitute induction var and emit body
 */

typedef struct {
    uint32_t header_block_idx;
    uint32_t merge_block_id;
    uint32_t continue_block_id;
    uint32_t induction_var_id;
    uint32_t induction_init_val;  /* SSIR ID of initial value */
    uint32_t induction_step_id;   /* SSIR ID of step instruction */
    int32_t  init_val;
    int32_t  bound_val;
    int32_t  step_val;
    uint32_t iteration_count;
    SsirOpcode cmp_op;           /* comparison in BRANCH_COND */
    bool     found;
} LoopInfo;

static int32_t v3d_try_get_const_i32(V3dCtx *ctx, uint32_t ssir_id)
{
    SsirConstant *c = ssir_get_constant((SsirModule *)ctx->mod, ssir_id);
    if (c) {
        switch (c->kind) {
        case SSIR_CONST_I32: return c->i32_val;
        case SSIR_CONST_U32: return (int32_t)c->u32_val;
        default: break;
        }
    }
    return -1; /* sentinel */
}

static LoopInfo v3d_analyze_loop(V3dCtx *ctx, uint32_t header_blk_idx)
{
    LoopInfo info;
    memset(&info, 0, sizeof(info));
    info.header_block_idx = header_blk_idx;

    const SsirBlock *header = &ctx->fn->blocks[header_blk_idx];

    /* Find LOOP_MERGE in header */
    uint32_t merge_id = 0, continue_id = 0;
    bool found_merge = false;
    for (uint32_t i = 0; i < header->inst_count; i++) {
        if (header->insts[i].op == SSIR_OP_LOOP_MERGE) {
            merge_id = header->insts[i].operands[0];
            continue_id = header->insts[i].operands[1];
            found_merge = true;
            break;
        }
    }
    if (!found_merge) return info;
    info.merge_block_id = merge_id;
    info.continue_block_id = continue_id;

    /* Find PHI in header -- induction variable */
    for (uint32_t i = 0; i < header->inst_count; i++) {
        const SsirInst *phi = &header->insts[i];
        if (phi->op != SSIR_OP_PHI) continue;
        if (phi->operand_count < 4) continue;

        /* operands: [val0, blk0, val1, blk1]
         * One source from outside the loop (initial value),
         * one from the continue block (updated value).
         */
        uint32_t val0 = phi->operands[0], blk0 = phi->operands[1];
        uint32_t val1 = phi->operands[2], blk1 = phi->operands[3];

        uint32_t init_val_id, step_val_id;
        if (blk1 == continue_id) {
            init_val_id = val0;
            step_val_id = val1;
        } else if (blk0 == continue_id) {
            init_val_id = val1;
            step_val_id = val0;
        } else {
            continue;
        }

        info.induction_var_id = phi->result;
        info.induction_init_val = init_val_id;
        info.induction_step_id = step_val_id;

        /* Try to resolve init value as constant */
        int32_t iv = v3d_try_get_const_i32(ctx, init_val_id);
        if (iv < 0) { info.induction_var_id = 0; continue; }
        info.init_val = iv;

        break;
    }
    if (!info.induction_var_id) return info;

    /* Find the step in the continue block */
    const SsirBlock *cont_blk = NULL;
    for (uint32_t bi = 0; bi < ctx->fn->block_count; bi++) {
        if (ctx->fn->blocks[bi].id == continue_id) {
            cont_blk = &ctx->fn->blocks[bi];
            break;
        }
    }
    if (!cont_blk) return info;

    /* Look for ADD/SUB with one operand being the induction var */
    for (uint32_t i = 0; i < cont_blk->inst_count; i++) {
        const SsirInst *si = &cont_blk->insts[i];
        if (si->result == info.induction_step_id) {
            if (si->op == SSIR_OP_ADD) {
                /* Which operand is the induction var? */
                if (si->operands[0] == info.induction_var_id) {
                    int32_t sv = v3d_try_get_const_i32(ctx, si->operands[1]);
                    if (sv >= 0) { info.step_val = sv; break; }
                } else if (si->operands[1] == info.induction_var_id) {
                    int32_t sv = v3d_try_get_const_i32(ctx, si->operands[0]);
                    if (sv >= 0) { info.step_val = sv; break; }
                }
            }
        }
    }
    if (info.step_val == 0) return info;

    /* Find BRANCH_COND in header comparing induction var to bound */
    for (uint32_t i = 0; i < header->inst_count; i++) {
        const SsirInst *br = &header->insts[i];
        if (br->op != SSIR_OP_BRANCH_COND) continue;

        /* The condition is operands[0] which references a comparison */
        SsirInst *cmp = sw_find_inst(ctx->inst_map, ctx->inst_map_cap,
                                     br->operands[0]);
        if (!cmp) continue;

        if (cmp->op >= SSIR_OP_LT && cmp->op <= SSIR_OP_GE) {
            info.cmp_op = cmp->op;
            /* Find the bound (the operand that's not the induction var) */
            if (cmp->operands[0] == info.induction_var_id) {
                int32_t bv = v3d_try_get_const_i32(ctx, cmp->operands[1]);
                if (bv >= 0) {
                    info.bound_val = bv;
                    info.found = true;
                }
            } else if (cmp->operands[1] == info.induction_var_id) {
                int32_t bv = v3d_try_get_const_i32(ctx, cmp->operands[0]);
                if (bv >= 0) {
                    info.bound_val = bv;
                    /* Swap comparison direction since induction var
                     * is on the right */
                    switch (info.cmp_op) {
                    case SSIR_OP_LT: info.cmp_op = SSIR_OP_GT; break;
                    case SSIR_OP_GT: info.cmp_op = SSIR_OP_LT; break;
                    case SSIR_OP_LE: info.cmp_op = SSIR_OP_GE; break;
                    case SSIR_OP_GE: info.cmp_op = SSIR_OP_LE; break;
                    default: break;
                    }
                    info.found = true;
                }
            }
        }
        break;
    }

    /* Compute iteration count */
    if (info.found && info.step_val != 0) {
        int32_t range = info.bound_val - info.init_val;
        if (info.step_val > 0 && range > 0) {
            info.iteration_count = (uint32_t)((range + info.step_val - 1) / info.step_val);
        } else if (info.step_val < 0 && range < 0) {
            info.iteration_count = (uint32_t)((-range + (-info.step_val) - 1) / (-info.step_val));
        } else {
            info.iteration_count = 0;
            info.found = false;
        }
    }

    return info;
}

static SsirToV3dResult v3d_emit_loop_unrolled(V3dCtx *ctx, const LoopInfo *loop)
{
    int max_iter = ctx->opts.max_unroll_iterations;
    if (max_iter <= 0) max_iter = 256;

    if (loop->iteration_count > (uint32_t)max_iter)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_CONTROL_FLOW,
                "loop has %u iterations, max unroll is %d",
                loop->iteration_count, max_iter);

    /* Find loop body blocks (between header and merge, excluding continue) */
    uint32_t body_start = loop->header_block_idx + 1;
    uint32_t body_end = 0;
    for (uint32_t bi = body_start; bi < ctx->fn->block_count; bi++) {
        if (ctx->fn->blocks[bi].id == loop->merge_block_id) {
            body_end = bi;
            break;
        }
    }
    if (body_end == 0)
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_CONTROL_FLOW, "loop merge block not found");

    /* Unroll: for each iteration, substitute the induction variable
     * with the concrete value and emit the body blocks */
    for (uint32_t iter = 0; iter < loop->iteration_count; iter++) {
        int32_t iv_val = loop->init_val + (int32_t)iter * loop->step_val;

        /* Load induction variable value into its RF */
        uint8_t iv_rf;
        if (ctx->ssir_to_rf[loop->induction_var_id] != V3D_RF_UNASSIGNED) {
            iv_rf = ctx->ssir_to_rf[loop->induction_var_id];
        } else {
            iv_rf = alloc_rf(ctx);
            V3D_CHECK_RF(ctx, iv_rf);
            ctx->ssir_to_rf[loop->induction_var_id] = iv_rf;
        }
        int r = v3d_load_const_rf(ctx, iv_rf, (uint32_t)iv_val);
        if (r < 0) V3D_ERR(ctx, SSIR_TO_V3D_ERR_OOM, "loop iv load failed");

        /* Emit body blocks, skipping control flow instructions */
        for (uint32_t bi = body_start; bi < body_end; bi++) {
            const SsirBlock *blk = &ctx->fn->blocks[bi];
            if (blk->id == loop->continue_block_id) {
                /* Skip continue block -- we handle the induction var update */
                continue;
            }

            for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
                const SsirInst *inst = &blk->insts[ii];
                /* Skip loop control flow ops */
                if (inst->op == SSIR_OP_BRANCH ||
                    inst->op == SSIR_OP_BRANCH_COND ||
                    inst->op == SSIR_OP_LOOP_MERGE ||
                    inst->op == SSIR_OP_PHI)
                    continue;

                pending_free_count = 0;
                SsirToV3dResult res = v3d_emit_inst(ctx, inst);
                flush_pending_frees(ctx);
                if (res != SSIR_TO_V3D_OK) return res;
            }
        }

        /* Reset use counts for instructions inside the loop body
         * (they'll be used again in the next iteration) */
        if (iter + 1 < loop->iteration_count) {
            for (uint32_t bi = body_start; bi < body_end; bi++) {
                const SsirBlock *blk = &ctx->fn->blocks[bi];
                if (blk->id == loop->continue_block_id) continue;
                for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
                    uint32_t rid = blk->insts[ii].result;
                    if (rid && rid < ctx->inst_map_cap) {
                        ctx->remaining_uses[rid] = ctx->use_counts[rid];
                        if (ctx->ssir_to_rf[rid] != V3D_RF_UNASSIGNED &&
                            rid != loop->induction_var_id) {
                            free_rf(ctx, ctx->ssir_to_rf[rid]);
                            ctx->ssir_to_rf[rid] = V3D_RF_UNASSIGNED;
                        }
                    }
                }
            }
        }
    }

    return SSIR_TO_V3D_OK;
}

/* ========================================================================== */
/* Instruction dispatch                                                       */
/* ========================================================================== */

static SsirToV3dResult v3d_emit_inst(V3dCtx *ctx, const SsirInst *inst)
{
    switch (inst->op) {
    /* Arithmetic */
    case SSIR_OP_ADD:         return v3d_emit_add(ctx, inst);
    case SSIR_OP_SUB:         return v3d_emit_sub(ctx, inst);
    case SSIR_OP_MUL:         return v3d_emit_mul(ctx, inst);
    case SSIR_OP_DIV:         return v3d_emit_div(ctx, inst);
    case SSIR_OP_NEG:         return v3d_emit_neg(ctx, inst);

    /* Bitwise */
    case SSIR_OP_BIT_AND:
    case SSIR_OP_BIT_OR:
    case SSIR_OP_BIT_XOR:
    case SSIR_OP_BIT_NOT:     return v3d_emit_bitwise(ctx, inst);

    /* Shifts */
    case SSIR_OP_SHL:
    case SSIR_OP_SHR:
    case SSIR_OP_SHR_LOGICAL: return v3d_emit_shift(ctx, inst);

    /* Memory */
    case SSIR_OP_ACCESS:      return v3d_emit_access(ctx, inst);
    case SSIR_OP_LOAD:        return v3d_emit_load(ctx, inst);
    case SSIR_OP_STORE:       return v3d_emit_store(ctx, inst);

    /* Comparisons */
    case SSIR_OP_EQ: case SSIR_OP_NE:
    case SSIR_OP_LT: case SSIR_OP_LE:
    case SSIR_OP_GT: case SSIR_OP_GE: return v3d_emit_cmp(ctx, inst);

    /* Logical */
    case SSIR_OP_AND: case SSIR_OP_OR:
    case SSIR_OP_NOT:         return v3d_emit_logical(ctx, inst);

    /* Builtins */
    case SSIR_OP_BUILTIN:     return v3d_emit_builtin(ctx, inst);

    /* Conversions */
    case SSIR_OP_CONVERT:     return v3d_emit_convert(ctx, inst);

    /* EXTRACT: for vec3 builtin loads, extract .x = global_id */
    case SSIR_OP_EXTRACT: {
        /* operands[0] = composite, operands[1] = index (constant) */
        uint8_t src = get_operand_rf(ctx, inst->operands[0]);
        V3D_CHECK_RF(ctx, src);
        /* For vec3<u32> from global_invocation_id, .x is the only
         * component we support. The composite RF IS global_id.x. */
        uint8_t d = alloc_result_rf(ctx, inst->result);
        V3D_CHECK_RF(ctx, d);
        int r = qpu_emit(&ctx->prog, qpu_a_mov(d, src));
        V3D_CHECK_EMIT(ctx, r);
        return SSIR_TO_V3D_OK;
    }

    /* BITCAST: no-op, same RF (Step 7).
     * Don't use get_operand_rf (it decrements use counts). Just alias. */
    case SSIR_OP_BITCAST: {
        uint32_t src_id = inst->operands[0];
        uint8_t a = V3D_RF_UNASSIGNED;
        if (src_id < ctx->inst_map_cap)
            a = ctx->ssir_to_rf[src_id];
        V3D_CHECK_RF(ctx, a);
        if (inst->result < ctx->inst_map_cap)
            ctx->ssir_to_rf[inst->result] = a;
        /* Transfer ownership: source's remaining uses transfer to result.
         * The result now owns the RF. Zero source remaining_uses so it
         * won't be freed when source is "consumed" later. */
        if (src_id < ctx->inst_map_cap && ctx->remaining_uses[src_id] > 0) {
            if (inst->result < ctx->inst_map_cap)
                ctx->remaining_uses[inst->result] = ctx->remaining_uses[src_id];
            ctx->remaining_uses[src_id] = 0;
        }
        return SSIR_TO_V3D_OK;
    }

    /* PHI */
    case SSIR_OP_PHI:         return v3d_emit_phi(ctx, inst);

    /* Control flow -- mostly handled at block level */
    case SSIR_OP_BRANCH:
    case SSIR_OP_BRANCH_COND:
    case SSIR_OP_LOOP_MERGE:
    case SSIR_OP_SELECTION_MERGE:
        return SSIR_TO_V3D_OK;  /* handled at block level */

    /* Termination (Step 7) */
    case SSIR_OP_RETURN_VOID:
    case SSIR_OP_UNREACHABLE:
        return SSIR_TO_V3D_OK;  /* epilogue handles thread end */

    default:
        V3D_ERR(ctx, SSIR_TO_V3D_ERR_UNSUPPORTED,
                "unhandled opcode %d (%s)", inst->op,
                ssir_opcode_name(inst->op));
    }
}

/* ========================================================================== */
/* Block-level emission with structured control flow handling                 */
/* ========================================================================== */

static SsirToV3dResult v3d_emit_block(V3dCtx *ctx, const SsirBlock *blk)
{
    for (uint32_t i = 0; i < blk->inst_count; i++) {
        pending_free_count = 0;
        SsirToV3dResult res = v3d_emit_inst(ctx, &blk->insts[i]);
        flush_pending_frees(ctx);
        if (res != SSIR_TO_V3D_OK) return res;
    }
    return SSIR_TO_V3D_OK;
}

static SsirToV3dResult v3d_emit_body(V3dCtx *ctx)
{
    /*
     * Walk blocks in order. When a LOOP_MERGE is encountered, analyze
     * the loop and unroll it. Skip blocks that are part of the unrolled
     * loop body (they've been emitted by the unroller).
     */
    uint32_t skip_until_merge = 0;

    for (uint32_t bi = 0; bi < ctx->fn->block_count; bi++) {
        const SsirBlock *blk = &ctx->fn->blocks[bi];

        /* Skip blocks that were part of an unrolled loop */
        if (skip_until_merge && blk->id != skip_until_merge)
            continue;
        if (blk->id == skip_until_merge)
            skip_until_merge = 0;

        /* Check for LOOP_MERGE in this block */
        bool has_loop = false;
        for (uint32_t i = 0; i < blk->inst_count; i++) {
            if (blk->insts[i].op == SSIR_OP_LOOP_MERGE) {
                has_loop = true;
                break;
            }
        }

        if (has_loop) {
            LoopInfo loop = v3d_analyze_loop(ctx, bi);
            if (loop.found) {
                SsirToV3dResult res = v3d_emit_loop_unrolled(ctx, &loop);
                if (res != SSIR_TO_V3D_OK) return res;
                skip_until_merge = loop.merge_block_id;
                continue;
            }
            /* If loop analysis failed, fall through and emit the header
             * block normally. This will emit the LOOP_MERGE as a no-op
             * and subsequent blocks will be emitted in order.
             * This may produce incorrect code for non-trivial loops. */
        }

        /* Emit block normally */
        SsirToV3dResult res = v3d_emit_block(ctx, blk);
        if (res != SSIR_TO_V3D_OK) return res;
    }

    return SSIR_TO_V3D_OK;
}

/* ========================================================================== */
/* Main API entry point                                                       */
/* ========================================================================== */

SsirToV3dResult ssir_to_v3d(const SsirModule *mod, const SsirToV3dOptions *opts,
                            SsirToV3dOutput *out, char **out_error)
{
    if (!mod || !out)
        return SSIR_TO_V3D_ERR_INVALID_INPUT;

    memset(out, 0, sizeof(*out));

    V3dCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.mod = mod;
    ctx.max_rf = V3D_MAX_RF;

    if (opts)
        ctx.opts = *opts;
    if (ctx.opts.max_unroll_iterations <= 0)
        ctx.opts.max_unroll_iterations = 256;

    /* Initialize QPU program buffer */
    if (qpu_prog_init(&ctx.prog) < 0) {
        if (out_error) *out_error = strdup("qpu_prog_init failed");
        return SSIR_TO_V3D_ERR_OOM;
    }

    SsirToV3dResult result;

    /* Step 1: Validate */
    result = v3d_validate(&ctx);
    if (result != SSIR_TO_V3D_OK) goto fail;

    /* Step 2: Setup context */
    result = v3d_setup(&ctx);
    if (result != SSIR_TO_V3D_OK) goto fail;

    /* Step 2: Prologue */
    result = v3d_emit_prologue(&ctx);
    if (result != SSIR_TO_V3D_OK) goto fail;

    /* Steps 3-7: Body */
    result = v3d_emit_body(&ctx);
    if (result != SSIR_TO_V3D_OK) goto fail;

    /* Step 2: Epilogue */
    result = v3d_emit_epilogue(&ctx);
    if (result != SSIR_TO_V3D_OK) goto fail;

    /* Copy output */
    out->instruction_count = ctx.prog.len;
    out->instructions = (uint64_t *)malloc(ctx.prog.len * sizeof(uint64_t));
    if (!out->instructions) { result = SSIR_TO_V3D_ERR_OOM; goto fail; }
    memcpy(out->instructions, ctx.prog.instr, ctx.prog.len * sizeof(uint64_t));

    out->uniform_count = ctx.uniform_count;
    if (ctx.uniform_count > 0) {
        out->uniforms = (uint32_t *)malloc(ctx.uniform_count * sizeof(uint32_t));
        if (!out->uniforms) { result = SSIR_TO_V3D_ERR_OOM; goto fail; }
        memcpy(out->uniforms, ctx.uniforms, ctx.uniform_count * sizeof(uint32_t));
    }

    out->workgroup_size[0] = ctx.ep->workgroup_size[0];
    out->workgroup_size[1] = ctx.ep->workgroup_size[1];
    out->workgroup_size[2] = ctx.ep->workgroup_size[2];

    /* Build binding map */
    out->binding_count = ctx.binding_count;
    for (uint32_t i = 0; i < ctx.binding_count; i++) {
        out->binding_map[i].group = ctx.bindings[i].group;
        out->binding_map[i].binding = ctx.bindings[i].binding;
        out->binding_map[i].uniform_index = i; /* Uniform index = binding index */
    }

    result = SSIR_TO_V3D_OK;

fail:
    if (result != SSIR_TO_V3D_OK && out_error && ctx.error[0])
        *out_error = strdup(ctx.error);

    /* Cleanup context */
    qpu_prog_free(&ctx.prog);
    free(ctx.uniforms);
    free(ctx.ssir_to_rf);
    free(ctx.inst_map);
    free(ctx.use_counts);
    free(ctx.remaining_uses);

    if (result != SSIR_TO_V3D_OK) {
        free(out->instructions);
        free(out->uniforms);
        memset(out, 0, sizeof(*out));
    }

    return result;
}
