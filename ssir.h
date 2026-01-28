/*
 * SSIR (Simple Shader IR) - Header
 *
 * A type-driven intermediate representation for shader code.
 * One instruction per operation - operand types determine target opcode.
 */

#ifndef SSIR_H
#define SSIR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory Allocation Macros (Customizable)
 * ============================================================================ */

#ifndef SSIR_MALLOC
#include <stdlib.h>
#define SSIR_MALLOC(sz) calloc(1, (sz))
#endif

#ifndef SSIR_REALLOC
#include <stdlib.h>
#define SSIR_REALLOC(p, sz) realloc((p), (sz))
#endif

#ifndef SSIR_FREE
#include <stdlib.h>
#define SSIR_FREE(p) free((p))
#endif

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

typedef struct SsirModule SsirModule;
typedef struct SsirType SsirType;
typedef struct SsirInst SsirInst;
typedef struct SsirFunction SsirFunction;
typedef struct SsirBlock SsirBlock;
typedef struct SsirConstant SsirConstant;
typedef struct SsirGlobalVar SsirGlobalVar;
typedef struct SsirEntryPoint SsirEntryPoint;

/* ============================================================================
 * Type System
 * ============================================================================ */

typedef enum SsirTypeKind {
    SSIR_TYPE_VOID,
    SSIR_TYPE_BOOL,
    SSIR_TYPE_I32,
    SSIR_TYPE_U32,
    SSIR_TYPE_F32,
    SSIR_TYPE_F16,
    SSIR_TYPE_VEC,
    SSIR_TYPE_MAT,
    SSIR_TYPE_ARRAY,
    SSIR_TYPE_RUNTIME_ARRAY,
    SSIR_TYPE_STRUCT,
    SSIR_TYPE_PTR,
    SSIR_TYPE_SAMPLER,
    SSIR_TYPE_SAMPLER_COMPARISON,
    SSIR_TYPE_TEXTURE,
    SSIR_TYPE_TEXTURE_STORAGE,
    SSIR_TYPE_TEXTURE_DEPTH,
} SsirTypeKind;

typedef enum SsirAddressSpace {
    SSIR_ADDR_FUNCTION,
    SSIR_ADDR_PRIVATE,
    SSIR_ADDR_WORKGROUP,
    SSIR_ADDR_UNIFORM,
    SSIR_ADDR_UNIFORM_CONSTANT,  /* For samplers/textures */
    SSIR_ADDR_STORAGE,
    SSIR_ADDR_INPUT,
    SSIR_ADDR_OUTPUT,
    SSIR_ADDR_PUSH_CONSTANT,
} SsirAddressSpace;

typedef enum SsirTextureDim {
    SSIR_TEX_1D,
    SSIR_TEX_2D,
    SSIR_TEX_3D,
    SSIR_TEX_CUBE,
    SSIR_TEX_2D_ARRAY,
    SSIR_TEX_CUBE_ARRAY,
    SSIR_TEX_MULTISAMPLED_2D,
} SsirTextureDim;

typedef enum SsirAccessMode {
    SSIR_ACCESS_READ,
    SSIR_ACCESS_WRITE,
    SSIR_ACCESS_READ_WRITE,
} SsirAccessMode;

typedef enum SsirInterpolation {
    SSIR_INTERP_NONE,
    SSIR_INTERP_PERSPECTIVE,
    SSIR_INTERP_LINEAR,
    SSIR_INTERP_FLAT,
} SsirInterpolation;

typedef enum SsirBuiltinVar {
    SSIR_BUILTIN_NONE = 0,
    /* Vertex */
    SSIR_BUILTIN_VERTEX_INDEX,
    SSIR_BUILTIN_INSTANCE_INDEX,
    SSIR_BUILTIN_POSITION,
    /* Fragment */
    SSIR_BUILTIN_FRONT_FACING,
    SSIR_BUILTIN_FRAG_DEPTH,
    SSIR_BUILTIN_SAMPLE_INDEX,
    SSIR_BUILTIN_SAMPLE_MASK,
    /* Compute */
    SSIR_BUILTIN_LOCAL_INVOCATION_ID,
    SSIR_BUILTIN_LOCAL_INVOCATION_INDEX,
    SSIR_BUILTIN_GLOBAL_INVOCATION_ID,
    SSIR_BUILTIN_WORKGROUP_ID,
    SSIR_BUILTIN_NUM_WORKGROUPS,
} SsirBuiltinVar;

typedef enum SsirStage {
    SSIR_STAGE_VERTEX,
    SSIR_STAGE_FRAGMENT,
    SSIR_STAGE_COMPUTE,
} SsirStage;

struct SsirType {
    uint32_t id;         /* unique ID from module's next_id */
    SsirTypeKind kind;
    union {
        /* SSIR_TYPE_VEC */
        struct {
            uint32_t elem;   /* element type ID */
            uint8_t size;    /* 2, 3, or 4 */
        } vec;

        /* SSIR_TYPE_MAT */
        struct {
            uint32_t elem;   /* column type ID (vec type) */
            uint8_t cols;    /* number of columns */
            uint8_t rows;    /* number of rows */
        } mat;

        /* SSIR_TYPE_ARRAY */
        struct {
            uint32_t elem;   /* element type ID */
            uint32_t length; /* array length */
        } array;

        /* SSIR_TYPE_RUNTIME_ARRAY */
        struct {
            uint32_t elem;   /* element type ID */
        } runtime_array;

        /* SSIR_TYPE_STRUCT */
        struct {
            const char *name;
            uint32_t *members;       /* member type IDs */
            uint32_t member_count;
            uint32_t *offsets;       /* byte offset of each member (for layout) */
        } struc;

        /* SSIR_TYPE_PTR */
        struct {
            uint32_t pointee;        /* pointee type ID */
            SsirAddressSpace space;
        } ptr;

        /* SSIR_TYPE_TEXTURE */
        struct {
            SsirTextureDim dim;
            uint32_t sampled_type;   /* type ID of sampled component */
        } texture;

        /* SSIR_TYPE_TEXTURE_STORAGE */
        struct {
            SsirTextureDim dim;
            uint32_t format;         /* texture format enum */
            SsirAccessMode access;
        } texture_storage;

        /* SSIR_TYPE_TEXTURE_DEPTH */
        struct {
            SsirTextureDim dim;
        } texture_depth;
    };
};

/* ============================================================================
 * Instructions
 * ============================================================================ */

typedef enum SsirOpcode {
    /* Arithmetic (6) */
    SSIR_OP_ADD,                /* %r = add %a, %b */
    SSIR_OP_SUB,                /* %r = sub %a, %b */
    SSIR_OP_MUL,                /* %r = mul %a, %b */
    SSIR_OP_DIV,                /* %r = div %a, %b */
    SSIR_OP_MOD,                /* %r = mod %a, %b */
    SSIR_OP_NEG,                /* %r = neg %a */

    /* Matrix (2) */
    SSIR_OP_MAT_MUL,            /* %r = mat_mul %a, %b */
    SSIR_OP_MAT_TRANSPOSE,      /* %r = mat_transpose %m */

    /* Bitwise (7) */
    SSIR_OP_BIT_AND,            /* %r = bit_and %a, %b */
    SSIR_OP_BIT_OR,             /* %r = bit_or %a, %b */
    SSIR_OP_BIT_XOR,            /* %r = bit_xor %a, %b */
    SSIR_OP_BIT_NOT,            /* %r = bit_not %a */
    SSIR_OP_SHL,                /* %r = shl %a, %b */
    SSIR_OP_SHR,                /* %r = shr %a, %b (arithmetic) */
    SSIR_OP_SHR_LOGICAL,        /* %r = shr_logical %a, %b */

    /* Comparison (6) */
    SSIR_OP_EQ,                 /* %r = eq %a, %b */
    SSIR_OP_NE,                 /* %r = ne %a, %b */
    SSIR_OP_LT,                 /* %r = lt %a, %b */
    SSIR_OP_LE,                 /* %r = le %a, %b */
    SSIR_OP_GT,                 /* %r = gt %a, %b */
    SSIR_OP_GE,                 /* %r = ge %a, %b */

    /* Logical (3) */
    SSIR_OP_AND,                /* %r = and %a, %b (bool) */
    SSIR_OP_OR,                 /* %r = or %a, %b (bool) */
    SSIR_OP_NOT,                /* %r = not %a (bool) */

    /* Composite (6) */
    SSIR_OP_CONSTRUCT,          /* %r = construct %a, %b, ... */
    SSIR_OP_EXTRACT,            /* %r = extract %v, idx */
    SSIR_OP_INSERT,             /* %r = insert %v, %val, idx */
    SSIR_OP_SHUFFLE,            /* %r = shuffle %a, %b, i0, i1, ... */
    SSIR_OP_SPLAT,              /* %r = splat %scalar */
    SSIR_OP_EXTRACT_DYN,        /* %r = extract_dyn %v, %idx */

    /* Memory (4) */
    SSIR_OP_LOAD,               /* %r = load %ptr */
    SSIR_OP_STORE,              /* store %ptr, %val */
    SSIR_OP_ACCESS,             /* %r = access %base, idx0, idx1, ... */
    SSIR_OP_ARRAY_LEN,          /* %r = array_len %ptr */

    /* Control flow (7) */
    SSIR_OP_BRANCH,             /* branch label */
    SSIR_OP_BRANCH_COND,        /* branch_cond %c, label_t, label_f */
    SSIR_OP_SWITCH,             /* switch %sel, default, [val, label]... */
    SSIR_OP_PHI,                /* %r = phi [%v0, label0], [%v1, label1], ... */
    SSIR_OP_RETURN,             /* return %val */
    SSIR_OP_RETURN_VOID,        /* return (no value) */
    SSIR_OP_UNREACHABLE,        /* unreachable */

    /* Call (2) */
    SSIR_OP_CALL,               /* %r = call %func, %a0, %a1, ... */
    SSIR_OP_BUILTIN,            /* %r = builtin <id>, %a0, %a1, ... */

    /* Conversion (2) */
    SSIR_OP_CONVERT,            /* %r = convert %val (to result type) */
    SSIR_OP_BITCAST,            /* %r = bitcast %val (to result type) */

    /* Texture (8) */
    SSIR_OP_TEX_SAMPLE,         /* %r = tex_sample %t, %s, %coord */
    SSIR_OP_TEX_SAMPLE_BIAS,    /* %r = tex_sample_bias %t, %s, %coord, %bias */
    SSIR_OP_TEX_SAMPLE_LEVEL,   /* %r = tex_sample_level %t, %s, %coord, %lod */
    SSIR_OP_TEX_SAMPLE_GRAD,    /* %r = tex_sample_grad %t, %s, %coord, %ddx, %ddy */
    SSIR_OP_TEX_SAMPLE_CMP,     /* %r = tex_sample_cmp %t, %s, %coord, %ref */
    SSIR_OP_TEX_LOAD,           /* %r = tex_load %t, %coord, %level */
    SSIR_OP_TEX_STORE,          /* tex_store %t, %coord, %val */
    SSIR_OP_TEX_SIZE,           /* %r = tex_size %t, (%level)? */

    /* Sync (2) */
    SSIR_OP_BARRIER,            /* barrier <scope> */
    SSIR_OP_ATOMIC,             /* %r = atomic <op>, %ptr, (%val, %cmp)? */

    /* Structured control flow (1) */
    SSIR_OP_LOOP_MERGE,         /* loop_merge %merge_block, %continue_block */

    SSIR_OP_COUNT               /* sentinel */
} SsirOpcode;

/* Built-in function IDs */
typedef enum SsirBuiltinId {
    /* Trigonometric */
    SSIR_BUILTIN_SIN,
    SSIR_BUILTIN_COS,
    SSIR_BUILTIN_TAN,
    SSIR_BUILTIN_ASIN,
    SSIR_BUILTIN_ACOS,
    SSIR_BUILTIN_ATAN,
    SSIR_BUILTIN_ATAN2,
    SSIR_BUILTIN_SINH,
    SSIR_BUILTIN_COSH,
    SSIR_BUILTIN_TANH,
    SSIR_BUILTIN_ASINH,
    SSIR_BUILTIN_ACOSH,
    SSIR_BUILTIN_ATANH,

    /* Exponential */
    SSIR_BUILTIN_EXP,
    SSIR_BUILTIN_EXP2,
    SSIR_BUILTIN_LOG,
    SSIR_BUILTIN_LOG2,
    SSIR_BUILTIN_POW,
    SSIR_BUILTIN_SQRT,
    SSIR_BUILTIN_INVERSESQRT,

    /* Common */
    SSIR_BUILTIN_ABS,
    SSIR_BUILTIN_SIGN,
    SSIR_BUILTIN_FLOOR,
    SSIR_BUILTIN_CEIL,
    SSIR_BUILTIN_ROUND,
    SSIR_BUILTIN_TRUNC,
    SSIR_BUILTIN_FRACT,
    SSIR_BUILTIN_MIN,
    SSIR_BUILTIN_MAX,
    SSIR_BUILTIN_CLAMP,
    SSIR_BUILTIN_SATURATE,
    SSIR_BUILTIN_MIX,
    SSIR_BUILTIN_STEP,
    SSIR_BUILTIN_SMOOTHSTEP,

    /* Geometric */
    SSIR_BUILTIN_DOT,
    SSIR_BUILTIN_CROSS,
    SSIR_BUILTIN_LENGTH,
    SSIR_BUILTIN_DISTANCE,
    SSIR_BUILTIN_NORMALIZE,
    SSIR_BUILTIN_FACEFORWARD,
    SSIR_BUILTIN_REFLECT,
    SSIR_BUILTIN_REFRACT,

    /* Relational */
    SSIR_BUILTIN_ALL,
    SSIR_BUILTIN_ANY,
    SSIR_BUILTIN_SELECT,

    /* Integer */
    SSIR_BUILTIN_COUNTBITS,
    SSIR_BUILTIN_REVERSEBITS,
    SSIR_BUILTIN_FIRSTLEADINGBIT,
    SSIR_BUILTIN_FIRSTTRAILINGBIT,
    SSIR_BUILTIN_EXTRACTBITS,
    SSIR_BUILTIN_INSERTBITS,

    /* Derivative (fragment only) */
    SSIR_BUILTIN_DPDX,
    SSIR_BUILTIN_DPDY,
    SSIR_BUILTIN_FWIDTH,
    SSIR_BUILTIN_DPDX_COARSE,
    SSIR_BUILTIN_DPDY_COARSE,
    SSIR_BUILTIN_DPDX_FINE,
    SSIR_BUILTIN_DPDY_FINE,

    SSIR_BUILTIN_COUNT           /* sentinel */
} SsirBuiltinId;

/* Atomic operation kinds */
typedef enum SsirAtomicOp {
    SSIR_ATOMIC_LOAD,
    SSIR_ATOMIC_STORE,
    SSIR_ATOMIC_ADD,
    SSIR_ATOMIC_SUB,
    SSIR_ATOMIC_MAX,
    SSIR_ATOMIC_MIN,
    SSIR_ATOMIC_AND,
    SSIR_ATOMIC_OR,
    SSIR_ATOMIC_XOR,
    SSIR_ATOMIC_EXCHANGE,
    SSIR_ATOMIC_COMPARE_EXCHANGE,
} SsirAtomicOp;

/* Barrier scope */
typedef enum SsirBarrierScope {
    SSIR_BARRIER_WORKGROUP,
    SSIR_BARRIER_STORAGE,
} SsirBarrierScope;

#define SSIR_MAX_OPERANDS 8

struct SsirInst {
    SsirOpcode op;
    uint32_t result;            /* result ID (0 if none) */
    uint32_t type;              /* result type ID (0 if none) */
    uint32_t operands[SSIR_MAX_OPERANDS]; /* fixed max operands */
    uint8_t operand_count;
    uint32_t *extra;            /* for variable-length (phi, switch, construct) */
    uint16_t extra_count;
};

/* ============================================================================
 * Constants
 * ============================================================================ */

typedef enum SsirConstantKind {
    SSIR_CONST_BOOL,
    SSIR_CONST_I32,
    SSIR_CONST_U32,
    SSIR_CONST_F32,
    SSIR_CONST_F16,
    SSIR_CONST_COMPOSITE,       /* array, vector, matrix, struct */
    SSIR_CONST_NULL,            /* null constant for pointer types */
} SsirConstantKind;

struct SsirConstant {
    uint32_t id;
    uint32_t type;              /* type ID */
    SsirConstantKind kind;
    union {
        bool bool_val;
        int32_t i32_val;
        uint32_t u32_val;
        float f32_val;
        uint16_t f16_val;       /* IEEE 754 half-precision bits */
        struct {
            uint32_t *components; /* IDs of component constants */
            uint32_t count;
        } composite;
    };
};

/* ============================================================================
 * Basic Blocks
 * ============================================================================ */

struct SsirBlock {
    uint32_t id;                /* label ID */
    const char *name;           /* optional debug name */
    SsirInst *insts;
    uint32_t inst_count;
    uint32_t inst_capacity;
};

/* ============================================================================
 * Functions
 * ============================================================================ */

typedef struct SsirFunctionParam {
    uint32_t id;
    uint32_t type;              /* type ID */
    const char *name;           /* optional debug name */
} SsirFunctionParam;

typedef struct SsirLocalVar {
    uint32_t id;
    uint32_t type;              /* must be ptr type with function address space */
    const char *name;           /* optional debug name */
    bool has_initializer;
    uint32_t initializer;       /* constant ID */
} SsirLocalVar;

struct SsirFunction {
    uint32_t id;
    const char *name;
    uint32_t return_type;       /* type ID */

    SsirFunctionParam *params;
    uint32_t param_count;

    SsirLocalVar *locals;
    uint32_t local_count;
    uint32_t local_capacity;

    SsirBlock *blocks;
    uint32_t block_count;
    uint32_t block_capacity;
};

/* ============================================================================
 * Global Variables
 * ============================================================================ */

struct SsirGlobalVar {
    uint32_t id;
    const char *name;
    uint32_t type;              /* must be ptr type */

    /* Attributes */
    bool has_group;
    uint32_t group;
    bool has_binding;
    uint32_t binding;
    bool has_location;
    uint32_t location;
    SsirBuiltinVar builtin;
    SsirInterpolation interp;

    /* Initializer (optional) */
    bool has_initializer;
    uint32_t initializer;       /* constant ID */
};

/* ============================================================================
 * Entry Points
 * ============================================================================ */

struct SsirEntryPoint {
    SsirStage stage;
    uint32_t function;          /* function ID */
    const char *name;           /* entry point name */

    /* Interface variables (IDs of global vars used) */
    uint32_t *interface;
    uint32_t interface_count;

    /* Compute-specific */
    uint32_t workgroup_size[3]; /* only for COMPUTE */
};

/* ============================================================================
 * Module
 * ============================================================================ */

struct SsirModule {
    /* Types (deduplicated, ID = index) */
    SsirType *types;
    uint32_t type_count;
    uint32_t type_capacity;

    /* Constants */
    SsirConstant *constants;
    uint32_t constant_count;
    uint32_t constant_capacity;

    /* Global variables */
    SsirGlobalVar *globals;
    uint32_t global_count;
    uint32_t global_capacity;

    /* Functions */
    SsirFunction *functions;
    uint32_t function_count;
    uint32_t function_capacity;

    /* Entry points */
    SsirEntryPoint *entry_points;
    uint32_t entry_point_count;
    uint32_t entry_point_capacity;

    /* ID counter */
    uint32_t next_id;
};

/* ============================================================================
 * Result Codes
 * ============================================================================ */

typedef enum SsirResult {
    SSIR_OK = 0,
    SSIR_ERROR_OUT_OF_MEMORY,
    SSIR_ERROR_INVALID_TYPE,
    SSIR_ERROR_INVALID_ID,
    SSIR_ERROR_INVALID_OPERAND,
    SSIR_ERROR_TYPE_MISMATCH,
    SSIR_ERROR_INVALID_BLOCK,
    SSIR_ERROR_INVALID_FUNCTION,
    SSIR_ERROR_SSA_VIOLATION,
    SSIR_ERROR_TERMINATOR_MISSING,
    SSIR_ERROR_PHI_PLACEMENT,
    SSIR_ERROR_ADDRESS_SPACE,
    SSIR_ERROR_ENTRY_POINT,
} SsirResult;

/* ============================================================================
 * Module API
 * ============================================================================ */

/* Create/destroy module */
SsirModule *ssir_module_create(void);
void ssir_module_destroy(SsirModule *mod);

/* Allocate a new unique ID */
uint32_t ssir_module_alloc_id(SsirModule *mod);

/* ============================================================================
 * Type API
 * ============================================================================ */

/* Create scalar types (returns type ID, deduplicates) */
uint32_t ssir_type_void(SsirModule *mod);
uint32_t ssir_type_bool(SsirModule *mod);
uint32_t ssir_type_i32(SsirModule *mod);
uint32_t ssir_type_u32(SsirModule *mod);
uint32_t ssir_type_f32(SsirModule *mod);
uint32_t ssir_type_f16(SsirModule *mod);

/* Create composite types */
uint32_t ssir_type_vec(SsirModule *mod, uint32_t elem_type, uint8_t size);
uint32_t ssir_type_mat(SsirModule *mod, uint32_t col_type, uint8_t cols, uint8_t rows);
uint32_t ssir_type_array(SsirModule *mod, uint32_t elem_type, uint32_t length);
uint32_t ssir_type_runtime_array(SsirModule *mod, uint32_t elem_type);
uint32_t ssir_type_struct(SsirModule *mod, const char *name,
                          const uint32_t *members, uint32_t member_count,
                          const uint32_t *offsets);
uint32_t ssir_type_ptr(SsirModule *mod, uint32_t pointee_type, SsirAddressSpace space);

/* Create sampler/texture types */
uint32_t ssir_type_sampler(SsirModule *mod);
uint32_t ssir_type_sampler_comparison(SsirModule *mod);
uint32_t ssir_type_texture(SsirModule *mod, SsirTextureDim dim, uint32_t sampled_type);
uint32_t ssir_type_texture_storage(SsirModule *mod, SsirTextureDim dim,
                                   uint32_t format, SsirAccessMode access);
uint32_t ssir_type_texture_depth(SsirModule *mod, SsirTextureDim dim);

/* Get type by ID */
SsirType *ssir_get_type(SsirModule *mod, uint32_t type_id);

/* Type classification helpers */
bool ssir_type_is_scalar(const SsirType *t);
bool ssir_type_is_integer(const SsirType *t);
bool ssir_type_is_signed(const SsirType *t);
bool ssir_type_is_unsigned(const SsirType *t);
bool ssir_type_is_float(const SsirType *t);
bool ssir_type_is_bool(const SsirType *t);
bool ssir_type_is_vector(const SsirType *t);
bool ssir_type_is_matrix(const SsirType *t);
bool ssir_type_is_numeric(const SsirType *t);

/* Get the scalar element type from vector/matrix (returns type ID) */
uint32_t ssir_type_scalar_of(SsirModule *mod, uint32_t type_id);

/* ============================================================================
 * Constant API
 * ============================================================================ */

uint32_t ssir_const_bool(SsirModule *mod, bool val);
uint32_t ssir_const_i32(SsirModule *mod, int32_t val);
uint32_t ssir_const_u32(SsirModule *mod, uint32_t val);
uint32_t ssir_const_f32(SsirModule *mod, float val);
uint32_t ssir_const_f16(SsirModule *mod, uint16_t val);
uint32_t ssir_const_composite(SsirModule *mod, uint32_t type_id,
                              const uint32_t *components, uint32_t count);
uint32_t ssir_const_null(SsirModule *mod, uint32_t type_id);

/* Get constant by ID (returns NULL if not found or not a constant) */
SsirConstant *ssir_get_constant(SsirModule *mod, uint32_t const_id);

/* ============================================================================
 * Global Variable API
 * ============================================================================ */

uint32_t ssir_global_var(SsirModule *mod, const char *name, uint32_t ptr_type);
SsirGlobalVar *ssir_get_global(SsirModule *mod, uint32_t global_id);

/* Set global attributes */
void ssir_global_set_group(SsirModule *mod, uint32_t global_id, uint32_t group);
void ssir_global_set_binding(SsirModule *mod, uint32_t global_id, uint32_t binding);
void ssir_global_set_location(SsirModule *mod, uint32_t global_id, uint32_t location);
void ssir_global_set_builtin(SsirModule *mod, uint32_t global_id, SsirBuiltinVar builtin);
void ssir_global_set_interpolation(SsirModule *mod, uint32_t global_id, SsirInterpolation interp);
void ssir_global_set_initializer(SsirModule *mod, uint32_t global_id, uint32_t const_id);

/* ============================================================================
 * Function API
 * ============================================================================ */

uint32_t ssir_function_create(SsirModule *mod, const char *name, uint32_t return_type);
SsirFunction *ssir_get_function(SsirModule *mod, uint32_t func_id);

/* Add parameter to function (returns parameter ID) */
uint32_t ssir_function_add_param(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t type);

/* Add local variable to function (returns variable ID) */
uint32_t ssir_function_add_local(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t ptr_type);

/* ============================================================================
 * Block API
 * ============================================================================ */

/* Create block in function (returns block label ID) */
uint32_t ssir_block_create(SsirModule *mod, uint32_t func_id, const char *name);
/* Create block with pre-allocated ID (for deferred block creation) */
uint32_t ssir_block_create_with_id(SsirModule *mod, uint32_t func_id, uint32_t block_id, const char *name);
SsirBlock *ssir_get_block(SsirModule *mod, uint32_t func_id, uint32_t block_id);

/* ============================================================================
 * Instruction Builder API
 * ============================================================================ */

/* All instruction builders take: module, function ID, block ID
 * Returns result ID (or 0 for void instructions) */

/* Arithmetic */
uint32_t ssir_build_add(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_sub(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_div(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_mod(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_neg(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a);

/* Matrix */
uint32_t ssir_build_mat_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_mat_transpose(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t type, uint32_t m);

/* Bitwise */
uint32_t ssir_build_bit_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_bit_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_bit_xor(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_bit_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a);
uint32_t ssir_build_shl(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_shr(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_shr_logical(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t a, uint32_t b);

/* Comparison */
uint32_t ssir_build_eq(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_ne(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_lt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_le(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_gt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_ge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);

/* Logical */
uint32_t ssir_build_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a);

/* Composite */
uint32_t ssir_build_construct(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t type, const uint32_t *components, uint32_t count);
uint32_t ssir_build_extract(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t composite, uint32_t index);
uint32_t ssir_build_insert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t composite, uint32_t value, uint32_t index);
uint32_t ssir_build_shuffle(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t v1, uint32_t v2,
                            const uint32_t *indices, uint32_t index_count);
uint32_t ssir_build_splat(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t type, uint32_t scalar);
uint32_t ssir_build_extract_dyn(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t composite, uint32_t index);

/* Memory */
uint32_t ssir_build_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t ptr);
void ssir_build_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                      uint32_t ptr, uint32_t value);
uint32_t ssir_build_access(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t base,
                           const uint32_t *indices, uint32_t index_count);
uint32_t ssir_build_array_len(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t ptr);

/* Control flow */
void ssir_build_branch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t target_block);
void ssir_build_branch_cond(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t cond, uint32_t true_block, uint32_t false_block);
void ssir_build_branch_cond_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t cond, uint32_t true_block, uint32_t false_block,
                                  uint32_t merge_block);
void ssir_build_loop_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t merge_block, uint32_t continue_block);
void ssir_build_switch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t selector, uint32_t default_block,
                       const uint32_t *cases, uint32_t case_count);
uint32_t ssir_build_phi(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, const uint32_t *incoming, uint32_t count);
void ssir_build_return(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t value);
void ssir_build_return_void(SsirModule *mod, uint32_t func_id, uint32_t block_id);
void ssir_build_unreachable(SsirModule *mod, uint32_t func_id, uint32_t block_id);

/* Call */
uint32_t ssir_build_call(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t callee,
                         const uint32_t *args, uint32_t arg_count);
uint32_t ssir_build_builtin(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, SsirBuiltinId builtin,
                            const uint32_t *args, uint32_t arg_count);

/* Conversion */
uint32_t ssir_build_convert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value);
uint32_t ssir_build_bitcast(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value);

/* Texture */
uint32_t ssir_build_tex_sample(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t texture, uint32_t sampler,
                               uint32_t coord);
uint32_t ssir_build_tex_sample_bias(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t bias);
uint32_t ssir_build_tex_sample_level(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                     uint32_t type, uint32_t texture, uint32_t sampler,
                                     uint32_t coord, uint32_t lod);
uint32_t ssir_build_tex_sample_grad(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t ddx, uint32_t ddy);
uint32_t ssir_build_tex_sample_cmp(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                   uint32_t type, uint32_t texture, uint32_t sampler,
                                   uint32_t coord, uint32_t ref);
uint32_t ssir_build_tex_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t coord, uint32_t level);
void ssir_build_tex_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t texture, uint32_t coord, uint32_t value);
uint32_t ssir_build_tex_size(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t level);

/* Sync */
void ssir_build_barrier(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        SsirBarrierScope scope);
uint32_t ssir_build_atomic(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, SsirAtomicOp op, uint32_t ptr,
                           uint32_t value, uint32_t comparator);

/* ============================================================================
 * Entry Point API
 * ============================================================================ */

uint32_t ssir_entry_point_create(SsirModule *mod, SsirStage stage,
                                 uint32_t func_id, const char *name);
SsirEntryPoint *ssir_get_entry_point(SsirModule *mod, uint32_t index);

void ssir_entry_point_add_interface(SsirModule *mod, uint32_t ep_index, uint32_t global_id);
void ssir_entry_point_set_workgroup_size(SsirModule *mod, uint32_t ep_index,
                                         uint32_t x, uint32_t y, uint32_t z);

/* ============================================================================
 * Validation API
 * ============================================================================ */

typedef struct SsirValidationError {
    SsirResult code;
    const char *message;
    uint32_t func_id;           /* 0 if not function-specific */
    uint32_t block_id;          /* 0 if not block-specific */
    uint32_t inst_index;        /* index within block, or 0 */
} SsirValidationError;

typedef struct SsirValidationResult {
    bool valid;
    SsirValidationError *errors;
    uint32_t error_count;
    uint32_t error_capacity;
} SsirValidationResult;

/* Validate module, returns validation result (caller owns and must free) */
SsirValidationResult *ssir_validate(SsirModule *mod);
void ssir_validation_result_free(SsirValidationResult *result);

/* ============================================================================
 * Debug/Utility API
 * ============================================================================ */

/* Get opcode name as string */
const char *ssir_opcode_name(SsirOpcode op);
const char *ssir_builtin_name(SsirBuiltinId id);
const char *ssir_type_kind_name(SsirTypeKind kind);

/* Print module to string (for debugging) - caller must free returned string */
char *ssir_module_to_string(SsirModule *mod);

#ifdef __cplusplus
}
#endif

#endif /* SSIR_H */
