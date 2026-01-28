/*
 * SPIR-V to SSIR Converter - Implementation
 */

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>

#ifndef SPIRV_TO_SSIR_MALLOC
#define SPIRV_TO_SSIR_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef SPIRV_TO_SSIR_REALLOC
#define SPIRV_TO_SSIR_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef SPIRV_TO_SSIR_FREE
#define SPIRV_TO_SSIR_FREE(p) free((p))
#endif

#define SPV_MAGIC 0x07230203

typedef enum {
    SPV_ID_UNKNOWN = 0,
    SPV_ID_TYPE,
    SPV_ID_CONSTANT,
    SPV_ID_VARIABLE,
    SPV_ID_FUNCTION,
    SPV_ID_LABEL,
    SPV_ID_INSTRUCTION,
    SPV_ID_EXT_INST_IMPORT,
    SPV_ID_PARAM,
} SpvIdKind;

typedef enum {
    SPV_TYPE_VOID = 0,
    SPV_TYPE_BOOL,
    SPV_TYPE_INT,
    SPV_TYPE_FLOAT,
    SPV_TYPE_VECTOR,
    SPV_TYPE_MATRIX,
    SPV_TYPE_ARRAY,
    SPV_TYPE_RUNTIME_ARRAY,
    SPV_TYPE_STRUCT,
    SPV_TYPE_POINTER,
    SPV_TYPE_FUNCTION,
    SPV_TYPE_IMAGE,
    SPV_TYPE_SAMPLED_IMAGE,
    SPV_TYPE_SAMPLER,
} SpvTypeKind;

typedef struct {
    SpvTypeKind kind;
    union {
        struct { uint32_t width; uint32_t signedness; } int_type;
        struct { uint32_t width; } float_type;
        struct { uint32_t component_type; uint32_t count; } vector;
        struct { uint32_t column_type; uint32_t columns; } matrix;
        struct { uint32_t element_type; uint32_t length_id; } array;
        struct { uint32_t element_type; } runtime_array;
        struct { uint32_t *member_types; int member_count; } struct_type;
        struct { uint32_t pointee_type; SpvStorageClass storage; } pointer;
        struct { uint32_t return_type; uint32_t *param_types; int param_count; } function;
        struct { uint32_t sampled_type; SpvDim dim; uint32_t depth; uint32_t arrayed; uint32_t ms; uint32_t sampled; SpvImageFormat format; } image;
        struct { uint32_t image_type; } sampled_image;
    };
} SpvTypeInfo;

typedef struct {
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} SpvDecorationEntry;

typedef struct {
    uint32_t member_index;
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} SpvMemberDecoration;

typedef struct {
    SpvIdKind kind;
    uint32_t id;
    char *name;
    SpvTypeInfo type_info;
    uint32_t ssir_id;

    union {
        struct { uint32_t type_id; uint32_t *values; int value_count; int is_composite; } constant;
        struct { uint32_t type_id; SpvStorageClass storage_class; uint32_t initializer; } variable;
        struct { uint32_t return_type; uint32_t func_type; } function;
        struct { uint32_t type_id; SpvOp opcode; uint32_t *operands; int operand_count; } instruction;
        struct { uint32_t type_id; } param;
    };

    SpvDecorationEntry *decorations;
    int decoration_count;
    SpvMemberDecoration *member_decorations;
    int member_decoration_count;
    char **member_names;
    int member_name_count;
} SpvIdInfo;

typedef struct {
    uint32_t label_id;
    uint32_t *instructions;
    int instruction_count;
    int instruction_cap;
    uint32_t merge_block;
    uint32_t continue_block;
    int is_loop_header;
    int is_selection_header;
} SpvBasicBlock;

typedef struct {
    uint32_t id;
    char *name;
    uint32_t return_type;
    uint32_t func_type;
    uint32_t *params;
    int param_count;
    SpvBasicBlock *blocks;
    int block_count;
    int block_cap;
    SpvExecutionModel exec_model;
    int is_entry_point;
    uint32_t *interface_vars;
    int interface_var_count;
    int workgroup_size[3];
    uint32_t *local_vars;
    int local_var_count;
    int local_var_cap;
} SpvFunction;

typedef struct {
    uint32_t func_id;
    SpvExecutionModel exec_model;
    char *name;
    uint32_t *interface_vars;
    int interface_var_count;
} PendingEntryPoint;

typedef struct {
    uint32_t func_id;
    int workgroup_size[3];
} PendingWorkgroupSize;

typedef struct {
    const uint32_t *spirv;
    size_t word_count;
    uint32_t version;
    uint32_t generator;
    uint32_t id_bound;

    SpvIdInfo *ids;
    SpvFunction *functions;
    int function_count;
    int function_cap;

    PendingEntryPoint *pending_eps;
    int pending_ep_count;
    int pending_ep_cap;

    PendingWorkgroupSize *pending_wgs;
    int pending_wg_count;
    int pending_wg_cap;

    uint32_t glsl_ext_id;

    SsirModule *mod;
    const SpirvToSsirOptions *opts;
    char last_error[512];
} Converter;

static void set_error(Converter *c, const char *msg) {
    size_t n = strlen(msg);
    if (n >= sizeof(c->last_error)) n = sizeof(c->last_error) - 1;
    memcpy(c->last_error, msg, n);
    c->last_error[n] = 0;
}

static char *read_string(const uint32_t *words, int word_count, int *out_words_read) {
    if (word_count <= 0) { *out_words_read = 0; return NULL; }
    int max_chars = word_count * 4;
    const char *str = (const char*)words;
    int len = 0;
    while (len < max_chars && str[len]) len++;
    *out_words_read = (len + 4) / 4;
    char *copy = (char*)SPIRV_TO_SSIR_MALLOC(len + 1);
    if (!copy) return NULL;
    memcpy(copy, str, len);
    copy[len] = 0;
    return copy;
}

static void add_decoration(Converter *c, uint32_t target, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    if (target >= c->id_bound) return;
    SpvIdInfo *info = &c->ids[target];
    int idx = info->decoration_count++;
    info->decorations = (SpvDecorationEntry*)SPIRV_TO_SSIR_REALLOC(info->decorations, info->decoration_count * sizeof(SpvDecorationEntry));
    info->decorations[idx].decoration = decor;
    info->decorations[idx].literal_count = lit_count;
    if (lit_count > 0) {
        info->decorations[idx].literals = (uint32_t*)SPIRV_TO_SSIR_MALLOC(lit_count * sizeof(uint32_t));
        memcpy(info->decorations[idx].literals, literals, lit_count * sizeof(uint32_t));
    } else {
        info->decorations[idx].literals = NULL;
    }
}

static void add_member_decoration(Converter *c, uint32_t struct_id, uint32_t member, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    if (struct_id >= c->id_bound) return;
    SpvIdInfo *info = &c->ids[struct_id];
    int idx = info->member_decoration_count++;
    info->member_decorations = (SpvMemberDecoration*)SPIRV_TO_SSIR_REALLOC(info->member_decorations, info->member_decoration_count * sizeof(SpvMemberDecoration));
    info->member_decorations[idx].member_index = member;
    info->member_decorations[idx].decoration = decor;
    info->member_decorations[idx].literal_count = lit_count;
    if (lit_count > 0) {
        info->member_decorations[idx].literals = (uint32_t*)SPIRV_TO_SSIR_MALLOC(lit_count * sizeof(uint32_t));
        memcpy(info->member_decorations[idx].literals, literals, lit_count * sizeof(uint32_t));
    } else {
        info->member_decorations[idx].literals = NULL;
    }
}

static int has_decoration(Converter *c, uint32_t id, SpvDecoration decor, uint32_t *out_value) {
    if (id >= c->id_bound) return 0;
    SpvIdInfo *info = &c->ids[id];
    for (int i = 0; i < info->decoration_count; i++) {
        if (info->decorations[i].decoration == decor) {
            if (out_value && info->decorations[i].literal_count > 0) {
                *out_value = info->decorations[i].literals[0];
            }
            return 1;
        }
    }
    return 0;
}

static int get_member_offset(Converter *c, uint32_t struct_id, uint32_t member, uint32_t *out_offset) {
    if (struct_id >= c->id_bound) return 0;
    SpvIdInfo *info = &c->ids[struct_id];
    for (int i = 0; i < info->member_decoration_count; i++) {
        if (info->member_decorations[i].member_index == member &&
            info->member_decorations[i].decoration == SpvDecorationOffset &&
            info->member_decorations[i].literal_count > 0) {
            *out_offset = info->member_decorations[i].literals[0];
            return 1;
        }
    }
    return 0;
}

static SpvFunction *add_function(Converter *c) {
    if (c->function_count >= c->function_cap) {
        int ncap = c->function_cap ? c->function_cap * 2 : 8;
        c->functions = (SpvFunction*)SPIRV_TO_SSIR_REALLOC(c->functions, ncap * sizeof(SpvFunction));
        c->function_cap = ncap;
    }
    SpvFunction *fn = &c->functions[c->function_count++];
    memset(fn, 0, sizeof(SpvFunction));
    fn->workgroup_size[0] = 1;
    fn->workgroup_size[1] = 1;
    fn->workgroup_size[2] = 1;
    return fn;
}

static SpvBasicBlock *add_block(SpvFunction *fn, uint32_t label_id) {
    if (fn->block_count >= fn->block_cap) {
        int ncap = fn->block_cap ? fn->block_cap * 2 : 8;
        fn->blocks = (SpvBasicBlock*)SPIRV_TO_SSIR_REALLOC(fn->blocks, ncap * sizeof(SpvBasicBlock));
        fn->block_cap = ncap;
    }
    SpvBasicBlock *blk = &fn->blocks[fn->block_count++];
    memset(blk, 0, sizeof(SpvBasicBlock));
    blk->label_id = label_id;
    return blk;
}

static void add_block_instr(SpvBasicBlock *blk, uint32_t instr_start) {
    if (blk->instruction_count >= blk->instruction_cap) {
        int ncap = blk->instruction_cap ? blk->instruction_cap * 2 : 16;
        blk->instructions = (uint32_t*)SPIRV_TO_SSIR_REALLOC(blk->instructions, ncap * sizeof(uint32_t));
        blk->instruction_cap = ncap;
    }
    blk->instructions[blk->instruction_count++] = instr_start;
}

static void add_function_local_var(SpvFunction *fn, uint32_t var_id) {
    if (fn->local_var_count >= fn->local_var_cap) {
        int ncap = fn->local_var_cap ? fn->local_var_cap * 2 : 8;
        fn->local_vars = (uint32_t*)SPIRV_TO_SSIR_REALLOC(fn->local_vars, ncap * sizeof(uint32_t));
        fn->local_var_cap = ncap;
    }
    fn->local_vars[fn->local_var_count++] = var_id;
}

static SpirvToSsirResult parse_spirv(Converter *c) {
    size_t pos = 5;
    SpvFunction *current_fn = NULL;
    SpvBasicBlock *current_block = NULL;

    while (pos < c->word_count) {
        uint32_t word0 = c->spirv[pos];
        uint16_t opcode = word0 & 0xFFFF;
        uint16_t wc = word0 >> 16;
        if (wc == 0 || pos + wc > c->word_count) {
            set_error(c, "Invalid instruction");
            return SPIRV_TO_SSIR_INVALID_SPIRV;
        }

        const uint32_t *operands = &c->spirv[pos + 1];
        int operand_count = wc - 1;

        switch ((SpvOp)opcode) {
        case SpvOpName:
            if (operand_count >= 2) {
                uint32_t target = operands[0];
                int str_words;
                char *name = read_string(&operands[1], operand_count - 1, &str_words);
                if (target < c->id_bound && name) {
                    c->ids[target].name = name;
                }
            }
            break;

        case SpvOpMemberName:
            if (operand_count >= 3) {
                uint32_t struct_id = operands[0];
                uint32_t member = operands[1];
                int str_words;
                char *name = read_string(&operands[2], operand_count - 2, &str_words);
                if (struct_id < c->id_bound && name) {
                    SpvIdInfo *info = &c->ids[struct_id];
                    if (member >= (uint32_t)info->member_name_count) {
                        int new_count = member + 1;
                        info->member_names = (char**)SPIRV_TO_SSIR_REALLOC(info->member_names, new_count * sizeof(char*));
                        for (int i = info->member_name_count; i < new_count; i++) {
                            info->member_names[i] = NULL;
                        }
                        info->member_name_count = new_count;
                    }
                    if (info->member_names[member]) SPIRV_TO_SSIR_FREE(info->member_names[member]);
                    info->member_names[member] = name;
                }
            }
            break;

        case SpvOpDecorate:
            if (operand_count >= 2) {
                uint32_t target = operands[0];
                SpvDecoration decor = (SpvDecoration)operands[1];
                add_decoration(c, target, decor, &operands[2], operand_count - 2);
            }
            break;

        case SpvOpMemberDecorate:
            if (operand_count >= 3) {
                uint32_t struct_id = operands[0];
                uint32_t member = operands[1];
                SpvDecoration decor = (SpvDecoration)operands[2];
                add_member_decoration(c, struct_id, member, decor, &operands[3], operand_count - 3);
            }
            break;

        case SpvOpExtInstImport:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_EXT_INST_IMPORT;
                    int str_words;
                    char *name = read_string(&operands[1], operand_count - 1, &str_words);
                    if (name && strcmp(name, "GLSL.std.450") == 0) {
                        c->glsl_ext_id = id;
                    }
                    SPIRV_TO_SSIR_FREE(name);
                }
            }
            break;

        case SpvOpTypeVoid:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_VOID;
                }
            }
            break;

        case SpvOpTypeBool:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_BOOL;
                }
            }
            break;

        case SpvOpTypeInt:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_INT;
                    c->ids[id].type_info.int_type.width = operands[1];
                    c->ids[id].type_info.int_type.signedness = operands[2];
                }
            }
            break;

        case SpvOpTypeFloat:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_FLOAT;
                    c->ids[id].type_info.float_type.width = operands[1];
                }
            }
            break;

        case SpvOpTypeVector:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_VECTOR;
                    c->ids[id].type_info.vector.component_type = operands[1];
                    c->ids[id].type_info.vector.count = operands[2];
                }
            }
            break;

        case SpvOpTypeMatrix:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_MATRIX;
                    c->ids[id].type_info.matrix.column_type = operands[1];
                    c->ids[id].type_info.matrix.columns = operands[2];
                }
            }
            break;

        case SpvOpTypeArray:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_ARRAY;
                    c->ids[id].type_info.array.element_type = operands[1];
                    c->ids[id].type_info.array.length_id = operands[2];
                }
            }
            break;

        case SpvOpTypeRuntimeArray:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_RUNTIME_ARRAY;
                    c->ids[id].type_info.runtime_array.element_type = operands[1];
                }
            }
            break;

        case SpvOpTypeStruct:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_STRUCT;
                    int mc = operand_count - 1;
                    c->ids[id].type_info.struct_type.member_count = mc;
                    if (mc > 0) {
                        c->ids[id].type_info.struct_type.member_types = (uint32_t*)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
                        memcpy(c->ids[id].type_info.struct_type.member_types, &operands[1], mc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpTypePointer:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_POINTER;
                    c->ids[id].type_info.pointer.storage = (SpvStorageClass)operands[1];
                    c->ids[id].type_info.pointer.pointee_type = operands[2];
                }
            }
            break;

        case SpvOpTypeFunction:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_FUNCTION;
                    c->ids[id].type_info.function.return_type = operands[1];
                    int pc = operand_count - 2;
                    c->ids[id].type_info.function.param_count = pc;
                    if (pc > 0) {
                        c->ids[id].type_info.function.param_types = (uint32_t*)SPIRV_TO_SSIR_MALLOC(pc * sizeof(uint32_t));
                        memcpy(c->ids[id].type_info.function.param_types, &operands[2], pc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpTypeImage:
            if (operand_count >= 8) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_IMAGE;
                    c->ids[id].type_info.image.sampled_type = operands[1];
                    c->ids[id].type_info.image.dim = (SpvDim)operands[2];
                    c->ids[id].type_info.image.depth = operands[3];
                    c->ids[id].type_info.image.arrayed = operands[4];
                    c->ids[id].type_info.image.ms = operands[5];
                    c->ids[id].type_info.image.sampled = operands[6];
                    c->ids[id].type_info.image.format = (SpvImageFormat)operands[7];
                }
            }
            break;

        case SpvOpTypeSampledImage:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_SAMPLED_IMAGE;
                    c->ids[id].type_info.sampled_image.image_type = operands[1];
                }
            }
            break;

        case SpvOpTypeSampler:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_TYPE;
                    c->ids[id].type_info.kind = SPV_TYPE_SAMPLER;
                }
            }
            break;

        case SpvOpConstant:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
            if (operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_CONSTANT;
                    c->ids[id].constant.type_id = type_id;
                    c->ids[id].constant.is_composite = 0;
                    int vc = operand_count - 2;
                    c->ids[id].constant.value_count = vc;
                    if (vc > 0) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(vc * sizeof(uint32_t));
                        memcpy(c->ids[id].constant.values, &operands[2], vc * sizeof(uint32_t));
                    }
                    if ((SpvOp)opcode == SpvOpConstantTrue) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(sizeof(uint32_t));
                        c->ids[id].constant.values[0] = 1;
                        c->ids[id].constant.value_count = 1;
                    } else if ((SpvOp)opcode == SpvOpConstantFalse) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(sizeof(uint32_t));
                        c->ids[id].constant.values[0] = 0;
                        c->ids[id].constant.value_count = 1;
                    }
                }
            }
            break;

        case SpvOpConstantComposite:
            if (operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_CONSTANT;
                    c->ids[id].constant.type_id = type_id;
                    c->ids[id].constant.is_composite = 1;
                    int vc = operand_count - 2;
                    c->ids[id].constant.value_count = vc;
                    if (vc > 0) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(vc * sizeof(uint32_t));
                        memcpy(c->ids[id].constant.values, &operands[2], vc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpVariable:
            if (operand_count >= 3) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                SpvStorageClass sc = (SpvStorageClass)operands[2];
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_VARIABLE;
                    c->ids[id].variable.type_id = type_id;
                    c->ids[id].variable.storage_class = sc;
                    c->ids[id].variable.initializer = (operand_count > 3) ? operands[3] : 0;
                    if (sc == SpvStorageClassFunction && current_fn) {
                        add_function_local_var(current_fn, id);
                    }
                }
            }
            break;

        case SpvOpEntryPoint:
            if (operand_count >= 3) {
                SpvExecutionModel model = (SpvExecutionModel)operands[0];
                uint32_t fn_id = operands[1];
                int str_words;
                char *name = read_string(&operands[2], operand_count - 2, &str_words);
                if (fn_id < c->id_bound) {
                    c->ids[fn_id].kind = SPV_ID_FUNCTION;
                    if (c->ids[fn_id].name) SPIRV_TO_SSIR_FREE(c->ids[fn_id].name);
                    c->ids[fn_id].name = name ? strdup(name) : NULL;
                }
                if (c->pending_ep_count >= c->pending_ep_cap) {
                    int ncap = c->pending_ep_cap ? c->pending_ep_cap * 2 : 4;
                    c->pending_eps = (PendingEntryPoint*)SPIRV_TO_SSIR_REALLOC(c->pending_eps, ncap * sizeof(PendingEntryPoint));
                    c->pending_ep_cap = ncap;
                }
                PendingEntryPoint *ep = &c->pending_eps[c->pending_ep_count++];
                ep->func_id = fn_id;
                ep->exec_model = model;
                ep->name = name ? strdup(name) : NULL;
                int iface_start = 2 + str_words;
                int iface_count = operand_count - iface_start;
                if (iface_count > 0) {
                    ep->interface_vars = (uint32_t*)SPIRV_TO_SSIR_MALLOC(iface_count * sizeof(uint32_t));
                    memcpy(ep->interface_vars, &operands[iface_start], iface_count * sizeof(uint32_t));
                    ep->interface_var_count = iface_count;
                } else {
                    ep->interface_vars = NULL;
                    ep->interface_var_count = 0;
                }
                SPIRV_TO_SSIR_FREE(name);
            }
            break;

        case SpvOpExecutionMode:
            if (operand_count >= 2) {
                uint32_t fn_id = operands[0];
                SpvExecutionMode mode = (SpvExecutionMode)operands[1];
                if (mode == SpvExecutionModeLocalSize && operand_count >= 5) {
                    if (c->pending_wg_count >= c->pending_wg_cap) {
                        int ncap = c->pending_wg_cap ? c->pending_wg_cap * 2 : 4;
                        c->pending_wgs = (PendingWorkgroupSize*)SPIRV_TO_SSIR_REALLOC(c->pending_wgs, ncap * sizeof(PendingWorkgroupSize));
                        c->pending_wg_cap = ncap;
                    }
                    PendingWorkgroupSize *wg = &c->pending_wgs[c->pending_wg_count++];
                    wg->func_id = fn_id;
                    wg->workgroup_size[0] = operands[2];
                    wg->workgroup_size[1] = operands[3];
                    wg->workgroup_size[2] = operands[4];
                }
            }
            break;

        case SpvOpFunction:
            if (operand_count >= 4) {
                uint32_t ret_type = operands[0];
                uint32_t id = operands[1];
                uint32_t func_type = operands[3];
                current_fn = add_function(c);
                current_fn->id = id;
                current_fn->return_type = ret_type;
                current_fn->func_type = func_type;
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_FUNCTION;
                    c->ids[id].function.return_type = ret_type;
                    c->ids[id].function.func_type = func_type;
                    if (c->ids[id].name) {
                        current_fn->name = strdup(c->ids[id].name);
                    }
                }
                current_block = NULL;
            }
            break;

        case SpvOpFunctionParameter:
            if (current_fn && operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                int idx = current_fn->param_count++;
                current_fn->params = (uint32_t*)SPIRV_TO_SSIR_REALLOC(current_fn->params, current_fn->param_count * sizeof(uint32_t));
                current_fn->params[idx] = id;
                if (id < c->id_bound) {
                    c->ids[id].kind = SPV_ID_PARAM;
                    c->ids[id].param.type_id = type_id;
                }
            }
            break;

        case SpvOpFunctionEnd:
            current_fn = NULL;
            current_block = NULL;
            break;

        case SpvOpLabel:
            if (current_fn && operand_count >= 1) {
                uint32_t label_id = operands[0];
                current_block = add_block(current_fn, label_id);
                if (label_id < c->id_bound) {
                    c->ids[label_id].kind = SPV_ID_LABEL;
                }
            }
            break;

        case SpvOpLoopMerge:
            if (current_block && operand_count >= 2) {
                current_block->merge_block = operands[0];
                current_block->continue_block = operands[1];
                current_block->is_loop_header = 1;
            }
            break;

        case SpvOpSelectionMerge:
            if (current_block && operand_count >= 1) {
                current_block->merge_block = operands[0];
                current_block->is_selection_header = 1;
            }
            break;

        default:
            if (current_block) {
                add_block_instr(current_block, pos);
            }
            if (operand_count >= 2) {
                SpvOp op = (SpvOp)opcode;
                int has_result_type = 0;
                int has_result = 0;
                switch (op) {
                case SpvOpLoad: case SpvOpAccessChain: case SpvOpInBoundsAccessChain:
                case SpvOpFAdd: case SpvOpFSub: case SpvOpFMul: case SpvOpFDiv: case SpvOpFRem: case SpvOpFMod:
                case SpvOpIAdd: case SpvOpISub: case SpvOpIMul: case SpvOpSDiv: case SpvOpUDiv: case SpvOpSRem: case SpvOpUMod: case SpvOpSMod:
                case SpvOpFNegate: case SpvOpSNegate:
                case SpvOpFOrdEqual: case SpvOpFOrdNotEqual: case SpvOpFOrdLessThan: case SpvOpFOrdGreaterThan:
                case SpvOpFOrdLessThanEqual: case SpvOpFOrdGreaterThanEqual:
                case SpvOpFUnordEqual: case SpvOpFUnordNotEqual: case SpvOpFUnordLessThan: case SpvOpFUnordGreaterThan:
                case SpvOpFUnordLessThanEqual: case SpvOpFUnordGreaterThanEqual:
                case SpvOpIEqual: case SpvOpINotEqual: case SpvOpSLessThan: case SpvOpSGreaterThan:
                case SpvOpSLessThanEqual: case SpvOpSGreaterThanEqual: case SpvOpULessThan: case SpvOpUGreaterThan:
                case SpvOpULessThanEqual: case SpvOpUGreaterThanEqual:
                case SpvOpLogicalAnd: case SpvOpLogicalOr: case SpvOpLogicalNot: case SpvOpLogicalEqual: case SpvOpLogicalNotEqual:
                case SpvOpBitwiseAnd: case SpvOpBitwiseOr: case SpvOpBitwiseXor: case SpvOpNot:
                case SpvOpShiftLeftLogical: case SpvOpShiftRightLogical: case SpvOpShiftRightArithmetic:
                case SpvOpCompositeConstruct: case SpvOpCompositeExtract: case SpvOpCompositeInsert: case SpvOpVectorShuffle:
                case SpvOpConvertFToS: case SpvOpConvertFToU: case SpvOpConvertSToF: case SpvOpConvertUToF:
                case SpvOpBitcast: case SpvOpFConvert: case SpvOpSConvert: case SpvOpUConvert:
                case SpvOpSelect: case SpvOpPhi: case SpvOpDot: case SpvOpVectorTimesScalar: case SpvOpMatrixTimesScalar:
                case SpvOpVectorTimesMatrix: case SpvOpMatrixTimesVector: case SpvOpMatrixTimesMatrix:
                case SpvOpExtInst: case SpvOpImageSampleImplicitLod: case SpvOpImageSampleExplicitLod:
                case SpvOpSampledImage: case SpvOpFunctionCall: case SpvOpTranspose:
                case SpvOpCopyObject: case SpvOpVectorExtractDynamic:
                case SpvOpArrayLength:
                    has_result_type = 1;
                    has_result = 1;
                    break;
                default:
                    break;
                }
                if (has_result_type && has_result && operand_count >= 2) {
                    uint32_t type_id = operands[0];
                    uint32_t result_id = operands[1];
                    if (result_id < c->id_bound) {
                        c->ids[result_id].kind = SPV_ID_INSTRUCTION;
                        c->ids[result_id].instruction.type_id = type_id;
                        c->ids[result_id].instruction.opcode = op;
                        int remaining = operand_count - 2;
                        if (remaining > 0) {
                            c->ids[result_id].instruction.operands = (uint32_t*)SPIRV_TO_SSIR_MALLOC(remaining * sizeof(uint32_t));
                            memcpy(c->ids[result_id].instruction.operands, &operands[2], remaining * sizeof(uint32_t));
                        }
                        c->ids[result_id].instruction.operand_count = remaining;
                    }
                }
            }
            break;
        }

        pos += wc;
    }

    for (int i = 0; i < c->pending_ep_count; i++) {
        PendingEntryPoint *ep = &c->pending_eps[i];
        for (int j = 0; j < c->function_count; j++) {
            if (c->functions[j].id == ep->func_id) {
                c->functions[j].exec_model = ep->exec_model;
                c->functions[j].is_entry_point = 1;
                if (ep->name && !c->functions[j].name) {
                    c->functions[j].name = strdup(ep->name);
                }
                c->functions[j].interface_vars = ep->interface_vars;
                c->functions[j].interface_var_count = ep->interface_var_count;
                ep->interface_vars = NULL;
                break;
            }
        }
    }

    for (int i = 0; i < c->pending_wg_count; i++) {
        PendingWorkgroupSize *wg = &c->pending_wgs[i];
        for (int j = 0; j < c->function_count; j++) {
            if (c->functions[j].id == wg->func_id) {
                c->functions[j].workgroup_size[0] = wg->workgroup_size[0];
                c->functions[j].workgroup_size[1] = wg->workgroup_size[1];
                c->functions[j].workgroup_size[2] = wg->workgroup_size[2];
                break;
            }
        }
    }

    return SPIRV_TO_SSIR_SUCCESS;
}

static uint32_t convert_type(Converter *c, uint32_t spv_type_id);

static uint32_t convert_scalar_type(Converter *c, uint32_t spv_type_id) {
    if (spv_type_id >= c->id_bound) return 0;
    SpvIdInfo *info = &c->ids[spv_type_id];
    if (info->kind != SPV_ID_TYPE) return 0;

    switch (info->type_info.kind) {
    case SPV_TYPE_VOID:
        return ssir_type_void(c->mod);
    case SPV_TYPE_BOOL:
        return ssir_type_bool(c->mod);
    case SPV_TYPE_INT:
        if (info->type_info.int_type.width == 32) {
            return info->type_info.int_type.signedness ? ssir_type_i32(c->mod) : ssir_type_u32(c->mod);
        }
        return 0;
    case SPV_TYPE_FLOAT:
        if (info->type_info.float_type.width == 32) return ssir_type_f32(c->mod);
        if (info->type_info.float_type.width == 16) return ssir_type_f16(c->mod);
        return 0;
    default:
        return 0;
    }
}

static SsirAddressSpace storage_class_to_addr_space(SpvStorageClass sc) {
    switch (sc) {
    case SpvStorageClassFunction: return SSIR_ADDR_FUNCTION;
    case SpvStorageClassPrivate: return SSIR_ADDR_PRIVATE;
    case SpvStorageClassWorkgroup: return SSIR_ADDR_WORKGROUP;
    case SpvStorageClassUniform: return SSIR_ADDR_UNIFORM;
    case SpvStorageClassUniformConstant: return SSIR_ADDR_UNIFORM_CONSTANT;
    case SpvStorageClassStorageBuffer: return SSIR_ADDR_STORAGE;
    case SpvStorageClassInput: return SSIR_ADDR_INPUT;
    case SpvStorageClassOutput: return SSIR_ADDR_OUTPUT;
    case SpvStorageClassPushConstant: return SSIR_ADDR_PUSH_CONSTANT;
    default: return SSIR_ADDR_FUNCTION;
    }
}

static SsirTextureDim spv_dim_to_ssir(SpvDim dim, uint32_t arrayed, uint32_t ms) {
    if (ms) return SSIR_TEX_MULTISAMPLED_2D;
    switch (dim) {
    case SpvDim1D: return SSIR_TEX_1D;
    case SpvDim2D: return arrayed ? SSIR_TEX_2D_ARRAY : SSIR_TEX_2D;
    case SpvDim3D: return SSIR_TEX_3D;
    case SpvDimCube: return arrayed ? SSIR_TEX_CUBE_ARRAY : SSIR_TEX_CUBE;
    default: return SSIR_TEX_2D;
    }
}

static uint32_t convert_type(Converter *c, uint32_t spv_type_id) {
    if (spv_type_id >= c->id_bound) return 0;
    SpvIdInfo *info = &c->ids[spv_type_id];
    if (info->kind != SPV_ID_TYPE) return 0;
    if (info->ssir_id) return info->ssir_id;

    uint32_t result = 0;

    switch (info->type_info.kind) {
    case SPV_TYPE_VOID:
    case SPV_TYPE_BOOL:
    case SPV_TYPE_INT:
    case SPV_TYPE_FLOAT:
        result = convert_scalar_type(c, spv_type_id);
        break;

    case SPV_TYPE_VECTOR: {
        uint32_t elem = convert_type(c, info->type_info.vector.component_type);
        result = ssir_type_vec(c->mod, elem, (uint8_t)info->type_info.vector.count);
        break;
    }

    case SPV_TYPE_MATRIX: {
        uint32_t col = convert_type(c, info->type_info.matrix.column_type);
        SpvIdInfo *col_info = &c->ids[info->type_info.matrix.column_type];
        uint8_t rows = (uint8_t)col_info->type_info.vector.count;
        result = ssir_type_mat(c->mod, col, (uint8_t)info->type_info.matrix.columns, rows);
        break;
    }

    case SPV_TYPE_ARRAY: {
        uint32_t elem = convert_type(c, info->type_info.array.element_type);
        uint32_t len_id = info->type_info.array.length_id;
        uint32_t len = 0;
        if (len_id < c->id_bound && c->ids[len_id].kind == SPV_ID_CONSTANT) {
            if (c->ids[len_id].constant.value_count > 0) {
                len = c->ids[len_id].constant.values[0];
            }
        }
        result = ssir_type_array(c->mod, elem, len);
        break;
    }

    case SPV_TYPE_RUNTIME_ARRAY: {
        uint32_t elem = convert_type(c, info->type_info.runtime_array.element_type);
        result = ssir_type_runtime_array(c->mod, elem);
        break;
    }

    case SPV_TYPE_STRUCT: {
        int mc = info->type_info.struct_type.member_count;
        uint32_t *members = NULL;
        uint32_t *offsets = NULL;
        if (mc > 0) {
            members = (uint32_t*)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
            offsets = (uint32_t*)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
            for (int i = 0; i < mc; i++) {
                members[i] = convert_type(c, info->type_info.struct_type.member_types[i]);
                if (!get_member_offset(c, spv_type_id, i, &offsets[i])) {
                    offsets[i] = 0;
                }
            }
        }
        const char *name = info->name;
        result = ssir_type_struct(c->mod, name, members, mc, offsets);
        SPIRV_TO_SSIR_FREE(members);
        SPIRV_TO_SSIR_FREE(offsets);
        break;
    }

    case SPV_TYPE_POINTER: {
        uint32_t pointee = convert_type(c, info->type_info.pointer.pointee_type);
        SsirAddressSpace space = storage_class_to_addr_space(info->type_info.pointer.storage);
        result = ssir_type_ptr(c->mod, pointee, space);
        break;
    }

    case SPV_TYPE_SAMPLER:
        result = ssir_type_sampler(c->mod);
        break;

    case SPV_TYPE_IMAGE: {
        SsirTextureDim dim = spv_dim_to_ssir(info->type_info.image.dim,
                                              info->type_info.image.arrayed,
                                              info->type_info.image.ms);
        if (info->type_info.image.depth == 1) {
            result = ssir_type_texture_depth(c->mod, dim);
        } else if (info->type_info.image.sampled == 2) {
            result = ssir_type_texture_storage(c->mod, dim, info->type_info.image.format, SSIR_ACCESS_READ_WRITE);
        } else {
            uint32_t sampled = convert_type(c, info->type_info.image.sampled_type);
            result = ssir_type_texture(c->mod, dim, sampled);
        }
        break;
    }

    case SPV_TYPE_SAMPLED_IMAGE: {
        result = convert_type(c, info->type_info.sampled_image.image_type);
        break;
    }

    case SPV_TYPE_FUNCTION:
        return 0;

    default:
        return 0;
    }

    info->ssir_id = result;
    return result;
}

static uint32_t convert_constant(Converter *c, uint32_t spv_const_id) {
    if (spv_const_id >= c->id_bound) return 0;
    SpvIdInfo *info = &c->ids[spv_const_id];
    if (info->kind != SPV_ID_CONSTANT) return 0;
    if (info->ssir_id) return info->ssir_id;

    uint32_t type_id = convert_type(c, info->constant.type_id);
    SsirType *type = ssir_get_type(c->mod, type_id);
    if (!type) return 0;

    uint32_t result = 0;

    if (info->constant.is_composite) {
        int count = info->constant.value_count;
        uint32_t *components = NULL;
        if (count > 0) {
            components = (uint32_t*)SPIRV_TO_SSIR_MALLOC(count * sizeof(uint32_t));
            for (int i = 0; i < count; i++) {
                components[i] = convert_constant(c, info->constant.values[i]);
            }
        }
        result = ssir_const_composite(c->mod, type_id, components, count);
        SPIRV_TO_SSIR_FREE(components);
    } else {
        switch (type->kind) {
        case SSIR_TYPE_BOOL:
            result = ssir_const_bool(c->mod, info->constant.value_count > 0 && info->constant.values[0] != 0);
            break;
        case SSIR_TYPE_I32:
            result = ssir_const_i32(c->mod, info->constant.value_count > 0 ? (int32_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_U32:
            result = ssir_const_u32(c->mod, info->constant.value_count > 0 ? info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_F32: {
            float fval = 0.0f;
            if (info->constant.value_count > 0) {
                memcpy(&fval, &info->constant.values[0], sizeof(float));
            }
            result = ssir_const_f32(c->mod, fval);
            break;
        }
        case SSIR_TYPE_F16:
            result = ssir_const_f16(c->mod, info->constant.value_count > 0 ? (uint16_t)info->constant.values[0] : 0);
            break;
        default:
            return 0;
        }
    }

    info->ssir_id = result;
    return result;
}

static SsirBuiltinVar spv_builtin_to_ssir(SpvBuiltIn builtin) {
    switch (builtin) {
    case SpvBuiltInVertexIndex: return SSIR_BUILTIN_VERTEX_INDEX;
    case SpvBuiltInInstanceIndex: return SSIR_BUILTIN_INSTANCE_INDEX;
    case SpvBuiltInPosition: return SSIR_BUILTIN_POSITION;
    case SpvBuiltInFrontFacing: return SSIR_BUILTIN_FRONT_FACING;
    case SpvBuiltInFragDepth: return SSIR_BUILTIN_FRAG_DEPTH;
    case SpvBuiltInSampleId: return SSIR_BUILTIN_SAMPLE_INDEX;
    case SpvBuiltInSampleMask: return SSIR_BUILTIN_SAMPLE_MASK;
    case SpvBuiltInLocalInvocationId: return SSIR_BUILTIN_LOCAL_INVOCATION_ID;
    case SpvBuiltInLocalInvocationIndex: return SSIR_BUILTIN_LOCAL_INVOCATION_INDEX;
    case SpvBuiltInGlobalInvocationId: return SSIR_BUILTIN_GLOBAL_INVOCATION_ID;
    case SpvBuiltInWorkgroupId: return SSIR_BUILTIN_WORKGROUP_ID;
    case SpvBuiltInNumWorkgroups: return SSIR_BUILTIN_NUM_WORKGROUPS;
    default: return SSIR_BUILTIN_NONE;
    }
}

static void convert_global_vars(Converter *c) {
    for (uint32_t i = 1; i < c->id_bound; i++) {
        SpvIdInfo *info = &c->ids[i];
        if (info->kind != SPV_ID_VARIABLE) continue;
        if (info->variable.storage_class == SpvStorageClassFunction) continue;

        uint32_t ptr_type = convert_type(c, info->variable.type_id);
        if (!ptr_type) continue;

        const char *name = info->name;
        uint32_t gid = ssir_global_var(c->mod, name, ptr_type);
        info->ssir_id = gid;

        uint32_t group_val, binding_val, location_val, builtin_val;
        if (has_decoration(c, i, SpvDecorationDescriptorSet, &group_val)) {
            ssir_global_set_group(c->mod, gid, group_val);
        }
        if (has_decoration(c, i, SpvDecorationBinding, &binding_val)) {
            ssir_global_set_binding(c->mod, gid, binding_val);
        }
        if (has_decoration(c, i, SpvDecorationLocation, &location_val)) {
            ssir_global_set_location(c->mod, gid, location_val);
        }
        if (has_decoration(c, i, SpvDecorationBuiltIn, &builtin_val)) {
            SsirBuiltinVar bv = spv_builtin_to_ssir((SpvBuiltIn)builtin_val);
            ssir_global_set_builtin(c->mod, gid, bv);
        }
        if (has_decoration(c, i, SpvDecorationFlat, NULL)) {
            ssir_global_set_interpolation(c->mod, gid, SSIR_INTERP_FLAT);
        } else if (has_decoration(c, i, SpvDecorationNoPerspective, NULL)) {
            ssir_global_set_interpolation(c->mod, gid, SSIR_INTERP_LINEAR);
        }

        if (info->variable.initializer) {
            uint32_t init_id = convert_constant(c, info->variable.initializer);
            if (init_id) {
                ssir_global_set_initializer(c->mod, gid, init_id);
            }
        }
    }
}

static uint32_t get_ssir_id(Converter *c, uint32_t spv_id) {
    if (spv_id >= c->id_bound) return 0;
    SpvIdInfo *info = &c->ids[spv_id];
    if (info->ssir_id) return info->ssir_id;

    switch (info->kind) {
    case SPV_ID_TYPE:
        return convert_type(c, spv_id);
    case SPV_ID_CONSTANT:
        return convert_constant(c, spv_id);
    case SPV_ID_VARIABLE:
    case SPV_ID_INSTRUCTION:
    case SPV_ID_PARAM:
        return info->ssir_id;
    default:
        return 0;
    }
}

static SsirBuiltinId glsl_ext_to_ssir_builtin(enum GLSLstd450 glsl_op) {
    switch (glsl_op) {
    case GLSLstd450Sin: return SSIR_BUILTIN_SIN;
    case GLSLstd450Cos: return SSIR_BUILTIN_COS;
    case GLSLstd450Tan: return SSIR_BUILTIN_TAN;
    case GLSLstd450Asin: return SSIR_BUILTIN_ASIN;
    case GLSLstd450Acos: return SSIR_BUILTIN_ACOS;
    case GLSLstd450Atan: return SSIR_BUILTIN_ATAN;
    case GLSLstd450Atan2: return SSIR_BUILTIN_ATAN2;
    case GLSLstd450Sinh: return SSIR_BUILTIN_SINH;
    case GLSLstd450Cosh: return SSIR_BUILTIN_COSH;
    case GLSLstd450Tanh: return SSIR_BUILTIN_TANH;
    case GLSLstd450Asinh: return SSIR_BUILTIN_ASINH;
    case GLSLstd450Acosh: return SSIR_BUILTIN_ACOSH;
    case GLSLstd450Atanh: return SSIR_BUILTIN_ATANH;
    case GLSLstd450Exp: return SSIR_BUILTIN_EXP;
    case GLSLstd450Exp2: return SSIR_BUILTIN_EXP2;
    case GLSLstd450Log: return SSIR_BUILTIN_LOG;
    case GLSLstd450Log2: return SSIR_BUILTIN_LOG2;
    case GLSLstd450Pow: return SSIR_BUILTIN_POW;
    case GLSLstd450Sqrt: return SSIR_BUILTIN_SQRT;
    case GLSLstd450InverseSqrt: return SSIR_BUILTIN_INVERSESQRT;
    case GLSLstd450FAbs:
    case GLSLstd450SAbs: return SSIR_BUILTIN_ABS;
    case GLSLstd450FSign:
    case GLSLstd450SSign: return SSIR_BUILTIN_SIGN;
    case GLSLstd450Floor: return SSIR_BUILTIN_FLOOR;
    case GLSLstd450Ceil: return SSIR_BUILTIN_CEIL;
    case GLSLstd450Round: return SSIR_BUILTIN_ROUND;
    case GLSLstd450Trunc: return SSIR_BUILTIN_TRUNC;
    case GLSLstd450Fract: return SSIR_BUILTIN_FRACT;
    case GLSLstd450FMin:
    case GLSLstd450SMin:
    case GLSLstd450UMin: return SSIR_BUILTIN_MIN;
    case GLSLstd450FMax:
    case GLSLstd450SMax:
    case GLSLstd450UMax: return SSIR_BUILTIN_MAX;
    case GLSLstd450FClamp:
    case GLSLstd450SClamp:
    case GLSLstd450UClamp: return SSIR_BUILTIN_CLAMP;
    case GLSLstd450FMix: return SSIR_BUILTIN_MIX;
    case GLSLstd450Step: return SSIR_BUILTIN_STEP;
    case GLSLstd450SmoothStep: return SSIR_BUILTIN_SMOOTHSTEP;
    case GLSLstd450Length: return SSIR_BUILTIN_LENGTH;
    case GLSLstd450Distance: return SSIR_BUILTIN_DISTANCE;
    case GLSLstd450Cross: return SSIR_BUILTIN_CROSS;
    case GLSLstd450Normalize: return SSIR_BUILTIN_NORMALIZE;
    case GLSLstd450FaceForward: return SSIR_BUILTIN_FACEFORWARD;
    case GLSLstd450Reflect: return SSIR_BUILTIN_REFLECT;
    case GLSLstd450Refract: return SSIR_BUILTIN_REFRACT;
    default: return SSIR_BUILTIN_COUNT;
    }
}

static void convert_function(Converter *c, SpvFunction *fn) {
    uint32_t ret_type = convert_type(c, fn->return_type);
    const char *name = fn->name ? fn->name : "fn";
    uint32_t func_id = ssir_function_create(c->mod, name, ret_type);
    c->ids[fn->id].ssir_id = func_id;

    for (int i = 0; i < fn->param_count; i++) {
        uint32_t param_spv = fn->params[i];
        if (param_spv >= c->id_bound) continue;
        SpvIdInfo *param_info = &c->ids[param_spv];
        uint32_t param_type = convert_type(c, param_info->param.type_id);
        const char *param_name = param_info->name;
        uint32_t param_id = ssir_function_add_param(c->mod, func_id, param_name, param_type);
        param_info->ssir_id = param_id;
    }

    for (int i = 0; i < fn->local_var_count; i++) {
        uint32_t var_spv = fn->local_vars[i];
        if (var_spv >= c->id_bound) continue;
        SpvIdInfo *var_info = &c->ids[var_spv];
        uint32_t var_type = convert_type(c, var_info->variable.type_id);
        const char *var_name = var_info->name;
        uint32_t local_id = ssir_function_add_local(c->mod, func_id, var_name, var_type);
        var_info->ssir_id = local_id;
    }

    for (int bi = 0; bi < fn->block_count; bi++) {
        SpvBasicBlock *blk = &fn->blocks[bi];
        const char *blk_name = NULL;
        if (blk->label_id < c->id_bound) {
            blk_name = c->ids[blk->label_id].name;
        }
        uint32_t block_id = ssir_block_create(c->mod, func_id, blk_name);
        if (blk->label_id < c->id_bound) {
            c->ids[blk->label_id].ssir_id = block_id;
        }
    }

    for (int bi = 0; bi < fn->block_count; bi++) {
        SpvBasicBlock *blk = &fn->blocks[bi];
        uint32_t block_id = c->ids[blk->label_id].ssir_id;

        if (blk->is_loop_header) {
            uint32_t merge_id = 0, cont_id = 0;
            if (blk->merge_block < c->id_bound) merge_id = c->ids[blk->merge_block].ssir_id;
            if (blk->continue_block < c->id_bound) cont_id = c->ids[blk->continue_block].ssir_id;
            if (merge_id && cont_id) {
                ssir_build_loop_merge(c->mod, func_id, block_id, merge_id, cont_id);
            }
        }

        for (int ii = 0; ii < blk->instruction_count; ii++) {
            uint32_t inst_pos = blk->instructions[ii];
            uint32_t word0 = c->spirv[inst_pos];
            uint16_t opcode = word0 & 0xFFFF;
            uint16_t wc = word0 >> 16;
            const uint32_t *operands = &c->spirv[inst_pos + 1];
            int operand_count = wc - 1;

            SpvOp op = (SpvOp)opcode;

            switch (op) {
            case SpvOpStore:
                if (operand_count >= 2) {
                    uint32_t ptr = get_ssir_id(c, operands[0]);
                    uint32_t val = get_ssir_id(c, operands[1]);
                    ssir_build_store(c->mod, func_id, block_id, ptr, val);
                }
                break;

            case SpvOpLoad:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t ptr = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_load(c->mod, func_id, block_id, type_id, ptr);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t base = get_ssir_id(c, operands[2]);
                    int idx_count = operand_count - 3;
                    uint32_t *indices = NULL;
                    if (idx_count > 0) {
                        indices = (uint32_t*)SPIRV_TO_SSIR_MALLOC(idx_count * sizeof(uint32_t));
                        for (int j = 0; j < idx_count; j++) {
                            indices[j] = get_ssir_id(c, operands[3 + j]);
                        }
                    }
                    uint32_t r = ssir_build_access(c->mod, func_id, block_id, type_id, base, indices, idx_count);
                    c->ids[result_id].ssir_id = r;
                    SPIRV_TO_SSIR_FREE(indices);
                }
                break;

            case SpvOpFAdd: case SpvOpIAdd:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_add(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFSub: case SpvOpISub:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_sub(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFMul: case SpvOpIMul:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mul(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFDiv: case SpvOpSDiv: case SpvOpUDiv:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_div(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFRem: case SpvOpFMod: case SpvOpSRem: case SpvOpSMod: case SpvOpUMod:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mod(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFNegate: case SpvOpSNegate:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_neg(c->mod, func_id, block_id, type_id, a);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpMatrixTimesMatrix:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mat_mul(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpMatrixTimesVector:
            case SpvOpVectorTimesMatrix:
            case SpvOpVectorTimesScalar:
            case SpvOpMatrixTimesScalar:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mul(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpTranspose:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t m = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_mat_transpose(c->mod, func_id, block_id, type_id, m);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpBitwiseAnd:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_bit_and(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpBitwiseOr:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_bit_or(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpBitwiseXor:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_bit_xor(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpNot:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_bit_not(c->mod, func_id, block_id, type_id, a);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpShiftLeftLogical:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_shl(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpShiftRightLogical:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_shr_logical(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpShiftRightArithmetic:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_shr(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFOrdEqual: case SpvOpFUnordEqual: case SpvOpIEqual: case SpvOpLogicalEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_eq(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFOrdNotEqual: case SpvOpFUnordNotEqual: case SpvOpINotEqual: case SpvOpLogicalNotEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_ne(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFOrdLessThan: case SpvOpFUnordLessThan: case SpvOpSLessThan: case SpvOpULessThan:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_lt(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFOrdLessThanEqual: case SpvOpFUnordLessThanEqual: case SpvOpSLessThanEqual: case SpvOpULessThanEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_le(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFOrdGreaterThan: case SpvOpFUnordGreaterThan: case SpvOpSGreaterThan: case SpvOpUGreaterThan:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_gt(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpFOrdGreaterThanEqual: case SpvOpFUnordGreaterThanEqual: case SpvOpSGreaterThanEqual: case SpvOpUGreaterThanEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_ge(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpLogicalAnd:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_and(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpLogicalOr:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_or(c->mod, func_id, block_id, type_id, a, b);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpLogicalNot:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_not(c->mod, func_id, block_id, type_id, a);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpCompositeConstruct:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    int comp_count = operand_count - 2;
                    uint32_t *comps = NULL;
                    if (comp_count > 0) {
                        comps = (uint32_t*)SPIRV_TO_SSIR_MALLOC(comp_count * sizeof(uint32_t));
                        for (int j = 0; j < comp_count; j++) {
                            comps[j] = get_ssir_id(c, operands[2 + j]);
                        }
                    }
                    uint32_t r = ssir_build_construct(c->mod, func_id, block_id, type_id, comps, comp_count);
                    c->ids[result_id].ssir_id = r;
                    SPIRV_TO_SSIR_FREE(comps);
                }
                break;

            case SpvOpCompositeExtract:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t composite = get_ssir_id(c, operands[2]);
                    uint32_t index = operands[3];
                    uint32_t r = ssir_build_extract(c->mod, func_id, block_id, type_id, composite, index);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpCompositeInsert:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t value = get_ssir_id(c, operands[2]);
                    uint32_t composite = get_ssir_id(c, operands[3]);
                    uint32_t index = operands[4];
                    uint32_t r = ssir_build_insert(c->mod, func_id, block_id, type_id, composite, value, index);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpVectorShuffle:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t v1 = get_ssir_id(c, operands[2]);
                    uint32_t v2 = get_ssir_id(c, operands[3]);
                    int idx_count = operand_count - 4;
                    uint32_t *indices = NULL;
                    if (idx_count > 0) {
                        indices = (uint32_t*)SPIRV_TO_SSIR_MALLOC(idx_count * sizeof(uint32_t));
                        memcpy(indices, &operands[4], idx_count * sizeof(uint32_t));
                    }
                    uint32_t r = ssir_build_shuffle(c->mod, func_id, block_id, type_id, v1, v2, indices, idx_count);
                    c->ids[result_id].ssir_id = r;
                    SPIRV_TO_SSIR_FREE(indices);
                }
                break;

            case SpvOpVectorExtractDynamic:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t vec = get_ssir_id(c, operands[2]);
                    uint32_t idx = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_extract_dyn(c->mod, func_id, block_id, type_id, vec, idx);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpSelect:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t cond = get_ssir_id(c, operands[2]);
                    uint32_t t = get_ssir_id(c, operands[3]);
                    uint32_t f = get_ssir_id(c, operands[4]);
                    uint32_t args[3] = { cond, t, f };
                    uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, SSIR_BUILTIN_SELECT, args, 3);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpDot:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t args[2] = { a, b };
                    uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, SSIR_BUILTIN_DOT, args, 2);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpConvertFToS: case SpvOpConvertFToU: case SpvOpConvertSToF: case SpvOpConvertUToF:
            case SpvOpFConvert: case SpvOpSConvert: case SpvOpUConvert:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t val = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_convert(c->mod, func_id, block_id, type_id, val);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpBitcast:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t val = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_bitcast(c->mod, func_id, block_id, type_id, val);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpCopyObject:
                if (operand_count >= 3) {
                    uint32_t result_id = operands[1];
                    uint32_t src = get_ssir_id(c, operands[2]);
                    c->ids[result_id].ssir_id = src;
                }
                break;

            case SpvOpPhi:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    int pair_count = (operand_count - 2) / 2;
                    uint32_t *incoming = NULL;
                    if (pair_count > 0) {
                        incoming = (uint32_t*)SPIRV_TO_SSIR_MALLOC(pair_count * 2 * sizeof(uint32_t));
                        for (int j = 0; j < pair_count; j++) {
                            incoming[j * 2] = get_ssir_id(c, operands[2 + j * 2]);
                            incoming[j * 2 + 1] = c->ids[operands[3 + j * 2]].ssir_id;
                        }
                    }
                    uint32_t r = ssir_build_phi(c->mod, func_id, block_id, type_id, incoming, pair_count * 2);
                    c->ids[result_id].ssir_id = r;
                    SPIRV_TO_SSIR_FREE(incoming);
                }
                break;

            case SpvOpBranch:
                if (operand_count >= 1) {
                    uint32_t target = c->ids[operands[0]].ssir_id;
                    ssir_build_branch(c->mod, func_id, block_id, target);
                }
                break;

            case SpvOpBranchConditional:
                if (operand_count >= 3) {
                    uint32_t cond = get_ssir_id(c, operands[0]);
                    uint32_t true_block = c->ids[operands[1]].ssir_id;
                    uint32_t false_block = c->ids[operands[2]].ssir_id;
                    if (blk->is_selection_header && blk->merge_block) {
                        uint32_t merge = c->ids[blk->merge_block].ssir_id;
                        ssir_build_branch_cond_merge(c->mod, func_id, block_id, cond, true_block, false_block, merge);
                    } else {
                        ssir_build_branch_cond(c->mod, func_id, block_id, cond, true_block, false_block);
                    }
                }
                break;

            case SpvOpReturn:
                ssir_build_return_void(c->mod, func_id, block_id);
                break;

            case SpvOpReturnValue:
                if (operand_count >= 1) {
                    uint32_t val = get_ssir_id(c, operands[0]);
                    ssir_build_return(c->mod, func_id, block_id, val);
                }
                break;

            case SpvOpUnreachable:
                ssir_build_unreachable(c->mod, func_id, block_id);
                break;

            case SpvOpFunctionCall:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t callee = c->ids[operands[2]].ssir_id;
                    int arg_count = operand_count - 3;
                    uint32_t *args = NULL;
                    if (arg_count > 0) {
                        args = (uint32_t*)SPIRV_TO_SSIR_MALLOC(arg_count * sizeof(uint32_t));
                        for (int j = 0; j < arg_count; j++) {
                            args[j] = get_ssir_id(c, operands[3 + j]);
                        }
                    }
                    uint32_t r = ssir_build_call(c->mod, func_id, block_id, type_id, callee, args, arg_count);
                    c->ids[result_id].ssir_id = r;
                    SPIRV_TO_SSIR_FREE(args);
                }
                break;

            case SpvOpExtInst:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t set = operands[2];
                    uint32_t inst = operands[3];
                    if (set == c->glsl_ext_id) {
                        SsirBuiltinId bid = glsl_ext_to_ssir_builtin((enum GLSLstd450)inst);
                        if (bid != SSIR_BUILTIN_COUNT) {
                            int arg_count = operand_count - 4;
                            uint32_t *args = NULL;
                            if (arg_count > 0) {
                                args = (uint32_t*)SPIRV_TO_SSIR_MALLOC(arg_count * sizeof(uint32_t));
                                for (int j = 0; j < arg_count; j++) {
                                    args[j] = get_ssir_id(c, operands[4 + j]);
                                }
                            }
                            uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, bid, args, arg_count);
                            c->ids[result_id].ssir_id = r;
                            SPIRV_TO_SSIR_FREE(args);
                        }
                    }
                }
                break;

            case SpvOpArrayLength:
                if (operand_count >= 4) {
                    uint32_t result_id = operands[1];
                    uint32_t ptr = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_array_len(c->mod, func_id, block_id, ptr);
                    c->ids[result_id].ssir_id = r;
                }
                break;

            case SpvOpControlBarrier:
                ssir_build_barrier(c->mod, func_id, block_id, SSIR_BARRIER_WORKGROUP);
                break;

            default:
                break;
            }
        }
    }
}

static SsirStage exec_model_to_stage(SpvExecutionModel model) {
    switch (model) {
    case SpvExecutionModelVertex: return SSIR_STAGE_VERTEX;
    case SpvExecutionModelFragment: return SSIR_STAGE_FRAGMENT;
    case SpvExecutionModelGLCompute: return SSIR_STAGE_COMPUTE;
    default: return SSIR_STAGE_COMPUTE;
    }
}

static void convert_entry_points(Converter *c) {
    for (int i = 0; i < c->function_count; i++) {
        SpvFunction *fn = &c->functions[i];
        if (!fn->is_entry_point) continue;

        uint32_t func_ssir_id = c->ids[fn->id].ssir_id;
        SsirStage stage = exec_model_to_stage(fn->exec_model);
        const char *name = fn->name ? fn->name : "main";

        uint32_t ep_idx = ssir_entry_point_create(c->mod, stage, func_ssir_id, name);

        for (int j = 0; j < fn->interface_var_count; j++) {
            uint32_t var_id = fn->interface_vars[j];
            if (var_id < c->id_bound) {
                uint32_t ssir_var = c->ids[var_id].ssir_id;
                if (ssir_var) {
                    ssir_entry_point_add_interface(c->mod, ep_idx, ssir_var);
                }
            }
        }

        if (stage == SSIR_STAGE_COMPUTE) {
            ssir_entry_point_set_workgroup_size(c->mod, ep_idx,
                fn->workgroup_size[0], fn->workgroup_size[1], fn->workgroup_size[2]);
        }
    }
}

static void free_converter(Converter *c) {
    if (!c) return;
    if (c->ids) {
        for (uint32_t i = 0; i < c->id_bound; i++) {
            SPIRV_TO_SSIR_FREE(c->ids[i].name);
            if (c->ids[i].decorations) {
                for (int j = 0; j < c->ids[i].decoration_count; j++) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].decorations[j].literals);
                }
                SPIRV_TO_SSIR_FREE(c->ids[i].decorations);
            }
            if (c->ids[i].member_decorations) {
                for (int j = 0; j < c->ids[i].member_decoration_count; j++) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].member_decorations[j].literals);
                }
                SPIRV_TO_SSIR_FREE(c->ids[i].member_decorations);
            }
            if (c->ids[i].member_names) {
                for (int j = 0; j < c->ids[i].member_name_count; j++) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].member_names[j]);
                }
                SPIRV_TO_SSIR_FREE(c->ids[i].member_names);
            }
            if (c->ids[i].kind == SPV_ID_TYPE) {
                if (c->ids[i].type_info.kind == SPV_TYPE_STRUCT && c->ids[i].type_info.struct_type.member_types) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].type_info.struct_type.member_types);
                }
                if (c->ids[i].type_info.kind == SPV_TYPE_FUNCTION && c->ids[i].type_info.function.param_types) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].type_info.function.param_types);
                }
            }
            if (c->ids[i].kind == SPV_ID_CONSTANT && c->ids[i].constant.values) {
                SPIRV_TO_SSIR_FREE(c->ids[i].constant.values);
            }
            if (c->ids[i].kind == SPV_ID_INSTRUCTION && c->ids[i].instruction.operands) {
                SPIRV_TO_SSIR_FREE(c->ids[i].instruction.operands);
            }
        }
        SPIRV_TO_SSIR_FREE(c->ids);
    }
    if (c->functions) {
        for (int i = 0; i < c->function_count; i++) {
            SPIRV_TO_SSIR_FREE(c->functions[i].name);
            SPIRV_TO_SSIR_FREE(c->functions[i].params);
            SPIRV_TO_SSIR_FREE(c->functions[i].interface_vars);
            SPIRV_TO_SSIR_FREE(c->functions[i].local_vars);
            for (int j = 0; j < c->functions[i].block_count; j++) {
                SPIRV_TO_SSIR_FREE(c->functions[i].blocks[j].instructions);
            }
            SPIRV_TO_SSIR_FREE(c->functions[i].blocks);
        }
        SPIRV_TO_SSIR_FREE(c->functions);
    }
    if (c->pending_eps) {
        for (int i = 0; i < c->pending_ep_count; i++) {
            SPIRV_TO_SSIR_FREE(c->pending_eps[i].name);
            SPIRV_TO_SSIR_FREE(c->pending_eps[i].interface_vars);
        }
        SPIRV_TO_SSIR_FREE(c->pending_eps);
    }
    SPIRV_TO_SSIR_FREE(c->pending_wgs);
}

SpirvToSsirResult spirv_to_ssir(
    const uint32_t *spirv,
    size_t word_count,
    const SpirvToSsirOptions *opts,
    SsirModule **out_module,
    char **out_error
) {
    if (!spirv || word_count < 5 || !out_module) {
        if (out_error) *out_error = strdup("Invalid input");
        return SPIRV_TO_SSIR_INVALID_SPIRV;
    }

    if (spirv[0] != SPV_MAGIC) {
        if (out_error) *out_error = strdup("Invalid SPIR-V magic number");
        return SPIRV_TO_SSIR_INVALID_SPIRV;
    }

    Converter c;
    memset(&c, 0, sizeof(c));
    c.spirv = spirv;
    c.word_count = word_count;
    c.version = spirv[1];
    c.generator = spirv[2];
    c.id_bound = spirv[3];
    c.opts = opts;

    c.ids = (SpvIdInfo*)SPIRV_TO_SSIR_MALLOC(c.id_bound * sizeof(SpvIdInfo));
    if (!c.ids) {
        if (out_error) *out_error = strdup("Out of memory");
        return SPIRV_TO_SSIR_INTERNAL_ERROR;
    }
    memset(c.ids, 0, c.id_bound * sizeof(SpvIdInfo));
    for (uint32_t i = 0; i < c.id_bound; i++) {
        c.ids[i].id = i;
    }

    c.mod = ssir_module_create();
    if (!c.mod) {
        free_converter(&c);
        if (out_error) *out_error = strdup("Failed to create SSIR module");
        return SPIRV_TO_SSIR_INTERNAL_ERROR;
    }

    SpirvToSsirResult result = parse_spirv(&c);
    if (result != SPIRV_TO_SSIR_SUCCESS) {
        ssir_module_destroy(c.mod);
        if (out_error) *out_error = strdup(c.last_error);
        free_converter(&c);
        return result;
    }

    for (uint32_t i = 1; i < c.id_bound; i++) {
        if (c.ids[i].kind == SPV_ID_TYPE) {
            convert_type(&c, i);
        }
    }

    for (uint32_t i = 1; i < c.id_bound; i++) {
        if (c.ids[i].kind == SPV_ID_CONSTANT) {
            convert_constant(&c, i);
        }
    }

    convert_global_vars(&c);

    for (int i = 0; i < c.function_count; i++) {
        convert_function(&c, &c.functions[i]);
    }

    convert_entry_points(&c);

    *out_module = c.mod;
    free_converter(&c);
    return SPIRV_TO_SSIR_SUCCESS;
}

const char *spirv_to_ssir_result_string(SpirvToSsirResult result) {
    switch (result) {
    case SPIRV_TO_SSIR_SUCCESS: return "Success";
    case SPIRV_TO_SSIR_INVALID_SPIRV: return "Invalid SPIR-V";
    case SPIRV_TO_SSIR_UNSUPPORTED_FEATURE: return "Unsupported feature";
    case SPIRV_TO_SSIR_INTERNAL_ERROR: return "Internal error";
    default: return "Unknown error";
    }
}

void spirv_to_ssir_free(void *p) {
    SPIRV_TO_SSIR_FREE(p);
}
