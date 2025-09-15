// BEGIN FILE wgsl_lower.h
#ifndef WGSL_LOWER_H
#define WGSL_LOWER_H

#include <stdint.h>
#include <stddef.h>
#include <spirv/unified1/spirv.h>

#include "wgsl_parser.h"
#include "wgsl_resolve.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque lowering context
typedef struct WgslLower WgslLower;

// Result code
typedef enum {
    WGSL_LOWER_OK = 0,
    WGSL_LOWER_ERR_INVALID_INPUT,
    WGSL_LOWER_ERR_UNSUPPORTED,
    WGSL_LOWER_ERR_INTERNAL,
    WGSL_LOWER_ERR_OOM
} WgslLowerResult;

// Target environment
typedef enum {
    WGSL_LOWER_ENV_VULKAN_1_1 = 1,
    WGSL_LOWER_ENV_VULKAN_1_2,
    WGSL_LOWER_ENV_VULKAN_1_3,
    WGSL_LOWER_ENV_WEBGPU
} WgslLowerEnv;

// Numeric packing for matrices/vectors in interface I/O
typedef enum {
    WGSL_LOWER_PACK_DEFAULT = 0,   // per target env
    WGSL_LOWER_PACK_STD430,
    WGSL_LOWER_PACK_STD140
} WgslLowerPacking;

// Options for lowering
typedef struct {
    uint32_t spirv_version;             // e.g. SPV_VERSION(1,5) -> 0x00010500; 0 = choose per env
    WgslLowerEnv env;                   // default WGSL_LOWER_ENV_WEBGPU
    WgslLowerPacking packing;           // default WGSL_LOWER_PACK_DEFAULT
    int enable_debug_names;             // OpName/OpMemberName
    int enable_line_info;               // OpLine
    int zero_initialize_vars;           // emit explicit initializers when needed
    int relax_block_layout;             // enable ScalarBlockLayout if supported
    int use_khr_shader_draw_parameters; // allow BuiltInBaseInstance etc.
    uint32_t id_bound_hint;             // 0 = auto
} WgslLowerOptions;

// Minimal reflection for produced module
typedef struct {
    const char *name;
    WgslStage stage;
    uint32_t function_id;    // SPIR-V ID for OpEntryPoint function
    uint32_t interface_count;
    const uint32_t *interface_ids; // IDs listed in OpEntryPoint
} WgslLowerEntrypointInfo;

typedef struct {
    uint32_t capability_count;
    const SpvCapability *capabilities; // pointers valid while WgslLower lives
    uint32_t extension_count;
    const char *const *extensions;     // null-terminated strings
} WgslLowerModuleFeatures;

// Build/destroy
WgslLower *wgsl_lower_create(const WgslAstNode *program,
                             const WgslResolver *resolver,
                             const WgslLowerOptions *opts);

void wgsl_lower_destroy(WgslLower *lower);

// One-shot convenience API
// Allocates and returns a SPIR-V word buffer. Caller frees with wgsl_lower_free.
WgslLowerResult wgsl_lower_emit_spirv(const WgslAstNode *program,
                                      const WgslResolver *resolver,
                                      const WgslLowerOptions *opts,
                                      uint32_t **out_words,
                                      size_t *out_word_count);

// Serialize module from an existing lowering context
// If out_words==NULL, only sets out_word_count.
WgslLowerResult wgsl_lower_serialize(const WgslLower *lower,
                                     uint32_t **out_words,
                                     size_t *out_word_count);

// Serialize into caller-provided buffer. Writes at most max_words.
// If buffer is NULL or too small, returns required word count in out_written and WGSL_LOWER_ERR_INVALID_INPUT.
WgslLowerResult wgsl_lower_serialize_into(const WgslLower *lower,
                                          uint32_t *out_words,
                                          size_t max_words,
                                          size_t *out_written);

// Introspection
const char *wgsl_lower_last_error(const WgslLower *lower);

const WgslLowerModuleFeatures *wgsl_lower_module_features(const WgslLower *lower);

const WgslLowerEntrypointInfo *wgsl_lower_entrypoints(const WgslLower *lower, int *out_count);

// Map WGSL AST nodes to resulting SPIR-V IDs when applicable.
// Returns 0 if the node does not materialize into an SSA/object result.
uint32_t wgsl_lower_node_result_id(const WgslLower *lower, const WgslAstNode *node);

// Map a symbol id (from resolver) to the declared SPIR-V ID, or 0 if not present.
uint32_t wgsl_lower_symbol_result_id(const WgslLower *lower, int symbol_id);

// Utilities
void wgsl_lower_free(void *p);

#ifdef __cplusplus
}
#endif

#endif // WGSL_LOWER_H
// END FILE wgsl_lower.h

