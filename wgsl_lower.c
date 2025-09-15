// BEGIN FILE wgsl_lower.c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "wgsl_lower.h"

// ---------- Internal helpers ----------

#ifndef WGSL_MALLOC
#define WGSL_MALLOC(SZ) calloc(1, (SZ))
#endif
#ifndef WGSL_REALLOC
#define WGSL_REALLOC(P, SZ) realloc((P), (SZ))
#endif
#ifndef WGSL_FREE
#define WGSL_FREE(P) free((P))
#endif

typedef struct {
    uint32_t *data;
    size_t    len;
    size_t    cap;
} WordBuf;

static void wb_init(WordBuf *wb) {
    wb->data = NULL;
    wb->len = 0;
    wb->cap = 0;
}
static void wb_free(WordBuf *wb) {
    WGSL_FREE(wb->data);
    wb->data = NULL;
    wb->len = wb->cap = 0;
}
static int wb_reserve(WordBuf *wb, size_t need) {
    if (wb->len + need <= wb->cap) return 1;
    size_t ncap = wb->cap ? wb->cap : 64;
    while (ncap < wb->len + need) ncap *= 2;
    void *nd = WGSL_REALLOC(wb->data, ncap * sizeof(uint32_t));
    if (!nd) return 0;
    wb->data = (uint32_t*)nd;
    wb->cap = ncap;
    return 1;
}
static int wb_push_u32(WordBuf *wb, uint32_t w) {
    if (!wb_reserve(wb, 1)) return 0;
    wb->data[wb->len++] = w;
    return 1;
}
static int wb_push_many(WordBuf *wb, const uint32_t *src, size_t n) {
    if (!wb_reserve(wb, n)) return 0;
    memcpy(wb->data + wb->len, src, n * sizeof(uint32_t));
    wb->len += n;
    return 1;
}

static uint32_t make_string_lit(const char *s, uint32_t **out_words, size_t *out_count) {
    // Returns number of words written through out_words/count.
    // SPIR-V strings are null-terminated and packed into 32-bit words.
    size_t n = strlen(s) + 1; // include null
    size_t words = (n + 3) / 4;
    uint32_t *buf = (uint32_t*)WGSL_MALLOC(words * sizeof(uint32_t));
    if (!buf) { *out_words = NULL; *out_count = 0; return 0; }
    memset(buf, 0, words * sizeof(uint32_t));
    memcpy(buf, s, n);
    *out_words = buf;
    *out_count = words;
    return (uint32_t)words;
}

typedef struct {
    WordBuf   words;        // full module words
    size_t    header_index; // index of header word 3 for bound patch
    uint32_t  next_id;      // next fresh id
} SpirvModule;

static int spv_init(SpirvModule *m, uint32_t version /* e.g. 0x00010400 */) {
    wb_init(&m->words);
    m->next_id = 1; // IDs start at 1
    // Header: magic, version, generator, bound, reserved
    // We will patch "bound" later
    if (!wb_push_u32(&m->words, 0x07230203)) return 0;
    if (!wb_push_u32(&m->words, version)) return 0;
    if (!wb_push_u32(&m->words, 0)) return 0;               // generator
    m->header_index = m->words.len;                         // position of bound
    if (!wb_push_u32(&m->words, 0)) return 0;               // bound (patch)
    if (!wb_push_u32(&m->words, 0)) return 0;               // reserved
    return 1;
}
static void spv_finish(SpirvModule *m) {
    // patch bound = next_id
    if (m->header_index < m->words.len) {
        m->words.data[m->header_index] = m->next_id;
    }
}
static void spv_free(SpirvModule *m) {
    wb_free(&m->words);
    m->header_index = 0;
    m->next_id = 1;
}

static uint32_t spv_fresh_id(SpirvModule *m) { return m->next_id++; }

static int spv_op_begin(WordBuf *wb, SpvOp op, uint16_t wc_placeholder_idx, uint16_t *out_wc_index) {
    (void)wc_placeholder_idx;
    if (!wb_reserve(wb, 1)) return 0;
    // push placeholder, will be overwritten when finalized
    *out_wc_index = (uint16_t)wb->len;
    wb->data[wb->len++] = ((uint32_t)1 << 16) | (uint16_t)op; // initial wc=1
    return 1;
}
static void spv_op_patch_wc(WordBuf *wb, uint16_t wc_index, uint16_t wc) {
    uint32_t op = wb->data[wc_index] & 0x0000FFFFu;
    wb->data[wc_index] = ((uint32_t)wc << 16) | op;
}
static int spv_op_end(WordBuf *wb, uint16_t wc_index) {
    // compute wc from difference
    uint16_t wc = (uint16_t)(wb->len - wc_index);
    spv_op_patch_wc(wb, wc_index, wc);
    return 1;
}
static int spv_emit_u32(WordBuf *wb, uint32_t v) { return wb_push_u32(wb, v); }
static int spv_emit_words(WordBuf *wb, const uint32_t *v, size_t n) { return wb_push_many(wb, v, n); }

// ---------- Lowering context ----------

struct WgslLower {
    const WgslAstNode     *program;
    const WgslResolver    *resolver;
    WgslLowerOptions       opts;

    SpirvModule            mod;

    // reflection
    WgslLowerModuleFeatures features;
    SpvCapability          cap_buf[4];
    const char            *ext_buf[2];

    WgslLowerEntrypointInfo *eps;
    int                      ep_count;

    char last_error[256];

    // basic types
    uint32_t id_void;
    uint32_t id_fn_void_void;
    uint32_t id_extinst_glsl; // optional
};

static void set_error(WgslLower *l, const char *msg) {
    if (!l) return;
    size_t n = strlen(msg);
    if (n >= sizeof(l->last_error)) n = sizeof(l->last_error) - 1;
    memcpy(l->last_error, msg, n);
    l->last_error[n] = 0;
}

static SpvExecutionModel stage_to_model(WgslStage s) {
    switch (s) {
        case WGSL_STAGE_VERTEX:   return SpvExecutionModelVertex;
        case WGSL_STAGE_FRAGMENT: return SpvExecutionModelFragment;
        case WGSL_STAGE_COMPUTE:  return SpvExecutionModelGLCompute;
        default:                  return SpvExecutionModelMax; // invalid
    }
}

// ---------- Minimal type system ----------

static int emit_type_void(WgslLower *l) {
    if (l->id_void) return 1;
    uint16_t idx;
    if (!spv_op_begin(&l->mod.words, SpvOpTypeVoid, 0, &idx)) return 0;
    l->id_void = spv_fresh_id(&l->mod);
    if (!spv_emit_u32(&l->mod.words, l->id_void)) return 0;
    spv_op_end(&l->mod.words, idx);
    return 1;
}

static int emit_type_function_void_void(WgslLower *l) {
    if (l->id_fn_void_void) return 1;
    if (!emit_type_void(l)) return 0;
    uint16_t idx;
    if (!spv_op_begin(&l->mod.words, SpvOpTypeFunction, 0, &idx)) return 0;
    l->id_fn_void_void = spv_fresh_id(&l->mod);
    if (!spv_emit_u32(&l->mod.words, l->id_fn_void_void)) return 0; // result id
    if (!spv_emit_u32(&l->mod.words, l->id_void)) return 0;         // return type
    // no parameters
    spv_op_end(&l->mod.words, idx);
    return 1;
}

// ---------- Module prolog ----------

static int emit_capabilities_memory_model(WgslLower *l) {
    // Capability Shader
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpCapability, 0, &idx)) return 0;
        if (!spv_emit_u32(&l->mod.words, SpvCapabilityShader)) return 0;
        spv_op_end(&l->mod.words, idx);
    }
#ifdef SpvCapabilityScalarBlockLayout
    if (l->opts.relax_block_layout) {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpCapability, 0, &idx)) return 0;
        if (!spv_emit_u32(&l->mod.words, SpvCapabilityScalarBlockLayout)) return 0;
        spv_op_end(&l->mod.words, idx);
    }
#endif

    // Import GLSL.std.450 for GLSL memory model selection and possible math
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpExtInstImport, 0, &idx)) return 0;
        l->id_extinst_glsl = spv_fresh_id(&l->mod);
        if (!spv_emit_u32(&l->mod.words, l->id_extinst_glsl)) return 0;
        uint32_t *strw = NULL; size_t wn = 0;
        make_string_lit("GLSL.std.450", &strw, &wn);
        int ok = spv_emit_words(&l->mod.words, strw, wn);
        WGSL_FREE(strw);
        if (!ok) return 0;
        spv_op_end(&l->mod.words, idx);
    }

    // Memory model
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpMemoryModel, 0, &idx)) return 0;
        if (!spv_emit_u32(&l->mod.words, SpvAddressingModelLogical)) return 0;
        if (!spv_emit_u32(&l->mod.words, SpvMemoryModelGLSL450)) return 0;
        spv_op_end(&l->mod.words, idx);
    }

    // Record features reflection
    l->features.capabilities = l->cap_buf;
    l->features.capability_count = 1 + (l->opts.relax_block_layout ? 1u : 0u);
    l->cap_buf[0] = SpvCapabilityShader;
    #ifdef SpvCapabilityScalarBlockLayout
    if (l->opts.relax_block_layout) l->cap_buf[1] = SpvCapabilityScalarBlockLayout;
    #endif

    l->features.extensions = (const char *const *)l->ext_buf;
    l->features.extension_count = 1;
    l->ext_buf[0] = "GLSL.std.450";

    return 1;
}

static int emit_debug_name(WgslLower *l, uint32_t target_id, const char *name) {
    if (!l->opts.enable_debug_names || !name || !*name) return 1;
    uint16_t idx;
    if (!spv_op_begin(&l->mod.words, SpvOpName, 0, &idx)) return 0;
    if (!spv_emit_u32(&l->mod.words, target_id)) return 0;
    uint32_t *strw = NULL; size_t wn = 0;
    make_string_lit(name, &strw, &wn);
    int ok = spv_emit_words(&l->mod.words, strw, wn);
    WGSL_FREE(strw);
    if (!ok) return 0;
    spv_op_end(&l->mod.words, idx);
    return 1;
}

// ---------- Minimal function emission ----------

static int emit_void_entry_function(WgslLower *l,
                                    const char *name,
                                    SpvExecutionModel model,
                                    uint32_t *out_fn_id) {
    if (!emit_type_function_void_void(l)) return 0;

    // OpFunction
    uint32_t fn_id = spv_fresh_id(&l->mod);
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpFunction, 0, &idx)) return 0;
        if (!spv_emit_u32(&l->mod.words, l->id_void)) return 0;            // result type
        if (!spv_emit_u32(&l->mod.words, fn_id)) return 0;                 // result id
        if (!spv_emit_u32(&l->mod.words, SpvFunctionControlMaskNone)) return 0;
        if (!spv_emit_u32(&l->mod.words, l->id_fn_void_void)) return 0;    // function type
        spv_op_end(&l->mod.words, idx);
    }
    // OpLabel
    uint32_t lbl = spv_fresh_id(&l->mod);
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpLabel, 0, &idx)) return 0;
        if (!spv_emit_u32(&l->mod.words, lbl)) return 0;
        spv_op_end(&l->mod.words, idx);
    }
    // OpReturn
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpReturn, 0, &idx)) return 0;
        spv_op_end(&l->mod.words, idx);
    }
    // OpFunctionEnd
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpFunctionEnd, 0, &idx)) return 0;
        spv_op_end(&l->mod.words, idx);
    }

    // OpEntryPoint
    {
        uint16_t idx;
        if (!spv_op_begin(&l->mod.words, SpvOpEntryPoint, 0, &idx)) return 0;
        if (!spv_emit_u32(&l->mod.words, model)) return 0;
        if (!spv_emit_u32(&l->mod.words, fn_id)) return 0;
        uint32_t *strw = NULL; size_t wn = 0;
        make_string_lit(name ? name : "main", &strw, &wn);
        int ok = spv_emit_words(&l->mod.words, strw, wn);
        WGSL_FREE(strw);
        if (!ok) return 0;
        // no interface ids for now
        spv_op_end(&l->mod.words, idx);
    }

    // OpExecutionMode if needed (none for now)

    // Debug name
    if (!emit_debug_name(l, fn_id, name)) return 0;

    *out_fn_id = fn_id;
    return 1;
}

// ---------- Public API ----------

WgslLower *wgsl_lower_create(const WgslAstNode *program,
                             const WgslResolver *resolver,
                             const WgslLowerOptions *opts) {
    WgslLower *l = (WgslLower*)WGSL_MALLOC(sizeof(WgslLower));
    if (!l) return NULL;
    memset(l, 0, sizeof(*l));

    l->program  = program;
    l->resolver = resolver;
    if (opts) l->opts = *opts;
    else {
        memset(&l->opts, 0, sizeof(l->opts));
        l->opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
        l->opts.spirv_version = 0x00010400; // SPIR-V 1.4 as requested
    }
    // Force assumptions requested by user
    if (l->opts.spirv_version == 0) l->opts.spirv_version = 0x00010400;
    l->opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    if (!spv_init(&l->mod, l->opts.spirv_version)) {
        set_error(l, "failed to initialize SPIR-V module");
        WGSL_FREE(l);
        return NULL;
    }

    if (!emit_capabilities_memory_model(l)) {
        set_error(l, "failed to emit capabilities/memory model");
        spv_free(&l->mod);
        WGSL_FREE(l);
        return NULL;
    }

    // Minimal types needed
    if (!emit_type_void(l) || !emit_type_function_void_void(l)) {
        set_error(l, "failed to emit basic types");
        spv_free(&l->mod);
        WGSL_FREE(l);
        return NULL;
    }

    // Entrypoints: build trivial functions matching resolver entrypoints
    int ep_count = 0;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(resolver, &ep_count);
    if (ep_count <= 0 || !eps) {
        // If none found, synthesize a single "main" as Fragment by default
        l->eps = (WgslLowerEntrypointInfo*)WGSL_MALLOC(sizeof(WgslLowerEntrypointInfo));
        if (!l->eps) {
            set_error(l, "oom");
            spv_free(&l->mod);
            WGSL_FREE(l);
            return NULL;
        }
        l->ep_count = 1;
        l->eps[0].name = "main";
        l->eps[0].stage = WGSL_STAGE_FRAGMENT;
        l->eps[0].interface_count = 0;
        l->eps[0].interface_ids = NULL;
        uint32_t fn_id = 0;
        if (!emit_void_entry_function(l, l->eps[0].name, SpvExecutionModelFragment, &fn_id)) {
            set_error(l, "failed to emit default entry");
            spv_free(&l->mod);
            WGSL_FREE(l->eps);
            WGSL_FREE(l);
            return NULL;
        }
        l->eps[0].function_id = fn_id;
    } else {
        l->eps = (WgslLowerEntrypointInfo*)WGSL_MALLOC(sizeof(WgslLowerEntrypointInfo) * (size_t)ep_count);
        if (!l->eps) {
            set_error(l, "oom");
            spv_free(&l->mod);
            WGSL_FREE(l);
            return NULL;
        }
        l->ep_count = ep_count;
        for (int i = 0; i < ep_count; ++i) {
            const char *nm = eps[i].name ? eps[i].name : "main";
            SpvExecutionModel model = stage_to_model(eps[i].stage);
            if (model == SpvExecutionModelMax) model = SpvExecutionModelFragment;
            uint32_t fn_id = 0;
            if (!emit_void_entry_function(l, nm, model, &fn_id)) {
                set_error(l, "failed to emit entry");
                spv_free(&l->mod);
                WGSL_FREE(l->eps);
                WGSL_FREE(l);
                return NULL;
            }
            l->eps[i].name = nm;
            l->eps[i].stage = eps[i].stage;
            l->eps[i].function_id = fn_id;
            l->eps[i].interface_count = 0;
            l->eps[i].interface_ids = NULL;
        }
    }

    // Finish module
    spv_finish(&l->mod);

    return l;
}

void wgsl_lower_destroy(WgslLower *lower) {
    if (!lower) return;
    spv_free(&lower->mod);
    WGSL_FREE(lower->eps);
    // features arrays are on stack buffers inside struct; nothing to free
    WGSL_FREE(lower);
}

WgslLowerResult wgsl_lower_emit_spirv(const WgslAstNode *program,
                                      const WgslResolver *resolver,
                                      const WgslLowerOptions *opts,
                                      uint32_t **out_words,
                                      size_t *out_word_count) {
    if (!out_word_count) return WGSL_LOWER_ERR_INVALID_INPUT;
    *out_word_count = 0;
    if (out_words) *out_words = NULL;

    WgslLower *l = wgsl_lower_create(program, resolver, opts);
    if (!l) return WGSL_LOWER_ERR_INTERNAL;

    size_t words = l->mod.words.len;
    if (out_words) {
        uint32_t *buf = (uint32_t*)WGSL_MALLOC(words * sizeof(uint32_t));
        if (!buf) {
            wgsl_lower_destroy(l);
            return WGSL_LOWER_ERR_OOM;
        }
        memcpy(buf, l->mod.words.data, words * sizeof(uint32_t));
        *out_words = buf;
    }
    *out_word_count = words;

    wgsl_lower_destroy(l);
    return WGSL_LOWER_OK;
}

WgslLowerResult wgsl_lower_serialize(const WgslLower *lower,
                                     uint32_t **out_words,
                                     size_t *out_word_count) {
    if (!lower || !out_word_count) return WGSL_LOWER_ERR_INVALID_INPUT;
    *out_word_count = lower->mod.words.len;
    if (out_words) {
        uint32_t *buf = (uint32_t*)WGSL_MALLOC(lower->mod.words.len * sizeof(uint32_t));
        if (!buf) return WGSL_LOWER_ERR_OOM;
        memcpy(buf, lower->mod.words.data, lower->mod.words.len * sizeof(uint32_t));
        *out_words = buf;
    }
    return WGSL_LOWER_OK;
}

WgslLowerResult wgsl_lower_serialize_into(const WgslLower *lower,
                                          uint32_t *out_words,
                                          size_t max_words,
                                          size_t *out_written) {
    if (!lower || !out_written) return WGSL_LOWER_ERR_INVALID_INPUT;
    size_t need = lower->mod.words.len;
    if (!out_words || max_words < need) {
        *out_written = need;
        return WGSL_LOWER_ERR_INVALID_INPUT;
    }
    memcpy(out_words, lower->mod.words.data, need * sizeof(uint32_t));
    *out_written = need;
    return WGSL_LOWER_OK;
}

const char *wgsl_lower_last_error(const WgslLower *lower) {
    if (!lower) return "invalid";
    return lower->last_error[0] ? lower->last_error : "";
}

const WgslLowerModuleFeatures *wgsl_lower_module_features(const WgslLower *lower) {
    if (!lower) return NULL;
    return &lower->features;
}

const WgslLowerEntrypointInfo *wgsl_lower_entrypoints(const WgslLower *lower, int *out_count) {
    if (!lower) return NULL;
    if (out_count) *out_count = lower->ep_count;
    return lower->eps;
}

uint32_t wgsl_lower_node_result_id(const WgslLower *lower, const WgslAstNode *node) {
    (void)lower; (void)node;
    // Not tracked in this minimal implementation
    return 0;
}

uint32_t wgsl_lower_symbol_result_id(const WgslLower *lower, int symbol_id) {
    (void)lower; (void)symbol_id;
    // Not tracked in this minimal implementation
    return 0;
}

void wgsl_lower_free(void *p) {
    WGSL_FREE(p);
}
// END FILE wgsl_lower.c
