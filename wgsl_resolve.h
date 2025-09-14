#ifndef WGSL_RESOLVE_H
#define WGSL_RESOLVE_H

#include "wgsl_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WGSL_SYM_GLOBAL = 1,
    WGSL_SYM_PARAM,
    WGSL_SYM_LOCAL
} WgslSymbolKind;

typedef struct {
    int id;
    WgslSymbolKind kind;
    const char *name;
    int has_group;
    int group_index;
    int has_binding;
    int binding_index;

    /* New: present when the symbol is a buffer binding (var<uniform|storage>)
       and a static minimum size could be computed. */
    int has_min_binding_size;
    int min_binding_size; /* bytes */

    const WgslAstNode *decl_node;
    const WgslAstNode *function_node;
} WgslSymbolInfo;

typedef enum {
    WGSL_NUM_UNKNOWN = 0,
    WGSL_NUM_F32,
    WGSL_NUM_I32,
    WGSL_NUM_U32,
    WGSL_NUM_BOOL
} WgslNumericType;

typedef struct {
    int location;
    int component_count;
    WgslNumericType numeric_type;
    int byte_size;
} WgslVertexSlot;

typedef enum {
    WGSL_STAGE_UNKNOWN = 0,
    WGSL_STAGE_VERTEX,
    WGSL_STAGE_FRAGMENT,
    WGSL_STAGE_COMPUTE
} WgslStage;

typedef struct {
    const char *name;
    WgslStage stage;
    const WgslAstNode *function_node;
} WgslResolverEntrypoint;

typedef struct WgslResolver WgslResolver;

WgslResolver *wgsl_resolver_build(const WgslAstNode *program);
void wgsl_resolver_free(WgslResolver *r);

const WgslSymbolInfo *wgsl_resolver_all_symbols(const WgslResolver *r, int *out_count);
const WgslSymbolInfo *wgsl_resolver_globals(const WgslResolver *r, int *out_count);
const WgslSymbolInfo *wgsl_resolver_binding_vars(const WgslResolver *r, int *out_count);

int wgsl_resolver_ident_symbol_id(const WgslResolver *r, const WgslAstNode *ident_node);

int wgsl_resolver_vertex_inputs(const WgslResolver *r, const char *vertex_entry_name, WgslVertexSlot **out_slots);

const WgslResolverEntrypoint *wgsl_resolver_entrypoints(const WgslResolver *r, int *out_count);

const WgslSymbolInfo *wgsl_resolver_entrypoint_globals(const WgslResolver *r, const char *entry_name, int *out_count);
const WgslSymbolInfo *wgsl_resolver_entrypoint_binding_vars(const WgslResolver *r, const char *entry_name, int *out_count);

void wgsl_resolve_free(void *p);

#ifdef __cplusplus
}
#endif
#endif

