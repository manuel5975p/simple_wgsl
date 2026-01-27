#ifndef WGSL_RAISE_H
#define WGSL_RAISE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WgslRaiser WgslRaiser;

typedef struct WgslRaiseOptions {
    int emit_debug_comments;
    int preserve_names;
    int inline_constants;
} WgslRaiseOptions;

typedef enum WgslRaiseResult {
    WGSL_RAISE_SUCCESS = 0,
    WGSL_RAISE_INVALID_SPIRV,
    WGSL_RAISE_UNSUPPORTED_FEATURE,
    WGSL_RAISE_INTERNAL_ERROR
} WgslRaiseResult;

WgslRaiseResult wgsl_raise_to_wgsl(
    const uint32_t *spirv,
    size_t word_count,
    const WgslRaiseOptions *options,
    char **out_wgsl,
    char **out_error
);

WgslRaiser *wgsl_raise_create(const uint32_t *spirv, size_t word_count);
WgslRaiseResult wgsl_raise_parse(WgslRaiser *r);
WgslRaiseResult wgsl_raise_analyze(WgslRaiser *r);
const char *wgsl_raise_emit(WgslRaiser *r, const WgslRaiseOptions *options);
void wgsl_raise_destroy(WgslRaiser *r);

int wgsl_raise_entry_point_count(WgslRaiser *r);
const char *wgsl_raise_entry_point_name(WgslRaiser *r, int index);

void wgsl_raise_free(void *p);

#ifdef __cplusplus
}
#endif

#endif
