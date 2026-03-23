#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *src = malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    /* Parse */
    WgslAstNode *ast = wgsl_parse(src);
    if (!ast) goto done;

    /* Resolve */
    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        goto done;
    }

    /* Lower to SPIR-V */
    uint32_t *spirv = NULL;
    size_t word_count = 0;
    WgslLowerOptions opts = {0};
    WgslLowerResult res = wgsl_lower_emit_spirv(ast, resolver, &opts,
                                                &spirv, &word_count);
    if (res == WGSL_LOWER_OK && spirv && word_count > 0) {
        /* Raise back to WGSL */
        char *wgsl_out = NULL;
        char *err = NULL;
        WgslRaiseOptions raise_opts = {0};
        wgsl_raise_to_wgsl(spirv, word_count, &raise_opts, &wgsl_out, &err);
        free(wgsl_out);
        free(err);
    }
    free(spirv);

    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

done:
    free(src);
    return 0;
}
