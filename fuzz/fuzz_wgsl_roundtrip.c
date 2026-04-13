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
    WgslParseResult pr = wgsl_parse(src);
    WgslAstNode *ast = pr.value;
    if (!ast) { wgsl_diagnostic_list_free(pr.diags); goto done; }

    /* Resolve */
    WgslResolveResult rr = wgsl_resolver_build(ast);
    WgslResolver *resolver = rr.value;
    if (!resolver) {
        wgsl_free_ast(ast);
        wgsl_diagnostic_list_free(pr.diags);
        wgsl_diagnostic_list_free(rr.diags);
        goto done;
    }

    /* Lower to SPIR-V */
    WgslLowerOptions opts = {0};
    WgslLowerSpirvResult lsr = wgsl_lower_emit_spirv(ast, resolver, &opts);
    if (lsr.code == SW_OK && lsr.words && lsr.word_count > 0) {
        /* Raise back to WGSL */
        char *wgsl_out = NULL;
        char *err = NULL;
        WgslRaiseOptions raise_opts = {0};
        wgsl_raise_to_wgsl(lsr.words, lsr.word_count, &raise_opts, &wgsl_out, &err);
        free(wgsl_out);
        free(err);
    }
    if (lsr.words) wgsl_lower_free(lsr.words);
    wgsl_diagnostic_list_free(lsr.diags);

    wgsl_resolver_free(resolver);
    wgsl_diagnostic_list_free(rr.diags);
    wgsl_free_ast(ast);
    wgsl_diagnostic_list_free(pr.diags);

done:
    free(src);
    return 0;
}
