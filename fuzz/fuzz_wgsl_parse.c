#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *src = malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    WgslParseResult pr = wgsl_parse(src);
    WgslAstNode *ast = pr.value;
    if (ast) wgsl_free_ast(ast);
    wgsl_diagnostic_list_free(pr.diags);

    free(src);
    return 0;
}
