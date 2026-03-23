#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *src = malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    WgslAstNode *ast = wgsl_parse(src);
    if (ast) wgsl_free_ast(ast);

    free(src);
    return 0;
}
