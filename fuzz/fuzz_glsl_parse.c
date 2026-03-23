#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *src = malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    /* Fuzz all three stage variants */
    WgslStage stages[] = {WGSL_STAGE_VERTEX, WGSL_STAGE_FRAGMENT, WGSL_STAGE_COMPUTE};
    for (int i = 0; i < 3; i++) {
        WgslAstNode *ast = glsl_parse(src, "fuzz.glsl", stages[i], NULL);
        if (ast) wgsl_free_ast(ast);
    }

    free(src);
    return 0;
}
