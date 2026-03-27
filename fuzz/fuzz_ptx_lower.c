#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *src = malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    char *err = NULL;
    PtxModule *mod = ptx_parse(src, &err);
    if (mod) {
        SsirModule *ssir = NULL;
        char *lower_err = NULL;
        PtxToSsirOptions opts = {0};
        PtxToSsirResult res = ptx_lower(mod, &opts, &ssir, &lower_err);
        (void)res;
        if (ssir) ssir_module_destroy(ssir);
        free(lower_err);
        ptx_parse_free(mod);
    }
    free(err);

    free(src);
    return 0;
}
