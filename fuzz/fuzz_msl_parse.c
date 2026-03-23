#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char *src = malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    SsirModule *mod = NULL;
    char *err = NULL;
    MslToSsirOptions opts = {0};

    MslToSsirResult res = msl_to_ssir(src, &opts, &mod, &err);
    (void)res;
    if (mod) ssir_module_destroy(mod);
    if (err) msl_to_ssir_free(err);

    free(src);
    return 0;
}
