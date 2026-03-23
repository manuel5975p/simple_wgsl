#include "simple_wgsl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* SPIR-V is uint32_t words — need at least 4 bytes for one word */
    if (size < 4) return 0;

    size_t word_count = size / 4;
    const uint32_t *words = (const uint32_t *)data;

    /* Fuzz spirv_to_ssir deserializer */
    SsirModule *mod = NULL;
    char *err = NULL;
    SpirvToSsirOptions opts = {0};

    SpirvToSsirResult res = spirv_to_ssir(words, word_count, &opts, &mod, &err);
    (void)res;
    if (mod) ssir_module_destroy(mod);
    free(err);

    /* Also fuzz the raise path */
    char *wgsl = NULL;
    char *raise_err = NULL;
    WgslRaiseOptions raise_opts = {0};
    WgslRaiseResult rres = wgsl_raise_to_wgsl(words, word_count, &raise_opts,
                                               &wgsl, &raise_err);
    (void)rres;
    free(wgsl);
    free(raise_err);

    return 0;
}
