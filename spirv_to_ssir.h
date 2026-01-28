/*
 * SPIR-V to SSIR Converter - Header
 *
 * Converts SPIR-V binary to SSIR (Simple Shader IR) module.
 */

#ifndef SPIRV_TO_SSIR_H
#define SPIRV_TO_SSIR_H

#include "ssir.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPIRV_TO_SSIR_SUCCESS = 0,
    SPIRV_TO_SSIR_INVALID_SPIRV,
    SPIRV_TO_SSIR_UNSUPPORTED_FEATURE,
    SPIRV_TO_SSIR_INTERNAL_ERROR
} SpirvToSsirResult;

typedef struct {
    bool preserve_names;
    bool preserve_locations;
} SpirvToSsirOptions;

SpirvToSsirResult spirv_to_ssir(
    const uint32_t *spirv,
    size_t word_count,
    const SpirvToSsirOptions *opts,
    SsirModule **out_module,
    char **out_error
);

const char *spirv_to_ssir_result_string(SpirvToSsirResult result);

void spirv_to_ssir_free(void *p);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV_TO_SSIR_H */
