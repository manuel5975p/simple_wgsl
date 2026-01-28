/*
 * SSIR to SPIR-V Converter - Header
 *
 * Converts SSIR modules to SPIR-V binary format.
 */

#ifndef SSIR_TO_SPIRV_H
#define SSIR_TO_SPIRV_H

#include "ssir.h"
#include <spirv/unified1/spirv.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Options
 * ============================================================================ */

typedef struct SsirToSpirvOptions {
    uint32_t spirv_version;         /* e.g., 0x00010400 for 1.4; 0 = auto */
    int enable_debug_names;         /* emit OpName/OpMemberName */
    int enable_line_info;           /* emit OpLine (not yet implemented) */
} SsirToSpirvOptions;

/* ============================================================================
 * Result
 * ============================================================================ */

typedef enum SsirToSpirvResult {
    SSIR_TO_SPIRV_OK = 0,
    SSIR_TO_SPIRV_ERR_INVALID_INPUT,
    SSIR_TO_SPIRV_ERR_UNSUPPORTED,
    SSIR_TO_SPIRV_ERR_INTERNAL,
    SSIR_TO_SPIRV_ERR_OOM,
} SsirToSpirvResult;

/* ============================================================================
 * Conversion API
 * ============================================================================ */

/*
 * Convert SSIR module to SPIR-V binary.
 *
 * Parameters:
 *   mod         - Input SSIR module (must be valid)
 *   opts        - Conversion options (can be NULL for defaults)
 *   out_words   - Output: allocated SPIR-V word buffer (caller frees with ssir_to_spirv_free)
 *   out_count   - Output: number of words in buffer
 *
 * Returns:
 *   SSIR_TO_SPIRV_OK on success, error code otherwise.
 */
SsirToSpirvResult ssir_to_spirv(const SsirModule *mod,
                                 const SsirToSpirvOptions *opts,
                                 uint32_t **out_words,
                                 size_t *out_count);

/* Free memory allocated by ssir_to_spirv */
void ssir_to_spirv_free(void *p);

/* Get error message for result code */
const char *ssir_to_spirv_result_string(SsirToSpirvResult result);

#ifdef __cplusplus
}
#endif

#endif /* SSIR_TO_SPIRV_H */
