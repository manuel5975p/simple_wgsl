/*
 * SSIR to WGSL Converter - Header
 *
 * Converts SSIR (Simple Shader IR) directly to WGSL text.
 */

#ifndef SSIR_TO_WGSL_H
#define SSIR_TO_WGSL_H

#include "ssir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Result Codes
 * ============================================================================ */

typedef enum SsirToWgslResult {
    SSIR_TO_WGSL_OK = 0,
    SSIR_TO_WGSL_ERR_INVALID_INPUT,
    SSIR_TO_WGSL_ERR_UNSUPPORTED,
    SSIR_TO_WGSL_ERR_INTERNAL,
    SSIR_TO_WGSL_ERR_OOM,
} SsirToWgslResult;

/* ============================================================================
 * Options
 * ============================================================================ */

typedef struct SsirToWgslOptions {
    int preserve_names;    /* Use SSIR debug names when available */
} SsirToWgslOptions;

/* ============================================================================
 * API
 * ============================================================================ */

/*
 * Convert an SSIR module to WGSL text.
 *
 * Parameters:
 *   mod       - Input SSIR module (must be valid, non-NULL)
 *   opts      - Conversion options (can be NULL for defaults)
 *   out_wgsl  - Output: allocated WGSL string (caller must free with ssir_to_wgsl_free)
 *   out_error - Output: error message on failure (caller must free, can be NULL)
 *
 * Returns:
 *   SSIR_TO_WGSL_OK on success, error code otherwise.
 */
SsirToWgslResult ssir_to_wgsl(const SsirModule *mod,
                              const SsirToWgslOptions *opts,
                              char **out_wgsl,
                              char **out_error);

/*
 * Free memory allocated by ssir_to_wgsl.
 */
void ssir_to_wgsl_free(void *p);

/*
 * Get human-readable error message for result code.
 */
const char *ssir_to_wgsl_result_string(SsirToWgslResult result);

#ifdef __cplusplus
}
#endif

#endif /* SSIR_TO_WGSL_H */
