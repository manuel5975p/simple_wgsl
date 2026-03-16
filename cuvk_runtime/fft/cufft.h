/*
 * cufft.h - cuFFT API compatibility header for cuvk
 */

#ifndef CUVK_CUFFT_H
#define CUVK_CUFFT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUFFT_SUCCESS            = 0,
    CUFFT_INVALID_PLAN       = 1,
    CUFFT_ALLOC_FAILED       = 2,
    CUFFT_INVALID_TYPE       = 3,
    CUFFT_INVALID_VALUE      = 4,
    CUFFT_INTERNAL_ERROR     = 5,
    CUFFT_EXEC_FAILED        = 6,
    CUFFT_SETUP_FAILED       = 7,
    CUFFT_INVALID_SIZE       = 8,
    CUFFT_UNALIGNED_DATA     = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE     = 11,
    CUFFT_PARSE_ERROR        = 12,
    CUFFT_NO_WORKSPACE       = 13,
    CUFFT_NOT_IMPLEMENTED    = 14,
    CUFFT_LICENSE_ERROR      = 15,
    CUFFT_NOT_SUPPORTED      = 16,
} cufftResult;

typedef enum {
    CUFFT_R2C = 0x2a,
    CUFFT_C2R = 0x2c,
    CUFFT_C2C = 0x29,
    CUFFT_D2Z = 0x6a,
    CUFFT_Z2D = 0x6c,
    CUFFT_Z2Z = 0x69,
} cufftType;

#define CUFFT_FORWARD (-1)
#define CUFFT_INVERSE  (1)

typedef int cufftHandle;
typedef float cufftReal;
typedef struct { float x, y; } cufftComplex;

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
                           int *inembed, int istride, int idist,
                           int *onembed, int ostride, int odist,
                           cufftType type, int batch);

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata,
                          cufftComplex *odata, int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata,
                          cufftComplex *odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata,
                          cufftReal *odata);

cufftResult cufftDestroy(cufftHandle plan);

#ifdef __cplusplus
}
#endif

#endif /* CUVK_CUFFT_H */
