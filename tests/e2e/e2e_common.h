/*
 * e2e_common.h - Shared helpers for end-to-end CUDA driver API tests
 */
#ifndef E2E_COMMON_H
#define E2E_COMMON_H

#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CU(call)                                                      \
    do {                                                                    \
        CUresult _r = (call);                                              \
        if (_r != CUDA_SUCCESS) {                                          \
            const char *_err = NULL;                                       \
            cuGetErrorString(_r, &_err);                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s (code %d)\n",        \
                    __FILE__, __LINE__, _err ? _err : "unknown", (int)_r); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static char *read_ptx_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open PTX file: %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); exit(1); }
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static int check_float_eq(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

#endif /* E2E_COMMON_H */
