/*
 * cuvk_jitlink.c - cuLink* API backed by nvJitLink
 *
 * When CUVK_NVJITLINK is enabled, this file provides cuLinkCreate,
 * cuLinkAddData, cuLinkComplete, cuLinkDestroy using the system's
 * libnvJitLink.so (dlopen'd at runtime).
 *
 * nvJitLink compiles LTO-IR (and PTX/cubin) into PTX output, which
 * then feeds into our normal PTX -> SSIR -> SPIR-V pipeline.
 */

#include "cuvk_internal.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* nvJitLink types (matching nvJitLink.h without requiring header) */
typedef int nvJitLinkResult;
typedef struct nvJitLink *nvJitLinkHandle;
typedef int nvJitLinkInputType;

#define NVJITLINK_SUCCESS 0
#define NVJITLINK_INPUT_CUBIN  1
#define NVJITLINK_INPUT_PTX    2
#define NVJITLINK_INPUT_LTOIR  3
#define NVJITLINK_INPUT_FATBIN 4
#define NVJITLINK_INPUT_ANY    10

/* Function pointer types */
typedef nvJitLinkResult (*PFN_nvJitLinkCreate)(
    nvJitLinkHandle *, uint32_t, const char *const *);
typedef nvJitLinkResult (*PFN_nvJitLinkDestroy)(nvJitLinkHandle *);
typedef nvJitLinkResult (*PFN_nvJitLinkAddData)(
    nvJitLinkHandle, nvJitLinkInputType, const void *, size_t, const char *);
typedef nvJitLinkResult (*PFN_nvJitLinkComplete)(nvJitLinkHandle);
typedef nvJitLinkResult (*PFN_nvJitLinkGetLinkedPtxSize)(
    nvJitLinkHandle, size_t *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetLinkedPtx)(
    nvJitLinkHandle, char *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetLinkedCubinSize)(
    nvJitLinkHandle, size_t *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetLinkedCubin)(
    nvJitLinkHandle, void *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetErrorLogSize)(
    nvJitLinkHandle, size_t *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetErrorLog)(
    nvJitLinkHandle, char *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetInfoLogSize)(
    nvJitLinkHandle, size_t *);
typedef nvJitLinkResult (*PFN_nvJitLinkGetInfoLog)(
    nvJitLinkHandle, char *);

static struct {
    void *lib;
    int   loaded;
    PFN_nvJitLinkCreate              Create;
    PFN_nvJitLinkDestroy             Destroy;
    PFN_nvJitLinkAddData             AddData;
    PFN_nvJitLinkComplete            Complete;
    PFN_nvJitLinkGetLinkedPtxSize    GetLinkedPtxSize;
    PFN_nvJitLinkGetLinkedPtx        GetLinkedPtx;
    PFN_nvJitLinkGetLinkedCubinSize  GetLinkedCubinSize;
    PFN_nvJitLinkGetLinkedCubin      GetLinkedCubin;
    PFN_nvJitLinkGetErrorLogSize     GetErrorLogSize;
    PFN_nvJitLinkGetErrorLog         GetErrorLog;
    PFN_nvJitLinkGetInfoLogSize      GetInfoLogSize;
    PFN_nvJitLinkGetInfoLog          GetInfoLog;
} g_nvjl;

static int nvjl_load(void) {
    if (g_nvjl.loaded) return g_nvjl.lib != NULL;

    g_nvjl.loaded = 1;
    static const char *names[] = {
        "libnvJitLink.so.13", "libnvJitLink.so.12", "libnvJitLink.so", NULL
    };
    for (int i = 0; names[i]; i++) {
        g_nvjl.lib = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
        if (g_nvjl.lib) {
            CUVK_LOG("[cuvk] nvJitLink loaded: %s\n", names[i]);
            break;
        }
    }
    if (!g_nvjl.lib) {
        CUVK_LOG("[cuvk] nvJitLink not found\n");
        return 0;
    }

    /* Versioned symbols (CUDA 13.1) */
#define LOAD(fn, sym) \
    g_nvjl.fn = (PFN_nvJitLink##fn)dlsym(g_nvjl.lib, sym); \
    if (!g_nvjl.fn) { \
        g_nvjl.fn = (PFN_nvJitLink##fn)dlsym(g_nvjl.lib, "nvJitLink" #fn); \
    }

    LOAD(Create,              "__nvJitLinkCreate_13_1");
    LOAD(Destroy,             "__nvJitLinkDestroy_13_1");
    LOAD(AddData,             "__nvJitLinkAddData_13_1");
    LOAD(Complete,            "__nvJitLinkComplete_13_1");
    LOAD(GetLinkedPtxSize,    "__nvJitLinkGetLinkedPtxSize_13_1");
    LOAD(GetLinkedPtx,        "__nvJitLinkGetLinkedPtx_13_1");
    LOAD(GetLinkedCubinSize,  "__nvJitLinkGetLinkedCubinSize_13_1");
    LOAD(GetLinkedCubin,      "__nvJitLinkGetLinkedCubin_13_1");
    LOAD(GetErrorLogSize,     "__nvJitLinkGetErrorLogSize_13_1");
    LOAD(GetErrorLog,         "__nvJitLinkGetErrorLog_13_1");
    LOAD(GetInfoLogSize,      "__nvJitLinkGetInfoLogSize_13_1");
    LOAD(GetInfoLog,          "__nvJitLinkGetInfoLog_13_1");
#undef LOAD

    if (!g_nvjl.Create || !g_nvjl.Destroy || !g_nvjl.AddData ||
        !g_nvjl.Complete) {
        CUVK_LOG("[cuvk] nvJitLink: missing required symbols\n");
        dlclose(g_nvjl.lib);
        g_nvjl.lib = NULL;
        return 0;
    }

    CUVK_LOG("[cuvk] nvJitLink ready\n");
    return 1;
}

static void nvjl_dump_logs(nvJitLinkHandle h) {
    if (!g_nvjl.GetErrorLogSize || !g_nvjl.GetErrorLog) return;
    size_t sz = 0;
    if (g_nvjl.GetErrorLogSize(h, &sz) == NVJITLINK_SUCCESS && sz > 1) {
        char *log = (char *)malloc(sz);
        if (log) {
            g_nvjl.GetErrorLog(h, log);
            CUVK_LOG("[cuvk] nvJitLink error log: %s\n", log);
            free(log);
        }
    }
    if (!g_nvjl.GetInfoLogSize || !g_nvjl.GetInfoLog) return;
    if (g_nvjl.GetInfoLogSize(h, &sz) == NVJITLINK_SUCCESS && sz > 1) {
        char *log = (char *)malloc(sz);
        if (log) {
            g_nvjl.GetInfoLog(h, log);
            CUVK_LOG("[cuvk] nvJitLink info log: %s\n", log);
            free(log);
        }
    }
}

/* ============================================================================
 * cuLink* state
 * ============================================================================ */

typedef struct CuvkLinkInput {
    int         cu_type;  /* CUjitInputType */
    void       *data;     /* owned copy */
    size_t      size;
    char       *name;     /* owned copy, may be NULL */
} CuvkLinkInput;

typedef struct CUlinkState_st {
    CuvkLinkInput  *inputs;
    uint32_t        input_count;
    uint32_t        input_capacity;
    char           *ptx_out;     /* result PTX (owned) */
    size_t          ptx_out_len;
} CUlinkState_st;

/* Map CUjitInputType -> nvJitLinkInputType */
static nvJitLinkInputType cu_to_nvjl_input(int cu_type) {
    switch (cu_type) {
    case 0: return NVJITLINK_INPUT_CUBIN;   /* CU_JIT_INPUT_CUBIN */
    case 1: return NVJITLINK_INPUT_PTX;     /* CU_JIT_INPUT_PTX */
    case 2: return NVJITLINK_INPUT_FATBIN;  /* CU_JIT_INPUT_FATBINARY */
    default: return NVJITLINK_INPUT_ANY;
    }
}

/* ============================================================================
 * cuLinkCreate
 * ============================================================================ */

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
    CUVK_LOG("[cuvk] cuLinkCreate: numOptions=%u\n", numOptions);
    (void)numOptions; (void)options; (void)optionValues;
    if (!stateOut) return CUDA_ERROR_INVALID_VALUE;
    if (!nvjl_load()) return CUDA_ERROR_NOT_SUPPORTED;

    CUlinkState_st *s = (CUlinkState_st *)calloc(1, sizeof(*s));
    if (!s) return CUDA_ERROR_OUT_OF_MEMORY;
    *stateOut = (CUlinkState)s;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuLinkAddData
 * ============================================================================ */

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
    (void)numOptions; (void)options; (void)optionValues;
    CUlinkState_st *s = (CUlinkState_st *)state;
    if (!s || !data || size == 0) return CUDA_ERROR_INVALID_VALUE;

    CUVK_LOG("[cuvk] cuLinkAddData: type=%d size=%zu name=%s\n",
             type, size, name ? name : "(null)");

    if (s->input_count >= s->input_capacity) {
        uint32_t cap = s->input_capacity ? s->input_capacity * 2 : 8;
        CuvkLinkInput *ni = (CuvkLinkInput *)realloc(
            s->inputs, cap * sizeof(CuvkLinkInput));
        if (!ni) return CUDA_ERROR_OUT_OF_MEMORY;
        s->inputs = ni;
        s->input_capacity = cap;
    }

    void *copy = malloc(size);
    if (!copy) return CUDA_ERROR_OUT_OF_MEMORY;
    memcpy(copy, data, size);

    CuvkLinkInput *inp = &s->inputs[s->input_count++];
    inp->cu_type = type;
    inp->data = copy;
    inp->size = size;
    inp->name = name ? strdup(name) : NULL;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuLinkAddFile
 * ============================================================================ */

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
    (void)numOptions; (void)options; (void)optionValues;
    CUlinkState_st *s = (CUlinkState_st *)state;
    if (!s || !path) return CUDA_ERROR_INVALID_VALUE;

    CUVK_LOG("[cuvk] cuLinkAddFile: type=%d path=%s\n", type, path);

    FILE *f = fopen(path, "rb");
    if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return CUDA_ERROR_INVALID_VALUE; }
    void *buf = malloc((size_t)len);
    if (!buf) { fclose(f); return CUDA_ERROR_OUT_OF_MEMORY; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    fclose(f);

    CUresult r = cuLinkAddData(state, type, buf, (size_t)len, path,
                                0, NULL, NULL);
    free(buf);
    return r;
}

/* ============================================================================
 * cuLinkComplete - do the actual linking via nvJitLink, produce PTX
 * ============================================================================ */

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut,
                                size_t *sizeOut) {
    CUlinkState_st *s = (CUlinkState_st *)state;
    if (!s) return CUDA_ERROR_INVALID_VALUE;

    CUVK_LOG("[cuvk] cuLinkComplete: %u inputs\n", s->input_count);

    if (!nvjl_load()) return CUDA_ERROR_NOT_SUPPORTED;
    if (s->input_count == 0) return CUDA_ERROR_INVALID_VALUE;

    /* Build nvJitLink options: -arch=sm_XX -lto -ptx */
    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "-arch=sm_75");

    const char *opts[] = {arch_opt, "-lto", "-ptx"};
    nvJitLinkHandle jl = NULL;
    nvJitLinkResult jr = g_nvjl.Create(&jl, 3, opts);
    if (jr != NVJITLINK_SUCCESS) {
        CUVK_LOG("[cuvk] nvJitLinkCreate failed: %d\n", jr);
        return CUDA_ERROR_INVALID_IMAGE;
    }

    for (uint32_t i = 0; i < s->input_count; i++) {
        CuvkLinkInput *inp = &s->inputs[i];
        nvJitLinkInputType ntype = cu_to_nvjl_input(inp->cu_type);
        jr = g_nvjl.AddData(jl, ntype, inp->data, inp->size, inp->name);
        if (jr != NVJITLINK_SUCCESS) {
            CUVK_LOG("[cuvk] nvJitLinkAddData[%u] failed: %d\n", i, jr);
            nvjl_dump_logs(jl);
            g_nvjl.Destroy(&jl);
            return CUDA_ERROR_INVALID_IMAGE;
        }
    }

    jr = g_nvjl.Complete(jl);
    if (jr != NVJITLINK_SUCCESS) {
        CUVK_LOG("[cuvk] nvJitLinkComplete failed: %d\n", jr);
        nvjl_dump_logs(jl);
        g_nvjl.Destroy(&jl);
        return CUDA_ERROR_INVALID_IMAGE;
    }

    /* Get linked PTX */
    size_t ptx_size = 0;
    if (g_nvjl.GetLinkedPtxSize &&
        g_nvjl.GetLinkedPtxSize(jl, &ptx_size) == NVJITLINK_SUCCESS &&
        ptx_size > 0) {
        char *ptx = (char *)malloc(ptx_size + 1);
        if (!ptx) { g_nvjl.Destroy(&jl); return CUDA_ERROR_OUT_OF_MEMORY; }
        g_nvjl.GetLinkedPtx(jl, ptx);
        ptx[ptx_size] = '\0';
        s->ptx_out = ptx;
        s->ptx_out_len = ptx_size;
        CUVK_LOG("[cuvk] nvJitLink produced %zu bytes of PTX\n", ptx_size);
    } else {
        CUVK_LOG("[cuvk] nvJitLink: no PTX output, trying cubin\n");
        size_t cubin_size = 0;
        if (g_nvjl.GetLinkedCubinSize &&
            g_nvjl.GetLinkedCubinSize(jl, &cubin_size) == NVJITLINK_SUCCESS &&
            cubin_size > 0) {
            CUVK_LOG("[cuvk] nvJitLink produced %zu bytes of cubin "
                     "(cannot use directly)\n", cubin_size);
        }
        g_nvjl.Destroy(&jl);
        return CUDA_ERROR_INVALID_IMAGE;
    }

    nvjl_dump_logs(jl);
    g_nvjl.Destroy(&jl);

    if (cubinOut) *cubinOut = s->ptx_out;
    if (sizeOut) *sizeOut = s->ptx_out_len;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuLinkDestroy
 * ============================================================================ */

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
    CUlinkState_st *s = (CUlinkState_st *)state;
    if (!s) return CUDA_SUCCESS;
    CUVK_LOG("[cuvk] cuLinkDestroy\n");
    for (uint32_t i = 0; i < s->input_count; i++) {
        free(s->inputs[i].data);
        free(s->inputs[i].name);
    }
    free(s->inputs);
    free(s->ptx_out);
    free(s);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Fatbin LTO-IR extraction helper
 *
 * Given a fatbin that has no PTX sections, try to extract LTO-IR sections,
 * compile them with nvJitLink, and return the resulting PTX.
 * ============================================================================ */

#define FATBIN_MAGIC   0xBA55ED50
#define FATBIN_KIND_PTX   1
#define FATBIN_KIND_CUBIN 2

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t fat_size;
} __attribute__((packed)) FatbinHdr;

typedef struct {
    uint16_t kind;
    uint16_t attr;
    uint32_t header_size;
    uint32_t padded_payload_size;
    uint32_t unknown0;
    uint32_t compressed_size;
} __attribute__((packed)) FatbinSecHdr;

char *cuvk_jitlink_compile_ltoir(const void *fatbin_data, size_t *ptx_len) {
    if (!fatbin_data || !nvjl_load()) return NULL;

    const uint8_t *data = (const uint8_t *)fatbin_data;
    const FatbinHdr *hdr = (const FatbinHdr *)data;
    if (hdr->magic != FATBIN_MAGIC) return NULL;

    uint64_t total = hdr->header_size + hdr->fat_size;
    const uint8_t *pos = data + hdr->header_size;
    const uint8_t *end = data + total;

    const char *opts[] = {"-arch=sm_75", "-lto", "-ptx"};
    nvJitLinkHandle jl = NULL;
    nvJitLinkResult jr = g_nvjl.Create(&jl, 3, opts);
    if (jr != NVJITLINK_SUCCESS) return NULL;

    /* Feed the entire fatbin to nvJitLink; it knows how to parse
     * fatbin containers and extract LTO-IR sections internally. */
    CUVK_LOG("[cuvk] feeding fatbin (%lu bytes) to nvJitLink\n",
             (unsigned long)total);
    jr = g_nvjl.AddData(jl, NVJITLINK_INPUT_FATBIN, data, total, NULL);
    if (jr == NVJITLINK_SUCCESS)
        jr = g_nvjl.Complete(jl);
    if (jr != NVJITLINK_SUCCESS) {
        CUVK_LOG("[cuvk] fatbin-as-whole failed: %d\n", jr);
        nvjl_dump_logs(jl);
        g_nvjl.Destroy(&jl);

        /* Retry: feed individual CUBIN sections as LTOIR
         * (modern CUDA ships LTO-IR inside kind=2 CUBIN sections). */
        jr = g_nvjl.Create(&jl, 3, opts);
        if (jr != NVJITLINK_SUCCESS) return NULL;

        int found_ltoir = 0;
        while (pos + sizeof(FatbinSecHdr) <= end) {
            const FatbinSecHdr *sec = (const FatbinSecHdr *)pos;
            if (sec->header_size == 0) break;

            const uint8_t *payload = pos + sec->header_size;
            uint32_t payload_size = sec->compressed_size > 0 ?
                sec->compressed_size : sec->padded_payload_size;
            CUVK_LOG("[cuvk] fatbin section: kind=%u attr=0x%x size=%u\n",
                     sec->kind, sec->attr, payload_size);
            if (payload_size > 0 && sec->kind == FATBIN_KIND_CUBIN) {
                jr = g_nvjl.AddData(jl, NVJITLINK_INPUT_LTOIR,
                                     payload, payload_size, NULL);
                if (jr == NVJITLINK_SUCCESS) {
                    found_ltoir = 1;
                } else {
                    CUVK_LOG("[cuvk] nvJitLink AddData LTOIR failed: %d\n", jr);
                    nvjl_dump_logs(jl);
                }
            }
            pos += sec->header_size + sec->padded_payload_size;
        }
        if (!found_ltoir) {
            g_nvjl.Destroy(&jl);
            return NULL;
        }
        jr = g_nvjl.Complete(jl);
    }
    if (jr != NVJITLINK_SUCCESS) {
        CUVK_LOG("[cuvk] nvJitLink LTO-IR compile failed: %d\n", jr);
        nvjl_dump_logs(jl);
        g_nvjl.Destroy(&jl);
        return NULL;
    }

    size_t ptx_size = 0;
    if (!g_nvjl.GetLinkedPtxSize ||
        g_nvjl.GetLinkedPtxSize(jl, &ptx_size) != NVJITLINK_SUCCESS ||
        ptx_size == 0) {
        g_nvjl.Destroy(&jl);
        return NULL;
    }

    char *ptx = (char *)malloc(ptx_size + 1);
    if (!ptx) { g_nvjl.Destroy(&jl); return NULL; }
    g_nvjl.GetLinkedPtx(jl, ptx);
    ptx[ptx_size] = '\0';

    CUVK_LOG("[cuvk] LTO-IR compiled to %zu bytes of PTX\n", ptx_size);
    nvjl_dump_logs(jl);
    g_nvjl.Destroy(&jl);

    if (ptx_len) *ptx_len = ptx_size;
    return ptx;
}

char *cuvk_jitlink_compile_raw(const void *data, size_t size,
                                size_t *ptx_len) {
    if (!data || !size || !nvjl_load()) return NULL;

    const char *opts[] = {"-arch=sm_75", "-lto", "-ptx"};
    nvJitLinkHandle jl = NULL;
    nvJitLinkResult jr = g_nvjl.Create(&jl, 3, opts);
    if (jr != NVJITLINK_SUCCESS) return NULL;

    jr = g_nvjl.AddData(jl, NVJITLINK_INPUT_LTOIR, data, size, NULL);
    if (jr != NVJITLINK_SUCCESS) {
        CUVK_LOG("[cuvk] nvJitLink AddData(LTOIR) failed: %d\n", jr);
        nvjl_dump_logs(jl);
        g_nvjl.Destroy(&jl);
        return NULL;
    }

    jr = g_nvjl.Complete(jl);
    if (jr != NVJITLINK_SUCCESS) {
        CUVK_LOG("[cuvk] nvJitLink compile_raw failed: %d\n", jr);
        nvjl_dump_logs(jl);
        g_nvjl.Destroy(&jl);
        return NULL;
    }

    size_t ptx_size = 0;
    if (!g_nvjl.GetLinkedPtxSize ||
        g_nvjl.GetLinkedPtxSize(jl, &ptx_size) != NVJITLINK_SUCCESS ||
        ptx_size == 0) {
        g_nvjl.Destroy(&jl);
        return NULL;
    }

    char *ptx = (char *)malloc(ptx_size + 1);
    if (!ptx) { g_nvjl.Destroy(&jl); return NULL; }
    g_nvjl.GetLinkedPtx(jl, ptx);
    ptx[ptx_size] = '\0';

    CUVK_LOG("[cuvk] raw LTO-IR compiled to %zu bytes of PTX\n", ptx_size);
    nvjl_dump_logs(jl);
    g_nvjl.Destroy(&jl);

    if (ptx_len) *ptx_len = ptx_size;
    return ptx;
}
