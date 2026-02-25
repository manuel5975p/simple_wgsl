/*
 * cuvk_fatbin.c - NVIDIA fatbin container parser
 *
 * Extracts PTX text from fatbin containers embedded by nvcc.
 * Fatbin format: 16-byte header (magic 0xBA55ED50), followed by
 * variable-length sections. Section kind=1 is PTX (ZSTD-compressed).
 */

#include "cuvk_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#define FATBIN_MAGIC 0xBA55ED50
#define FATBIN_SECTION_PTX  1
#define FATBIN_SECTION_CUBIN 2

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t fat_size;
} __attribute__((packed)) FatbinHeader;

typedef struct {
    uint16_t kind;
    uint16_t attr;
    uint32_t header_size;
    uint32_t padded_payload_size;
    uint32_t unknown0;
    uint32_t compressed_size;
} __attribute__((packed)) FatbinSectionHeader;

char *cuvk_fatbin_extract_ptx(const void *fatbin_data, size_t *ptx_len)
{
    if (!fatbin_data)
        return NULL;

    const uint8_t *data = (const uint8_t *)fatbin_data;
    const FatbinHeader *hdr = (const FatbinHeader *)data;

    if (hdr->magic != FATBIN_MAGIC)
        return NULL;

    uint64_t total_size = hdr->header_size + hdr->fat_size;
    const uint8_t *pos = data + hdr->header_size;
    const uint8_t *end = data + total_size;

    while (pos + sizeof(FatbinSectionHeader) <= end) {
        const FatbinSectionHeader *sec = (const FatbinSectionHeader *)pos;

        if (sec->header_size == 0)
            break;

        CUVK_LOG("[cuvk] fatbin section: kind=%u attr=0x%x hdr_size=%u "
                "padded=%u comp_size=%u\n",
                sec->kind, sec->attr, sec->header_size,
                sec->padded_payload_size, sec->compressed_size);

        if (sec->kind == FATBIN_SECTION_PTX) {
            const uint8_t *payload = pos + sec->header_size;
            uint32_t comp_size = sec->compressed_size;

            CUVK_LOG("[cuvk] PTX section: comp=%u padded=%u "
                    "first4=%02x%02x%02x%02x\n",
                    comp_size, sec->padded_payload_size,
                    payload[0], payload[1], payload[2], payload[3]);

            /* Check for ZSTD magic regardless of comp_size vs padded_payload_size,
             * since comp_size can equal padded_payload_size when padding is 0. */
            if (comp_size >= 4 && payload[0] == 0x28 && payload[1] == 0xB5
                && payload[2] == 0x2F && payload[3] == 0xFD) {
                uint64_t decomp_size = 0;
                if (sec->header_size >= 0x40) {
                    decomp_size = *(const uint64_t *)(pos + 0x38);
                }
                if (decomp_size == 0) {
                    decomp_size =
                        ZSTD_getFrameContentSize(payload, comp_size);
                    if (decomp_size == ZSTD_CONTENTSIZE_UNKNOWN ||
                        decomp_size == ZSTD_CONTENTSIZE_ERROR)
                        decomp_size = (uint64_t)sec->padded_payload_size * 4;
                }

                char *ptx = (char *)malloc(decomp_size + 1);
                if (!ptx)
                    return NULL;

                size_t result =
                    ZSTD_decompress(ptx, decomp_size, payload, comp_size);
                if (ZSTD_isError(result)) {
                    free(ptx);
                    return NULL;
                }

                ptx[result] = '\0';
                if (ptx_len)
                    *ptx_len = result;
                return ptx;
            }

            /* Uncompressed PTX */
            uint32_t payload_size = sec->padded_payload_size;
            size_t actual_len = strnlen((const char *)payload, payload_size);
            char *ptx = (char *)malloc(actual_len + 1);
            if (!ptx)
                return NULL;
            memcpy(ptx, payload, actual_len);
            ptx[actual_len] = '\0';
            if (ptx_len)
                *ptx_len = actual_len;
            return ptx;
        }

        pos += sec->header_size + sec->padded_payload_size;
    }

    return NULL;
}
