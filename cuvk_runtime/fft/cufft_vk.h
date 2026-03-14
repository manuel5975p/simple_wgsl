/*
 * cufft_vk.h - Vulkan-native cuFFT API for manual command buffer building
 *
 * Provides CuvkWorkPackage: a self-describing GPU command sequence that can be
 * inspected, recorded into any VkCommandBuffer, or submitted standalone.
 *
 * Usage:
 *   CuvkWorkPackage wp;
 *   cuvk_wp_init(&wp, ctx);
 *   vkCufftExecC2C(&wp, plan, src_buf, src_bda, dst_buf, dst_bda, CUFFT_FORWARD);
 *
 *   // Option A: encode into your own command buffer
 *   vkBeginCommandBuffer(your_cb, ...);
 *   cuvk_wp_encode(&wp, your_cb);
 *   vkEndCommandBuffer(your_cb);
 *
 *   // Option B: seal + submit standalone
 *   cuvk_wp_seal(&wp);
 *   cuvk_wp_submit(&wp);
 *
 *   cuvk_wp_destroy(&wp);
 */

#ifndef CUVK_CUFFT_VK_H
#define CUVK_CUFFT_VK_H

#include <vulkan/vulkan.h>
#include "cufft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
struct CUctx_st;

/* ============================================================================
 * Work package command types
 * ============================================================================ */

typedef enum {
    CUVK_WORK_BARRIER,       /* Pipeline memory barrier */
    CUVK_WORK_COPY,          /* Buffer-to-buffer copy */
    CUVK_WORK_DISPATCH,      /* Bind pipeline + push constants + dispatch */
} CuvkWorkCmdType;

typedef struct {
    CuvkWorkCmdType type;
    union {
        struct {
            VkPipelineStageFlags src_stage;
            VkPipelineStageFlags dst_stage;
            VkAccessFlags        src_access;
            VkAccessFlags        dst_access;
        } barrier;

        struct {
            VkBuffer     src;
            VkBuffer     dst;
            VkDeviceSize src_offset;
            VkDeviceSize dst_offset;
            VkDeviceSize size;
        } copy;

        struct {
            VkPipeline       pipeline;
            VkPipelineLayout layout;
            uint8_t          push_data[128];
            uint32_t         push_size;
            uint32_t         group_count_x;
            uint32_t         group_count_y;
            uint32_t         group_count_z;
        } dispatch;
    };
} CuvkWorkCmd;

/* ============================================================================
 * Buffer reference tracking
 * ============================================================================ */

#define CUVK_BUF_READ  0x1
#define CUVK_BUF_WRITE 0x2

typedef struct {
    VkBuffer        buffer;
    VkDeviceAddress bda;
    uint32_t        access;   /* CUVK_BUF_READ | CUVK_BUF_WRITE */
} CuvkWorkBufRef;

/* ============================================================================
 * CuvkWorkPackage
 * ============================================================================ */

typedef struct CuvkWorkPackage {
    struct CUctx_st *ctx;

    /* Pre-recorded command buffer (owned, allocated from ctx->cmd_pool) */
    VkCommandBuffer  cmd_buf;
    int              sealed;       /* 1 = cmd_buf matches cmds[], 0 = stale */

    /* Structured command list */
    CuvkWorkCmd     *cmds;
    int              n_cmds;
    int              cmd_cap;

    /* Deduplicated buffer references */
    CuvkWorkBufRef  *bufs;
    int              n_bufs;
    int              buf_cap;
} CuvkWorkPackage;

/* ============================================================================
 * Work package API
 * ============================================================================ */

/* Lifecycle */
cufftResult cuvk_wp_init(CuvkWorkPackage *wp, struct CUctx_st *ctx);
void        cuvk_wp_destroy(CuvkWorkPackage *wp);
void        cuvk_wp_clear(CuvkWorkPackage *wp);

/* Building (appends to cmds[]) */
void cuvk_wp_barrier(CuvkWorkPackage *wp,
                     VkPipelineStageFlags src_stage,
                     VkPipelineStageFlags dst_stage,
                     VkAccessFlags src_access,
                     VkAccessFlags dst_access);

void cuvk_wp_copy(CuvkWorkPackage *wp,
                  VkBuffer src, VkBuffer dst,
                  VkDeviceSize src_off, VkDeviceSize dst_off,
                  VkDeviceSize size);

void cuvk_wp_dispatch(CuvkWorkPackage *wp,
                      VkPipeline pipeline, VkPipelineLayout layout,
                      const void *push_data, uint32_t push_size,
                      uint32_t gx, uint32_t gy, uint32_t gz);

void cuvk_wp_ref_buf(CuvkWorkPackage *wp,
                     VkBuffer buf, VkDeviceAddress bda, uint32_t access);

/* Recording */
void        cuvk_wp_encode(const CuvkWorkPackage *wp, VkCommandBuffer cb);
cufftResult cuvk_wp_seal(CuvkWorkPackage *wp);

/* Submission */
cufftResult cuvk_wp_submit(CuvkWorkPackage *wp);

/* ============================================================================
 * Vulkan-native cuFFT execution API
 *
 * These populate a work package with FFT commands. The caller then uses
 * cuvk_wp_encode(), cuvk_wp_seal(), or cuvk_wp_submit() to execute.
 * ============================================================================ */

cufftResult vkCufftExecC2C(CuvkWorkPackage *wp, cufftHandle plan,
                            VkBuffer idata_buf, VkDeviceAddress idata_bda,
                            VkBuffer odata_buf, VkDeviceAddress odata_bda,
                            int direction);

cufftResult vkCufftExecR2C(CuvkWorkPackage *wp, cufftHandle plan,
                            VkBuffer idata_buf, VkDeviceAddress idata_bda,
                            VkBuffer odata_buf, VkDeviceAddress odata_bda);

cufftResult vkCufftExecC2R(CuvkWorkPackage *wp, cufftHandle plan,
                            VkBuffer idata_buf, VkDeviceAddress idata_bda,
                            VkBuffer odata_buf, VkDeviceAddress odata_bda);

#ifdef __cplusplus
}
#endif

#endif /* CUVK_CUFFT_VK_H */
