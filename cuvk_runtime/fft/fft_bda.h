#ifndef FFT_BDA_H
#define FFT_BDA_H

#include "fft_strbuf.h"

/*
 * FftDeviceCaps: runtime device capabilities for FFT shader generation.
 *
 *   use_bda:  1 = var<device> with push-constant addresses (BDA).
 *                 Requires VK_KHR_buffer_device_address + 64-bit support in NIR.
 *             0 = @group(0) @binding(N) var<storage> with descriptor sets.
 *                 Works on all Vulkan 1.1+ devices.
 *
 * BDA push constant layout (host side must match):
 *   offset 0:  u64  src/data address
 *   offset 8:  u64  dst address (if separate src/dst)
 *   offset 16: u64  lut/ctl address (if LUT or control buffer)
 *
 * Binding layout (non-BDA):
 *   binding 0: src / data
 *   binding 1: dst         (if separate src/dst)
 *   binding 2: lut / ctl   (if present)
 */

typedef struct FftDeviceCaps {
    int use_bda;
} FftDeviceCaps;

/* Default caps: BDA enabled (backward compatible) */
static inline FftDeviceCaps fft_default_caps(void) {
    FftDeviceCaps c;
    c.use_bda = 1;
    return c;
}

/* Emit struct + var declarations for src/dst pattern */
static inline void sb_emit_bufs_src_dst(StrBuf *sb, int has_lut, const FftDeviceCaps *caps) {
    if (caps->use_bda) {
        sb_printf(sb, "enable device_address;\n");
        sb_printf(sb, "struct SrcBuf { d: array<f32> };\n");
        sb_printf(sb, "struct DstBuf { d: array<f32> };\n");
        sb_printf(sb, "var<device, read> src: SrcBuf;\n");
        sb_printf(sb, "var<device, read_write> dst: DstBuf;\n");
        if (has_lut) {
            sb_printf(sb, "struct LutBuf { d: array<f32> };\n");
            sb_printf(sb, "var<device, read> lut: LutBuf;\n");
        }
    } else {
        sb_printf(sb, "struct SrcBuf { d: array<f32> };\n");
        sb_printf(sb, "struct DstBuf { d: array<f32> };\n");
        sb_printf(sb, "@group(0) @binding(0) var<storage, read> src: SrcBuf;\n");
        sb_printf(sb, "@group(0) @binding(1) var<storage, read_write> dst: DstBuf;\n");
        if (has_lut) {
            sb_printf(sb, "struct LutBuf { d: array<f32> };\n");
            sb_printf(sb, "@group(0) @binding(2) var<storage, read> lut: LutBuf;\n");
        }
    }
}

/* Emit struct + var declarations for in-place (data only) pattern */
static inline void sb_emit_bufs_inplace(StrBuf *sb, const FftDeviceCaps *caps) {
    if (caps->use_bda) {
        sb_printf(sb, "enable device_address;\n");
        sb_printf(sb, "struct DataBuf { d: array<f32> };\n");
        sb_printf(sb, "var<device, read_write> data: DataBuf;\n");
    } else {
        sb_printf(sb, "struct DataBuf { d: array<f32> };\n");
        sb_printf(sb, "@group(0) @binding(0) var<storage, read_write> data: DataBuf;\n");
    }
}

/* Emit struct + var declarations for in-place data + control buffer */
static inline void sb_emit_bufs_inplace_ctl(StrBuf *sb, const FftDeviceCaps *caps) {
    if (caps->use_bda) {
        sb_printf(sb, "enable device_address;\n");
        sb_printf(sb, "struct DataBuf { d: array<f32> };\n");
        sb_printf(sb, "var<device, read_write> data: DataBuf;\n");
        sb_printf(sb, "struct CtlBuf { d: array<f32> };\n");
        sb_printf(sb, "var<device, read> ctl: CtlBuf;\n");
    } else {
        sb_printf(sb, "struct DataBuf { d: array<f32> };\n");
        sb_printf(sb, "@group(0) @binding(0) var<storage, read_write> data: DataBuf;\n");
        sb_printf(sb, "struct CtlBuf { d: array<f32> };\n");
        sb_printf(sb, "@group(0) @binding(1) var<storage, read> ctl: CtlBuf;\n");
    }
}

/* Backward-compatible wrappers using default caps (BDA=1) */
static inline void sb_emit_bda_src_dst(StrBuf *sb, int has_lut) {
    FftDeviceCaps c = fft_default_caps();
    sb_emit_bufs_src_dst(sb, has_lut, &c);
}
static inline void sb_emit_bda_inplace(StrBuf *sb) {
    FftDeviceCaps c = fft_default_caps();
    sb_emit_bufs_inplace(sb, &c);
}
static inline void sb_emit_bda_inplace_ctl(StrBuf *sb) {
    FftDeviceCaps c = fft_default_caps();
    sb_emit_bufs_inplace_ctl(sb, &c);
}

/* Push constant byte sizes for pipeline layout creation (BDA mode only) */
#define FFT_BDA_PC_SIZE_1BUF   8   /* data only */
#define FFT_BDA_PC_SIZE_2BUF  16   /* src + dst */
#define FFT_BDA_PC_SIZE_3BUF  24   /* src + dst + lut/ctl */

#endif /* FFT_BDA_H */
