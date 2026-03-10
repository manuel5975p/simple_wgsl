#ifndef FFT_BDA_H
#define FFT_BDA_H

#include "fft_strbuf.h"

/*
 * Shared helpers for emitting var<device> declarations in FFT generators.
 * Replaces @group(0) @binding(N) var<storage, ...> with var<device, ...>.
 *
 * Push constant layout (host side must match):
 *   offset 0:  u64  src/data address
 *   offset 8:  u64  dst address (if separate src/dst)
 *   offset 16: u64  lut/ctl address (if LUT or control buffer)
 */

/* Emit enable + struct + var declarations for src/dst pattern */
static inline void sb_emit_bda_src_dst(StrBuf *sb, int has_lut) {
    sb_printf(sb, "enable device_address;\n");
    sb_printf(sb, "struct SrcBuf { d: array<f32> };\n");
    sb_printf(sb, "struct DstBuf { d: array<f32> };\n");
    sb_printf(sb, "var<device, read> src: SrcBuf;\n");
    sb_printf(sb, "var<device, read_write> dst: DstBuf;\n");
    if (has_lut) {
        sb_printf(sb, "struct LutBuf { d: array<f32> };\n");
        sb_printf(sb, "var<device, read> lut: LutBuf;\n");
    }
}

/* Emit enable + struct + var declarations for in-place (data only) pattern */
static inline void sb_emit_bda_inplace(StrBuf *sb) {
    sb_printf(sb, "enable device_address;\n");
    sb_printf(sb, "struct DataBuf { d: array<f32> };\n");
    sb_printf(sb, "var<device, read_write> data: DataBuf;\n");
}

/* Emit enable + struct + var declarations for in-place data + control buffer */
static inline void sb_emit_bda_inplace_ctl(StrBuf *sb) {
    sb_printf(sb, "enable device_address;\n");
    sb_printf(sb, "struct DataBuf { d: array<f32> };\n");
    sb_printf(sb, "var<device, read_write> data: DataBuf;\n");
    sb_printf(sb, "struct CtlBuf { d: array<f32> };\n");
    sb_printf(sb, "var<device, read> ctl: CtlBuf;\n");
}

/* Push constant byte sizes for pipeline layout creation */
#define FFT_BDA_PC_SIZE_1BUF   8   /* data only */
#define FFT_BDA_PC_SIZE_2BUF  16   /* src + dst */
#define FFT_BDA_PC_SIZE_3BUF  24   /* src + dst + lut/ctl */

#endif /* FFT_BDA_H */
