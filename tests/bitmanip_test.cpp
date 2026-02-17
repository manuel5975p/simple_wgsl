#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// Bit manipulation builtin functions
// countOneBits, reverseBits, countLeadingZeros, countTrailingZeros,
// firstLeadingBit, firstTrailingBit, extractBits, insertBits
// =============================================================================

// -----------------------------------------------------------------------------
// countOneBits(x) - counts set bits
// SPIR-V: SpvOpBitCount
// Works on i32, u32, and vectors thereof
// -----------------------------------------------------------------------------

TEST(BitManipTest, CountOneBits_ScalarU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: u32 = 0xF0F0u;     // 0b1111000011110000 -> 8 bits set
    output.data[0] = countOneBits(v);
    output.data[1] = countOneBits(0u);
    output.data[2] = countOneBits(0xFFFFFFFFu);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountOneBits_ScalarI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: i32 = 7;           // 0b111 -> 3 bits set
    output.data[0] = countOneBits(v);
    output.data[1] = countOneBits(-1i);  // all bits set -> 32
    output.data[2] = countOneBits(0i);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountOneBits_Vec2U32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = vec2<u32>(0xAAAAu, 0x5555u);  // alternating bits: 8 each
    let result = countOneBits(v);
    output.data[0] = result.x;
    output.data[1] = result.y;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// reverseBits(x) - reverses bit order
// SPIR-V: SpvOpBitReverse
// Works on i32, u32, and vectors thereof
// -----------------------------------------------------------------------------

TEST(BitManipTest, ReverseBits_ScalarU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: u32 = 1u;          // bit 0 set -> bit 31 set after reverse
    output.data[0] = reverseBits(v);
    output.data[1] = reverseBits(0x80000000u);   // bit 31 -> bit 0
    output.data[2] = reverseBits(0xFFFFFFFFu);   // all bits -> all bits
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ReverseBits_ScalarI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: i32 = 1i;
    output.data[0] = reverseBits(v);
    output.data[1] = reverseBits(-1i);   // all bits -> all bits
    output.data[2] = reverseBits(0i);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ReverseBits_Vec2U32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = vec2<u32>(1u, 0x80000000u);
    let result = reverseBits(v);
    output.data[0] = result.x;   // expect 0x80000000
    output.data[1] = result.y;   // expect 1
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// countLeadingZeros(x) - CLZ
// Emulated via GLSLstd450FindUMsb
// Works on i32, u32
// -----------------------------------------------------------------------------

TEST(BitManipTest, CountLeadingZeros_ScalarU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = countLeadingZeros(1u);             // 31 leading zeros
    output.data[1] = countLeadingZeros(0x80000000u);    // 0 leading zeros
    output.data[2] = countLeadingZeros(256u);            // 23 leading zeros (bit 8)
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountLeadingZeros_ScalarI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = countLeadingZeros(1i);    // 31 leading zeros
    output.data[1] = countLeadingZeros(-1i);   // 0 leading zeros (all bits set)
    output.data[2] = countLeadingZeros(16i);   // 27 leading zeros (bit 4)
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountLeadingZeros_ZeroValue) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = countLeadingZeros(0u);    // 32 leading zeros
    output.data[1] = countLeadingZeros(0u);    // 32 leading zeros (redundant, for coverage)
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// countTrailingZeros(x) - CTZ
// Emulated via GLSLstd450FindILsb
// Works on i32, u32
// -----------------------------------------------------------------------------

TEST(BitManipTest, CountTrailingZeros_ScalarU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = countTrailingZeros(1u);             // 0 trailing zeros
    output.data[1] = countTrailingZeros(0x80000000u);    // 31 trailing zeros
    output.data[2] = countTrailingZeros(8u);             // 3 trailing zeros
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountTrailingZeros_ScalarI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = countTrailingZeros(1i);      // 0 trailing zeros
    output.data[1] = countTrailingZeros(-4i);     // 2 trailing zeros (0x...FC)
    output.data[2] = countTrailingZeros(64i);     // 6 trailing zeros
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountTrailingZeros_ZeroValue) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = countTrailingZeros(0u);    // 32 trailing zeros
    output.data[1] = countTrailingZeros(0u);    // 32 trailing zeros (redundant, for coverage)
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// firstLeadingBit(x) - for unsigned: GLSLstd450FindUMsb, for signed: GLSLstd450FindSMsb
// Returns the bit index of the most significant 1 bit (unsigned) or
// most significant bit that differs from the sign bit (signed).
// Returns -1 (or 0xFFFFFFFF for unsigned) if no such bit exists.
// -----------------------------------------------------------------------------

TEST(BitManipTest, FirstLeadingBit_UnsignedU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = firstLeadingBit(1u);             // bit 0 -> returns 0
    output.data[1] = firstLeadingBit(0x80000000u);    // bit 31 -> returns 31
    output.data[2] = firstLeadingBit(255u);           // bit 7 -> returns 7
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstLeadingBit_SignedPositiveI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = firstLeadingBit(1i);    // bit 0 -> returns 0
    output.data[1] = firstLeadingBit(127i);  // bit 6 -> returns 6
    output.data[2] = firstLeadingBit(0i);    // no bit -> returns -1
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstLeadingBit_SignedNegativeI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = firstLeadingBit(-1i);    // all 1s -> returns -1 (no bit differs from sign)
    output.data[1] = firstLeadingBit(-2i);    // 0xFFFFFFFE -> bit 0 differs from sign -> returns 0
    output.data[2] = firstLeadingBit(-128i);  // first 0-bit from MSB at bit 6 -> returns 6
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// firstTrailingBit(x) - GLSLstd450FindILsb
// Returns the bit index of the least significant 1 bit.
// Returns -1 (or 0xFFFFFFFF) if the input is 0.
// -----------------------------------------------------------------------------

TEST(BitManipTest, FirstTrailingBit_ScalarU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = firstTrailingBit(1u);             // bit 0 -> returns 0
    output.data[1] = firstTrailingBit(0x80000000u);    // bit 31 -> returns 31
    output.data[2] = firstTrailingBit(12u);            // 0b1100 -> bit 2 -> returns 2
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstTrailingBit_ScalarI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = firstTrailingBit(1i);     // bit 0 -> returns 0
    output.data[1] = firstTrailingBit(-4i);    // 0x...FC -> bit 2 -> returns 2
    output.data[2] = firstTrailingBit(48i);    // 0b110000 -> bit 4 -> returns 4
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstTrailingBit_ZeroValue) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = firstTrailingBit(0u);     // no bit set -> returns 0xFFFFFFFF
    output.data[1] = firstTrailingBit(0u);     // redundant, for coverage
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// extractBits(v, offset, count)
// Extracts count bits from v starting at offset.
// unsigned: SpvOpBitFieldUExtract, signed: SpvOpBitFieldSExtract
// -----------------------------------------------------------------------------

TEST(BitManipTest, ExtractBits_UnsignedU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: u32 = 0xABCD1234u;
    output.data[0] = extractBits(v, 4u, 8u);    // extract bits 4..11 -> 0x23
    output.data[1] = extractBits(v, 0u, 4u);    // extract bits 0..3 -> 0x4
    output.data[2] = extractBits(v, 16u, 16u);  // extract bits 16..31 -> 0xABCD
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ExtractBits_SignedI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: i32 = 0x0000FF00i;
    output.data[0] = extractBits(v, 8u, 8u);    // extract bits 8..15 -> 0xFF -> sign-extended to -1
    output.data[1] = extractBits(v, 8u, 4u);    // extract bits 8..11 -> 0x0 (lower nibble of 0xFF)
    output.data[2] = extractBits(v, 0u, 16u);   // extract bits 0..15 -> 0xFF00 -> sign-extended to -256
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ExtractBits_Vec2U32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = vec2<u32>(0xFF00FF00u, 0x12345678u);
    let result = extractBits(v, 8u, 8u);
    output.data[0] = result.x;    // extract bits 8..15 of 0xFF00FF00 -> 0xFF
    output.data[1] = result.y;    // extract bits 8..15 of 0x12345678 -> 0x56
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// insertBits(v, newbits, offset, count)
// Inserts count bits from newbits into v at offset.
// SPIR-V: SpvOpBitFieldInsert
// -----------------------------------------------------------------------------

TEST(BitManipTest, InsertBits_ScalarU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: u32 = 0x00000000u;
    output.data[0] = insertBits(v, 0xFFu, 8u, 8u);   // insert 0xFF at bits 8..15 -> 0x0000FF00
    output.data[1] = insertBits(v, 0xAu, 0u, 4u);    // insert 0xA at bits 0..3 -> 0x0000000A
    output.data[2] = insertBits(0xFFFFFFFFu, 0u, 4u, 8u); // clear bits 4..11 -> 0xFFFFF00F
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, InsertBits_ScalarI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v: i32 = 0i;
    output.data[0] = insertBits(v, 0x7Fi, 0u, 8u);    // insert 0x7F at bits 0..7
    output.data[1] = insertBits(v, -1i, 4u, 4u);       // insert 0xF at bits 4..7 -> 0xF0
    output.data[2] = insertBits(255i, 0i, 0u, 4u);     // clear low 4 bits of 255 -> 0xF0
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, InsertBits_Vec2U32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = vec2<u32>(0u, 0xFFFFFFFFu);
    let newbits = vec2<u32>(0xABu, 0u);
    let result = insertBits(v, newbits, 8u, 8u);
    output.data[0] = result.x;    // insert 0xAB at bits 8..15 of 0 -> 0x0000AB00
    output.data[1] = result.y;    // insert 0 at bits 8..15 of 0xFFFFFFFF -> 0xFFFF00FF
})");
    EXPECT_TRUE(r.success) << r.error;
}
