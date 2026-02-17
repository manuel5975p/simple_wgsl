#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// WGSL Spec Compliance Tests — Core Language
// Tests parse + lower + spirv-val for each feature.
// =============================================================================

// =============================================================================
// 1. Boolean Literals (true / false)
// =============================================================================

TEST(BoolLiteralTest, LetTrue) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let b = true;
    if b { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BoolLiteralTest, LetFalse) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let b = false;
    if b { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BoolLiteralTest, BoolInCondition) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    if true { return vec4<f32>(1.0); }
    return vec4<f32>(0.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BoolLiteralTest, BoolVar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var flag: bool = true;
    flag = false;
    if flag { return vec4<f32>(1.0); }
    return vec4<f32>(0.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BoolLiteralTest, BoolLogicalOps) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = true;
    let b = false;
    let c = a && b;
    let d = a || b;
    let e = !a;
    if c || d || e { return vec4<f32>(1.0); }
    return vec4<f32>(0.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 2. Break and Continue in Loops
// =============================================================================

TEST(BreakContinueTest, BreakInWhile) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var i: i32 = 0;
    while (i < 10) {
        if (i == 5) { break; }
        i = i + 1;
    }
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BreakContinueTest, ContinueInWhile) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var sum: f32 = 0.0;
    var i: i32 = 0;
    while (i < 10) {
        i = i + 1;
        if (i == 3) { continue; }
        sum = sum + 1.0;
    }
    return vec4<f32>(sum, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BreakContinueTest, BreakInFor) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var total: f32 = 0.0;
    for (var i: i32 = 0; i < 100; i = i + 1) {
        if (i >= 5) { break; }
        total = total + 1.0;
    }
    return vec4<f32>(total, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BreakContinueTest, ContinueInFor) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var sum: f32 = 0.0;
    for (var i: i32 = 0; i < 10; i = i + 1) {
        if (i % 2 == 0) { continue; }
        sum = sum + 1.0;
    }
    return vec4<f32>(sum, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BreakContinueTest, NestedLoopBreak) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var count: f32 = 0.0;
    for (var i: i32 = 0; i < 5; i = i + 1) {
        for (var j: i32 = 0; j < 5; j = j + 1) {
            if (j == 2) { break; }
            count = count + 1.0;
        }
    }
    return vec4<f32>(count, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 3. Switch / Case / Default
// =============================================================================

TEST(SwitchTest, BasicSwitch) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: i32 = 2;
    var result: f32 = 0.0;
    switch x {
        case 0: { result = 0.0; }
        case 1: { result = 0.5; }
        case 2: { result = 1.0; }
        default: { result = -1.0; }
    }
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwitchTest, SwitchDefaultOnly) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: i32 = 42;
    var v: f32 = 0.0;
    switch x {
        default: { v = 1.0; }
    }
    return vec4<f32>(v, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwitchTest, SwitchMultipleCaseValues) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: i32 = 3;
    var v: f32 = 0.0;
    switch x {
        case 1, 2, 3: { v = 1.0; }
        default: { v = 0.0; }
    }
    return vec4<f32>(v, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwitchTest, SwitchUnsigned) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 1u;
    var v: f32 = 0.0;
    switch x {
        case 0u: { v = 0.0; }
        case 1u: { v = 1.0; }
        default: { v = 0.5; }
    }
    return vec4<f32>(v, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 4. Discard in Fragment Shaders
// =============================================================================

TEST(DiscardTest, BasicDiscard) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    if (uv.x < 0.5) {
        discard;
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DiscardTest, DiscardInLoop) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) val: f32) -> @location(0) vec4<f32> {
    var x = val;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        if (x < 0.0) { discard; }
        x = x - 0.25;
    }
    return vec4<f32>(x, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 5. Compound Assignment Operators
// =============================================================================

TEST(CompoundAssignTest, PlusEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: f32 = 1.0;
    x += 2.0;
    return vec4<f32>(x, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, MinusEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: f32 = 5.0;
    x -= 2.0;
    return vec4<f32>(x, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, TimesEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: f32 = 3.0;
    x *= 2.0;
    return vec4<f32>(x, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, DivEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: f32 = 6.0;
    x /= 2.0;
    return vec4<f32>(x, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, ModEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: i32 = 7;
    x %= 3;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, AndEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: u32 = 0xFFu;
    x &= 0x0Fu;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, OrEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: u32 = 0xF0u;
    x |= 0x0Fu;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, XorEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: u32 = 0xFFu;
    x ^= 0x0Fu;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, ShlEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: u32 = 1u;
    x <<= 4u;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, ShrEquals) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: u32 = 256u;
    x >>= 4u;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, PlusEqualsVec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var v = vec3<f32>(1.0, 2.0, 3.0);
    v += vec3<f32>(0.1, 0.2, 0.3);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(CompoundAssignTest, TimesEqualsVec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    v *= vec4<f32>(0.5, 0.5, 0.5, 0.5);
    return v;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 6. Increment and Decrement
// =============================================================================

TEST(IncrDecrTest, PostfixIncrement) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var i: i32 = 0;
    i++;
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IncrDecrTest, PostfixDecrement) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var i: i32 = 5;
    i--;
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IncrDecrTest, IncrementInFor) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var sum: f32 = 0.0;
    for (var i: i32 = 0; i < 5; i++) {
        sum += 1.0;
    }
    return vec4<f32>(sum, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IncrDecrTest, IncrementUnsigned) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    var x: u32 = 0u;
    x++;
    x++;
    x++;
    return vec4<f32>(f32(x), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 7. Type Alias
// =============================================================================

TEST(TypeAliasTest, SimpleAlias) {
    auto r = wgsl_test::CompileWgsl(R"(
alias float4 = vec4<f32>;

@fragment fn main() -> @location(0) float4 {
    return float4(1.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TypeAliasTest, AliasInStruct) {
    auto r = wgsl_test::CompileWgsl(R"(
alias Color = vec4<f32>;
alias Position = vec3<f32>;

struct Vertex {
    @location(0) pos: Position,
    @location(1) col: Color,
};

@fragment fn main(v: Vertex) -> @location(0) vec4<f32> {
    return v.col;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 8. Missing Numeric Builtins — select, all, any
// =============================================================================

TEST(LogicalBuiltinTest, Select_Scalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = 0.0;
    let b = 1.0;
    let s = select(a, b, true);
    return vec4<f32>(s, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LogicalBuiltinTest, Select_Vec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let mask = vec3<bool>(true, false, true);
    let s = select(a, b, mask);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LogicalBuiltinTest, All_Vec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<bool>(true, true, true);
    if all(v) { return vec4<f32>(1.0); }
    return vec4<f32>(0.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LogicalBuiltinTest, Any_Vec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<bool>(false, true, false);
    if any(v) { return vec4<f32>(1.0); }
    return vec4<f32>(0.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 8b. Missing Numeric Builtins — fma, saturate, faceForward, refract
// =============================================================================

TEST(NumericBuiltinTest, Fma) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 2.0, 3.0);
    let b = vec3<f32>(2.0, 3.0, 4.0);
    let c = vec3<f32>(0.5, 0.5, 0.5);
    let result = fma(a, b, c);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(NumericBuiltinTest, Saturate) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-0.5, 0.5, 1.5);
    let s = saturate(v);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(NumericBuiltinTest, FaceForward) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let i = vec3<f32>(0.0, -1.0, 0.0);
    let nref = vec3<f32>(0.0, 1.0, 0.0);
    let f = faceForward(n, i, nref);
    return vec4<f32>(f, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(NumericBuiltinTest, Refract) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let i = normalize(vec3<f32>(1.0, -1.0, 0.0));
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let r = refract(i, n, 1.5);
    return vec4<f32>(r, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 8c. Missing Numeric Builtins — transpose, determinant
// =============================================================================

TEST(MatrixBuiltinTest, Transpose) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat3x3<f32>(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
    let t = transpose(m);
    return vec4<f32>(t[0], 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(MatrixBuiltinTest, Determinant) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat3x3<f32>(
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0
    );
    let d = determinant(m);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 8d. Missing Numeric Builtins — arrayLength
// =============================================================================

TEST(ArrayBuiltinTest, ArrayLength) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Storage { data: array<f32>, };
@group(0) @binding(0) var<storage, read> buf: Storage;

@fragment fn main() -> @location(0) vec4<f32> {
    let len = arrayLength(&buf.data);
    return vec4<f32>(f32(len), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// 9. Bit Manipulation Builtins
// =============================================================================

TEST(BitManipTest, CountOneBits) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 0xABu;
    let c = countOneBits(x);
    return vec4<f32>(f32(c), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountLeadingZeros) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 16u;
    let c = countLeadingZeros(x);
    return vec4<f32>(f32(c), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountTrailingZeros) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 16u;
    let c = countTrailingZeros(x);
    return vec4<f32>(f32(c), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ReverseBits) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 1u;
    let rev = reverseBits(x);
    return vec4<f32>(f32(rev), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstLeadingBit_Unsigned) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 255u;
    let b = firstLeadingBit(x);
    return vec4<f32>(f32(b), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstLeadingBit_Signed) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: i32 = 255;
    let b = firstLeadingBit(x);
    return vec4<f32>(f32(b), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, FirstTrailingBit) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 12u;
    let b = firstTrailingBit(x);
    return vec4<f32>(f32(b), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ExtractBits_Unsigned) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: u32 = 0xABCDu;
    let bits = extractBits(x, 4u, 8u);
    return vec4<f32>(f32(bits), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ExtractBits_Signed) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x: i32 = -1;
    let bits = extractBits(x, 0u, 8u);
    return vec4<f32>(f32(bits), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, InsertBits) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base: u32 = 0u;
    let insert: u32 = 0xFFu;
    let result = insertBits(base, insert, 8u, 8u);
    return vec4<f32>(f32(result), 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, CountOneBits_Vec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<u32>(0xFFu, 0x0Fu, 0x01u);
    let c = countOneBits(v);
    return vec4<f32>(f32(c.x), f32(c.y), f32(c.z), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BitManipTest, ReverseBits_Vec) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<u32>(1u, 2u);
    let rev = reverseBits(v);
    return vec4<f32>(f32(rev.x), f32(rev.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
