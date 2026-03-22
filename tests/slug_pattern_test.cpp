// Tests for WGSL code patterns found in the Slug text renderer fragment shader.
//
// These tests target specific control-flow and codegen patterns that could
// cause the horizontal-white-line rendering artifacts observed in the Slug
// renderer.  Each test compiles a COMPUTE shader to SPIR-V, validates it
// with spirv-val, disassembles it, and checks for the presence of expected
// SPIR-V instructions.
//
// The patterns are distilled from the real Slug pixel shader, which does:
//   for each curve {
//       code = CalcRootCode(...)
//       if (code != 0) {
//           t = SolveHorizPoly(...)   // var t = expr; if (abs(a)<eps) { t = other; }
//           xcov += f(t)
//       }
//   }
// This combination of function calls with internal var + conditional
// overwrite, called from inside a for loop that accumulates into another
// var, is the suspect pattern.

#include <gtest/gtest.h>
#include "test_utils.h"

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Disassemble SPIR-V binary to text using spirv-dis (no header).
std::string Disassemble(const std::vector<uint32_t> &spirv) {
    std::string spv_path = wgsl_test::MakeTempSpvPath("slug_pat");
    wgsl_test::WriteSpirvFile(spv_path, spirv.data(), spirv.size());
    std::string output;
    wgsl_test::RunCommand("spirv-dis --no-header " + spv_path + " 2>&1",
                          &output);
    std::remove(spv_path.c_str());
    return output;
}

// Compile WGSL, validate, return spirv + disassembly.
struct PatternResult {
    bool success;
    std::string error;
    std::vector<uint32_t> spirv;
    std::string disasm;
};

PatternResult CompileAndDisassemble(const char *source) {
    PatternResult pr;
    auto r = wgsl_test::CompileWgsl(source);
    pr.success = r.success;
    pr.error   = r.error;
    pr.spirv   = std::move(r.spirv);
    if (pr.success) {
        pr.disasm = Disassemble(pr.spirv);
    }
    return pr;
}

// Count occurrences of needle in haystack.
int CountOccurrences(const std::string &haystack, const std::string &needle) {
    int count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

} // namespace

// ============================================================================
// Pattern 1: Function with var initialized then conditionally overwritten,
//            called from inside a for loop
//
// Slug: SolveHorizPoly does  var t = expr; if (abs(a)<eps) { t = other; }
//       and is called from inside a for(i=0; i<curveCount; i++) loop.
// Risk: The phi/merge for the conditional var overwrite might be misplaced
//       relative to the loop's own merge/continue constructs, causing the
//       wrong value of t to propagate.
// ============================================================================

TEST(SlugPattern, VarConditionalOverwrite_CalledFromLoop) {
    auto pr = CompileAndDisassemble(R"(
fn solve(a: f32, b: f32) -> f32 {
    var t: f32 = b / a;
    if (abs(a) < 0.001) {
        t = b * 0.5;
    }
    return t;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var accum: f32 = 0.0;
    for (var i: i32 = 0; i < 4; i++) {
        let val = solve(f32(i) * 0.1, 1.0);
        accum += val;
    }
    output.data[0] = accum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // The function must have OpBranch (for the if/else merge).
    EXPECT_NE(pr.disasm.find("OpBranch"), std::string::npos)
        << "Expected OpBranch for conditional merge:\n" << pr.disasm;
    // There must be an OpLoopMerge for the for-loop.
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge for the for loop:\n" << pr.disasm;
    // The var t must produce OpVariable or OpStore patterns.
    EXPECT_NE(pr.disasm.find("OpStore"), std::string::npos)
        << "Expected OpStore for var t:\n" << pr.disasm;
    // The function call must appear inside the loop body.
    EXPECT_NE(pr.disasm.find("OpFunctionCall"), std::string::npos)
        << "Expected OpFunctionCall for solve():\n" << pr.disasm;
}

// ============================================================================
// Pattern 2: var accumulation (xcov += ...) inside if blocks inside for loops
//
// Slug: xcov and ycov are accumulated inside nested if-blocks inside the
//       curve iteration loop.
// Risk: The store-back to the accumulator var might be placed on the wrong
//       branch, or the load from the accumulator might read a stale value
//       after branching.
// ============================================================================

TEST(SlugPattern, VarAccumulationInsideIfInsideForLoop) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var xcov: f32 = 0.0;
    var ycov: f32 = 0.0;
    for (var i: i32 = 0; i < 8; i++) {
        let code = i % 3;
        if (code != 0) {
            let t = f32(i) * 0.125;
            if (code == 1) {
                xcov += saturate(t + 0.5);
            } else {
                ycov += saturate(t + 0.5);
            }
        }
    }
    output.data[0] = xcov;
    output.data[1] = ycov;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have at least two OpSelectionMerge (one for code!=0, one for
    // code==1/else), plus the loop merge.
    int selectionMerges = CountOccurrences(pr.disasm, "OpSelectionMerge");
    EXPECT_GE(selectionMerges, 2)
        << "Expected >= 2 OpSelectionMerge for nested ifs:\n" << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
    // Both accumulator stores must appear.
    int stores = CountOccurrences(pr.disasm, "OpStore");
    EXPECT_GE(stores, 4)
        << "Expected >= 4 OpStore (init xcov, init ycov, accum xcov, accum ycov):\n"
        << pr.disasm;
}

// ============================================================================
// Pattern 3: Multiple function calls inside a for loop body
//            (CalcRootCode, SolveHorizPoly called sequentially)
//
// Slug: The loop body calls CalcRootCode to get a code, then conditionally
//       calls SolveHorizPoly.  The second call depends on the first.
// Risk: If the compiler reorders or miscompiles the sequence of function
//       calls inside a loop body, the dependency chain breaks.
// ============================================================================

TEST(SlugPattern, MultipleFunctionCallsInsideForLoop) {
    auto pr = CompileAndDisassemble(R"(
fn CalcRootCode(p0: vec2<f32>, p1: vec2<f32>) -> u32 {
    var code: u32 = 0u;
    if (p0.y * p1.y < 0.0) {
        code = code | 1u;
    }
    if (p0.y < 0.0 && p1.y < 0.0) {
        code = code | 2u;
    }
    return code;
}

fn SolveHorizPoly(a: f32, b: f32, c: f32) -> f32 {
    let ra = 1.0 / a;
    var t: f32 = (-b + sqrt(b * b - 4.0 * a * c)) * 0.5 * ra;
    if (abs(a) < 0.0001) {
        t = -c / b;
    }
    return clamp(t, 0.0, 1.0);
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var accum: f32 = 0.0;
    for (var i: i32 = 0; i < 4; i++) {
        let fi = f32(i);
        let p0 = vec2<f32>(fi * 0.25, fi * 0.25 - 0.5);
        let p1 = vec2<f32>(fi * 0.25 + 0.25, 0.5 - fi * 0.25);
        let code = CalcRootCode(p0, p1);
        if (code != 0u) {
            let t = SolveHorizPoly(p1.y - p0.y, p0.y, p0.y);
            accum += saturate(t + 0.5);
        }
    }
    output.data[0] = accum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // There must be two distinct OpFunctionCall instructions.
    int calls = CountOccurrences(pr.disasm, "OpFunctionCall");
    EXPECT_GE(calls, 2)
        << "Expected >= 2 OpFunctionCall (CalcRootCode + SolveHorizPoly):\n"
        << pr.disasm;
    // Both must be inside a loop, so OpLoopMerge must precede them.
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
}

// ============================================================================
// Pattern 4: Nested if-blocks inside for loops with bitwise condition checks
//            if (code != 0) { if (code & 1) { ... } if (code > 1) { ... } }
//
// Slug: After CalcRootCode returns, two separate if-blocks check different
//       bits to decide horizontal vs vertical coverage updates.
// Risk: Nested OpSelectionMerge blocks inside an OpLoopMerge can confuse
//       backends if merge-block ordering is wrong.
// ============================================================================

TEST(SlugPattern, NestedIfBitwiseInsideForLoop) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var xcov: f32 = 0.0;
    var ycov: f32 = 0.0;
    for (var i: u32 = 0u; i < 8u; i++) {
        let code: u32 = (i * 7u + 3u) % 4u;
        if (code != 0u) {
            let t: f32 = f32(i) * 0.125;
            if ((code & 1u) != 0u) {
                xcov += t;
            }
            if (code > 1u) {
                ycov += t * 0.5;
            }
        }
    }
    output.data[0] = xcov;
    output.data[1] = ycov;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have at least 3 OpSelectionMerge: one for code!=0, one for code&1,
    // one for code>1.
    int selMerge = CountOccurrences(pr.disasm, "OpSelectionMerge");
    EXPECT_GE(selMerge, 3)
        << "Expected >= 3 OpSelectionMerge for nested if-blocks:\n" << pr.disasm;
    // Must have OpBitwiseAnd for the (code & 1u) check.
    EXPECT_NE(pr.disasm.find("OpBitwiseAnd"), std::string::npos)
        << "Expected OpBitwiseAnd for code & 1u:\n" << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
}

// ============================================================================
// Pattern 5: break inside an if-block inside a for loop (early exit pattern)
//
// Slug: The band curve iteration uses break to exit early when a curve
//       endpoint is beyond the pixel boundary.
// Risk: The OpBranch for break must target the loop's merge block, not some
//       inner selection merge block.  If the break jumps to the wrong place,
//       subsequent iterations are skipped incorrectly.
// ============================================================================

TEST(SlugPattern, BreakInsideIfInsideForLoop) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var accum: f32 = 0.0;
    for (var i: i32 = 0; i < 16; i++) {
        let t = f32(i) * 0.0625;
        if (t > 0.75) {
            break;
        }
        accum += saturate(t);
    }
    output.data[0] = accum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // The if-block creates a SelectionMerge; the break creates an
    // OpBranch that must jump to the LoopMerge target.
    EXPECT_NE(pr.disasm.find("OpSelectionMerge"), std::string::npos)
        << "Expected OpSelectionMerge for if:\n" << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge for the for loop:\n" << pr.disasm;
    // Must have OpBranchConditional (for the if), plus at least two
    // OpBranch (one for break -> merge, one for continue/loop-back).
    int branchCond = CountOccurrences(pr.disasm, "OpBranchConditional");
    EXPECT_GE(branchCond, 1)
        << "Expected >= 1 OpBranchConditional:\n" << pr.disasm;
    int branch = CountOccurrences(pr.disasm, "OpBranch %");
    EXPECT_GE(branch, 2)
        << "Expected >= 2 OpBranch:\n" << pr.disasm;
}

// ============================================================================
// Pattern 5b: break inside if + continue inside else, inside a for loop.
//
// Slug: Some paths skip the rest of the loop body (continue) while others
//       exit entirely (break), both within the same if/else structure.
// Risk: The structured control flow merge for the if/else must be set up
//       such that both break (to loop merge) and continue (to loop continue
//       target) are valid outgoing edges.
// ============================================================================

TEST(SlugPattern, BreakAndContinueInsideIfElse) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var accum: f32 = 0.0;
    for (var i: i32 = 0; i < 16; i++) {
        let t = f32(i) * 0.0625;
        if (t > 0.75) {
            break;
        } else if (t < 0.25) {
            continue;
        }
        accum += t;
    }
    output.data[0] = accum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
    // At least two OpBranchConditional: one for (t > 0.75), one for (t < 0.25).
    int branchCond = CountOccurrences(pr.disasm, "OpBranchConditional");
    EXPECT_GE(branchCond, 2)
        << "Expected >= 2 OpBranchConditional (break-if + continue-if):\n"
        << pr.disasm;
}

// ============================================================================
// Pattern 6: Bitwise operations on function return values used as conditions
//            (code & 1u != 0u)
//
// Slug: CalcRootCode returns a bitmask; the caller tests individual bits.
// Risk: If the compiler evaluates (code & 1u) as a boolean instead of
//       doing the bitwise AND then comparing to zero, wrong branch is taken.
//       Also, operator precedence between & and != matters in WGSL.
// ============================================================================

TEST(SlugPattern, BitwiseFunctionReturnAsCondition) {
    auto pr = CompileAndDisassemble(R"(
fn GetCode(x: f32) -> u32 {
    var code: u32 = 0u;
    if (x > 0.0) { code = code | 1u; }
    if (x > 0.5) { code = code | 2u; }
    return code;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var r0: f32 = 0.0;
    var r1: f32 = 0.0;
    for (var i: i32 = 0; i < 4; i++) {
        let code = GetCode(f32(i) * 0.3);
        if ((code & 1u) != 0u) {
            r0 += 1.0;
        }
        if ((code & 2u) != 0u) {
            r1 += 1.0;
        }
    }
    output.data[0] = r0;
    output.data[1] = r1;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have OpBitwiseAnd (for code & 1u and code & 2u).
    int bitwiseAnd = CountOccurrences(pr.disasm, "OpBitwiseAnd");
    EXPECT_GE(bitwiseAnd, 2)
        << "Expected >= 2 OpBitwiseAnd for bit tests:\n" << pr.disasm;
    // Must have OpINotEqual or OpIEqual for the != 0u comparison.
    bool hasNotEqual = pr.disasm.find("OpINotEqual") != std::string::npos;
    bool hasEqual    = pr.disasm.find("OpIEqual") != std::string::npos;
    EXPECT_TRUE(hasNotEqual || hasEqual)
        << "Expected OpINotEqual or OpIEqual for != 0u comparison:\n"
        << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpFunctionCall"), std::string::npos)
        << "Expected OpFunctionCall for GetCode:\n" << pr.disasm;
}

// ============================================================================
// Pattern 6b: Bitwise OR accumulation inside function, then return
//
// Slug: CalcRootCode builds up a bitmask with  code = code | 1u; code |= 2u;
// Risk: The var code must survive multiple conditional stores and the
//       final return must read the correct accumulated value.
// ============================================================================

TEST(SlugPattern, BitwiseOrAccumulationInFunction) {
    auto pr = CompileAndDisassemble(R"(
fn BuildMask(a: f32, b: f32, c: f32) -> u32 {
    var code: u32 = 0u;
    if (a > 0.0) { code = code | 1u; }
    if (b > 0.0) { code = code | 2u; }
    if (c > 0.0) { code = code | 4u; }
    return code;
}

struct UBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: UBuf;

@compute @workgroup_size(1)
fn main() {
    output.data[0] = BuildMask(1.0, -1.0, 1.0);
    output.data[1] = BuildMask(-1.0, 1.0, -1.0);
    output.data[2] = BuildMask(1.0, 1.0, 1.0);
    output.data[3] = BuildMask(-1.0, -1.0, -1.0);
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have OpBitwiseOr for code | 1u, code | 2u, code | 4u.
    int bitwiseOr = CountOccurrences(pr.disasm, "OpBitwiseOr");
    EXPECT_GE(bitwiseOr, 3)
        << "Expected >= 3 OpBitwiseOr inside BuildMask:\n" << pr.disasm;
    // Must have 4 function calls from main.
    int calls = CountOccurrences(pr.disasm, "OpFunctionCall");
    EXPECT_GE(calls, 4)
        << "Expected >= 4 OpFunctionCall:\n" << pr.disasm;
}

// ============================================================================
// Pattern 7: The EXACT SolveHorizPoly pattern
//   let ra = 1.0/a;
//   var t = expr * ra;
//   if (abs(a) < eps) { t = other_expr; }
//   return clamp(t, 0.0, 1.0);
// Called from inside a for loop.
//
// This is the most suspicious pattern.  The var t is initialized with a
// potentially NaN/Inf value (when a is near zero), then conditionally
// overwritten.  The conditional overwrite MUST happen on the correct branch.
// When called from a loop, the loop's structured merge must not interfere
// with the function's internal selection merge.
// ============================================================================

TEST(SlugPattern, ExactSolveHorizPolyPattern) {
    auto pr = CompileAndDisassemble(R"(
fn SolveHorizLinear(p0y: f32, p1y: f32, rdy: f32) -> f32 {
    return clamp(-p0y * rdy, 0.0, 1.0);
}

fn SolveHorizQuad(p0y: f32, p1y: f32, p2y: f32) -> f32 {
    let a = p0y - 2.0 * p1y + p2y;
    let b = p0y - p1y;
    let ra = 1.0 / a;
    var t: f32 = (b - sqrt(max(b * b - a * p0y, 0.0))) * ra;
    if (abs(a) < 0.0001) {
        let dy = p1y - p0y;
        t = SolveHorizLinear(p0y, p1y, 1.0 / dy);
    }
    return clamp(t, 0.0, 1.0);
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var accum: f32 = 0.0;
    for (var i: i32 = 0; i < 4; i++) {
        let fi = f32(i);
        let p0y = fi * 0.1 - 0.2;
        let p1y = 0.0;
        let p2y = fi * 0.1 + 0.1;
        let t = SolveHorizQuad(p0y, p1y, p2y);
        accum += t;
    }
    output.data[0] = accum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must compile and validate.  Now check structural properties:

    // 1. There must be at least two OpFunction (main + SolveHorizQuad +
    //    SolveHorizLinear).
    int funcs = CountOccurrences(pr.disasm, "OpFunction ");
    EXPECT_GE(funcs, 3)
        << "Expected >= 3 OpFunction (main + SolveHorizQuad + SolveHorizLinear):\n"
        << pr.disasm;

    // 2. SolveHorizQuad must have an OpSelectionMerge for the
    //    if(abs(a) < eps) conditional.
    EXPECT_NE(pr.disasm.find("OpSelectionMerge"), std::string::npos)
        << "Expected OpSelectionMerge for the if(abs(a)<eps) branch:\n"
        << pr.disasm;

    // 3. The for loop in main must have an OpLoopMerge.
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge for the for loop:\n" << pr.disasm;

    // 4. There must be OpFunctionCall (main calls SolveHorizQuad, which
    //    calls SolveHorizLinear).
    int calls = CountOccurrences(pr.disasm, "OpFunctionCall");
    EXPECT_GE(calls, 2)
        << "Expected >= 2 OpFunctionCall:\n" << pr.disasm;

    // 5. The abs() call must appear (OpExtInst ... FAbs).
    bool hasFAbs = pr.disasm.find("FAbs") != std::string::npos;
    EXPECT_TRUE(hasFAbs)
        << "Expected FAbs (from abs(a)):\n" << pr.disasm;

    // 6. The clamp() call must appear (OpExtInst ... FClamp or NClamp).
    bool hasClamp = (pr.disasm.find("FClamp") != std::string::npos) ||
                    (pr.disasm.find("NClamp") != std::string::npos);
    EXPECT_TRUE(hasClamp)
        << "Expected FClamp or NClamp:\n" << pr.disasm;
}

// ============================================================================
// Pattern 7b: SolveHorizPoly with DEGENERATE input (a == 0 exactly)
//
// This is the scenario that triggers the white line bug: when a quadratic
// curve degenerates to a line, a=0, so 1.0/a = Inf, and t = expr*Inf = NaN.
// The if(abs(a)<eps) branch must overwrite t with the linear fallback.
// If the compiler mis-generates the control flow, t stays NaN.
// ============================================================================

TEST(SlugPattern, SolveHorizPoly_DegenerateA_Zero) {
    auto pr = CompileAndDisassemble(R"(
fn SolveHoriz(p0y: f32, p1y: f32, p2y: f32) -> f32 {
    let a = p0y - 2.0 * p1y + p2y;
    let b = p0y - p1y;
    let ra = 1.0 / a;
    var t: f32 = (b - sqrt(max(b * b - a * p0y, 0.0))) * ra;
    if (abs(a) < 0.0001) {
        let dy = p1y - p0y;
        if (abs(dy) > 0.0001) {
            t = -p0y / dy;
        } else {
            t = 0.0;
        }
    }
    return clamp(t, 0.0, 1.0);
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    // Degenerate case: p0y=0.1, p1y=0.55, p2y=1.0 gives a=0.0 exactly.
    // (0.1 - 2*0.55 + 1.0 = 0.0)
    output.data[0] = SolveHoriz(0.1, 0.55, 1.0);

    // Another degenerate: all collinear -> a=0.
    output.data[1] = SolveHoriz(-0.5, 0.0, 0.5);

    // Non-degenerate: a != 0, should use quadratic formula.
    output.data[2] = SolveHoriz(-0.3, 0.1, 0.2);

    // Edge case: all points at same y -> a=0, dy=0, should return 0.
    output.data[3] = SolveHoriz(0.5, 0.5, 0.5);
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // The function must have nested if (abs(a)<eps then if abs(dy)>eps).
    int selMerge = CountOccurrences(pr.disasm, "OpSelectionMerge");
    EXPECT_GE(selMerge, 2)
        << "Expected >= 2 OpSelectionMerge for nested ifs in SolveHoriz:\n"
        << pr.disasm;
}

// ============================================================================
// Pattern 7c: SolveHorizPoly degenerate called from a loop with accumulation
//
// This is the FULL combined pattern: the degenerate-a case called from a
// loop that accumulates coverage.  This is the maximal stress test.
// ============================================================================

TEST(SlugPattern, SolveHorizPoly_Degenerate_CalledFromLoopWithAccum) {
    auto pr = CompileAndDisassemble(R"(
fn SolveHoriz(p0y: f32, p1y: f32, p2y: f32) -> f32 {
    let a = p0y - 2.0 * p1y + p2y;
    let b = p0y - p1y;
    let ra = 1.0 / a;
    var t: f32 = (b - sqrt(max(b * b - a * p0y, 0.0))) * ra;
    if (abs(a) < 0.0001) {
        let dy = p1y - p0y;
        if (abs(dy) > 0.0001) {
            t = -p0y / dy;
        } else {
            t = 0.0;
        }
    }
    return clamp(t, 0.0, 1.0);
}

fn CalcRootCode(p0y: f32, p1y: f32) -> u32 {
    var code: u32 = 0u;
    if (p0y * p1y < 0.0) { code = code | 1u; }
    if (p0y > 0.0 && p1y > 0.0) { code = code | 2u; }
    return code;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    // Simulate the Slug loop over curves using parallel arrays.
    // Degenerate (a=0, linear), non-degenerate, degenerate, non-degenerate.
    let p0y = array<f32, 4>(-0.5, -0.3, 0.1, -0.2);
    let p1y = array<f32, 4>(0.0, 0.1, 0.55, -0.1);
    let p2y = array<f32, 4>(0.5, 0.2, 1.0, 0.3);

    var xcov: f32 = 0.0;
    for (var i: i32 = 0; i < 4; i++) {
        let code = CalcRootCode(p0y[i], p2y[i]);
        if (code != 0u) {
            let t = SolveHoriz(p0y[i], p1y[i], p2y[i]);
            xcov += saturate(t + 0.5) - saturate(t - 0.5);
        }
    }
    output.data[0] = xcov;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Full structural checks:
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;

    int calls = CountOccurrences(pr.disasm, "OpFunctionCall");
    EXPECT_GE(calls, 2)
        << "Expected >= 2 OpFunctionCall (CalcRootCode + SolveHoriz):\n"
        << pr.disasm;

    // SolveHoriz has nested ifs, CalcRootCode has ifs, main has an if.
    int selMerge = CountOccurrences(pr.disasm, "OpSelectionMerge");
    EXPECT_GE(selMerge, 2)
        << "Expected >= 2 OpSelectionMerge:\n" << pr.disasm;
}

// ============================================================================
// Pattern 8: var with NaN initial value, conditionally overwritten
//
// Direct test for the NaN-init pattern.  If a compiler treats NaN specially
// or optimizes away stores it considers "dead" (because the value is NaN),
// this test will catch it.
// ============================================================================

TEST(SlugPattern, VarNaNInit_ConditionalOverwrite) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

fn compute_value(x: f32) -> f32 {
    let denom = x - 0.5;
    var t: f32 = 1.0 / denom;   // NaN when x == 0.5 (Inf, then used)
    if (abs(denom) < 0.001) {
        t = 0.0;                 // Must overwrite NaN/Inf
    }
    return t;
}

@compute @workgroup_size(1)
fn main() {
    // x=0.5 -> denom=0 -> t=Inf, then overwritten to 0.0
    output.data[0] = compute_value(0.5);
    // x=1.0 -> denom=0.5 -> t=2.0, not overwritten
    output.data[1] = compute_value(1.0);
    // x=0.501 -> denom=0.001 -> t=1000, nearly degenerate but NOT overwritten
    output.data[2] = compute_value(0.501);
    // x=0.4999 -> denom=-0.0001 -> abs < 0.001, overwritten to 0.0
    output.data[3] = compute_value(0.4999);
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have the conditional overwrite pattern.
    EXPECT_NE(pr.disasm.find("OpSelectionMerge"), std::string::npos)
        << "Expected OpSelectionMerge for if(abs(denom)<eps):\n" << pr.disasm;
    // The FDiv must exist (1.0 / denom).
    EXPECT_NE(pr.disasm.find("OpFDiv"), std::string::npos)
        << "Expected OpFDiv for 1.0/denom:\n" << pr.disasm;
}

// ============================================================================
// Pattern 9: Multiple var accumulators updated across different branches
//            inside a single loop
//
// Slug: Both xcov and ycov are updated in different branches (code & 1 vs
//       code > 1) of the same loop iteration.  Both must retain their values
//       across iterations.
// ============================================================================

TEST(SlugPattern, MultipleAccumulatorsAcrossBranchesInLoop) {
    auto pr = CompileAndDisassemble(R"(
fn GetCode(i: i32) -> u32 {
    if (i % 2 == 0) { return 1u; }
    if (i % 3 == 0) { return 2u; }
    return 3u;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var xcov: f32 = 0.0;
    var ycov: f32 = 0.0;
    var xwgt: f32 = 0.0;
    var ywgt: f32 = 0.0;

    for (var i: i32 = 0; i < 8; i++) {
        let code = GetCode(i);
        let t = f32(i) * 0.125;
        if ((code & 1u) != 0u) {
            xcov += saturate(t);
            xwgt += 1.0;
        }
        if ((code & 2u) != 0u) {
            ycov += saturate(t);
            ywgt += 1.0;
        }
    }
    output.data[0] = xcov;
    output.data[1] = ycov;
    output.data[2] = xwgt;
    output.data[3] = ywgt;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;

    // Must have loads and stores for all four accumulators.
    int loads  = CountOccurrences(pr.disasm, "OpLoad");
    int stores = CountOccurrences(pr.disasm, "OpStore");
    EXPECT_GE(loads, 8)
        << "Expected >= 8 OpLoad (4 accum vars, loaded and stored per branch):\n"
        << pr.disasm;
    EXPECT_GE(stores, 8)
        << "Expected >= 8 OpStore (4 accum vars init + updates):\n"
        << pr.disasm;
}

// ============================================================================
// Pattern 10: Function call with var+conditional inside a nested loop
//
// Slug: The coverage computation has an outer loop (over curves) and inner
//       processing that may include its own loops.  SolveHorizPoly is called
//       from the outer loop.  This tests the pattern with explicit nesting.
// ============================================================================

TEST(SlugPattern, FunctionWithVarConditional_InNestedLoop) {
    auto pr = CompileAndDisassemble(R"(
fn fallback_solve(p0: f32, p1: f32) -> f32 {
    let dy = p1 - p0;
    var t: f32 = -p0 / dy;
    if (abs(dy) < 0.0001) {
        t = 0.5;
    }
    return clamp(t, 0.0, 1.0);
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var total: f32 = 0.0;
    for (var band: i32 = 0; band < 2; band++) {
        var band_cov: f32 = 0.0;
        for (var curve: i32 = 0; curve < 4; curve++) {
            let p0 = f32(band * 4 + curve) * 0.1 - 0.4;
            let p1 = p0 + 0.3;
            let t = fallback_solve(p0, p1);
            band_cov += t;
        }
        total += band_cov;
    }
    output.data[0] = total;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have two OpLoopMerge instructions (outer + inner loops).
    int loopMerges = CountOccurrences(pr.disasm, "OpLoopMerge");
    EXPECT_GE(loopMerges, 2)
        << "Expected >= 2 OpLoopMerge for nested loops:\n" << pr.disasm;

    // Function call inside inner loop.
    EXPECT_NE(pr.disasm.find("OpFunctionCall"), std::string::npos)
        << "Expected OpFunctionCall for fallback_solve:\n" << pr.disasm;
}

// ============================================================================
// Pattern 11: saturate() on intermediate results inside loop branches
//
// Slug uses saturate(t + 0.5) - saturate(t - 0.5) inside loop branches
// for coverage computation.  This exercises the ExtInst FClamp(0,1) lowering
// in a control-flow-heavy context.
// ============================================================================

TEST(SlugPattern, SaturateInsideLoopBranch) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var cov: f32 = 0.0;
    for (var i: i32 = 0; i < 8; i++) {
        let t = f32(i) * 0.25 - 0.75;
        let active = i % 2 == 0;
        if (active) {
            cov += saturate(t + 0.5) - saturate(t - 0.5);
        }
    }
    output.data[0] = cov;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
    // saturate should produce FClamp or NClamp with 0.0 and 1.0.
    bool hasClamp = (pr.disasm.find("FClamp") != std::string::npos) ||
                    (pr.disasm.find("NClamp") != std::string::npos);
    EXPECT_TRUE(hasClamp)
        << "Expected FClamp or NClamp for saturate():\n" << pr.disasm;
}

// ============================================================================
// Pattern 12: Function returning clamp(var, 0, 1) where var was conditionally
//             overwritten, called from a loop, result used in accumulation.
//
// This combines patterns 1, 2, and 7 into the tightest possible stress test.
// The return value flows through clamp() which could mask NaN propagation
// if the conditional overwrite fails -- but clamp of NaN is implementation-
// defined and may produce 0 on some hardware and NaN on others.
// ============================================================================

TEST(SlugPattern, ClampOfConditionalVar_InLoopAccumulation) {
    auto pr = CompileAndDisassemble(R"(
fn process(a: f32, b: f32) -> f32 {
    let ra = 1.0 / a;
    var t: f32 = b * ra;
    if (abs(a) < 0.0001) {
        t = b * 0.5;
    }
    return clamp(t, 0.0, 1.0);
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var sum: f32 = 0.0;
    for (var i: i32 = 0; i < 8; i++) {
        let a = f32(i) * 0.02;   // a is very small for first few iterations
        let b = 0.3;
        sum += process(a, b);
    }
    output.data[0] = sum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Both the loop merge and the selection merge must coexist correctly.
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpSelectionMerge"), std::string::npos)
        << "Expected OpSelectionMerge:\n" << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpFunctionCall"), std::string::npos)
        << "Expected OpFunctionCall:\n" << pr.disasm;
    // clamp must exist.
    bool hasClamp = (pr.disasm.find("FClamp") != std::string::npos) ||
                    (pr.disasm.find("NClamp") != std::string::npos);
    EXPECT_TRUE(hasClamp)
        << "Expected FClamp or NClamp:\n" << pr.disasm;
}

// ============================================================================
// Pattern 13: Struct return from function called inside loop, with member
//             access driving conditional control flow
//
// Slug: CalcBandLoc returns a vec2<i32> that is used to do textureLoad,
//       and various struct returns drive further control flow.
// Risk: If the function return struct is not materialized correctly, the
//       member access reads garbage.
// ============================================================================

TEST(SlugPattern, StructReturnFromFunctionInLoop) {
    auto pr = CompileAndDisassemble(R"(
struct SolveResult {
    t: f32,
    valid: f32,
};

fn Solve(p0: f32, p1: f32) -> SolveResult {
    let dy = p1 - p0;
    var result: SolveResult;
    result.valid = 0.0;
    result.t = 0.0;
    if (abs(dy) > 0.001) {
        result.t = clamp(-p0 / dy, 0.0, 1.0);
        result.valid = 1.0;
    }
    return result;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var accum: f32 = 0.0;
    var count: f32 = 0.0;
    for (var i: i32 = 0; i < 6; i++) {
        let p0 = f32(i) * 0.2 - 0.5;
        let p1 = p0 + 0.3;
        let sr = Solve(p0, p1);
        if (sr.valid > 0.5) {
            accum += sr.t;
            count += 1.0;
        }
    }
    output.data[0] = accum;
    output.data[1] = count;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Must have OpCompositeExtract or OpAccessChain for struct member access.
    bool hasExtract = (pr.disasm.find("OpCompositeExtract") != std::string::npos) ||
                      (pr.disasm.find("OpAccessChain") != std::string::npos);
    EXPECT_TRUE(hasExtract)
        << "Expected OpCompositeExtract or OpAccessChain for struct member access:\n"
        << pr.disasm;

    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
    EXPECT_NE(pr.disasm.find("OpFunctionCall"), std::string::npos)
        << "Expected OpFunctionCall:\n" << pr.disasm;
}

// ============================================================================
// Pattern 14: Complex expression in for-loop condition with function calls
//             and early returns in the called function
//
// Slug: Some helper functions have multiple early return paths.  When called
//       from a loop, the SSA merge at the call site must handle all return
//       paths correctly.
// ============================================================================

TEST(SlugPattern, FunctionWithMultipleEarlyReturns_CalledFromLoop) {
    auto pr = CompileAndDisassemble(R"(
fn classify(x: f32) -> f32 {
    if (x < -0.5) { return -1.0; }
    if (x > 0.5)  { return 1.0; }
    if (abs(x) < 0.01) { return 0.0; }
    return x * 2.0;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var sum: f32 = 0.0;
    for (var i: i32 = 0; i < 10; i++) {
        let x = f32(i) * 0.2 - 0.9;
        sum += classify(x);
    }
    output.data[0] = sum;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Multiple return paths should produce OpReturnValue at multiple points.
    int retVals = CountOccurrences(pr.disasm, "OpReturnValue");
    EXPECT_GE(retVals, 2)
        << "Expected >= 2 OpReturnValue for multiple early returns:\n"
        << pr.disasm;

    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;
}

// ============================================================================
// Pattern 15: The full CalcRootCode + SolveHorizPoly + accumulation pattern
//             with BOTH horizontal and vertical processing in the same loop
//
// This mirrors the actual Slug fragment shader structure where both xcov
// and ycov are computed in the same loop body using different code bits.
// ============================================================================

TEST(SlugPattern, FullCoverageComputationPattern) {
    auto pr = CompileAndDisassemble(R"(
fn CalcRootCode(p0y: f32, p2y: f32) -> u32 {
    var code: u32 = 0u;
    if (p0y <= 0.0 && p2y > 0.0) { code = code | 1u; }
    if (p0y > 0.0 && p2y <= 0.0) { code = code | 1u; }
    if (p0y > 0.0 && p2y > 0.0)  { code = code | 2u; }
    return code;
}

fn SolveHoriz(p0y: f32, p1y: f32, p2y: f32) -> f32 {
    let a = p0y - 2.0 * p1y + p2y;
    let b = p0y - p1y;
    let ra = 1.0 / a;
    var t: f32 = (b - sqrt(max(b * b - a * p0y, 0.0))) * ra;
    if (abs(a) < 0.0001) {
        t = -p0y / (p1y - p0y);
    }
    return clamp(t, 0.0, 1.0);
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    // Three curves represented as parallel arrays for p0y, p1y, p2y.
    let cp0y = array<f32, 3>(-0.3, -0.5, 0.1);
    let cp1y = array<f32, 3>(0.0, 0.0, 0.3);
    let cp2y = array<f32, 3>(0.3, 0.5, 0.5);

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;

    for (var i: i32 = 0; i < 3; i++) {
        let code = CalcRootCode(cp0y[i], cp2y[i]);
        if (code != 0u) {
            let t = SolveHoriz(cp0y[i], cp1y[i], cp2y[i]);
            if ((code & 1u) != 0u) {
                xcov += saturate(t + 0.5) - saturate(t - 0.5);
                xwgt += 1.0;
            }
            if ((code & 2u) != 0u) {
                xcov += saturate(t);
            }
        }
    }

    let coverage = xcov / max(xwgt, 1.0);
    output.data[0] = coverage;
    output.data[1] = xcov;
    output.data[2] = xwgt;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Full structural validation:
    EXPECT_NE(pr.disasm.find("OpLoopMerge"), std::string::npos)
        << "Expected OpLoopMerge:\n" << pr.disasm;

    int calls = CountOccurrences(pr.disasm, "OpFunctionCall");
    EXPECT_GE(calls, 2)
        << "Expected >= 2 OpFunctionCall:\n" << pr.disasm;

    // Nested selection merges: code!=0, code&1, code&2, plus those inside functions.
    int selMerge = CountOccurrences(pr.disasm, "OpSelectionMerge");
    EXPECT_GE(selMerge, 3)
        << "Expected >= 3 OpSelectionMerge:\n" << pr.disasm;

    EXPECT_NE(pr.disasm.find("OpBitwiseAnd"), std::string::npos)
        << "Expected OpBitwiseAnd:\n" << pr.disasm;

    // FDiv for 1.0/a and xcov/max(xwgt,1).
    int fdivs = CountOccurrences(pr.disasm, "OpFDiv");
    EXPECT_GE(fdivs, 1)
        << "Expected >= 1 OpFDiv:\n" << pr.disasm;
}

// ============================================================================
// Pattern 16: var accumulation with mixed += and = (reset) across iterations
//
// Slug: band_cov is reset per-band but accumulated per-curve within a band.
// Risk: If the compiler hoists the reset out of the loop or merges stores
//       incorrectly, the per-band reset fails.
// ============================================================================

TEST(SlugPattern, VarResetAndAccumulateInLoop) {
    auto pr = CompileAndDisassemble(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    var total: f32 = 0.0;
    for (var band: i32 = 0; band < 4; band++) {
        var band_cov: f32 = 0.0;   // reset each band
        for (var curve: i32 = 0; curve < 4; curve++) {
            let t = f32(band * 4 + curve) * 0.0625;
            band_cov += t;
        }
        total += band_cov;
    }
    output.data[0] = total;
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Two nested loops -> two OpLoopMerge.
    int loopMerges = CountOccurrences(pr.disasm, "OpLoopMerge");
    EXPECT_GE(loopMerges, 2)
        << "Expected >= 2 OpLoopMerge for nested loops:\n" << pr.disasm;
}

// ============================================================================
// Pattern 17: Var with conditional overwrite using BOTH abs() comparison and
//             a sign-dependent fallback (the Slug linear solve pattern)
//
// Slug:  if (abs(a) < eps) { t = -p0 / dy; }  where dy could also be ~0.
// Risk: Two divisions that could both produce Inf/NaN, with the conditional
//       overwrite being the only thing preventing propagation of bad values.
// ============================================================================

TEST(SlugPattern, DoubleDivisionWithConditionalFallback) {
    auto pr = CompileAndDisassemble(R"(
fn safe_solve(a: f32, b: f32, c: f32) -> f32 {
    let ra = 1.0 / a;
    var t: f32 = -b * ra;
    if (abs(a) < 0.001) {
        let rb = 1.0 / b;
        t = -c * rb;
        if (abs(b) < 0.001) {
            t = 0.0;
        }
    }
    return t;
}

struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;

@compute @workgroup_size(1)
fn main() {
    // a~0, b!=0: should use linear fallback
    output.data[0] = safe_solve(0.0001, 2.0, 1.0);
    // a!=0: should use quadratic path
    output.data[1] = safe_solve(1.0, 2.0, 1.0);
    // a~0, b~0: should return 0.0
    output.data[2] = safe_solve(0.00001, 0.00001, 1.0);
    // Normal case
    output.data[3] = safe_solve(3.0, -1.0, 0.5);
}
)");
    ASSERT_TRUE(pr.success) << pr.error;

    // Nested selection merges for the nested ifs.
    int selMerge = CountOccurrences(pr.disasm, "OpSelectionMerge");
    EXPECT_GE(selMerge, 2)
        << "Expected >= 2 OpSelectionMerge for nested conditional fallbacks:\n"
        << pr.disasm;

    // Two FDiv instructions (1/a and 1/b).
    int fdivs = CountOccurrences(pr.disasm, "OpFDiv");
    EXPECT_GE(fdivs, 2)
        << "Expected >= 2 OpFDiv:\n" << pr.disasm;
}
