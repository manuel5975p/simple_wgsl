#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// Atomic builtin functions
// atomicLoad, atomicStore, atomicAdd, atomicSub, atomicMax, atomicMin,
// atomicAnd, atomicOr, atomicXor, atomicExchange, atomicCompareExchangeWeak
// =============================================================================

// -----------------------------------------------------------------------------
// atomicLoad(ptr) - loads the value from an atomic variable
// SPIR-V: OpAtomicLoad
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicLoad_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let val = atomicLoad(&buf.counter);
    output.data[0] = val;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicLoad_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let val = atomicLoad(&buf.counter);
    output.data[0] = val;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicLoad_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_counter: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main() {
    let val = atomicLoad(&shared_counter);
    output.data[0] = val;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicStore(ptr, val) - stores a value into an atomic variable
// SPIR-V: OpAtomicStore
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicStore_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
@compute @workgroup_size(1) fn main() {
    atomicStore(&buf.counter, 42u);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicStore_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
@compute @workgroup_size(1) fn main() {
    atomicStore(&buf.counter, -7i);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicStore_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_val: atomic<u32>;
@compute @workgroup_size(64) fn main() {
    atomicStore(&shared_val, 100u);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicAdd(ptr, val) - atomically adds val, returns old value
// SPIR-V: OpAtomicIAdd
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicAdd_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicAdd(&buf.counter, 5u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicAdd_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicAdd(&buf.counter, 10i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicAdd_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_counter: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicAdd(&shared_counter, 1u);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicSub(ptr, val) - atomically subtracts val, returns old value
// SPIR-V: OpAtomicISub
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicSub_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicSub(&buf.counter, 3u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicSub_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicSub(&buf.counter, 2i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicSub_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_counter: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicSub(&shared_counter, 1u);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicMax(ptr, val) - atomically computes max, returns old value
// SPIR-V: OpAtomicUMax (unsigned), OpAtomicSMax (signed)
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicMax_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicMax(&buf.counter, 100u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicMax_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicMax(&buf.counter, -5i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicMax_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_max: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicMax(&shared_max, idx);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicMin(ptr, val) - atomically computes min, returns old value
// SPIR-V: OpAtomicUMin (unsigned), OpAtomicSMin (signed)
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicMin_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicMin(&buf.counter, 10u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicMin_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicMin(&buf.counter, 3i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicMin_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_min: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicMin(&shared_min, idx);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicAnd(ptr, val) - atomically computes bitwise AND, returns old value
// SPIR-V: OpAtomicAnd
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicAnd_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { flags: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicAnd(&buf.flags, 0xFF00FF00u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicAnd_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { flags: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicAnd(&buf.flags, 0x0Fi);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicAnd_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_flags: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicAnd(&shared_flags, 0xFFFFFFFEu);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicOr(ptr, val) - atomically computes bitwise OR, returns old value
// SPIR-V: OpAtomicOr
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicOr_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { flags: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicOr(&buf.flags, 0x0000FF00u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicOr_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { flags: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicOr(&buf.flags, 0xF0i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicOr_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_flags: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let bit = 1u << idx;
    let old = atomicOr(&shared_flags, bit);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicXor(ptr, val) - atomically computes bitwise XOR, returns old value
// SPIR-V: OpAtomicXor
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicXor_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { flags: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicXor(&buf.flags, 0xFFFFFFFFu);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicXor_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { flags: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicXor(&buf.flags, -1i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicXor_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_flags: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicXor(&shared_flags, 1u);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicExchange(ptr, val) - atomically replaces value, returns old value
// SPIR-V: OpAtomicExchange
// Works on atomic<u32>, atomic<i32>
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicExchange_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicExchange(&buf.counter, 999u);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicExchange_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let old = atomicExchange(&buf.counter, -42i);
    output.data[0] = old;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicExchange_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_lock: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let old = atomicExchange(&shared_lock, 1u);
    if idx == 0u {
        output.data[0] = old;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// atomicCompareExchangeWeak(ptr, cmp, val)
// Atomically compares *ptr with cmp; if equal, stores val.
// Returns __atomic_compare_exchange_result_<type> with members:
//   old_value: the value that was in *ptr
//   exchanged: bool, whether the exchange happened
// SPIR-V: OpAtomicCompareExchange
// Works on atomic<u32>, atomic<i32>
// NOTE: Only 2 tests -- the return type is a struct which adds complexity
// -----------------------------------------------------------------------------

TEST(AtomicTest, AtomicCompareExchangeWeak_StorageU32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<u32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let result = atomicCompareExchangeWeak(&buf.counter, 0u, 1u);
    output.data[0] = result.old_value;
    if result.exchanged {
        output.data[1] = 1u;
    } else {
        output.data[1] = 0u;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicCompareExchangeWeak_StorageI32) {
    auto r = wgsl_test::CompileWgsl(R"(
struct AtomicBuf { counter: atomic<i32> };
@group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
struct OutBuf { data: array<i32> };
@group(0) @binding(1) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(1) fn main() {
    let result = atomicCompareExchangeWeak(&buf.counter, 0i, 42i);
    output.data[0] = result.old_value;
    if result.exchanged {
        output.data[1] = 1i;
    } else {
        output.data[1] = 0i;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AtomicTest, AtomicCompareExchangeWeak_Workgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_lock: atomic<u32>;
struct OutBuf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: OutBuf;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) idx: u32) {
    let result = atomicCompareExchangeWeak(&shared_lock, 0u, idx);
    if result.exchanged {
        output.data[0] = idx;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}
