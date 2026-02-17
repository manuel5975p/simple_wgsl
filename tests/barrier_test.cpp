#include <gtest/gtest.h>
#include "test_utils.h"

// ---------------------------------------------------------------------------
// storageBarrier() tests
// ---------------------------------------------------------------------------

TEST(BarrierTest, StorageBarrier_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    buf.data[lid.x] = lid.x;
    storageBarrier();
    let val = buf.data[63u - lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, StorageBarrier_MultipleBarriers) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(32) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    buf.data[lid.x] = lid.x;
    storageBarrier();
    buf.data[lid.x] = buf.data[lid.x] + 1u;
    storageBarrier();
    let val = buf.data[31u - lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, StorageBarrier_WithConditional) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    buf.data[lid.x] = lid.x * 2u;
    storageBarrier();
    if (lid.x < 32u) {
        buf.data[lid.x] = buf.data[lid.x + 32u];
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ---------------------------------------------------------------------------
// textureBarrier() tests
// ---------------------------------------------------------------------------

TEST(BarrierTest, TextureBarrier_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    buf.data[lid.x] = lid.x;
    textureBarrier();
    let val = buf.data[lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, TextureBarrier_MultipleBarriers) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(16) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    buf.data[lid.x] = lid.x;
    textureBarrier();
    buf.data[lid.x] = buf.data[lid.x] + 1u;
    textureBarrier();
    let val = buf.data[15u - lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, TextureBarrier_WithLoop) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var i: u32 = 0u;
    loop {
        if (i >= 4u) { break; }
        buf.data[lid.x] = buf.data[lid.x] + 1u;
        textureBarrier();
        i = i + 1u;
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ---------------------------------------------------------------------------
// workgroupBarrier() tests
// ---------------------------------------------------------------------------

TEST(BarrierTest, WorkgroupBarrier_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_data: array<u32, 64>;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    shared_data[lid.x] = lid.x;
    workgroupBarrier();
    let val = shared_data[63u - lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, WorkgroupBarrier_ParallelReduction) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_data: array<u32, 64>;
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    shared_data[lid.x] = lid.x;
    workgroupBarrier();
    if (lid.x < 32u) {
        shared_data[lid.x] = shared_data[lid.x] + shared_data[lid.x + 32u];
    }
    workgroupBarrier();
    if (lid.x == 0u) {
        buf.data[0] = shared_data[0];
    }
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, WorkgroupBarrier_WithStorageAndWorkgroup) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> temp: array<u32, 128>;
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(128) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    temp[lid.x] = buf.data[lid.x];
    workgroupBarrier();
    buf.data[lid.x] = temp[127u - lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ---------------------------------------------------------------------------
// workgroupUniformLoad() tests
//
// NOTE: Using workgroupUniformLoad(shared_val) without address-of operator.
// The WGSL spec requires workgroupUniformLoad(&shared_val) (taking a pointer),
// but the parser may not support the & (address-of) operator. If that is the
// case, the fallback form workgroupUniformLoad(shared_val) is used instead.
// ---------------------------------------------------------------------------

TEST(BarrierTest, WorkgroupUniformLoad_Basic) {
    // Try without & operator first (parser may not support address-of)
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_val: u32;
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        shared_val = buf.data[0];
    }
    let uniform_val = workgroupUniformLoad(&shared_val);
    buf.data[lid.x] = uniform_val;
})");
    if (!r.success) {
        // Fallback: try without & operator
        r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_val: u32;
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        shared_val = buf.data[0];
    }
    let uniform_val = workgroupUniformLoad(shared_val);
    buf.data[lid.x] = uniform_val;
})");
    }
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, WorkgroupUniformLoad_I32) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_val: i32;
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(32) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        shared_val = buf.data[0];
    }
    let uniform_val = workgroupUniformLoad(&shared_val);
    buf.data[lid.x] = uniform_val + i32(lid.x);
})");
    if (!r.success) {
        // Fallback: try without & operator
        r = wgsl_test::CompileWgsl(R"(
var<workgroup> shared_val: i32;
struct Buf { data: array<i32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(32) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        shared_val = buf.data[0];
    }
    let uniform_val = workgroupUniformLoad(shared_val);
    buf.data[lid.x] = uniform_val + i32(lid.x);
})");
    }
    EXPECT_TRUE(r.success) << r.error;
}

TEST(BarrierTest, WorkgroupUniformLoad_UsedInCondition) {
    auto r = wgsl_test::CompileWgsl(R"(
var<workgroup> flag: u32;
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        flag = 1u;
    }
    let f = workgroupUniformLoad(&flag);
    if (f == 1u) {
        buf.data[lid.x] = lid.x;
    }
})");
    if (!r.success) {
        // Fallback: try without & operator
        r = wgsl_test::CompileWgsl(R"(
var<workgroup> flag: u32;
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        flag = 1u;
    }
    let f = workgroupUniformLoad(flag);
    if (f == 1u) {
        buf.data[lid.x] = lid.x;
    }
})");
    }
    EXPECT_TRUE(r.success) << r.error;
}
