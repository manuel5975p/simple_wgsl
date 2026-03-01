/*
 * cuvk_graph_test.cpp - CUDA Graph API tests
 *
 * Tests graph creation, node addition, dependency management, instantiation,
 * execution, and update. Runs against cuvk_runtime_static (Vulkan backend).
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
extern "C" {
#include "cuda.h"
}

/* PTX kernel: vecAdd(A, B, C) — one element per block, reqntid 1,1,1 */
static const char *VECTOR_ADD_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vecAdd(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<7>;\n"
    "    .reg .f32 %f<3>;\n"
    "\n"
    "    ld.param.u64 %rd0, [A];\n"
    "    ld.param.u64 %rd1, [B];\n"
    "    ld.param.u64 %rd2, [C];\n"
    "\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd3, %r0;\n"
    "    mul.lo.u64 %rd3, %rd3, 4;\n"
    "\n"
    "    add.u64 %rd4, %rd0, %rd3;\n"
    "    ld.global.f32 %f0, [%rd4];\n"
    "\n"
    "    add.u64 %rd5, %rd1, %rd3;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "\n"
    "    add.f32 %f2, %f0, %f1;\n"
    "    add.u64 %rd6, %rd2, %rd3;\n"
    "    st.global.f32 [%rd6], %f2;\n"
    "\n"
    "    ret;\n"
    "}\n";

/* PTX kernel: scaleVec(data, scalar) — multiplies each element by scalar */
static const char *SCALE_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry scaleVec(\n"
    "    .param .u64 data,\n"
    "    .param .f32 scalar\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<3>;\n"
    "    .reg .f32 %f<3>;\n"
    "\n"
    "    ld.param.u64 %rd0, [data];\n"
    "    ld.param.f32 %f0, [scalar];\n"
    "\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd1, %r0;\n"
    "    mul.lo.u64 %rd1, %rd1, 4;\n"
    "    add.u64 %rd2, %rd0, %rd1;\n"
    "\n"
    "    ld.global.f32 %f1, [%rd2];\n"
    "    mul.f32 %f2, %f1, %f0;\n"
    "    st.global.f32 [%rd2], %f2;\n"
    "\n"
    "    ret;\n"
    "}\n";

class CuvkGraphTest : public ::testing::Test {
protected:
    CUcontext ctx = NULL;
    void SetUp() override {
        ASSERT_EQ(CUDA_SUCCESS, cuInit(0));
        CUdevice dev;
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
        ASSERT_EQ(CUDA_SUCCESS, cuCtxCreate(&ctx, NULL, 0, dev));
    }
    void TearDown() override {
        if (ctx) cuCtxDestroy(ctx);
    }
};

/* ============================================================================
 * Graph lifecycle
 * ============================================================================ */

TEST_F(CuvkGraphTest, CreateDestroy) {
    CUgraph graph = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));
    EXPECT_NE(nullptr, graph);
    EXPECT_EQ(CUDA_SUCCESS, cuGraphDestroy(graph));
}

TEST_F(CuvkGraphTest, CreateNullPtr) {
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuGraphCreate(NULL, 0));
}

TEST_F(CuvkGraphTest, DestroyNull) {
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuGraphDestroy(NULL));
}

/* ============================================================================
 * Empty node
 * ============================================================================ */

TEST_F(CuvkGraphTest, AddEmptyNode) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode node = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphAddEmptyNode(&node, graph, NULL, 0));
    EXPECT_NE(nullptr, node);

    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, AddEmptyNodeWithDeps) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a = NULL, b = NULL;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddEmptyNode(&a, graph, NULL, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddEmptyNode(&b, graph, &a, 1));

    cuGraphDestroy(graph);
}

/* ============================================================================
 * Node type query
 * ============================================================================ */

TEST_F(CuvkGraphTest, NodeGetTypeEmpty) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddEmptyNode(&node, graph, NULL, 0));

    CUgraphNodeType type;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphNodeGetType(node, &type));
    EXPECT_EQ(CU_GRAPH_NODE_TYPE_EMPTY, type);

    cuGraphDestroy(graph);
}

/* ============================================================================
 * Graph query: GetNodes, GetEdges
 * ============================================================================ */

TEST_F(CuvkGraphTest, GetNodesEmpty) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    size_t numNodes = 99;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphGetNodes(graph, NULL, &numNodes));
    EXPECT_EQ(0u, numNodes);

    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, GetNodesCount) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b, c;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, NULL, 0);
    cuGraphAddEmptyNode(&c, graph, &a, 1);

    size_t numNodes = 0;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphGetNodes(graph, NULL, &numNodes));
    EXPECT_EQ(3u, numNodes);

    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, GetNodesList) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, &a, 1);

    size_t numNodes = 2;
    CUgraphNode nodes[2] = {};
    EXPECT_EQ(CUDA_SUCCESS, cuGraphGetNodes(graph, nodes, &numNodes));
    EXPECT_EQ(2u, numNodes);
    EXPECT_NE(nullptr, nodes[0]);
    EXPECT_NE(nullptr, nodes[1]);

    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, GetEdgesCount) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b, c;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, &a, 1);
    cuGraphAddEmptyNode(&c, graph, &a, 1);

    size_t numEdges = 0;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges));
    EXPECT_EQ(2u, numEdges);

    cuGraphDestroy(graph);
}

/* ============================================================================
 * Dependency management
 * ============================================================================ */

TEST_F(CuvkGraphTest, AddDependencies) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, NULL, 0);

    /* Initially 0 edges */
    size_t numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ(0u, numEdges);

    /* Add edge a -> b */
    EXPECT_EQ(CUDA_SUCCESS, cuGraphAddDependencies(graph, &a, &b, NULL, 1));

    numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ(1u, numEdges);

    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, RemoveDependencies) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, &a, 1);

    size_t numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ(1u, numEdges);

    EXPECT_EQ(CUDA_SUCCESS, cuGraphRemoveDependencies(graph, &a, &b, NULL, 1));

    numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ(0u, numEdges);

    cuGraphDestroy(graph);
}

/* ============================================================================
 * Instantiation
 * ============================================================================ */

TEST_F(CuvkGraphTest, InstantiateEmptyGraph) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphExec exec = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    EXPECT_NE(nullptr, exec);

    EXPECT_EQ(CUDA_SUCCESS, cuGraphExecDestroy(exec));
    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, InstantiateWithEmptyNodes) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b, c;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, &a, 1);
    cuGraphAddEmptyNode(&c, graph, &b, 1);

    CUgraphExec exec = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    EXPECT_NE(nullptr, exec);

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, InstantiateNullArgs) {
    CUgraph graph;
    cuGraphCreate(&graph, 0);

    EXPECT_NE(CUDA_SUCCESS, cuGraphInstantiate(NULL, graph, 0));
    EXPECT_NE(CUDA_SUCCESS, cuGraphInstantiate(NULL, NULL, 0));

    cuGraphDestroy(graph);
}

/* ============================================================================
 * Launch empty graph
 * ============================================================================ */

TEST_F(CuvkGraphTest, LaunchEmptyGraph) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));

    EXPECT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, LaunchEmptyNodes) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUgraphNode a, b, c;
    cuGraphAddEmptyNode(&a, graph, NULL, 0);
    cuGraphAddEmptyNode(&b, graph, &a, 1);
    cuGraphAddEmptyNode(&c, graph, &b, 1);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    EXPECT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

/* ============================================================================
 * Memset node
 * ============================================================================ */

TEST_F(CuvkGraphTest, MemsetNodeD32) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int N = 64;
    CUdeviceptr d_buf;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_buf, N * sizeof(uint32_t)));

    CUDA_MEMSET_NODE_PARAMS memsetParams = {};
    memsetParams.dst = d_buf;
    memsetParams.value = 0xCAFEBABE;
    memsetParams.elementSize = 4;
    memsetParams.width = N;
    memsetParams.height = 1;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemsetNode(&node, graph, NULL, 0,
                                                  &memsetParams, ctx));

    CUgraphNodeType type;
    cuGraphNodeGetType(node, &type);
    EXPECT_EQ(CU_GRAPH_NODE_TYPE_MEMSET, type);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    uint32_t h_buf[N];
    cuMemcpyDtoH(h_buf, d_buf, N * sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        EXPECT_EQ(0xCAFEBABEu, h_buf[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_buf);
}

TEST_F(CuvkGraphTest, MemsetNodeD8) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int N = 128;
    CUdeviceptr d_buf;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_buf, N));

    CUDA_MEMSET_NODE_PARAMS memsetParams = {};
    memsetParams.dst = d_buf;
    memsetParams.value = 0x42;
    memsetParams.elementSize = 1;
    memsetParams.width = N;
    memsetParams.height = 1;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemsetNode(&node, graph, NULL, 0,
                                                  &memsetParams, ctx));

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    uint8_t h_buf[N];
    cuMemcpyDtoH(h_buf, d_buf, N);
    for (int i = 0; i < N; i++)
        EXPECT_EQ(0x42, h_buf[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_buf);
}

TEST_F(CuvkGraphTest, MemsetNodeD16) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int N = 64;
    CUdeviceptr d_buf;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_buf, N * sizeof(uint16_t)));

    CUDA_MEMSET_NODE_PARAMS memsetParams = {};
    memsetParams.dst = d_buf;
    memsetParams.value = 0xBEEF;
    memsetParams.elementSize = 2;
    memsetParams.width = N;
    memsetParams.height = 1;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemsetNode(&node, graph, NULL, 0,
                                                  &memsetParams, ctx));

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    uint16_t h_buf[N];
    cuMemcpyDtoH(h_buf, d_buf, N * sizeof(uint16_t));
    for (int i = 0; i < N; i++)
        EXPECT_EQ(0xBEEF, h_buf[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_buf);
}

/* ============================================================================
 * Memcpy node
 * ============================================================================ */

TEST_F(CuvkGraphTest, MemcpyNodeHtoD) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int N = 64;
    float h_src[N];
    for (int i = 0; i < N; i++) h_src[i] = (float)i;

    CUdeviceptr d_buf;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_buf, N * sizeof(float)));

    CUDA_MEMCPY3D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.srcHost = h_src;
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstDevice = d_buf;
    copyParams.WidthInBytes = N * sizeof(float);
    copyParams.Height = 1;
    copyParams.Depth = 1;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemcpyNode(&node, graph, NULL, 0,
                                                  &copyParams, ctx));

    CUgraphNodeType type;
    cuGraphNodeGetType(node, &type);
    EXPECT_EQ(CU_GRAPH_NODE_TYPE_MEMCPY, type);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    float h_dst[N] = {};
    cuMemcpyDtoH(h_dst, d_buf, N * sizeof(float));
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ((float)i, h_dst[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_buf);
}

TEST_F(CuvkGraphTest, MemcpyNodeDtoH) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int N = 32;
    CUdeviceptr d_buf;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_buf, N * sizeof(float)));

    float h_src[N];
    for (int i = 0; i < N; i++) h_src[i] = (float)(i * 3);
    cuMemcpyHtoD(d_buf, h_src, N * sizeof(float));

    float h_dst[N] = {};

    CUDA_MEMCPY3D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.srcDevice = d_buf;
    copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.dstHost = h_dst;
    copyParams.WidthInBytes = N * sizeof(float);
    copyParams.Height = 1;
    copyParams.Depth = 1;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemcpyNode(&node, graph, NULL, 0,
                                                  &copyParams, ctx));

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ((float)(i * 3), h_dst[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_buf);
}

TEST_F(CuvkGraphTest, MemcpyNodeDtoD) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int N = 64;
    CUdeviceptr d_src, d_dst;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_src, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_dst, N * sizeof(float)));

    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i * 7);
    cuMemcpyHtoD(d_src, h_data, N * sizeof(float));

    CUDA_MEMCPY3D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.srcDevice = d_src;
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstDevice = d_dst;
    copyParams.WidthInBytes = N * sizeof(float);
    copyParams.Height = 1;
    copyParams.Depth = 1;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemcpyNode(&node, graph, NULL, 0,
                                                  &copyParams, ctx));

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    float h_result[N] = {};
    cuMemcpyDtoH(h_result, d_dst, N * sizeof(float));
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ((float)(i * 7), h_result[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_src);
    cuMemFree(d_dst);
}

/* ============================================================================
 * Host node
 * ============================================================================ */

static void CUDA_CB host_callback(void *userData) {
    int *counter = (int *)userData;
    (*counter)++;
}

TEST_F(CuvkGraphTest, HostNode) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    int counter = 0;
    CUDA_HOST_NODE_PARAMS hostParams = {};
    hostParams.fn = host_callback;
    hostParams.userData = &counter;

    CUgraphNode node;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddHostNode(&node, graph, NULL, 0, &hostParams));

    CUgraphNodeType type;
    cuGraphNodeGetType(node, &type);
    EXPECT_EQ(CU_GRAPH_NODE_TYPE_HOST, type);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    EXPECT_EQ(1, counter);

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

TEST_F(CuvkGraphTest, HostNodeMultipleLaunches) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    int counter = 0;
    CUDA_HOST_NODE_PARAMS hostParams = {};
    hostParams.fn = host_callback;
    hostParams.userData = &counter;

    CUgraphNode node;
    cuGraphAddHostNode(&node, graph, NULL, 0, &hostParams);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));

    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
        cuCtxSynchronize();
    }

    EXPECT_EQ(5, counter);

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

/* ============================================================================
 * Kernel node
 * ============================================================================ */

TEST_F(CuvkGraphTest, KernelNodeVectorAdd) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAdd"));

    const int N = 64;
    std::vector<float> h_a(N), h_b(N), h_c(N, 0.0f);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 10);
    }

    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_c, N * sizeof(float));
    cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b.data(), N * sizeof(float));

    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUDA_KERNEL_NODE_PARAMS kernelParams = {};
    kernelParams.func = func;
    kernelParams.gridDimX = N;
    kernelParams.gridDimY = 1;
    kernelParams.gridDimZ = 1;
    kernelParams.blockDimX = 1;
    kernelParams.blockDimY = 1;
    kernelParams.blockDimZ = 1;
    kernelParams.sharedMemBytes = 0;
    void *params[] = { &d_a, &d_b, &d_c };
    kernelParams.kernelParams = params;

    CUgraphNode knode;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddKernelNode(&knode, graph, NULL, 0,
                                                  &kernelParams));

    CUgraphNodeType type;
    cuGraphNodeGetType(knode, &type);
    EXPECT_EQ(CU_GRAPH_NODE_TYPE_KERNEL, type);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    cuMemcpyDtoH(h_c.data(), d_c, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_a[i] + h_b[i], h_c[i]) << "index " << i;
    }

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod);
}

TEST_F(CuvkGraphTest, KernelNodeGetSetParams) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAdd"));

    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, 64);
    cuMemAlloc(&d_b, 64);
    cuMemAlloc(&d_c, 64);

    CUDA_KERNEL_NODE_PARAMS kernelParams = {};
    kernelParams.func = func;
    kernelParams.gridDimX = 16;
    kernelParams.gridDimY = 1;
    kernelParams.gridDimZ = 1;
    kernelParams.blockDimX = 1;
    kernelParams.blockDimY = 1;
    kernelParams.blockDimZ = 1;
    void *params[] = { &d_a, &d_b, &d_c };
    kernelParams.kernelParams = params;

    CUgraphNode knode;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddKernelNode(&knode, graph, NULL, 0,
                                                  &kernelParams));

    /* Read back */
    CUDA_KERNEL_NODE_PARAMS readParams = {};
    EXPECT_EQ(CUDA_SUCCESS, cuGraphKernelNodeGetParams(knode, &readParams));
    EXPECT_EQ(func, readParams.func);
    EXPECT_EQ(16u, readParams.gridDimX);

    /* Update params */
    kernelParams.gridDimX = 32;
    EXPECT_EQ(CUDA_SUCCESS, cuGraphKernelNodeSetParams(knode, &kernelParams));

    CUDA_KERNEL_NODE_PARAMS readParams2 = {};
    cuGraphKernelNodeGetParams(knode, &readParams2);
    EXPECT_EQ(32u, readParams2.gridDimX);

    cuGraphDestroy(graph);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Graph upload (no-op)
 * ============================================================================ */

TEST_F(CuvkGraphTest, GraphUpload) {
    CUgraph graph;
    cuGraphCreate(&graph, 0);
    CUgraphNode n;
    cuGraphAddEmptyNode(&n, graph, NULL, 0);

    CUgraphExec exec;
    cuGraphInstantiate(&exec, graph, 0);

    EXPECT_EQ(CUDA_SUCCESS, cuGraphUpload(exec, NULL));

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

/* ============================================================================
 * Multi-node graph with dependencies (memset -> kernel -> memcpy)
 * ============================================================================ */

TEST_F(CuvkGraphTest, MemsetThenKernelPipeline) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, SCALE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "scaleVec"));

    const int N = 32;
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, N * sizeof(float));

    /* Step 1: Memset to zero */
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    /* HtoD copy node: upload initial data */
    float h_init[N];
    for (int i = 0; i < N; i++) h_init[i] = (float)(i + 1);  /* 1..32 */

    CUDA_MEMCPY3D cpParams = {};
    cpParams.srcMemoryType = CU_MEMORYTYPE_HOST;
    cpParams.srcHost = h_init;
    cpParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpParams.dstDevice = d_data;
    cpParams.WidthInBytes = N * sizeof(float);
    cpParams.Height = 1;
    cpParams.Depth = 1;

    CUgraphNode copyNode;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddMemcpyNode(&copyNode, graph, NULL, 0,
                                                  &cpParams, ctx));

    /* Kernel node: scale by 2.0, depends on copy */
    float scalar = 2.0f;
    CUDA_KERNEL_NODE_PARAMS kp = {};
    kp.func = func;
    kp.gridDimX = N;
    kp.gridDimY = 1;
    kp.gridDimZ = 1;
    kp.blockDimX = 1;
    kp.blockDimY = 1;
    kp.blockDimZ = 1;
    void *kparams[] = { &d_data, &scalar };
    kp.kernelParams = kparams;

    CUgraphNode kernelNode;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphAddKernelNode(&kernelNode, graph, &copyNode, 1,
                                                  &kp));

    /* Instantiate and launch */
    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    float h_result[N] = {};
    cuMemcpyDtoH(h_result, d_data, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ((float)(i + 1) * 2.0f, h_result[i]) << "index " << i;
    }

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_data);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Complex DAG: diamond dependency pattern
 *
 *       root (memcpy HtoD)
 *      /    \
 *  scale1  scale2    (two independent kernels)
 *      \    /
 *     host_cb        (host callback, depends on both)
 *
 * ============================================================================ */

static void CUDA_CB diamond_callback(void *userData) {
    int *flag = (int *)userData;
    *flag = 42;
}

TEST_F(CuvkGraphTest, DiamondDependencyDAG) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, SCALE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "scaleVec"));

    const int N = 16;
    CUdeviceptr d_a, d_b;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));

    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1);

    /* Copy to both buffers */
    cuMemcpyHtoD(d_a, h_data, N * sizeof(float));
    cuMemcpyHtoD(d_b, h_data, N * sizeof(float));

    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    /* Root: empty node */
    CUgraphNode root;
    cuGraphAddEmptyNode(&root, graph, NULL, 0);

    /* Branch 1: scale d_a by 3.0 */
    float s1 = 3.0f;
    CUDA_KERNEL_NODE_PARAMS kp1 = {};
    kp1.func = func;
    kp1.gridDimX = N; kp1.gridDimY = 1; kp1.gridDimZ = 1;
    kp1.blockDimX = 1; kp1.blockDimY = 1; kp1.blockDimZ = 1;
    void *p1[] = { &d_a, &s1 };
    kp1.kernelParams = p1;

    CUgraphNode scale1;
    cuGraphAddKernelNode(&scale1, graph, &root, 1, &kp1);

    /* Branch 2: scale d_b by 5.0 */
    float s2 = 5.0f;
    CUDA_KERNEL_NODE_PARAMS kp2 = {};
    kp2.func = func;
    kp2.gridDimX = N; kp2.gridDimY = 1; kp2.gridDimZ = 1;
    kp2.blockDimX = 1; kp2.blockDimY = 1; kp2.blockDimZ = 1;
    void *p2[] = { &d_b, &s2 };
    kp2.kernelParams = p2;

    CUgraphNode scale2;
    cuGraphAddKernelNode(&scale2, graph, &root, 1, &kp2);

    /* Join: host callback depends on both branches */
    int callback_flag = 0;
    CUDA_HOST_NODE_PARAMS hp = {};
    hp.fn = diamond_callback;
    hp.userData = &callback_flag;

    CUgraphNode deps[] = { scale1, scale2 };
    CUgraphNode hostNode;
    cuGraphAddHostNode(&hostNode, graph, deps, 2, &hp);

    /* Verify edge count: 3 edges (root->scale1, root->scale2, scale1->host, scale2->host) */
    size_t numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ(4u, numEdges);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    /* Verify results */
    EXPECT_EQ(42, callback_flag);

    float h_a[N], h_b[N];
    cuMemcpyDtoH(h_a, d_a, N * sizeof(float));
    cuMemcpyDtoH(h_b, d_b, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ((float)(i + 1) * 3.0f, h_a[i]) << "d_a[" << i << "]";
        EXPECT_FLOAT_EQ((float)(i + 1) * 5.0f, h_b[i]) << "d_b[" << i << "]";
    }

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Multiple launches of the same graph
 * ============================================================================ */

TEST_F(CuvkGraphTest, RepeatedLaunch) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, SCALE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "scaleVec"));

    const int N = 32;
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, N * sizeof(float));

    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cuMemcpyHtoD(d_data, h_data, N * sizeof(float));

    CUgraph graph;
    cuGraphCreate(&graph, 0);

    float scalar = 2.0f;
    CUDA_KERNEL_NODE_PARAMS kp = {};
    kp.func = func;
    kp.gridDimX = N; kp.gridDimY = 1; kp.gridDimZ = 1;
    kp.blockDimX = 1; kp.blockDimY = 1; kp.blockDimZ = 1;
    void *params[] = { &d_data, &scalar };
    kp.kernelParams = params;

    CUgraphNode knode;
    cuGraphAddKernelNode(&knode, graph, NULL, 0, &kp);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));

    /* Launch 4 times: data should be multiplied by 2^4 = 16 */
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
        cuCtxSynchronize();
    }

    float h_result[N];
    cuMemcpyDtoH(h_result, d_data, N * sizeof(float));
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ(16.0f, h_result[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_data);
    cuModuleUnload(mod);
}

/* ============================================================================
 * GraphExecUpdate
 * ============================================================================ */

TEST_F(CuvkGraphTest, ExecUpdate) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, SCALE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "scaleVec"));

    const int N = 16;
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, N * sizeof(float));

    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1);
    cuMemcpyHtoD(d_data, h_data, N * sizeof(float));

    /* Original graph: scale by 2 */
    CUgraph graph;
    cuGraphCreate(&graph, 0);

    float scalar = 2.0f;
    CUDA_KERNEL_NODE_PARAMS kp = {};
    kp.func = func;
    kp.gridDimX = N; kp.gridDimY = 1; kp.gridDimZ = 1;
    kp.blockDimX = 1; kp.blockDimY = 1; kp.blockDimZ = 1;
    void *params[] = { &d_data, &scalar };
    kp.kernelParams = params;

    CUgraphNode knode;
    cuGraphAddKernelNode(&knode, graph, NULL, 0, &kp);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    /* Verify scale by 2 */
    float h_result[N];
    cuMemcpyDtoH(h_result, d_data, N * sizeof(float));
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ((float)(i + 1) * 2.0f, h_result[i]);

    /* Update graph: now scale by 3 instead */
    float scalar3 = 3.0f;
    CUDA_KERNEL_NODE_PARAMS kp2 = kp;
    void *params2[] = { &d_data, &scalar3 };
    kp2.kernelParams = params2;
    cuGraphKernelNodeSetParams(knode, &kp2);

    CUgraphExecUpdateResultInfo updateResult = {};
    EXPECT_EQ(CUDA_SUCCESS, cuGraphExecUpdate(exec, graph, &updateResult));

    /* Reset data */
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1);
    cuMemcpyHtoD(d_data, h_data, N * sizeof(float));

    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    cuMemcpyDtoH(h_result, d_data, N * sizeof(float));
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ((float)(i + 1) * 3.0f, h_result[i]);

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_data);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Stream capture status
 * ============================================================================ */

TEST_F(CuvkGraphTest, StreamIsCapturingNone) {
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    EXPECT_EQ(CUDA_SUCCESS, cuStreamIsCapturing(NULL, &status));
    EXPECT_EQ(CU_STREAM_CAPTURE_STATUS_NONE, status);
}

/* ============================================================================
 * Large graph: linear chain of many nodes
 * ============================================================================ */

TEST_F(CuvkGraphTest, LinearChain50Nodes) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int CHAIN_LEN = 50;
    CUgraphNode prev = NULL;
    for (int i = 0; i < CHAIN_LEN; i++) {
        CUgraphNode node;
        if (prev)
            cuGraphAddEmptyNode(&node, graph, &prev, 1);
        else
            cuGraphAddEmptyNode(&node, graph, NULL, 0);
        prev = node;
    }

    size_t numNodes = 0;
    cuGraphGetNodes(graph, NULL, &numNodes);
    EXPECT_EQ((size_t)CHAIN_LEN, numNodes);

    size_t numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ((size_t)(CHAIN_LEN - 1), numEdges);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

/* ============================================================================
 * Wide graph: many independent nodes
 * ============================================================================ */

TEST_F(CuvkGraphTest, WideGraph32IndependentNodes) {
    CUgraph graph;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphCreate(&graph, 0));

    const int WIDTH = 32;
    for (int i = 0; i < WIDTH; i++) {
        CUgraphNode node;
        cuGraphAddEmptyNode(&node, graph, NULL, 0);
    }

    size_t numNodes = 0;
    cuGraphGetNodes(graph, NULL, &numNodes);
    EXPECT_EQ((size_t)WIDTH, numNodes);

    size_t numEdges = 0;
    cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges);
    EXPECT_EQ(0u, numEdges);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

/* ============================================================================
 * Mixed node types in one graph: memset + kernel + host callback
 * ============================================================================ */

static void CUDA_CB verify_callback(void *userData) {
    int *flag = (int *)userData;
    *flag = 1;
}

TEST_F(CuvkGraphTest, MixedNodeTypePipeline) {
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, SCALE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "scaleVec"));

    const int N = 16;
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, N * sizeof(float));

    CUgraph graph;
    cuGraphCreate(&graph, 0);

    /* Node 1: Memset to pattern (we'll reinterpret as float) */
    CUDA_MEMSET_NODE_PARAMS msParams = {};
    msParams.dst = d_data;
    msParams.value = 0x3F800000; /* IEEE 754 for 1.0f */
    msParams.elementSize = 4;
    msParams.width = N;
    msParams.height = 1;

    CUgraphNode memsetNode;
    cuGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &msParams, ctx);

    /* Node 2: Scale by 7.0, depends on memset */
    float scalar = 7.0f;
    CUDA_KERNEL_NODE_PARAMS kp = {};
    kp.func = func;
    kp.gridDimX = N; kp.gridDimY = 1; kp.gridDimZ = 1;
    kp.blockDimX = 1; kp.blockDimY = 1; kp.blockDimZ = 1;
    void *params[] = { &d_data, &scalar };
    kp.kernelParams = params;

    CUgraphNode kernelNode;
    cuGraphAddKernelNode(&kernelNode, graph, &memsetNode, 1, &kp);

    /* Node 3: Host callback, depends on kernel */
    int done_flag = 0;
    CUDA_HOST_NODE_PARAMS hp = {};
    hp.fn = verify_callback;
    hp.userData = &done_flag;

    CUgraphNode hostNode;
    cuGraphAddHostNode(&hostNode, graph, &kernelNode, 1, &hp);

    /* 3 nodes, 2 edges */
    size_t nn = 0, ne = 0;
    cuGraphGetNodes(graph, NULL, &nn);
    cuGraphGetEdges(graph, NULL, NULL, NULL, &ne);
    EXPECT_EQ(3u, nn);
    EXPECT_EQ(2u, ne);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    EXPECT_EQ(1, done_flag);

    /* d_data should be all 7.0f (1.0f * 7.0f) */
    float h_result[N];
    cuMemcpyDtoH(h_result, d_data, N * sizeof(float));
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ(7.0f, h_result[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_data);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Memset + DtoD copy pipeline
 * ============================================================================ */

TEST_F(CuvkGraphTest, MemsetThenDtoDCopy) {
    const int N = 32;
    CUdeviceptr d_src, d_dst;
    cuMemAlloc(&d_src, N * sizeof(uint32_t));
    cuMemAlloc(&d_dst, N * sizeof(uint32_t));

    CUgraph graph;
    cuGraphCreate(&graph, 0);

    /* Memset src buffer */
    CUDA_MEMSET_NODE_PARAMS msParams = {};
    msParams.dst = d_src;
    msParams.value = 0x12345678;
    msParams.elementSize = 4;
    msParams.width = N;
    msParams.height = 1;

    CUgraphNode msNode;
    cuGraphAddMemsetNode(&msNode, graph, NULL, 0, &msParams, ctx);

    /* DtoD copy: src -> dst, depends on memset */
    CUDA_MEMCPY3D cpParams = {};
    cpParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpParams.srcDevice = d_src;
    cpParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpParams.dstDevice = d_dst;
    cpParams.WidthInBytes = N * sizeof(uint32_t);
    cpParams.Height = 1;
    cpParams.Depth = 1;

    CUgraphNode cpNode;
    cuGraphAddMemcpyNode(&cpNode, graph, &msNode, 1, &cpParams, ctx);

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));
    cuCtxSynchronize();

    uint32_t h_result[N];
    cuMemcpyDtoH(h_result, d_dst, N * sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        EXPECT_EQ(0x12345678u, h_result[i]) << "index " << i;

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
    cuMemFree(d_src);
    cuMemFree(d_dst);
}

/* ============================================================================
 * Fan-out / Fan-in topology
 *
 *          root
 *      /  |  |  \
 *     n1  n2 n3  n4   (fan-out)
 *      \  |  |  /
 *         join         (fan-in)
 *
 * ============================================================================ */

TEST_F(CuvkGraphTest, FanOutFanInTopology) {
    CUgraph graph;
    cuGraphCreate(&graph, 0);

    CUgraphNode root;
    cuGraphAddEmptyNode(&root, graph, NULL, 0);

    const int FANOUT = 4;
    CUgraphNode branches[FANOUT];
    for (int i = 0; i < FANOUT; i++)
        cuGraphAddEmptyNode(&branches[i], graph, &root, 1);

    CUgraphNode join;
    cuGraphAddEmptyNode(&join, graph, branches, FANOUT);

    size_t nn = 0, ne = 0;
    cuGraphGetNodes(graph, NULL, &nn);
    cuGraphGetEdges(graph, NULL, NULL, NULL, &ne);
    EXPECT_EQ(6u, nn);        /* root + 4 branches + join */
    EXPECT_EQ(8u, ne);        /* root->4 + 4->join */

    CUgraphExec exec;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, NULL));

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);
}

/* ============================================================================
 * Multiple distinct graphs with separate execs
 * ============================================================================ */

TEST_F(CuvkGraphTest, TwoIndependentGraphs) {
    const int N = 32;
    CUdeviceptr d_a, d_b;
    cuMemAlloc(&d_a, N * sizeof(uint32_t));
    cuMemAlloc(&d_b, N * sizeof(uint32_t));

    /* Graph 1: memset d_a to 0xAAAA */
    CUgraph g1;
    cuGraphCreate(&g1, 0);
    CUDA_MEMSET_NODE_PARAMS ms1 = {};
    ms1.dst = d_a; ms1.value = 0xAAAAAAAA;
    ms1.elementSize = 4; ms1.width = N; ms1.height = 1;
    CUgraphNode n1;
    cuGraphAddMemsetNode(&n1, g1, NULL, 0, &ms1, ctx);

    /* Graph 2: memset d_b to 0xBBBB */
    CUgraph g2;
    cuGraphCreate(&g2, 0);
    CUDA_MEMSET_NODE_PARAMS ms2 = {};
    ms2.dst = d_b; ms2.value = 0xBBBBBBBB;
    ms2.elementSize = 4; ms2.width = N; ms2.height = 1;
    CUgraphNode n2;
    cuGraphAddMemsetNode(&n2, g2, NULL, 0, &ms2, ctx);

    CUgraphExec e1, e2;
    cuGraphInstantiate(&e1, g1, 0);
    cuGraphInstantiate(&e2, g2, 0);

    cuGraphLaunch(e1, NULL);
    cuGraphLaunch(e2, NULL);
    cuCtxSynchronize();

    uint32_t h_a[N], h_b[N];
    cuMemcpyDtoH(h_a, d_a, N * sizeof(uint32_t));
    cuMemcpyDtoH(h_b, d_b, N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(0xAAAAAAAAu, h_a[i]);
        EXPECT_EQ(0xBBBBBBBBu, h_b[i]);
    }

    cuGraphExecDestroy(e1);
    cuGraphExecDestroy(e2);
    cuGraphDestroy(g1);
    cuGraphDestroy(g2);
    cuMemFree(d_a);
    cuMemFree(d_b);
}
