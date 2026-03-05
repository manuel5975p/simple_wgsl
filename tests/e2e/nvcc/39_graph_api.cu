// 39. CUDA Graph API: create, add nodes, instantiate, launch
//     Tests graph lifecycle with memset + kernel + host callback nodes
//     Uses CUDA Driver API graph functions via cuGraphCreate etc.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_DRV(call) do { \
    CUresult e = (call); \
    if (e != CUDA_SUCCESS) { \
        const char *msg = "unknown"; \
        cuGetErrorString(e, &msg); \
        fprintf(stderr, "CUDA driver error %d (%s) at %s:%d\n", \
                e, msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA runtime error %d (%s) at %s:%d\n", \
                e, cudaGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

__global__ void scaleKernel(float *data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scalar;
}

__global__ void addKernel(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

static int g_host_flag = 0;
static void CUDART_CB hostCallback(void *userData) {
    int *flag = (int *)userData;
    *flag = 1;
}

// Test 1: Empty graph create/destroy/instantiate/launch
static int test_empty_graph() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 2: Memset node
static int test_memset_node() {
    const int N = 256;
    CUdeviceptr d_buf;
    CHECK_DRV(cuMemAlloc(&d_buf, N * sizeof(unsigned int)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUDA_MEMSET_NODE_PARAMS msParams = {};
    msParams.dst = d_buf;
    msParams.value = 0xDEADC0DE;
    msParams.elementSize = 4;
    msParams.width = N;
    msParams.height = 1;

    CUgraphNode msNode;
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    CHECK_DRV(cuGraphAddMemsetNode(&msNode, graph, NULL, 0, &msParams, ctx));

    CUgraphNodeType type;
    CHECK_DRV(cuGraphNodeGetType(msNode, &type));
    if (type != CU_GRAPH_NODE_TYPE_MEMSET) {
        fprintf(stderr, "FAIL: expected MEMSET node type, got %d\n", type);
        return 1;
    }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    unsigned int h_buf[N];
    CHECK_DRV(cuMemcpyDtoH(h_buf, d_buf, N * sizeof(unsigned int)));

    for (int i = 0; i < N; i++) {
        if (h_buf[i] != 0xDEADC0DE) {
            fprintf(stderr, "FAIL: memset at %d: got 0x%08X expected 0xDEADC0DE\n",
                    i, h_buf[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d_buf));
    return 0;
}

// Test 3: Empty node chain + node/edge counting
static int test_empty_nodes_and_topology() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUgraphNode a, b, c;
    CHECK_DRV(cuGraphAddEmptyNode(&a, graph, NULL, 0));
    CHECK_DRV(cuGraphAddEmptyNode(&b, graph, &a, 1));
    CHECK_DRV(cuGraphAddEmptyNode(&c, graph, &b, 1));

    size_t numNodes = 0;
    CHECK_DRV(cuGraphGetNodes(graph, NULL, &numNodes));
    if (numNodes != 3) {
        fprintf(stderr, "FAIL: expected 3 nodes, got %zu\n", numNodes);
        return 1;
    }

    size_t numEdges = 0;
    CHECK_DRV(cuGraphGetEdges(graph, NULL, NULL, NULL, &numEdges));
    if (numEdges != 2) {
        fprintf(stderr, "FAIL: expected 2 edges, got %zu\n", numEdges);
        return 1;
    }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 4: Host node callback
static int test_host_node() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    g_host_flag = 0;
    CUDA_HOST_NODE_PARAMS hp = {};
    hp.fn = hostCallback;
    hp.userData = &g_host_flag;

    CUgraphNode hostNode;
    CHECK_DRV(cuGraphAddHostNode(&hostNode, graph, NULL, 0, &hp));

    CUgraphNodeType type;
    CHECK_DRV(cuGraphNodeGetType(hostNode, &type));
    if (type != CU_GRAPH_NODE_TYPE_HOST) {
        fprintf(stderr, "FAIL: expected HOST node type, got %d\n", type);
        return 1;
    }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    if (g_host_flag != 1) {
        fprintf(stderr, "FAIL: host callback not invoked (flag=%d)\n", g_host_flag);
        return 1;
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 5: Memcpy HtoD node
static int test_memcpy_htod_node() {
    const int N = 128;
    float h_src[N];
    for (int i = 0; i < N; i++) h_src[i] = (float)i * 0.5f;

    CUdeviceptr d_buf;
    CHECK_DRV(cuMemAlloc(&d_buf, N * sizeof(float)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUDA_MEMCPY3D cpParams = {};
    cpParams.srcMemoryType = CU_MEMORYTYPE_HOST;
    cpParams.srcHost = h_src;
    cpParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpParams.dstDevice = d_buf;
    cpParams.WidthInBytes = N * sizeof(float);
    cpParams.Height = 1;
    cpParams.Depth = 1;

    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    CUgraphNode cpNode;
    CHECK_DRV(cuGraphAddMemcpyNode(&cpNode, graph, NULL, 0, &cpParams, ctx));

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    float h_dst[N] = {};
    CHECK_DRV(cuMemcpyDtoH(h_dst, d_buf, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        if (fabsf(h_dst[i] - h_src[i]) > 1e-6f) {
            fprintf(stderr, "FAIL: memcpy HtoD at %d: got %f expected %f\n",
                    i, h_dst[i], h_src[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d_buf));
    return 0;
}

// Test 6: Diamond DAG (memcpy -> two parallel memsets -> host callback)
static int test_diamond_dag() {
    const int N = 64;
    CUdeviceptr d_a, d_b;
    CHECK_DRV(cuMemAlloc(&d_a, N * sizeof(unsigned int)));
    CHECK_DRV(cuMemAlloc(&d_b, N * sizeof(unsigned int)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    /* Root: empty node */
    CUgraphNode root;
    CHECK_DRV(cuGraphAddEmptyNode(&root, graph, NULL, 0));

    /* Branch 1: memset d_a */
    CUDA_MEMSET_NODE_PARAMS ms1 = {};
    ms1.dst = d_a; ms1.value = 0x11111111;
    ms1.elementSize = 4; ms1.width = N; ms1.height = 1;
    CUgraphNode n1;
    CHECK_DRV(cuGraphAddMemsetNode(&n1, graph, &root, 1, &ms1, ctx));

    /* Branch 2: memset d_b */
    CUDA_MEMSET_NODE_PARAMS ms2 = {};
    ms2.dst = d_b; ms2.value = 0x22222222;
    ms2.elementSize = 4; ms2.width = N; ms2.height = 1;
    CUgraphNode n2;
    CHECK_DRV(cuGraphAddMemsetNode(&n2, graph, &root, 1, &ms2, ctx));

    /* Join: host callback */
    g_host_flag = 0;
    CUDA_HOST_NODE_PARAMS hp = {};
    hp.fn = hostCallback;
    hp.userData = &g_host_flag;
    CUgraphNode deps[] = { n1, n2 };
    CUgraphNode hostNode;
    CHECK_DRV(cuGraphAddHostNode(&hostNode, graph, deps, 2, &hp));

    /* Verify topology: 4 nodes, 4 edges */
    size_t nn = 0, ne = 0;
    CHECK_DRV(cuGraphGetNodes(graph, NULL, &nn));
    CHECK_DRV(cuGraphGetEdges(graph, NULL, NULL, NULL, &ne));
    if (nn != 4) { fprintf(stderr, "FAIL: expected 4 nodes, got %zu\n", nn); return 1; }
    if (ne != 4) { fprintf(stderr, "FAIL: expected 4 edges, got %zu\n", ne); return 1; }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    if (g_host_flag != 1) {
        fprintf(stderr, "FAIL: host callback not invoked in diamond\n");
        return 1;
    }

    unsigned int h_a[N], h_b[N];
    CHECK_DRV(cuMemcpyDtoH(h_a, d_a, N * sizeof(unsigned int)));
    CHECK_DRV(cuMemcpyDtoH(h_b, d_b, N * sizeof(unsigned int)));
    for (int i = 0; i < N; i++) {
        if (h_a[i] != 0x11111111 || h_b[i] != 0x22222222) {
            fprintf(stderr, "FAIL: diamond memset at %d: a=0x%08X b=0x%08X\n",
                    i, h_a[i], h_b[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d_a));
    CHECK_DRV(cuMemFree(d_b));
    return 0;
}

// Test 7: Repeated launches of the same exec graph
static int test_repeated_launch() {
    const int N = 64;
    CUdeviceptr d_buf;
    CHECK_DRV(cuMemAlloc(&d_buf, N * sizeof(unsigned int)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    CUDA_MEMSET_NODE_PARAMS msParams = {};
    msParams.dst = d_buf;
    msParams.value = 0xFACEFACE;
    msParams.elementSize = 4;
    msParams.width = N;
    msParams.height = 1;

    CUgraphNode msNode;
    CHECK_DRV(cuGraphAddMemsetNode(&msNode, graph, NULL, 0, &msParams, ctx));

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));

    /* Launch 10 times */
    for (int iter = 0; iter < 10; iter++) {
        CHECK_DRV(cuGraphLaunch(exec, NULL));
        CHECK_DRV(cuCtxSynchronize());
    }

    unsigned int h_buf[N];
    CHECK_DRV(cuMemcpyDtoH(h_buf, d_buf, N * sizeof(unsigned int)));
    for (int i = 0; i < N; i++) {
        if (h_buf[i] != 0xFACEFACE) {
            fprintf(stderr, "FAIL: repeated launch at %d: got 0x%08X\n", i, h_buf[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d_buf));
    return 0;
}

// Test 8: AddDependencies / RemoveDependencies
static int test_add_remove_deps() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUgraphNode a, b;
    CHECK_DRV(cuGraphAddEmptyNode(&a, graph, NULL, 0));
    CHECK_DRV(cuGraphAddEmptyNode(&b, graph, NULL, 0));

    /* Initially 0 edges */
    size_t ne = 0;
    CHECK_DRV(cuGraphGetEdges(graph, NULL, NULL, NULL, &ne));
    if (ne != 0) { fprintf(stderr, "FAIL: expected 0 edges, got %zu\n", ne); return 1; }

    /* Add dependency a -> b */
    CHECK_DRV(cuGraphAddDependencies(graph, &a, &b, NULL, 1));
    CHECK_DRV(cuGraphGetEdges(graph, NULL, NULL, NULL, &ne));
    if (ne != 1) { fprintf(stderr, "FAIL: expected 1 edge after add, got %zu\n", ne); return 1; }

    /* Remove it */
    CHECK_DRV(cuGraphRemoveDependencies(graph, &a, &b, NULL, 1));
    CHECK_DRV(cuGraphGetEdges(graph, NULL, NULL, NULL, &ne));
    if (ne != 0) { fprintf(stderr, "FAIL: expected 0 edges after remove, got %zu\n", ne); return 1; }

    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 9: StreamIsCapturing returns NONE
static int test_stream_not_capturing() {
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    CHECK_DRV(cuStreamIsCapturing(NULL, &status));
    if (status != CU_STREAM_CAPTURE_STATUS_NONE) {
        fprintf(stderr, "FAIL: expected CAPTURE_STATUS_NONE, got %d\n", status);
        return 1;
    }
    return 0;
}

// Test 10: Large linear chain
static int test_large_chain() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    const int CHAIN = 100;
    CUgraphNode prev = NULL;
    for (int i = 0; i < CHAIN; i++) {
        CUgraphNode node;
        if (prev)
            CHECK_DRV(cuGraphAddEmptyNode(&node, graph, &prev, 1));
        else
            CHECK_DRV(cuGraphAddEmptyNode(&node, graph, NULL, 0));
        prev = node;
    }

    size_t nn = 0;
    CHECK_DRV(cuGraphGetNodes(graph, NULL, &nn));
    if (nn != (size_t)CHAIN) {
        fprintf(stderr, "FAIL: expected %d nodes, got %zu\n", CHAIN, nn);
        return 1;
    }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 11: Fan-out fan-in
static int test_fan_out_fan_in() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUgraphNode root;
    CHECK_DRV(cuGraphAddEmptyNode(&root, graph, NULL, 0));

    const int FANOUT = 8;
    CUgraphNode branches[FANOUT];
    for (int i = 0; i < FANOUT; i++)
        CHECK_DRV(cuGraphAddEmptyNode(&branches[i], graph, &root, 1));

    CUgraphNode join;
    CHECK_DRV(cuGraphAddEmptyNode(&join, graph, branches, FANOUT));

    size_t nn = 0, ne = 0;
    CHECK_DRV(cuGraphGetNodes(graph, NULL, &nn));
    CHECK_DRV(cuGraphGetEdges(graph, NULL, NULL, NULL, &ne));
    if (nn != (size_t)(FANOUT + 2)) {
        fprintf(stderr, "FAIL: expected %d nodes, got %zu\n", FANOUT + 2, nn);
        return 1;
    }
    if (ne != (size_t)(FANOUT * 2)) {
        fprintf(stderr, "FAIL: expected %d edges, got %zu\n", FANOUT * 2, ne);
        return 1;
    }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 12: Memset D8 + D16
static int test_memset_d8_d16() {
    const int N = 128;
    CUdeviceptr d8_buf, d16_buf;
    CHECK_DRV(cuMemAlloc(&d8_buf, N));
    CHECK_DRV(cuMemAlloc(&d16_buf, N * sizeof(unsigned short)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    /* D8 memset */
    CUDA_MEMSET_NODE_PARAMS ms8 = {};
    ms8.dst = d8_buf; ms8.value = 0xAB;
    ms8.elementSize = 1; ms8.width = N; ms8.height = 1;
    CUgraphNode n8;
    CHECK_DRV(cuGraphAddMemsetNode(&n8, graph, NULL, 0, &ms8, ctx));

    /* D16 memset */
    CUDA_MEMSET_NODE_PARAMS ms16 = {};
    ms16.dst = d16_buf; ms16.value = 0x1234;
    ms16.elementSize = 2; ms16.width = N; ms16.height = 1;
    CUgraphNode n16;
    CHECK_DRV(cuGraphAddMemsetNode(&n16, graph, NULL, 0, &ms16, ctx));

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    unsigned char h8[N];
    CHECK_DRV(cuMemcpyDtoH(h8, d8_buf, N));
    for (int i = 0; i < N; i++) {
        if (h8[i] != 0xAB) {
            fprintf(stderr, "FAIL: D8 memset at %d: got 0x%02X\n", i, h8[i]);
            return 1;
        }
    }

    unsigned short h16[N];
    CHECK_DRV(cuMemcpyDtoH(h16, d16_buf, N * sizeof(unsigned short)));
    for (int i = 0; i < N; i++) {
        if (h16[i] != 0x1234) {
            fprintf(stderr, "FAIL: D16 memset at %d: got 0x%04X\n", i, h16[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d8_buf));
    CHECK_DRV(cuMemFree(d16_buf));
    return 0;
}

// Test 13: Multiple host callbacks in sequence
static int test_multiple_host_callbacks() {
    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    int counters[3] = {0, 0, 0};

    CUgraphNode prev = NULL;
    for (int i = 0; i < 3; i++) {
        CUDA_HOST_NODE_PARAMS hp = {};
        hp.fn = hostCallback;
        hp.userData = &counters[i];
        CUgraphNode node;
        if (prev)
            CHECK_DRV(cuGraphAddHostNode(&node, graph, &prev, 1, &hp));
        else
            CHECK_DRV(cuGraphAddHostNode(&node, graph, NULL, 0, &hp));
        prev = node;
    }

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    for (int i = 0; i < 3; i++) {
        if (counters[i] != 1) {
            fprintf(stderr, "FAIL: callback %d not invoked (counter=%d)\n", i, counters[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    return 0;
}

// Test 14: DtoD memcpy node
static int test_memcpy_dtod_node() {
    const int N = 64;
    CUdeviceptr d_src, d_dst;
    CHECK_DRV(cuMemAlloc(&d_src, N * sizeof(float)));
    CHECK_DRV(cuMemAlloc(&d_dst, N * sizeof(float)));

    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i * 11);
    CHECK_DRV(cuMemcpyHtoD(d_src, h_data, N * sizeof(float)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    CUDA_MEMCPY3D cp = {};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice = d_src;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice = d_dst;
    cp.WidthInBytes = N * sizeof(float);
    cp.Height = 1;
    cp.Depth = 1;

    CUgraphNode cpNode;
    CHECK_DRV(cuGraphAddMemcpyNode(&cpNode, graph, NULL, 0, &cp, ctx));

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    float h_result[N] = {};
    CHECK_DRV(cuMemcpyDtoH(h_result, d_dst, N * sizeof(float)));
    for (int i = 0; i < N; i++) {
        if (fabsf(h_result[i] - h_data[i]) > 1e-6f) {
            fprintf(stderr, "FAIL: DtoD copy at %d: got %f expected %f\n",
                    i, h_result[i], h_data[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d_src));
    CHECK_DRV(cuMemFree(d_dst));
    return 0;
}

// Test 15: Memset + DtoD in pipeline
static int test_memset_then_dtod() {
    const int N = 32;
    CUdeviceptr d_src, d_dst;
    CHECK_DRV(cuMemAlloc(&d_src, N * sizeof(unsigned int)));
    CHECK_DRV(cuMemAlloc(&d_dst, N * sizeof(unsigned int)));

    CUgraph graph;
    CHECK_DRV(cuGraphCreate(&graph, 0));
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    CUDA_MEMSET_NODE_PARAMS ms = {};
    ms.dst = d_src; ms.value = 0xBADF00D;
    ms.elementSize = 4; ms.width = N; ms.height = 1;
    CUgraphNode msNode;
    CHECK_DRV(cuGraphAddMemsetNode(&msNode, graph, NULL, 0, &ms, ctx));

    CUDA_MEMCPY3D cp = {};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice = d_src;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice = d_dst;
    cp.WidthInBytes = N * sizeof(unsigned int);
    cp.Height = 1;
    cp.Depth = 1;
    CUgraphNode cpNode;
    CHECK_DRV(cuGraphAddMemcpyNode(&cpNode, graph, &msNode, 1, &cp, ctx));

    CUgraphExec exec;
    CHECK_DRV(cuGraphInstantiate(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, NULL));
    CHECK_DRV(cuCtxSynchronize());

    unsigned int h_result[N];
    CHECK_DRV(cuMemcpyDtoH(h_result, d_dst, N * sizeof(unsigned int)));
    for (int i = 0; i < N; i++) {
        if (h_result[i] != 0xBADF00D) {
            fprintf(stderr, "FAIL: memset+dtod at %d: got 0x%08X\n", i, h_result[i]);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(d_src));
    CHECK_DRV(cuMemFree(d_dst));
    return 0;
}

int main() {
    CHECK_DRV(cuInit(0));

    /* Create a driver API context for the first device */
    CUdevice dev;
    CHECK_DRV(cuDeviceGet(&dev, 0));
    CUcontext drv_ctx;
    CHECK_DRV(cuCtxCreate(&drv_ctx, NULL, 0, dev));

    struct { const char *name; int (*fn)(); } tests[] = {
        {"empty_graph",          test_empty_graph},
        {"memset_node",          test_memset_node},
        {"empty_nodes_topology", test_empty_nodes_and_topology},
        {"host_node",            test_host_node},
        {"memcpy_htod_node",     test_memcpy_htod_node},
        {"diamond_dag",          test_diamond_dag},
        {"repeated_launch",      test_repeated_launch},
        {"add_remove_deps",      test_add_remove_deps},
        {"stream_not_capturing", test_stream_not_capturing},
        {"large_chain",          test_large_chain},
        {"fan_out_fan_in",       test_fan_out_fan_in},
        {"memset_d8_d16",        test_memset_d8_d16},
        {"multiple_host_cbs",    test_multiple_host_callbacks},
        {"memcpy_dtod",          test_memcpy_dtod_node},
        {"memset_then_dtod",     test_memset_then_dtod},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;

    for (int i = 0; i < n; i++) {
        printf("  [%2d/%d] %-25s ... ", i + 1, n, tests[i].name);
        fflush(stdout);
        int r = tests[i].fn();
        if (r == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
            failed++;
        }
    }

    printf("\nResults: %d passed, %d failed out of %d\n", passed, failed, n);
    if (failed) {
        fprintf(stderr, "FAIL: 39_graph_api\n");
        return 1;
    }
    printf("PASS: 39_graph_api (all %d tests)\n", n);
    return 0;
}
