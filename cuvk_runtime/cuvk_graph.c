/*
 * cuvk_graph.c - CUDA Graph API implementation
 *
 * Implements CUDA Graph lifecycle, node creation, instantiation (topological
 * sort), and execution using Vulkan timeline semaphores for dependency
 * enforcement between GPU work nodes.
 *
 * Graph nodes are individually heap-allocated so CUgraphNode handles (pointers)
 * remain stable across additions. The graph stores an array of pointers.
 * The executable graph (CUgraphExec) uses a flat, contiguous, topologically
 * sorted copy for efficient iteration during launch.
 */

#include "cuvk_internal.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

/* Find index of a node pointer in a graph's pointer array. */
static int32_t graph_find_node_index(struct CUgraph_st *g, CUgraphNode node) {
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i] == node)
            return (int32_t)i;
    }
    return -1;
}

/* Add a dependency edge: from_idx -> to_idx (using the graph's node ptrs) */
static CUresult graph_add_edge_ptrs(struct CUgraphNode_st **nodes,
                                     uint32_t from_idx, uint32_t to_idx) {
    struct CUgraphNode_st *from = nodes[from_idx];
    struct CUgraphNode_st *to   = nodes[to_idx];

    /* Add from_idx to to->deps */
    if (to->dep_count >= to->dep_capacity) {
        uint32_t new_cap = to->dep_capacity == 0 ? 4 : to->dep_capacity * 2;
        uint32_t *d = (uint32_t *)realloc(to->deps, new_cap * sizeof(uint32_t));
        if (!d) return CUDA_ERROR_OUT_OF_MEMORY;
        to->deps = d;
        to->dep_capacity = new_cap;
    }
    to->deps[to->dep_count++] = from_idx;

    /* Add to_idx to from->dependents */
    if (from->dependent_count >= from->dependent_capacity) {
        uint32_t new_cap = from->dependent_capacity == 0 ? 4 : from->dependent_capacity * 2;
        uint32_t *d = (uint32_t *)realloc(from->dependents, new_cap * sizeof(uint32_t));
        if (!d) return CUDA_ERROR_OUT_OF_MEMORY;
        from->dependents = d;
        from->dependent_capacity = new_cap;
    }
    from->dependents[from->dependent_count++] = to_idx;

    return CUDA_SUCCESS;
}

/* Deep-copy kernel params (the void** array and each pointed-to value) */
static CUresult deep_copy_kernel_params(struct CUgraphNode_st *dst,
                                         const CUDA_KERNEL_NODE_PARAMS *src) {
    dst->params.kernel = *src;
    dst->kernel_params_copy = NULL;
    dst->kernel_params_count = 0;

    if (!src->kernelParams || !src->func)
        return CUDA_SUCCESS;

    struct CUfunc_st *f = src->func;
    uint32_t n = f->param_count;
    if (n == 0)
        return CUDA_SUCCESS;

    void **copy = (void **)calloc(n, sizeof(void *));
    if (!copy) return CUDA_ERROR_OUT_OF_MEMORY;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t sz = f->params[i].size;
        copy[i] = malloc(sz);
        if (!copy[i]) {
            for (uint32_t j = 0; j < i; j++) free(copy[j]);
            free(copy);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(copy[i], src->kernelParams[i], sz);
    }

    dst->params.kernel.kernelParams = copy;
    dst->kernel_params_copy = copy;
    dst->kernel_params_count = n;
    return CUDA_SUCCESS;
}

static void free_kernel_params(struct CUgraphNode_st *node) {
    if (node->kernel_params_copy) {
        for (size_t i = 0; i < node->kernel_params_count; i++)
            free(node->kernel_params_copy[i]);
        free(node->kernel_params_copy);
        node->kernel_params_copy = NULL;
        node->kernel_params_count = 0;
    }
}

static void free_node_internals(struct CUgraphNode_st *node) {
    free_kernel_params(node);
    free(node->deps);
    free(node->dependents);
    node->deps = NULL;
    node->dependents = NULL;
}

/* Allocate a new node in the graph, wire up dependencies.
 * Node is individually heap-allocated so the returned pointer stays stable. */
static CUresult graph_alloc_node(CUgraph hGraph, CUgraphNode *phNode,
                                  const CUgraphNode *dependencies,
                                  size_t numDependencies,
                                  CUgraphNodeType type) {
    if (!hGraph || !phNode)
        return CUDA_ERROR_INVALID_VALUE;

    /* Allocate the node on the heap */
    struct CUgraphNode_st *node = (struct CUgraphNode_st *)calloc(1, sizeof(*node));
    if (!node) return CUDA_ERROR_OUT_OF_MEMORY;
    node->type = type;

    /* Grow the pointer array if needed */
    if (hGraph->node_count >= hGraph->node_capacity) {
        uint32_t new_cap = hGraph->node_capacity == 0 ? 8 : hGraph->node_capacity * 2;
        struct CUgraphNode_st **new_ptrs = (struct CUgraphNode_st **)realloc(
            hGraph->nodes, new_cap * sizeof(struct CUgraphNode_st *));
        if (!new_ptrs) { free(node); return CUDA_ERROR_OUT_OF_MEMORY; }
        hGraph->nodes = new_ptrs;
        hGraph->node_capacity = new_cap;
    }

    uint32_t idx = hGraph->node_count;
    hGraph->nodes[idx] = node;
    hGraph->node_count++;

    /* Wire dependencies */
    for (size_t i = 0; i < numDependencies; i++) {
        if (!dependencies || !dependencies[i]) continue;
        int32_t dep_idx = graph_find_node_index(hGraph, dependencies[i]);
        if (dep_idx < 0) continue;
        CUresult r = graph_add_edge_ptrs(hGraph->nodes, (uint32_t)dep_idx, idx);
        if (r != CUDA_SUCCESS) return r;
    }

    *phNode = node;
    return CUDA_SUCCESS;
}

/* Deep-copy a node (for instantiation). Does NOT copy deps/dependents. */
static CUresult deep_copy_node(struct CUgraphNode_st *dst,
                                const struct CUgraphNode_st *src) {
    memset(dst, 0, sizeof(*dst));
    dst->type = src->type;

    switch (src->type) {
    case CU_GRAPH_NODE_TYPE_KERNEL:
        return deep_copy_kernel_params(dst, &src->params.kernel);
    case CU_GRAPH_NODE_TYPE_MEMCPY:
        dst->params.memcpy = src->params.memcpy;
        return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_MEMSET:
        dst->params.memset = src->params.memset;
        return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_HOST:
        dst->params.host = src->params.host;
        return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_EMPTY:
    default:
        return CUDA_SUCCESS;
    }
}

/* ============================================================================
 * Graph lifecycle
 * ============================================================================ */

CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
    (void)flags;
    if (!phGraph) return CUDA_ERROR_INVALID_VALUE;

    struct CUgraph_st *g = (struct CUgraph_st *)calloc(1, sizeof(*g));
    if (!g) return CUDA_ERROR_OUT_OF_MEMORY;

    *phGraph = g;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph) {
    if (!hGraph) return CUDA_ERROR_INVALID_VALUE;

    for (uint32_t i = 0; i < hGraph->node_count; i++) {
        free_node_internals(hGraph->nodes[i]);
        free(hGraph->nodes[i]);
    }
    free(hGraph->nodes);
    free(hGraph);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Node creation
 * ============================================================================ */

CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode *phGraphNode,
                                       CUgraph hGraph,
                                       const CUgraphNode *dependencies,
                                       size_t numDependencies,
                                       const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    if (!nodeParams) return CUDA_ERROR_INVALID_VALUE;

    CUresult r = graph_alloc_node(hGraph, phGraphNode, dependencies,
                                   numDependencies, CU_GRAPH_NODE_TYPE_KERNEL);
    if (r != CUDA_SUCCESS) return r;

    return deep_copy_kernel_params(*phGraphNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode,
                                       CUgraph hGraph,
                                       const CUgraphNode *dependencies,
                                       size_t numDependencies,
                                       const CUDA_MEMCPY3D *copyParams,
                                       CUcontext ctx) {
    (void)ctx;
    if (!copyParams) return CUDA_ERROR_INVALID_VALUE;

    CUresult r = graph_alloc_node(hGraph, phGraphNode, dependencies,
                                   numDependencies, CU_GRAPH_NODE_TYPE_MEMCPY);
    if (r != CUDA_SUCCESS) return r;

    (*phGraphNode)->params.memcpy = *copyParams;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode *phGraphNode,
                                       CUgraph hGraph,
                                       const CUgraphNode *dependencies,
                                       size_t numDependencies,
                                       const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                                       CUcontext ctx) {
    (void)ctx;
    if (!memsetParams) return CUDA_ERROR_INVALID_VALUE;

    CUresult r = graph_alloc_node(hGraph, phGraphNode, dependencies,
                                   numDependencies, CU_GRAPH_NODE_TYPE_MEMSET);
    if (r != CUDA_SUCCESS) return r;

    (*phGraphNode)->params.memset = *memsetParams;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode *phGraphNode,
                                     CUgraph hGraph,
                                     const CUgraphNode *dependencies,
                                     size_t numDependencies,
                                     const CUDA_HOST_NODE_PARAMS *nodeParams) {
    if (!nodeParams) return CUDA_ERROR_INVALID_VALUE;

    CUresult r = graph_alloc_node(hGraph, phGraphNode, dependencies,
                                   numDependencies, CU_GRAPH_NODE_TYPE_HOST);
    if (r != CUDA_SUCCESS) return r;

    (*phGraphNode)->params.host = *nodeParams;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode *phGraphNode,
                                      CUgraph hGraph,
                                      const CUgraphNode *dependencies,
                                      size_t numDependencies) {
    return graph_alloc_node(hGraph, phGraphNode, dependencies,
                             numDependencies, CU_GRAPH_NODE_TYPE_EMPTY);
}

/* ============================================================================
 * Dependency management
 * ============================================================================ */

CUresult CUDAAPI cuGraphAddDependencies(CUgraph hGraph,
                                         const CUgraphNode *from,
                                         const CUgraphNode *to,
                                         const CUgraphEdgeData *edgeData,
                                         size_t numDependencies) {
    (void)edgeData;
    if (!hGraph) return CUDA_ERROR_INVALID_VALUE;
    if (numDependencies == 0) return CUDA_SUCCESS;
    if (!from || !to) return CUDA_ERROR_INVALID_VALUE;

    for (size_t i = 0; i < numDependencies; i++) {
        int32_t from_idx = graph_find_node_index(hGraph, from[i]);
        int32_t to_idx   = graph_find_node_index(hGraph, to[i]);
        if (from_idx < 0 || to_idx < 0)
            return CUDA_ERROR_INVALID_VALUE;
        CUresult r = graph_add_edge_ptrs(hGraph->nodes, (uint32_t)from_idx, (uint32_t)to_idx);
        if (r != CUDA_SUCCESS) return r;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph hGraph,
                                            const CUgraphNode *from,
                                            const CUgraphNode *to,
                                            const CUgraphEdgeData *edgeData,
                                            size_t numDependencies) {
    (void)edgeData;
    if (!hGraph) return CUDA_ERROR_INVALID_VALUE;
    if (numDependencies == 0) return CUDA_SUCCESS;
    if (!from || !to) return CUDA_ERROR_INVALID_VALUE;

    for (size_t i = 0; i < numDependencies; i++) {
        int32_t from_idx = graph_find_node_index(hGraph, from[i]);
        int32_t to_idx   = graph_find_node_index(hGraph, to[i]);
        if (from_idx < 0 || to_idx < 0)
            return CUDA_ERROR_INVALID_VALUE;

        struct CUgraphNode_st *to_node   = hGraph->nodes[to_idx];
        struct CUgraphNode_st *from_node = hGraph->nodes[from_idx];

        for (uint32_t j = 0; j < to_node->dep_count; j++) {
            if (to_node->deps[j] == (uint32_t)from_idx) {
                to_node->deps[j] = to_node->deps[--to_node->dep_count];
                break;
            }
        }

        for (uint32_t j = 0; j < from_node->dependent_count; j++) {
            if (from_node->dependents[j] == (uint32_t)to_idx) {
                from_node->dependents[j] = from_node->dependents[--from_node->dependent_count];
                break;
            }
        }
    }
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Query functions
 * ============================================================================ */

CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes,
                                  size_t *numNodes) {
    if (!hGraph || !numNodes) return CUDA_ERROR_INVALID_VALUE;

    if (!nodes) {
        *numNodes = hGraph->node_count;
        return CUDA_SUCCESS;
    }

    size_t count = *numNodes < hGraph->node_count ? *numNodes : hGraph->node_count;
    for (size_t i = 0; i < count; i++)
        nodes[i] = hGraph->nodes[i];
    *numNodes = count;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from,
                                  CUgraphNode *to, CUgraphEdgeData *edgeData,
                                  size_t *numEdges) {
    (void)edgeData;
    if (!hGraph || !numEdges) return CUDA_ERROR_INVALID_VALUE;

    size_t total = 0;
    for (uint32_t i = 0; i < hGraph->node_count; i++)
        total += hGraph->nodes[i]->dependent_count;

    if (!from || !to) {
        *numEdges = total;
        return CUDA_SUCCESS;
    }

    size_t limit = *numEdges < total ? *numEdges : total;
    size_t idx = 0;
    for (uint32_t i = 0; i < hGraph->node_count && idx < limit; i++) {
        struct CUgraphNode_st *node = hGraph->nodes[i];
        for (uint32_t j = 0; j < node->dependent_count && idx < limit; j++) {
            from[idx] = hGraph->nodes[i];
            to[idx]   = hGraph->nodes[node->dependents[j]];
            idx++;
        }
    }
    *numEdges = idx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
    if (!hNode || !type) return CUDA_ERROR_INVALID_VALUE;
    *type = hNode->type;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode hNode,
                                             CUgraphNode *dependencies,
                                             CUgraphEdgeData *edgeData,
                                             size_t *numDependencies) {
    (void)edgeData;
    if (!hNode || !numDependencies) return CUDA_ERROR_INVALID_VALUE;

    if (!dependencies) {
        *numDependencies = hNode->dep_count;
        return CUDA_SUCCESS;
    }

    size_t count = *numDependencies < hNode->dep_count ?
                   *numDependencies : hNode->dep_count;
    *numDependencies = count;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                               CUgraphNode *dependentNodes,
                                               CUgraphEdgeData *edgeData,
                                               size_t *numDependentNodes) {
    (void)edgeData;
    if (!hNode || !numDependentNodes) return CUDA_ERROR_INVALID_VALUE;

    if (!dependentNodes) {
        *numDependentNodes = hNode->dependent_count;
        return CUDA_SUCCESS;
    }

    size_t count = *numDependentNodes < hNode->dependent_count ?
                   *numDependentNodes : hNode->dependent_count;
    *numDependentNodes = count;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Kernel node params
 * ============================================================================ */

CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode hNode,
                                             CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    if (!hNode || !nodeParams) return CUDA_ERROR_INVALID_VALUE;
    if (hNode->type != CU_GRAPH_NODE_TYPE_KERNEL)
        return CUDA_ERROR_INVALID_VALUE;
    *nodeParams = hNode->params.kernel;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode hNode,
                                             const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    if (!hNode || !nodeParams) return CUDA_ERROR_INVALID_VALUE;
    if (hNode->type != CU_GRAPH_NODE_TYPE_KERNEL)
        return CUDA_ERROR_INVALID_VALUE;

    free_kernel_params(hNode);
    return deep_copy_kernel_params(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec,
                                                 CUgraphNode hNode,
                                                 const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    if (!hGraphExec || !hNode || !nodeParams)
        return CUDA_ERROR_INVALID_VALUE;

    for (uint32_t i = 0; i < hGraphExec->node_count; i++) {
        if (&hGraphExec->nodes[i] == hNode) {
            if (hNode->type != CU_GRAPH_NODE_TYPE_KERNEL)
                return CUDA_ERROR_INVALID_VALUE;
            free_kernel_params(hNode);
            return deep_copy_kernel_params(hNode, nodeParams);
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
}

/* ============================================================================
 * Instantiation — deep copy + topological sort + assign semaphore values
 * ============================================================================ */

CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                     unsigned long long flags) {
    (void)flags;
    if (!phGraphExec || !hGraph) return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;

    uint32_t n = hGraph->node_count;

    struct CUgraphExec_st *exec = (struct CUgraphExec_st *)calloc(1, sizeof(*exec));
    if (!exec) return CUDA_ERROR_OUT_OF_MEMORY;

    exec->ctx = ctx;
    exec->node_count = n;

    if (n == 0) {
        *phGraphExec = exec;
        return CUDA_SUCCESS;
    }

    exec->nodes = (struct CUgraphNode_st *)calloc(n, sizeof(struct CUgraphNode_st));
    exec->sem_values = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (!exec->nodes || !exec->sem_values) {
        free(exec->nodes);
        free(exec->sem_values);
        free(exec);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    /* Kahn's algorithm for topological sort */
    uint32_t *in_degree  = (uint32_t *)calloc(n, sizeof(uint32_t));
    uint32_t *queue      = (uint32_t *)calloc(n, sizeof(uint32_t));
    uint32_t *order      = (uint32_t *)calloc(n, sizeof(uint32_t));
    uint32_t *old_to_new = (uint32_t *)calloc(n, sizeof(uint32_t));

    if (!in_degree || !queue || !order || !old_to_new) {
        free(in_degree); free(queue); free(order); free(old_to_new);
        free(exec->nodes); free(exec->sem_values); free(exec);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < n; i++)
        in_degree[i] = hGraph->nodes[i]->dep_count;

    uint32_t head = 0, tail = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (in_degree[i] == 0)
            queue[tail++] = i;
    }

    uint32_t sorted_count = 0;
    while (head < tail) {
        uint32_t cur = queue[head++];
        order[sorted_count] = cur;
        old_to_new[cur] = sorted_count;
        sorted_count++;

        struct CUgraphNode_st *node = hGraph->nodes[cur];
        for (uint32_t j = 0; j < node->dependent_count; j++) {
            uint32_t dep_idx = node->dependents[j];
            if (--in_degree[dep_idx] == 0)
                queue[tail++] = dep_idx;
        }
    }

    free(in_degree);
    free(queue);

    if (sorted_count != n) {
        free(order); free(old_to_new);
        free(exec->nodes); free(exec->sem_values); free(exec);
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Deep-copy nodes in topological order, remapping dep indices */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t orig = order[i];
        CUresult r = deep_copy_node(&exec->nodes[i], hGraph->nodes[orig]);
        if (r != CUDA_SUCCESS) {
            for (uint32_t j = 0; j < i; j++)
                free_node_internals(&exec->nodes[j]);
            free(order); free(old_to_new);
            free(exec->nodes); free(exec->sem_values); free(exec);
            return r;
        }

        struct CUgraphNode_st *src_node = hGraph->nodes[orig];
        if (src_node->dep_count > 0) {
            exec->nodes[i].deps = (uint32_t *)malloc(src_node->dep_count * sizeof(uint32_t));
            if (!exec->nodes[i].deps) {
                for (uint32_t j = 0; j <= i; j++)
                    free_node_internals(&exec->nodes[j]);
                free(order); free(old_to_new);
                free(exec->nodes); free(exec->sem_values); free(exec);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            exec->nodes[i].dep_count = src_node->dep_count;
            exec->nodes[i].dep_capacity = src_node->dep_count;
            for (uint32_t j = 0; j < src_node->dep_count; j++)
                exec->nodes[i].deps[j] = old_to_new[src_node->deps[j]];
        }

        if (src_node->dependent_count > 0) {
            exec->nodes[i].dependents = (uint32_t *)malloc(
                src_node->dependent_count * sizeof(uint32_t));
            if (!exec->nodes[i].dependents) {
                for (uint32_t j = 0; j <= i; j++)
                    free_node_internals(&exec->nodes[j]);
                free(order); free(old_to_new);
                free(exec->nodes); free(exec->sem_values); free(exec);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            exec->nodes[i].dependent_count = src_node->dependent_count;
            exec->nodes[i].dependent_capacity = src_node->dependent_count;
            for (uint32_t j = 0; j < src_node->dependent_count; j++)
                exec->nodes[i].dependents[j] = old_to_new[src_node->dependents[j]];
        }
    }

    free(order);
    free(old_to_new);

    *phGraphExec = exec;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Execution — walk sorted nodes, execute with timeline semaphore sync
 * ============================================================================ */

CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    (void)hStream;
    if (!hGraphExec) return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = hGraphExec->ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;

    for (uint32_t i = 0; i < hGraphExec->node_count; i++) {
        struct CUgraphNode_st *node = &hGraphExec->nodes[i];

        uint64_t wait_value = 0;
        for (uint32_t j = 0; j < node->dep_count; j++) {
            uint32_t dep = node->deps[j];
            if (hGraphExec->sem_values[dep] > wait_value)
                wait_value = hGraphExec->sem_values[dep];
        }

        uint64_t signal_value = ++ctx->timeline_value;
        hGraphExec->sem_values[i] = signal_value;

        CUresult r = CUDA_SUCCESS;

        switch (node->type) {
        case CU_GRAPH_NODE_TYPE_KERNEL: {
            CUDA_KERNEL_NODE_PARAMS *kp = &node->params.kernel;

            if (wait_value > 0) {
                VkSemaphoreWaitInfo wi = {0};
                wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wi.semaphoreCount = 1;
                wi.pSemaphores = &ctx->timeline_sem;
                wi.pValues = &wait_value;
                g_cuvk.vk.vkWaitSemaphores(ctx->device, &wi, UINT64_MAX);
            }

            r = cuLaunchKernel(kp->func,
                               kp->gridDimX, kp->gridDimY, kp->gridDimZ,
                               kp->blockDimX, kp->blockDimY, kp->blockDimZ,
                               kp->sharedMemBytes, NULL,
                               kp->kernelParams, kp->extra);
            if (r != CUDA_SUCCESS) return r;

            VkSemaphoreSignalInfo sig = {0};
            sig.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            sig.semaphore = ctx->timeline_sem;
            sig.value = signal_value;
            g_cuvk.vk.vkSignalSemaphore(ctx->device, &sig);
            break;
        }

        case CU_GRAPH_NODE_TYPE_MEMCPY: {
            CUDA_MEMCPY3D *mc = &node->params.memcpy;

            if (wait_value > 0) {
                VkSemaphoreWaitInfo wi = {0};
                wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wi.semaphoreCount = 1;
                wi.pSemaphores = &ctx->timeline_sem;
                wi.pValues = &wait_value;
                g_cuvk.vk.vkWaitSemaphores(ctx->device, &wi, UINT64_MAX);
            }

            size_t byte_count = mc->WidthInBytes * (mc->Height ? mc->Height : 1) *
                                (mc->Depth ? mc->Depth : 1);

            if (mc->srcMemoryType == CU_MEMORYTYPE_HOST &&
                mc->dstMemoryType == CU_MEMORYTYPE_DEVICE) {
                r = cuMemcpyHtoD_v2(mc->dstDevice + mc->dstXInBytes,
                                     (const char *)mc->srcHost + mc->srcXInBytes,
                                     byte_count);
            } else if (mc->srcMemoryType == CU_MEMORYTYPE_DEVICE &&
                       mc->dstMemoryType == CU_MEMORYTYPE_HOST) {
                r = cuMemcpyDtoH_v2((char *)mc->dstHost + mc->dstXInBytes,
                                     mc->srcDevice + mc->srcXInBytes,
                                     byte_count);
            } else if (mc->srcMemoryType == CU_MEMORYTYPE_DEVICE &&
                       mc->dstMemoryType == CU_MEMORYTYPE_DEVICE) {
                r = cuMemcpyDtoD_v2(mc->dstDevice + mc->dstXInBytes,
                                     mc->srcDevice + mc->srcXInBytes,
                                     byte_count);
            } else {
                r = CUDA_ERROR_INVALID_VALUE;
            }
            if (r != CUDA_SUCCESS) return r;

            VkSemaphoreSignalInfo sig = {0};
            sig.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            sig.semaphore = ctx->timeline_sem;
            sig.value = signal_value;
            g_cuvk.vk.vkSignalSemaphore(ctx->device, &sig);
            break;
        }

        case CU_GRAPH_NODE_TYPE_MEMSET: {
            CUDA_MEMSET_NODE_PARAMS *ms = &node->params.memset;

            if (wait_value > 0) {
                VkSemaphoreWaitInfo wi = {0};
                wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wi.semaphoreCount = 1;
                wi.pSemaphores = &ctx->timeline_sem;
                wi.pValues = &wait_value;
                g_cuvk.vk.vkWaitSemaphores(ctx->device, &wi, UINT64_MAX);
            }

            size_t count = ms->width * (ms->height ? ms->height : 1);

            switch (ms->elementSize) {
            case 1:
                r = cuMemsetD8_v2(ms->dst, (unsigned char)(ms->value & 0xFF), count);
                break;
            case 2:
                r = cuMemsetD16_v2(ms->dst, (unsigned short)(ms->value & 0xFFFF), count);
                break;
            case 4:
                r = cuMemsetD32_v2(ms->dst, ms->value, count);
                break;
            default:
                r = CUDA_ERROR_INVALID_VALUE;
            }
            if (r != CUDA_SUCCESS) return r;

            VkSemaphoreSignalInfo sig = {0};
            sig.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            sig.semaphore = ctx->timeline_sem;
            sig.value = signal_value;
            g_cuvk.vk.vkSignalSemaphore(ctx->device, &sig);
            break;
        }

        case CU_GRAPH_NODE_TYPE_HOST: {
            if (wait_value > 0) {
                VkSemaphoreWaitInfo wi = {0};
                wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wi.semaphoreCount = 1;
                wi.pSemaphores = &ctx->timeline_sem;
                wi.pValues = &wait_value;
                g_cuvk.vk.vkWaitSemaphores(ctx->device, &wi, UINT64_MAX);
            }

            cuCtxSynchronize();

            if (node->params.host.fn)
                node->params.host.fn(node->params.host.userData);

            VkSemaphoreSignalInfo sig = {0};
            sig.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            sig.semaphore = ctx->timeline_sem;
            sig.value = signal_value;
            g_cuvk.vk.vkSignalSemaphore(ctx->device, &sig);
            break;
        }

        case CU_GRAPH_NODE_TYPE_EMPTY:
        default: {
            if (wait_value > 0) {
                VkSemaphoreWaitInfo wi = {0};
                wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wi.semaphoreCount = 1;
                wi.pSemaphores = &ctx->timeline_sem;
                wi.pValues = &wait_value;
                g_cuvk.vk.vkWaitSemaphores(ctx->device, &wi, UINT64_MAX);
            }

            VkSemaphoreSignalInfo sig = {0};
            sig.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            sig.semaphore = ctx->timeline_sem;
            sig.value = signal_value;
            g_cuvk.vk.vkSignalSemaphore(ctx->device, &sig);
            break;
        }
        }
    }

    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    (void)hGraphExec; (void)hStream;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Cleanup
 * ============================================================================ */

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
    if (!hGraphExec) return CUDA_ERROR_INVALID_VALUE;

    for (uint32_t i = 0; i < hGraphExec->node_count; i++)
        free_node_internals(&hGraphExec->nodes[i]);
    free(hGraphExec->nodes);
    free(hGraphExec->sem_values);
    free(hGraphExec);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Graph update
 * ============================================================================ */

CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                    CUgraphExecUpdateResultInfo *resultInfo) {
    if (!hGraphExec || !hGraph) return CUDA_ERROR_INVALID_VALUE;

    for (uint32_t i = 0; i < hGraphExec->node_count; i++)
        free_node_internals(&hGraphExec->nodes[i]);
    free(hGraphExec->nodes);
    free(hGraphExec->sem_values);
    hGraphExec->nodes = NULL;
    hGraphExec->sem_values = NULL;
    hGraphExec->node_count = 0;

    CUgraphExec temp = NULL;
    CUresult r = cuGraphInstantiate(&temp, hGraph, 0);
    if (r != CUDA_SUCCESS) {
        if (resultInfo)
            resultInfo->result = CU_GRAPH_EXEC_UPDATE_ERROR;
        return r;
    }

    hGraphExec->nodes = temp->nodes;
    hGraphExec->node_count = temp->node_count;
    hGraphExec->sem_values = temp->sem_values;
    free(temp);

    if (resultInfo)
        resultInfo->result = CU_GRAPH_EXEC_UPDATE_SUCCESS;

    return CUDA_SUCCESS;
}
