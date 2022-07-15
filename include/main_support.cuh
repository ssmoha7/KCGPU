#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"
#include "../include/GraphDataStructure.cuh"



template<typename T>
__global__ void init(graph::COOCSRGraph_d<T> g, T* asc, bool* keep)
{
    uint tx = threadIdx.x;
    uint bx = blockIdx.x;

    uint ptx = tx + bx * blockDim.x;

    for (uint i = ptx; i < g.numEdges; i += blockDim.x * gridDim.x)
    {
        const T src = g.rowInd[i];
        const T dst = g.colInd[i];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;


        keep[i] = (dstLen < srcLen || ((dstLen == srcLen) && src < dst));// Some simple graph orientation
        //src[i] < dst[i];
        asc[i] = i;
    }
}


template<typename T, typename PeelT>
__global__ void initAsc(T* asc, T count)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < count; i += blockDim.x * gridDim.x)
    {

        asc[i] = i;
    }
}

template<typename T, typename PeelT>
__global__ void init(graph::COOCSRGraph_d<T> g, T* asc, bool* keep, PeelT* degeneracy)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numEdges; i += blockDim.x * gridDim.x)
    {
        const T src = g.rowInd[i];
        const T dst = g.colInd[i];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        keep[i] = false;
        if (degeneracy[src] < degeneracy[dst])
            keep[i] = true;
        else if (degeneracy[src] == degeneracy[dst] && dstLen < srcLen)
            keep[i] = true;
        else if (degeneracy[src] == degeneracy[dst] && dstLen == srcLen && src < dst)
            keep[i] = true;

        asc[i] = i;
    }
}


template<typename T, typename PeelT>
__global__ void init(graph::COOCSRGraph_d<T> g, T* asc, bool* keep, PeelT* nodeDegen, T* nodePriority)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numEdges; i += blockDim.x * gridDim.x)
    {
        const T src = g.rowInd[i];
        const T dst = g.colInd[i];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        keep[i] = false;
        // if (nodeDegen[src] < nodeDegen[dst])
        // 	keep[i] = true;
        // else if (nodeDegen[src] == nodeDegen[dst])
        {
            if (nodePriority[src] < nodePriority[dst])
                keep[i] = true;
            else if (nodePriority[src] == nodePriority[dst])
            {
                if (src < dst)
                    keep[i] = true;
            }
        }


        asc[i] = i;
    }
}



//Overloaded form Ktruss
template<typename T>
__global__
void warp_detect_deleted_edges(
    T* rowPtr, T numRows,
    bool* keep,
    T* histogram)
{

    __shared__ uint32_t cnts[WARPS_PER_BLOCK];

    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gtnum = blockDim.x * gridDim.x;
    auto gwid = gtid >> WARP_BITS;
    auto gwnum = gtnum >> WARP_BITS;
    auto lane = threadIdx.x & WARP_MASK;
    auto lwid = threadIdx.x >> WARP_BITS;

    for (auto u = gwid; u < numRows; u += gwnum) {
        if (0 == lane)
            cnts[lwid] = 0;
        __syncwarp();

        auto start = rowPtr[u];
        auto end = rowPtr[u + 1];
        for (auto v_idx = start + lane; v_idx < end; v_idx += WARP_SIZE)
        {
            if (keep[v_idx])
                atomicAdd(&cnts[lwid], 1);
        }
        __syncwarp();

        if (0 == lane)
            histogram[u] = cnts[lwid];
    }
}

template<typename T>
__global__ void InitEid(T numEdges, T* asc, T* newSrc, T* newDst, T* rowPtr, T* colInd, T* eid)
{
    uint tx = threadIdx.x;
    uint bx = blockIdx.x;

    uint ptx = tx + bx * blockDim.x;

    for (uint i = ptx; i < numEdges; i += blockDim.x * gridDim.x)
    {
        //i : is the new index of the edge !!
        T srcnode = newSrc[i];
        T dstnode = newDst[i];



        T olduV = asc[i];
        T oldUv = getEdgeId(rowPtr, colInd, dstnode, srcnode); //Search for it please !!


        eid[olduV] = i;
        eid[oldUv] = i;
    }
}




struct Node
{
    uint val;
    int i;
    int r;
    int l;
    int p;
};

template<typename T>
__global__ void split_inverse(bool* keep, T m)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < m; i += blockDim.x * gridDim.x)
    {
        keep[i] ^= 1;
    }
}

template<typename T>
__global__ void split_acc(graph::COOCSRGraph_d<T> g, T* split_ptr)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numNodes; i += blockDim.x * gridDim.x)
    {
        split_ptr[i] += g.rowPtr[i];
    }
}

template<typename T>
__global__ void split_child(graph::COOCSRGraph_d<T> g, T* tmp_row, T* tmp_col, T* split_col, T* split_ptr)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numEdges / 2; i += blockDim.x * gridDim.x)
    {
        const T src = tmp_row[i];
        const T dst = tmp_col[i];
        split_col[g.rowPtr[src + 1] - (split_ptr[src + 1] - i)] = dst;
    }
}

template<typename T>
__global__ void split_parent(graph::COOCSRGraph_d<T> g, T* tmp_row, T* tmp_col, T* split_col, T* split_ptr)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numEdges / 2; i += blockDim.x * gridDim.x)
    {
        const T src = tmp_row[i];
        const T dst = tmp_col[i];
        split_col[g.rowPtr[src] + (i - split_ptr[src])] = dst;
    }
}