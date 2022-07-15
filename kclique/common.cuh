#pragma once

#define NodeStartLevelOr 3
#define EdgeStartLevelOr 4
#define NodeStartLevelPiv 2
#define EdgeStartLevelPiv 3

__constant__ uint KCCOUNT;
__constant__ uint MAXDEG;
__constant__ uint PARTSIZE;
__constant__ uint NUMPART;
__constant__ uint MAXLEVEL;
__constant__ uint NUMDIVS;
__constant__ uint CBPSM;


///////////////////////////// Global ////////////////////////////////////////////
template <typename T, int BLOCK_DIM_X>
__global__ void getEdgeDegree_kernel(graph::COOCSRGraph_d<T> g, T* edgePtr, T* maxDegree)
{
    const T gtid = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;

    if (gtid < g.numEdges)
    {
        T src = g.rowInd[gtid];
        T dst = g.colInd[gtid];

        T srcDeg = g.rowPtr[src + 1] - g.rowPtr[src];
        T dstDeg = g.rowPtr[dst + 1] - g.rowPtr[dst];

        degree = srcDeg > dstDeg ? srcDeg : dstDeg;
        edgePtr[gtid] = degree;
    }

    __syncthreads();
    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());


    if (threadIdx.x == 0)
        atomicMax(maxDegree, aggregate);


}
template<typename T, int BLOCK_DIM_X>
__global__ void getNodeDegree_kernel(T* nodeDegree, graph::COOCSRGraph_d<T> g, T* maxDegree)
{
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;
    if (gtid < g.numNodes)
    {
        degree = g.rowPtr[gtid + 1] - g.rowPtr[gtid];
        nodeDegree[gtid] = degree;
    }

    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());
    if (threadIdx.x == 0)
        atomicMax(maxDegree, aggregate);
}


////////////////////////////// Device ////////////////////////////////////////////
__device__ __inline__ uint32_t __mysmid() {
    unsigned int r;
    asm("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

template<typename T>
__device__ __inline__ void reserve_space(uint32_t& sm_id, uint32_t& levelPtr, T* levelStats)
{
    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
}

template<typename T>
__device__ __inline__ void release_space(uint32_t sm_id, uint32_t levelPtr, T* levelStats)
{
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template<typename T>
__device__ T get_next_sibling_index(T* from, T maskIndex, T& maskBlock)
{
    T newIndex = __ffs(from[maskBlock] & maskIndex);
    while(newIndex == 0)
    {
        maskIndex = 0xFFFFFFFF;
        maskBlock++;
        newIndex = __ffs(from[maskBlock] & maskIndex);
    }
    newIndex =  32*maskBlock + newIndex - 1;
    return newIndex;
}

template<typename T, uint CPARTSIZE>
__device__ T get_thread_group_mask(T wx)
{
    T partMask = (1 << CPARTSIZE) - 1;
    partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
    return partMask;
}

