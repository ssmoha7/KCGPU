#pragma once
#include "kckernels.cuh"

/* AtomicAdd to SharedMemory with accumulation through levels + Deep:
 *		where we accumulate the counter when we backtrack to the previous level,
 *		instead of incrementing all counters included by the clique by 1 when a k-clique is found.
 *      This kernel is built on top of baseline_deep.
 * Extra shared memory: __shared__ uint64 local_clique_count[1024];
 *						__shared__ uint64 root_count;
 *						__shared__ uint64 level_local_clique_count[numPartitions][9];
 * Difference from baseline_deep:
 * 		Clear local_clique counter in shared memory:
 *			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				local_clique_count[idx] = 0;
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				root_count = 0;
 *			}
 *			__syncthreads();
 *		Clear the first level when starting from a new branch:
 * 			level_local_clique_count[wx][0] = 0;
 * 		Whenever we find a clique:
 *			atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 3] - 1]], 1);
 *			level_local_clique_count[wx][l[wx] - 3] ++;
 * 		When backtracking:
 *			uint64 cur = level_local_clique_count[wx][l[wx] - 3];
 *			level_local_clique_count[wx][l[wx] - 4] += cur;
 *			atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 4] - 1], cur);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&root_count, clique_count[wx]);
 *			atomicAdd(&local_clique_count[j], clique_count[wx]);
 *		After exploring a source node:
 * 			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				atomicAdd(&cpn[src], root_count);
 *			}
 *			__syncthreads();
 */
/*
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_lazy_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    // shared memory counter for local_clique
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;
    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

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
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                level_local_clique_count[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                    {
                        clique_count[wx] ++;
                        atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 3] - 1], 1);
                        level_local_clique_count[wx][l[wx] - 3] ++;
                    }	
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                         atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 4] - 1], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
*/

/* AtomicAdd to GlobalMemory with accumulation through levels + Deep:
 * 		Use atomicAdd to increase the local_clique counter in global memory,
 *		where we accumulate the counter when we backtrack to the previous level,
 *		instead of incrementing all counters included by the clique by 1 when a k-clique is found.
 *      This kernel is built on top of baseline_deep.
 * Extra shared memory:
 * 		__shared__ uint64 level_local_clique_count[numPartitions][9];
 * Difference from baseline_deep:
 * 		Clear the first level when starting from a new branch:
 * 			level_local_clique_count[wx][0] = 0;
 * 		Whenever we find a clique:
 *			atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 3] - 1]], 1);
 *			level_local_clique_count[wx][l[wx] - 3] ++;
 *		Clear the new level When branching:
 *			level_local_clique_count[wx][l[wx] - 3] = 0;
 *		When backtracking:
 *			uint64 cur = level_local_clique_count[wx][l[wx] - 3];
 *			level_local_clique_count[wx][l[wx] - 4] += cur;
 *			atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 4] - 1]], cur);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
 *			atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
 */
/*
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_lazy_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

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
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                level_local_clique_count[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                    {
                        clique_count[wx] ++;
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 3] - 1]], 1);
                        level_local_clique_count[wx][l[wx] - 3] ++;
                    }
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 4] - 1]], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
*/

/* AtomicAdd to SharedMemory + Deep: Use atomicAdd to increase the local_clique counter in shared memory,
 *                                   whenever we find a clique. And increase the global counter later.
 *                                   This kernel is built on top of baseline_deep.
 * Extra shared memory: __shared__ uint64 local_clique_count[1024];
 *						__shared__ uint64 root_count;
 * Difference from baseline_deep:
 * 		Clear local_clique counter in shared memory:
 *			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				local_clique_count[idx] = 0;
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				root_count = 0;
 *			}
 *			__syncthreads();
 * 		Whenever we find a clique:
 *			for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
 *			{
 *				atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], 1);
 *			}
 *			__syncwarp(partMask);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&root_count, clique_count[wx]);
 *			atomicAdd(&local_clique_count[j], clique_count[wx]);
 *		After exploring a source node:
 * 			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				atomicAdd(&cpn[src], root_count);
 *			}
 *			__syncthreads();
 */
/*
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_direct_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    // shared memory counter for local_clique
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;

    __syncthreads();

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
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                // accumulate local counter
                if (l[wx] == KCCOUNT)
                {
                    for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
                    {
                        atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], 1);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                        clique_count[wx] ++;
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
*/

/* AtomicAdd to GlobalMemory + Deep: Use atomicAdd to increase the local_clique counter in global memory,
 *                                   whenever we find a clique. This kernel is built on top of baseline_deep.
 * Extra shared memory: none
 * Difference from baseline_deep:
 * 		Whenever we find a clique:
 *			for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
 *			{
 *				atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][k] - 1]], 1);
 *			}
 *			__syncwarp(partMask);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
 *			atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
 */
/*
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_direct_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __syncthreads();

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
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                // accumulate local counter
                if (l[wx] == KCCOUNT)
                {
                    for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
                    {
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][k] - 1]], 1);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                        clique_count[wx] ++;
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
*/

/* Baseline + Deep
 * Intuition: In the original k-kclique counting kernel, we didn't search the k'th level,
 *            but stop at the (k - 1)'th level and accumulate the number of neighbors into the clique counter.
 *            This baseline_deep kernel reformulates the code to get the total clique count by getting into the k'th level,
 *            and increase the counted by 1 for each on-bit in the binary encoded list.
 * Expect:    The execution time of this kernel for k-clique should be similar as that
 *            of counting (k + 1)-clique in the original kernel. But should be a bit faster since we skip the last intersection 
 *            done in the original kernel. 
 */
/*
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_baseline_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __syncthreads();

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
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                        clique_count[wx] ++;
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
*/

/* AtomicAdd to SharedMemory with accumulation through levels + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;
    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;

            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            //Local Specific
            root_count = 0;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (T idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
                level_local_clique_count[wx][0] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            //Local count: increment where ever you find bit value = 1
            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&local_clique_count[idx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&local_clique_count[idx], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                        atomicAdd(&local_clique_count[newIndex], warpCount);
                        level_local_clique_count[wx][l[wx] - 3] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                        atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 4] - 1], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (T idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to GlobalMemory with accumulation through levels + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
                level_local_clique_count[wx][0] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                        atomicAdd(&cpn[g.colInd[srcStart + newIndex]], warpCount);
                        level_local_clique_count[wx][l[wx] - 3] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 4] - 1]], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to SharedMemory Direct + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ uint64 shared_warp_count[numPartitions];
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;
    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;

            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            //Local Specific
            root_count = 0;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (T idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&local_clique_count[idx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    shared_warp_count[wx] = warpCount;
                }
                __syncwarp(partMask);

                if (l[wx] + 1 == KCCOUNT && shared_warp_count[wx] != 0)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&local_clique_count[idx], 1);
                        }
                    }
                    for (T k = lx; k < KCCOUNT - 3; k += CPARTSIZE)
                    {
                        atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], shared_warp_count[wx]);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (T idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to GlobalMemory Direct + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ uint64 shared_warp_count[numPartitions];
    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    shared_warp_count[wx] = warpCount;
                }
                __syncwarp(partMask);

                if (l[wx] + 1 == KCCOUNT && shared_warp_count[wx] != 0)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                        }
                    }
                    for (T k = lx; k < KCCOUNT - 3; k += CPARTSIZE)
                    {
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][k] - 1]], shared_warp_count[wx]);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}


/* Baseline Node + Loop */
/*
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_baseline_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __syncthreads();

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
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&clique_count[wx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&clique_count[wx], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {

                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
*/

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_local_globalmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int lx = threadIdx.x % CPARTSIZE;
    
    __shared__ unsigned short level_count[numPartitions][8];
    __shared__ unsigned short level_prev_index[numPartitions][8];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ T src2, src2Start, src2Len;
    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;
    __shared__ uint64 shared_warp_count[numPartitions];
    
    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;
            tc = 0;
        }
        __syncthreads();

        //get tri list: by block :!!
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        if(KCCOUNT == 3)
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(counter, scounter);
                atomicAdd(&cpn[src], scounter);
                atomicAdd(&cpn[src2], scounter);
            }
            __syncthreads();

            for (T k = threadIdx.x; k < scounter; k += blockDim.x)
            {
                atomicAdd(&cpn[tri[k]], 1);
            }
            __syncthreads();
        }

        //Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        if(lx == 0)
            wtc[wx] = atomicAdd(&(tc), 1);
        __syncwarp(partMask);

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 8];

        while(wtc[wx] < scounter)
        //for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            T j = wtc[wx];
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 4;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
                {
                    level_count[wx][0] = warpCount;
                }
            }
             __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&cpn[tri[idx]], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 4] > 0)
            {
                // First Index
                T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
                T* to = &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
                
                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 4] = newIndex + 1;
                    level_count[wx][l[wx] - 4]--; 
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    shared_warp_count[wx] = warpCount;
                }
                __syncwarp(partMask);

                if (l[wx] + 1 == KCCOUNT && shared_warp_count[wx] != 0)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&cpn[tri[idx]], 1);
                        }
                    }
                    for (T k = lx; k < KCCOUNT - 4; k += CPARTSIZE)
                    {
                        atomicAdd(&cpn[tri[level_prev_index[wx][k] - 1]], shared_warp_count[wx]);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        level_prev_index[wx][l[wx] - 4] = 0;
                    }
                
                    while (l[wx] > 4 && level_count[wx][l[wx] - 4] <= 0)
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[src], clique_count[wx]);
                atomicAdd(&cpn[src2], clique_count[wx]);
                atomicAdd(&cpn[tri[j]], clique_count[wx]);
                wtc[wx] = atomicAdd(&(tc), 1);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_local_globalmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int lx = threadIdx.x % CPARTSIZE;
    
    __shared__ unsigned short level_count[numPartitions][8];
    __shared__ unsigned short level_prev_index[numPartitions][8];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ T src2, src2Start, src2Len;
    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;
    
    __shared__ uint64 level_local_clique_count[numPartitions][8];
    
    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;
            tc = 0;
        }
        __syncthreads();

        //get tri list: by block :!!
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        if(KCCOUNT == 3)
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(counter, scounter);
                atomicAdd(&cpn[src], scounter);
                atomicAdd(&cpn[src2], scounter);
            }
            __syncthreads();

            for (T k = threadIdx.x; k < scounter; k += blockDim.x)
            {
                atomicAdd(&cpn[tri[k]], 1);
            }
            __syncthreads();
        }

        //Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        if(lx == 0)
            wtc[wx] = atomicAdd(&(tc), 1);
        __syncwarp(partMask);

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 8];

        while(wtc[wx] < scounter)
        //for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            T j = wtc[wx];
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 4;
                clique_count[wx] = 0;
                level_local_clique_count[wx][0] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
                {
                    level_count[wx][0] = warpCount;
                }
            }
             __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&cpn[tri[idx]], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 4] > 0)
            {
                // First Index
                T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
                T* to = &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
                
                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 4] = newIndex + 1;
                    level_count[wx][l[wx] - 4]--; 
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&cpn[tri[idx]], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                        atomicAdd(&cpn[tri[newIndex]], warpCount);
                        level_local_clique_count[wx][l[wx] - 4] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        level_prev_index[wx][l[wx] - 4] = 0;
                        level_local_clique_count[wx][l[wx] - 4] = 0;
                    }
                
                    while (l[wx] > 4 && level_count[wx][l[wx] - 4] <= 0)
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 4];
                        level_local_clique_count[wx][l[wx] - 5] += cur;
                        atomicAdd(&cpn[tri[level_prev_index[wx][l[wx] - 5] - 1]], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[src], clique_count[wx]);
                atomicAdd(&cpn[src2], clique_count[wx]);
                atomicAdd(&cpn[tri[j]], clique_count[wx]);
                wtc[wx] = atomicAdd(&(tc), 1);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_local_sharedmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int lx = threadIdx.x % CPARTSIZE;
    
    __shared__ unsigned short level_count[numPartitions][8];
    __shared__ unsigned short level_prev_index[numPartitions][8];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ T src2, src2Start, src2Len;
    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;
    __shared__ uint64 shared_warp_count[numPartitions];
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 src_count, src2_count;

    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;
            tc = 0;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (T idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            src_count = 0;
            src2_count = 0;
        }
        __syncthreads();

        //get tri list: by block :!!
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        if(KCCOUNT == 3)
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(counter, scounter);
                atomicAdd(&src_count, scounter);
                atomicAdd(&src2_count, scounter);
            }
            __syncthreads();

            for (T k = threadIdx.x; k < scounter; k += blockDim.x)
            {
                atomicAdd(&local_clique_count[k], 1);
            }
            __syncthreads();
        }

        //Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        if(lx == 0)
            wtc[wx] = atomicAdd(&(tc), 1);
        __syncwarp(partMask);

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 8];

        while(wtc[wx] < scounter)
        //for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            T j = wtc[wx];
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 4;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
                {
                    level_count[wx][0] = warpCount;
                }
            }
             __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&local_clique_count[idx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 4] > 0)
            {
                // First Index
                T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
                T* to = &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
                
                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 4] = newIndex + 1;
                    level_count[wx][l[wx] - 4]--; 
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    shared_warp_count[wx] = warpCount;
                }
                __syncwarp(partMask);

                if (l[wx] + 1 == KCCOUNT && shared_warp_count[wx] != 0)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&local_clique_count[idx], 1);
                        }
                    }
                    for (T k = lx; k < KCCOUNT - 4; k += CPARTSIZE)
                    {
                        atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], shared_warp_count[wx]);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        level_prev_index[wx][l[wx] - 4] = 0;
                    }
                
                    while (l[wx] > 4 && level_count[wx][l[wx] - 4] <= 0)
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&src_count, clique_count[wx]);
                atomicAdd(&src2_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
                wtc[wx] = atomicAdd(&(tc), 1);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (T idx = threadIdx.x; idx < scounter; idx += blockDim.x)
        {
            atomicAdd(&cpn[tri[idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], src_count);
            atomicAdd(&cpn[src2], src2_count);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_local_sharedmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int lx = threadIdx.x % CPARTSIZE;
    
    __shared__ unsigned short level_count[numPartitions][8];
    __shared__ unsigned short level_prev_index[numPartitions][8];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ T src2, src2Start, src2Len;
    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;
    
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 src_count, src2_count;
    __shared__ uint64 level_local_clique_count[numPartitions][8];
    
    __syncthreads();

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;
            tc = 0;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (T idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            src_count = 0;
            src2_count = 0;
        }
        __syncthreads();

        //get tri list: by block :!!
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        if(KCCOUNT == 3)
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(counter, scounter);
                atomicAdd(&src_count, scounter);
                atomicAdd(&src2_count, scounter);
            }
            __syncthreads();

            for (T k = threadIdx.x; k < scounter; k += blockDim.x)
            {
                atomicAdd(&local_clique_count[k], 1);
            }
            __syncthreads();
        }

        //Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        if(lx == 0)
            wtc[wx] = atomicAdd(&(tc), 1);
        __syncwarp(partMask);

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 8];

        while(wtc[wx] < scounter)
        //for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            T j = wtc[wx];
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 4;
                clique_count[wx] = 0;
                level_local_clique_count[wx][0] = 0;
            }

            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
                {
                    level_count[wx][0] = warpCount;
                }
            }
             __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&local_clique_count[idx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 4] > 0)
            {
                // First Index
                T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
                T* to = &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
                
                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 4] = newIndex + 1;
                    level_count[wx][l[wx] - 4]--; 
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&local_clique_count[idx], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                        atomicAdd(&local_clique_count[newIndex], warpCount);
                        level_local_clique_count[wx][l[wx] - 4] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        level_prev_index[wx][l[wx] - 4] = 0;
                        level_local_clique_count[wx][l[wx] - 4] = 0;
                    }
                
                    while (l[wx] > 4 && level_count[wx][l[wx] - 4] <= 0)
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 4];
                        level_local_clique_count[wx][l[wx] - 5] += cur;
                        atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 5] - 1], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&src_count, clique_count[wx]);
                atomicAdd(&src2_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
                wtc[wx] = atomicAdd(&(tc), 1);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (T idx = threadIdx.x; idx < scounter; idx += blockDim.x)
        {
            atomicAdd(&cpn[tri[idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], src_count);
            atomicAdd(&cpn[src2], src2_count);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_pivot_count_local_base(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ bool partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ T lastMask_i, lastMask_ii;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        // block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
        }
        __syncthreads();

        // Encode Clear
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
            partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();
    
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }	
        }
        if(lx == 0)
        {
            atomicMax(&(maxIntersection), maxCount[wx]);
        }
        __syncthreads();

        if(lx == 0)
        {
            if(maxIntersection == maxCount[wx])
            {
                atomicMin(&(level_pivot[0]), maxIndex[wx]);
            }
        }
        __syncthreads();

        // Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
    
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = m;
            warpCount += __popc(pl[j]);
        }
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) - 1);
            T newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l - 2) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-1] = drop[l-2];
                if(newIndex == level_pivot[l-2])
                    drop[l-1] = drop[l-2] + 1;
            }
            __syncthreads();

            if(l - drop[l-1] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 2)]);
                T* to =  &(cl[num_divs_local * (l - 1)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = srcLen + 1; // make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < srcLen; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == srcLen + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; // shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();
                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr = nCR[drop[l-1] * 401 + c];
                            atomicAdd(counter, ncr);
                        }
                        
                        while (l > 2 && level_count[l - 2] == 0)
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	                            
                            while (l > 2 && level_count[l - 2] == 0)
                            {
                                (l)--;
                            }
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-2] = 0;
                            level_prev_index[l-2] = 0;
                        }
                        __syncthreads();

                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 2]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_pivot_count_local_globalmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ bool partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ T lastMask_i, lastMask_ii;

    __shared__ uint64 local_choose_pivot, local_choose_hold;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        // block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
        }
        __syncthreads();

        // Encode Clear
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
            partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();
    
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }	
        }
        if(lx == 0)
        {
            atomicMax(&(maxIntersection), maxCount[wx]);
        }
        __syncthreads();

        if(lx == 0)
        {
            if(maxIntersection == maxCount[wx])
            {
                atomicMin(&(level_pivot[0]), maxIndex[wx]);
            }
        }
        __syncthreads();

        // Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
    
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = m;
            warpCount += __popc(pl[j]);
        }
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) - 1);
            T newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l - 2) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-1] = drop[l-2];
                if(newIndex == level_pivot[l-2])
                    drop[l-1] = drop[l-2] + 1;
            }
            __syncthreads();

            if(l - drop[l-1] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 2)]);
                T* to =  &(cl[num_divs_local * (l - 1)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = srcLen + 1; // make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < srcLen; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == srcLen + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; // shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();

                    if(l >= KCCOUNT)
                    {
                        if (threadIdx.x == 0)
                        {
                            T c = l - KCCOUNT;
                            local_choose_hold = nCR[drop[l - 1] * 401 + c];
                            local_choose_pivot = (drop[l - 1] == 0 ? 0 : nCR[(drop[l - 1] - 1) * 401 + c]);
                        }
                        __syncthreads();

                        for (T j = threadIdx.x; j < l - 1; j += BLOCK_DIM_X)
                        {
                            T cur = level_prev_index[j] - 1;
                            if(cur == level_pivot[j])
                            {
                                atomicAdd(&cpn[g.colInd[srcStart + cur]], local_choose_pivot);
                            }
                            else
                            {
                                atomicAdd(&cpn[g.colInd[srcStart + cur]], local_choose_hold);
                            }
                        }
                    }
                    
                    __syncthreads();

                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr = nCR[drop[l-1] * 401 + c];
                            atomicAdd(counter, ncr);
                            atomicAdd(&cpn[src], ncr);
                        }
                        
                        while (l > 2 && level_count[l - 2] == 0)
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	                            
                            while (l > 2 && level_count[l - 2] == 0)
                            {
                                (l)--;
                            }
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-2] = 0;
                            level_prev_index[l-2] = 0;
                        }
                        __syncthreads();

                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 2]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_pivot_count_local_sharedmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ bool partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ T lastMask_i, lastMask_ii;

    __shared__ uint64 local_choose_pivot, local_choose_hold;
    __shared__ uint64 local_clique_count[1024], root_count;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        // block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory:
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        // Encode Clear
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
            partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();
    
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }	
        }
        if(lx == 0)
        {
            atomicMax(&(maxIntersection), maxCount[wx]);
        }
        __syncthreads();

        if(lx == 0)
        {
            if(maxIntersection == maxCount[wx])
            {
                atomicMin(&(level_pivot[0]), maxIndex[wx]);
            }
        }
        __syncthreads();

        // Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
    
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = m;
            warpCount += __popc(pl[j]);
        }
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) - 1);
            T newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l - 2) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-1] = drop[l-2];
                if(newIndex == level_pivot[l-2])
                    drop[l-1] = drop[l-2] + 1;
            }
            __syncthreads();

            if(l - drop[l-1] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 2)]);
                T* to =  &(cl[num_divs_local * (l - 1)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = srcLen + 1; // make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < srcLen; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == srcLen + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; // shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();

                    if(l >= KCCOUNT)
                    {
                        if (threadIdx.x == 0)
                        {
                            T c = l - KCCOUNT;
                            local_choose_hold = nCR[drop[l - 1] * 401 + c];
                            local_choose_pivot = (drop[l - 1] == 0 ? 0 : nCR[(drop[l - 1] - 1) * 401 + c]);
                        }
                        __syncthreads();

                        for (T j = threadIdx.x; j < l - 1; j += BLOCK_DIM_X)
                        {
                            T cur = level_prev_index[j] - 1;
                            if(cur == level_pivot[j])
                            {
                                atomicAdd(&local_clique_count[cur], local_choose_pivot);
                            }
                            else
                            {
                                atomicAdd(&local_clique_count[cur], local_choose_hold);
                            }
                        }
                    }
                    
                    __syncthreads();

                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr = nCR[drop[l-1] * 401 + c];
                            atomicAdd(counter, ncr);
                            atomicAdd(&root_count, ncr);
                        }
                        
                        while (l > 2 && level_count[l - 2] == 0)
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	                            
                            while (l > 2 && level_count[l - 2] == 0)
                            {
                                (l)--;
                            }
                        }
                        __syncthreads();
                    }
                    else
                    {

                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-2] = 0;
                            level_prev_index[l-2] = 0;
                        }
                        __syncthreads();

                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 2]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_pivot_count_local_globalmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ bool partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ T lastMask_i, lastMask_ii;

    __shared__ uint64 local_level_choose_pivot[1024], local_level_choose_hold[1024];

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        // block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;

            local_level_choose_pivot[0] = 0;
            local_level_choose_hold[0] = 0;
        }
        __syncthreads();

        // Encode Clear
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
            partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();
    
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }	
        }
        if(lx == 0)
        {
            atomicMax(&(maxIntersection), maxCount[wx]);
        }
        __syncthreads();

        if(lx == 0)
        {
            if(maxIntersection == maxCount[wx])
            {
                atomicMin(&(level_pivot[0]), maxIndex[wx]);
            }
        }
        __syncthreads();

        // Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
    
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = m;
            warpCount += __popc(pl[j]);
        }
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) - 1);
            T newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l - 2) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-1] = drop[l-2];
                if(newIndex == level_pivot[l-2])
                    drop[l-1] = drop[l-2] + 1;
            }
            __syncthreads();

            if(l - drop[l-1] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        uint64 cur_pivot = local_level_choose_pivot[l - 2];
                        uint64 cur_hold = local_level_choose_hold[l - 2];
                        local_level_choose_pivot[l - 3] += cur_pivot;
                        local_level_choose_hold[l - 3] += cur_hold;
                        T cur = level_prev_index[l - 3] - 1;
                        if(cur == level_pivot[l - 3])
                        {
                            atomicAdd(&cpn[g.colInd[srcStart + cur]], cur_pivot);
                        }
                        else
                        {
                            atomicAdd(&cpn[g.colInd[srcStart + cur]], cur_hold);
                        }
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 2)]);
                T* to =  &(cl[num_divs_local * (l - 1)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = srcLen + 1; // make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < srcLen; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == srcLen + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; // shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {   
                    __syncthreads();

                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr_hold = nCR[drop[l-1] * 401 + c];
                            unsigned long long ncr_pivot = (drop[l - 1] == 0 ? 0 : nCR[(drop[l - 1] - 1) * 401 + c]);
                            atomicAdd(counter, ncr_hold);
                            atomicAdd(&cpn[src], ncr_hold);
                            T cur = level_prev_index[l - 2] - 1;
                            if(cur == level_pivot[l - 2])
                            {
                                atomicAdd(&cpn[g.colInd[srcStart + cur]], ncr_pivot);
                            }
                            else
                            {
                                atomicAdd(&cpn[g.colInd[srcStart + cur]], ncr_hold);
                            }
                            local_level_choose_pivot[l - 2] += ncr_pivot;
                            local_level_choose_hold[l - 2] += ncr_hold;
                        }
                        
                        while (l > 2 && level_count[l - 2] == 0)
                        {
                            uint64 cur_pivot = local_level_choose_pivot[l - 2];
                            uint64 cur_hold = local_level_choose_hold[l - 2];
                            local_level_choose_pivot[l - 3] += cur_pivot;
                            local_level_choose_hold[l - 3] += cur_hold;
                            T cur = level_prev_index[l - 3] - 1;
                            if(cur == level_pivot[l - 3])
                            {
                                atomicAdd(&cpn[g.colInd[srcStart + cur]], cur_pivot);
                            }
                            else
                            {
                                atomicAdd(&cpn[g.colInd[srcStart + cur]], cur_hold);
                            }
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	                            
                            while (l > 2 && level_count[l - 2] == 0)
                            {
                                uint64 cur_pivot = local_level_choose_pivot[l - 2];
                                uint64 cur_hold = local_level_choose_hold[l - 2];
                                local_level_choose_pivot[l - 3] += cur_pivot;
                                local_level_choose_hold[l - 3] += cur_hold;
                                T cur = level_prev_index[l - 3] - 1;
                                if(cur == level_pivot[l - 3])
                                {
                                    atomicAdd(&cpn[g.colInd[srcStart + cur]], cur_pivot);
                                }
                                else
                                {
                                    atomicAdd(&cpn[g.colInd[srcStart + cur]], cur_hold);
                                }
                                (l)--;
                            }
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-2] = 0;
                            level_prev_index[l-2] = 0;
                            local_level_choose_pivot[l - 2] = 0;
                            local_level_choose_hold[l - 2] = 0;
                        }
                        __syncthreads();

                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 2]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_pivot_count_local_sharedmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ bool partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ T lastMask_i, lastMask_ii;

    __shared__ uint64 local_level_choose_pivot[1024], local_level_choose_hold[1024];
    __shared__ uint64 local_clique_count[1024], root_count;


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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        // block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;

            local_level_choose_pivot[0] = 0;
            local_level_choose_hold[0] = 0;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory:
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        // Encode Clear
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
            partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();
    
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }	
        }
        if(lx == 0)
        {
            atomicMax(&(maxIntersection), maxCount[wx]);
        }
        __syncthreads();

        if(lx == 0)
        {
            if(maxIntersection == maxCount[wx])
            {
                atomicMin(&(level_pivot[0]), maxIndex[wx]);
            }
        }
        __syncthreads();

        // Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
    
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = m;
            warpCount += __popc(pl[j]);
        }
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) - 1);
            T newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l - 2) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-1] = drop[l-2];
                if(newIndex == level_pivot[l-2])
                    drop[l-1] = drop[l-2] + 1;
            }
            __syncthreads();

            if(l - drop[l-1] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        uint64 cur_pivot = local_level_choose_pivot[l - 2];
                        uint64 cur_hold = local_level_choose_hold[l - 2];
                        local_level_choose_pivot[l - 3] += cur_pivot;
                        local_level_choose_hold[l - 3] += cur_hold;
                        T cur = level_prev_index[l - 3] - 1;
                        if(cur == level_pivot[l - 3])
                        {
                            atomicAdd(&local_clique_count[cur], cur_pivot);
                        }
                        else
                        {
                            atomicAdd(&local_clique_count[cur], cur_hold);
                        }
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 2)]);
                T* to =  &(cl[num_divs_local * (l - 1)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = srcLen + 1; // make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < srcLen; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == srcLen + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; // shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {   
                    __syncthreads();

                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr_hold = nCR[drop[l-1] * 401 + c];
                            unsigned long long ncr_pivot = (drop[l - 1] == 0 ? 0 : nCR[(drop[l - 1] - 1) * 401 + c]);
                            atomicAdd(counter, ncr_hold);
                            atomicAdd(&root_count, ncr_hold);
                            T cur = level_prev_index[l - 2] - 1;
                            if(cur == level_pivot[l - 2])
                            {
                                atomicAdd(&local_clique_count[cur], ncr_pivot);
                            }
                            else
                            {
                                atomicAdd(&local_clique_count[cur], ncr_hold);
                            }
                            local_level_choose_pivot[l - 2] += ncr_pivot;
                            local_level_choose_hold[l - 2] += ncr_hold;
                        }
                        
                        while (l > 2 && level_count[l - 2] == 0)
                        {
                            uint64 cur_pivot = local_level_choose_pivot[l - 2];
                            uint64 cur_hold = local_level_choose_hold[l - 2];
                            local_level_choose_pivot[l - 3] += cur_pivot;
                            local_level_choose_hold[l - 3] += cur_hold;
                            T cur = level_prev_index[l - 3] - 1;
                            if(cur == level_pivot[l - 3])
                            {
                                atomicAdd(&local_clique_count[cur], cur_pivot);
                            }
                            else
                            {
                                atomicAdd(&local_clique_count[cur], cur_hold);
                            }
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	                            
                            while (l > 2 && level_count[l - 2] == 0)
                            {
                                uint64 cur_pivot = local_level_choose_pivot[l - 2];
                                uint64 cur_hold = local_level_choose_hold[l - 2];
                                local_level_choose_pivot[l - 3] += cur_pivot;
                                local_level_choose_hold[l - 3] += cur_hold;
                                T cur = level_prev_index[l - 3] - 1;
                                if(cur == level_pivot[l - 3])
                                {
                                    atomicAdd(&local_clique_count[cur], cur_pivot);
                                }
                                else
                                {
                                    atomicAdd(&local_clique_count[cur], cur_hold);
                                }
                                (l)--;
                            }
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-2] = 0;
                            level_prev_index[l-2] = 0;
                            local_level_choose_pivot[l - 2] = 0;
                            local_level_choose_hold[l - 2] = 0;
                        }
                        __syncthreads();

                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 2]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count_local_base(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
    __shared__ bool  partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ 	T lastMask_i, lastMask_ii;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
    
        __syncthreads();
        //Encode Clear
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        //Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                j, num_divs_local,  encode);
        }
        __syncthreads(); //Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]),maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
    
        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-2] = drop[l-3];
                if(newIndex == level_pivot[l-3])
                    drop[l-2] = drop[l-3] + 1;
            }
            __syncthreads();

            if(l - drop[l-2] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 3)]);
                T* to =  &(cl[num_divs_local * (l - 2)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF: sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = scounter + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < scounter; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == scounter + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; //shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();
                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr = nCR[drop[l-2] * 401 + c];
                            atomicAdd(counter, ncr);
                        }

                        while (l > 3 && level_count[l - 3] == 0)
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	   
                            while (l > 3 && level_count[l - 3] == 0)
                            {
                                (l)--;
                            }                         
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-3] = 0;
                            level_prev_index[l-3] = 0;
                        }

                        __syncthreads();
                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 3]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count_local_globalmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
    __shared__ bool  partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ T lastMask_i, lastMask_ii;

    __shared__ uint64 local_choose_pivot, local_choose_hold;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
    
        __syncthreads();
        //Encode Clear
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        //Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                j, num_divs_local,  encode);
        }
        __syncthreads(); //Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]),maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
    
        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-2] = drop[l-3];
                if(newIndex == level_pivot[l-3])
                    drop[l-2] = drop[l-3] + 1;
            }
            __syncthreads();

            if(l - drop[l-2] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 3)]);
                T* to =  &(cl[num_divs_local * (l - 2)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF: sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = scounter + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < scounter; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == scounter + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; //shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();

                    if(l >= KCCOUNT)
                    {
                        if (threadIdx.x == 0)
                        {
                            T c = l - KCCOUNT;
                            local_choose_hold = nCR[drop[l - 2] * 401 + c];
                            local_choose_pivot = (drop[l - 2] == 0 ? 0 : nCR[(drop[l - 2] - 1) * 401 + c]);
                        }
                        __syncthreads();

                        for (T j = threadIdx.x; j < l - 2; j += BLOCK_DIM_X)
                        {
                            T cur = level_prev_index[j] - 1;
                            if(cur == level_pivot[j])
                            {
                                atomicAdd(&cpn[tri[cur]], local_choose_pivot);
                            }
                            else
                            {
                                atomicAdd(&cpn[tri[cur]], local_choose_hold);
                            }
                        }
                    }
                    
                    __syncthreads();

                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr = nCR[drop[l-2] * 401 + c];
                            atomicAdd(counter, ncr);
                            atomicAdd(&cpn[src], ncr);
                            atomicAdd(&cpn[src2], ncr);
                        }

                        while (l > 3 && level_count[l - 3] == 0)
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	   
                            while (l > 3 && level_count[l - 3] == 0)
                            {
                                (l)--;
                            }                         
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-3] = 0;
                            level_prev_index[l-3] = 0;
                        }

                        __syncthreads();
                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 3]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count_local_sharedmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
    __shared__ bool  partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ T lastMask_i, lastMask_ii;

    __shared__ uint64 local_choose_pivot, local_choose_hold;
    __shared__ uint64 local_clique_count[1024], root_count;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        // Clear local_clique counter in shared memory:
        for (unsigned int idx = threadIdx.x; idx < scounter; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
             root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
    
        __syncthreads();
        //Encode Clear
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        //Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                j, num_divs_local,  encode);
        }
        __syncthreads(); //Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]),maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
    
        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-2] = drop[l-3];
                if(newIndex == level_pivot[l-3])
                    drop[l-2] = drop[l-3] + 1;
            }
            __syncthreads();

            if(l - drop[l-2] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 3)]);
                T* to =  &(cl[num_divs_local * (l - 2)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF: sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = scounter + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < scounter; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == scounter + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; //shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {      
                    __syncthreads();

                    if(l >= KCCOUNT)
                    {
                        if (threadIdx.x == 0)
                        {
                            T c = l - KCCOUNT;
                            local_choose_hold = nCR[drop[l - 2] * 401 + c];
                            local_choose_pivot = (drop[l - 2] == 0 ? 0 : nCR[(drop[l - 2] - 1) * 401 + c]);
                        }
                        __syncthreads();

                        for (T j = threadIdx.x; j < l - 2; j += BLOCK_DIM_X)
                        {
                            T cur = level_prev_index[j] - 1;
                            if(cur == level_pivot[j])
                            {
                                atomicAdd(&local_clique_count[cur], local_choose_pivot);
                            }
                            else
                            {
                                atomicAdd(&local_clique_count[cur], local_choose_hold);
                            }
                        }
                    }
                    
                    __syncthreads();

                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr = nCR[drop[l-2] * 401 + c];
                            atomicAdd(counter, ncr);
                            atomicAdd(&root_count, ncr);
                        }

                        while (l > 3 && level_count[l - 3] == 0)
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	   
                            while (l > 3 && level_count[l - 3] == 0)
                            {
                                (l)--;
                            }                         
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-3] = 0;
                            level_prev_index[l-3] = 0;
                        }

                        __syncthreads();
                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 3]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
        for (unsigned int idx = threadIdx.x; idx < scounter; idx += blockDim.x)
        {
            atomicAdd(&cpn[tri[idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
            atomicAdd(&cpn[src2], root_count);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count_local_globalmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
    __shared__ bool  partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ 	T lastMask_i, lastMask_ii;

    __shared__ uint64 local_level_choose_pivot[1024], local_level_choose_hold[1024];

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
    
        __syncthreads();
        //Encode Clear
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        //Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                j, num_divs_local,  encode);
        }
        __syncthreads(); //Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]),maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
    
        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-2] = drop[l-3];
                if(newIndex == level_pivot[l-3])
                    drop[l-2] = drop[l-3] + 1;
            }
            __syncthreads();

            if(l - drop[l-2] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        uint64 cur_pivot = local_level_choose_pivot[l - 3];
                        uint64 cur_hold = local_level_choose_hold[l - 3];
                        local_level_choose_pivot[l - 4] += cur_pivot;
                        local_level_choose_hold[l - 4] += cur_hold;
                        T cur = level_prev_index[l - 4] - 1;
                        if(cur == level_pivot[l - 4])
                        {
                            atomicAdd(&cpn[tri[cur]], cur_pivot);
                        }
                        else
                        {
                            atomicAdd(&cpn[tri[cur]], cur_hold);
                        }
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 3)]);
                T* to =  &(cl[num_divs_local * (l - 2)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF: sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = scounter + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < scounter; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == scounter + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; //shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();
                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr_hold = nCR[drop[l-2] * 401 + c];
                            unsigned long long ncr_pivot = (drop[l - 2] == 0 ? 0 : nCR[(drop[l - 2] - 1) * 401 + c]);
                            atomicAdd(counter, ncr_hold);
                            atomicAdd(&cpn[src], ncr_hold);
                            atomicAdd(&cpn[src2], ncr_hold);
                            T cur = level_prev_index[l - 3] - 1;
                            if(cur == level_pivot[l - 3])
                            {
                                atomicAdd(&cpn[tri[cur]], ncr_pivot);
                            }
                            else
                            {
                                atomicAdd(&cpn[tri[cur]], ncr_hold);
                            }
                            local_level_choose_pivot[l - 3] += ncr_pivot;
                            local_level_choose_hold[l - 3] += ncr_hold;
                        }

                        while (l > 3 && level_count[l - 3] == 0)
                        {
                            uint64 cur_pivot = local_level_choose_pivot[l - 3];
                            uint64 cur_hold = local_level_choose_hold[l - 3];
                            local_level_choose_pivot[l - 4] += cur_pivot;
                            local_level_choose_hold[l - 4] += cur_hold;
                            T cur = level_prev_index[l - 4] - 1;
                            if(cur == level_pivot[l - 4])
                            {
                                atomicAdd(&cpn[tri[cur]], cur_pivot);
                            }
                            else
                            {
                                atomicAdd(&cpn[tri[cur]], cur_hold);
                            }
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	   
                            while (l > 3 && level_count[l - 3] == 0)
                            {
                                uint64 cur_pivot = local_level_choose_pivot[l - 3];
                                uint64 cur_hold = local_level_choose_hold[l - 3];
                                local_level_choose_pivot[l - 4] += cur_pivot;
                                local_level_choose_hold[l - 4] += cur_hold;
                                T cur = level_prev_index[l - 4] - 1;
                                if(cur == level_pivot[l - 4])
                                {
                                    atomicAdd(&cpn[tri[cur]], cur_pivot);
                                }
                                else
                                {
                                    atomicAdd(&cpn[tri[cur]], cur_hold);
                                }
                                (l)--;
                            }                       
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-3] = 0;
                            level_prev_index[l-3] = 0;
                            local_level_choose_pivot[l - 3] = 0;
                            local_level_choose_hold[l - 3] = 0;
                        }

                        __syncthreads();
                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 3]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count_local_sharedmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,

    T* possible,
    T* level_count_g,
    T* level_prev_g,
    T* level_d,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
    __shared__ bool  partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_prev_index, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ 	T lastMask_i, lastMask_ii;

    __shared__ uint64 local_level_choose_pivot[1024], local_level_choose_hold[1024];
    __shared__ uint64 local_clique_count[1024], root_count;

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
    __syncthreads();

    for (T i = blockIdx.x; i < (T)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        //block things
        if (threadIdx.x == 0)
        {
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo];
            pl = &possible[lo];

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];
            drop = &level_d[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;

            local_level_choose_pivot[0] = 0;
            local_level_choose_hold[0] = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        // Clear local_clique counter in shared memory:
        for (unsigned int idx = threadIdx.x; idx < scounter; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
    
        __syncthreads();
        //Encode Clear
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        //Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                j, num_divs_local,  encode);
        }
        __syncthreads(); //Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]),maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
    
        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();

            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                drop[l-2] = drop[l-3];
                if(newIndex == level_pivot[l-3])
                    drop[l-2] = drop[l-3] + 1;
            }
            __syncthreads();

            if(l - drop[l-2] > KCCOUNT)
            {	
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        uint64 cur_pivot = local_level_choose_pivot[l - 3];
                        uint64 cur_hold = local_level_choose_hold[l - 3];
                        local_level_choose_pivot[l - 4] += cur_pivot;
                        local_level_choose_hold[l - 4] += cur_hold;
                        T cur = level_prev_index[l - 4] - 1;
                        if(cur == level_pivot[l - 4])
                        {
                            atomicAdd(&local_clique_count[cur], cur_pivot);
                        }
                        else
                        {
                            atomicAdd(&local_clique_count[cur], cur_hold);
                        }
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list
                T* from = &(cl[num_divs_local * (l - 3)]);
                T* to =  &(cl[num_divs_local * (l - 2)]);
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF: sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = scounter + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                
                for (T j = wx; j < scounter; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1 << ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == scounter + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; //shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();
                    if(threadIdx.x == 0)
                    {	
                        if(l >= KCCOUNT)
                        {
                            T c = l - KCCOUNT;
                            unsigned long long ncr_hold = nCR[drop[l-2] * 401 + c];
                            unsigned long long ncr_pivot = (drop[l - 2] == 0 ? 0 : nCR[(drop[l - 2] - 1) * 401 + c]);
                            atomicAdd(counter, ncr_hold);
                            atomicAdd(&root_count, ncr_hold);
                            T cur = level_prev_index[l - 3] - 1;
                            if(cur == level_pivot[l - 3])
                            {
                                atomicAdd(&local_clique_count[cur], ncr_pivot);
                            }
                            else
                            {
                                atomicAdd(&local_clique_count[cur], ncr_hold);
                            }
                            local_level_choose_pivot[l - 3] += ncr_pivot;
                            local_level_choose_hold[l - 3] += ncr_hold;
                        }

                        while (l > 3 && level_count[l - 3] == 0)
                        {
                            uint64 cur_pivot = local_level_choose_pivot[l - 3];
                            uint64 cur_hold = local_level_choose_hold[l - 3];
                            local_level_choose_pivot[l - 4] += cur_pivot;
                            local_level_choose_hold[l - 4] += cur_hold;
                            T cur = level_prev_index[l - 4] - 1;
                            if(cur == level_pivot[l - 4])
                            {
                                atomicAdd(&local_clique_count[cur], cur_pivot);
                            }
                            else
                            {
                                atomicAdd(&local_clique_count[cur], cur_hold);
                            }
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(l + maxIntersection + 1 < KCCOUNT)
                    {
                        __syncthreads();
                        if(threadIdx.x == 0)
                        {	   
                            while (l > 3 && level_count[l - 3] == 0)
                            {
                                uint64 cur_pivot = local_level_choose_pivot[l - 3];
                                uint64 cur_hold = local_level_choose_hold[l - 3];
                                local_level_choose_pivot[l - 4] += cur_pivot;
                                local_level_choose_hold[l - 4] += cur_hold;
                                T cur = level_prev_index[l - 4] - 1;
                                if(cur == level_pivot[l - 4])
                                {
                                    atomicAdd(&local_clique_count[cur], cur_pivot);
                                }
                                else
                                {
                                    atomicAdd(&local_clique_count[cur], cur_hold);
                                }
                                (l)--;
                            }                 
                        }
                        __syncthreads();
                    }
                    else
                    {
                        if(lx == 0 && maxIntersection == maxCount[wx])
                        {	
                            atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                        }
                        __syncthreads();

                        uint64 warpCount = 0;
                        for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                        {
                            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                            pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-3] = 0;
                            level_prev_index[l-3] = 0;
                            local_level_choose_pivot[l - 3] = 0;
                            local_level_choose_hold[l - 3] = 0;
                        }

                        __syncthreads();
                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l - 3]), warpCount);
                        }
                    }
                }
            }
            __syncthreads();
        }
        for (unsigned int idx = threadIdx.x; idx < scounter; idx += blockDim.x)
        {
            atomicAdd(&cpn[tri[idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
            atomicAdd(&cpn[src2], root_count);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}