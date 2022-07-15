#pragma once

/////////////////////////////////////////// Latest Kernels ///////////////////////////////////////////////////


template <typename T, uint BLOCK_DIM_X,  uint CPARTSIZE=32>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    unsigned short* current_level,
    T* levelStats
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ T level_index[numPartitions][10];
    __shared__ T level_count[numPartitions][10];
    __shared__ T level_prev_index[numPartitions][10];

    __shared__ T current_node_index[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ char l[numPartitions];
    __shared__ char new_level[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ T src2Start[numPartitions], src2Len[numPartitions], src2LenBlocks[numPartitions];
    __shared__ T refIndex[numPartitions], refLen[numPartitions], srcLenBlocks[numPartitions];

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
            T src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
    
        T partMask = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        __syncthreads();
        //warp loop
        for (unsigned long long j = wx; j < srcLen; j += numPartitions)
        {

            for(T k = lx; k < 10; k+= CPARTSIZE)
            {
                level_count[wx][k] = 0;
                level_index[wx][k] = 0;
                level_prev_index[wx][k] = 0;
            }
            
            
            if (lx == 0)
            {
                T src2 = g.colInd[srcStart + j];
                src2Start[wx] = g.rowPtr[src2];
                src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];

                refIndex[wx] = srcLen < src2Len[wx] ? srcStart : src2Start[wx];
                refLen[wx] = srcLen < src2Len[wx] ? srcLen : src2Len[wx];
                srcLenBlocks[wx] = (refLen[wx] + CPARTSIZE - 1) / CPARTSIZE;
                l[wx] = 2;
                new_level[wx] = 2;
                current_node_index[wx] = UINT_MAX;
                clique_count[wx] = 0;

            }

            __syncwarp(partMask);
            T blockOffset = sm_id * CBPSM * (numPartitions * MAXDEG)
                + levelPtr * (numPartitions * MAXDEG);
            unsigned short* cl = &current_level[blockOffset + wx * MAXDEG /*srcStart[wx]*/];
            for (unsigned long long k = lx; k < refLen[wx]; k += CPARTSIZE)
            {
                cl[k] = 0x01;
            }
            __syncwarp(partMask);

            uint64 warpCount = 0;
            if (src2Len[wx] >= KCCOUNT - l[wx])
            {
                if (srcLen < src2Len[wx])
                {
                    warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE, unsigned short>(&g.colInd[srcStart], srcLen,
                        &g.colInd[src2Start[wx]], src2Len[wx], true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 3);
                }
                else {
                    warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE, unsigned short>(&g.colInd[src2Start[wx]], src2Len[wx],
                        &g.colInd[srcStart], srcLen, true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 3);
                }

                __syncwarp(partMask);
                if (warpCount > 0)
                {
                    if (l[wx] + 1 == KCCOUNT && lx == 0)
                        clique_count[wx] += warpCount;
                    else if (l[wx] + 1 < KCCOUNT)
                    {
                        if (lx == 0)
                        {
                            (l[wx])++;
                            (new_level[wx])++;
                            level_count[wx][l[wx] - 2] = warpCount;
                            level_index[wx][l[wx] - 2] = 0;
                            level_prev_index[wx][l[wx] - 2] = 0;
                        }
                    }

                }
            }
            __syncwarp(partMask);
            while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
            {
                // for (T k = 0; k < srcLenBlocks[wx]; k++)
                // {
                // 	T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
                // 	int condition = index < refLen[wx] && (cl[index] & (0x01 << (l[wx] - 2)));
                // 	unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
                // 	if (newmask != 0)
                // 	{
                // 		uint elected_lane_deq = __ffs(newmask) - 1;
                // 		current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
                // 		break;
                // 	}
                // }


                T startIndex = level_prev_index[wx][l[wx]- 2];
                T newIndex = cl[startIndex] & (0x01 << (l[wx] - 2));
                while(newIndex == 0)
                {
                    startIndex++;
                    newIndex = cl[startIndex] & (0x01 << (l[wx] - 2));
                }

                if (lx == 0)
                {
                    //current_node_index[0] = finalIndex;
                    current_node_index[wx] = startIndex;
                    level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
                    level_index[wx][l[wx] - 2]++;
                    new_level[wx] = l[wx];
                }

                __syncwarp(partMask);

                uint64 warpCountIn = 0;
                const T dst = g.colInd[current_node_index[wx] + refIndex[wx]];
                const T dstStart = g.rowPtr[dst];
                const T dstLen = g.rowPtr[dst + 1] - dstStart;

                bool limit = ((l[wx] - 1 + level_count[wx][l[wx] - 2]) >= KCCOUNT) && (dstLen >= KCCOUNT - l[wx]);

                if (limit /*dstLen >= KCCOUNT - l[wx]*/)
                {
                    if (dstLen > refLen[wx])
                    {
                        warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE, unsigned short>(&g.colInd[refIndex[wx]], refLen[wx],
                            &g.colInd[dstStart], dstLen, true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 3);
                    }
                    else {
                        warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE, unsigned short>(&g.colInd[dstStart], dstLen,
                            &g.colInd[refIndex[wx]], refLen[wx], false, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 3);

                    }
                    __syncwarp(partMask);
                    if (lx == 0 && warpCountIn > 0)
                    {
                        if (l[wx] + 1 == KCCOUNT)
                            clique_count[wx] += warpCountIn;
                        else if ((l[wx] + 1 < KCCOUNT) /*&& ((l[wx] + warpCountIn) >= KCCOUNT)*/)
                        {
                            //if(warpCountIn >= KCCOUNT - l[wx])
                            {
                                (l[wx])++;
                                (new_level[wx])++;
                                level_count[wx][l[wx] - 2] = warpCountIn;
                                level_index[wx][l[wx] - 2] = 0;
                                level_prev_index[wx][l[wx] - 2] = 0;
                            }
                        }
                    }
                }


                __syncwarp(partMask);
                if (lx == 0)
                {
                    while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
                    {
                        (new_level[wx])--;
                    }
                }

                __syncwarp(partMask);
                if (new_level[wx] < l[wx])
                {
                    char clearMask = ~((1 << (l[wx] - 1)) - (1 << (new_level[wx] - 1)));
                    for (auto k = 0; k < srcLenBlocks[wx]; k++)
                    {
                        T index = k * CPARTSIZE + lx;
                        if (index < refLen[wx])
                            cl[index] = cl[index] & clearMask;
                    }

                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    l[wx] = new_level[wx];
                    current_node_index[wx] = UINT_MAX;
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}






template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE=32>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_subgraph_im_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    unsigned short* current_level,
    T* levelStats,
    T* adj_enc,
    T* im_level
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ T level_index[numPartitions][10];
    __shared__ T level_count[numPartitions][10];
    __shared__ T level_prev_index[numPartitions][10];

    __shared__ T current_node_index[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ char l[numPartitions];
    __shared__ char new_level[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ T encode_offset, *encode, im_size, *im;
    __shared__ unsigned short subgraph_counters[1024];
    __shared__ unsigned short intermediate_counters[numPartitions][10];
    __shared__ T  srcLenBlocks[numPartitions];

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
            encode_offset = sm_id * CBPSM * (MAXDEG * MAXDEG) + levelPtr * (MAXDEG * MAXDEG);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
        }

        T partMask = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        __syncthreads();

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < srcLen; k += CPARTSIZE)
            {
                encode[j * srcLen + k] = 0xFFFFFFFF;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_subgraph<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, srcLen, encode);
        }




        __syncthreads();
        //compact: simpler than cub
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if(lx == 0)
            {
                subgraph_counters[j] = 0;
                for(T k = 0; k< srcLen; k++)
                {
                    if(encode[j*srcLen + k] != 0xFFFFFFFF)
                    {
                        encode[j*srcLen + subgraph_counters[j]] = encode[j*srcLen + k];
                        subgraph_counters[j]++;
                    }
                }
            }
        }

        __syncthreads(); //Done encoding

        //warp loop
        for (T j = wx; j < srcLen; j += numPartitions)
        {


            for(T k = lx; k < 10; k+= CPARTSIZE)
            {
                level_count[wx][k] = 0;
                level_index[wx][k] = 0;
                level_prev_index[wx][k] = 0;
            }
            
            if (lx == 0)
            {
                srcLenBlocks[wx] = ( subgraph_counters[j] + CPARTSIZE - 1) / CPARTSIZE;
            
                l[wx] = 2;
                new_level[wx] = 2;
                current_node_index[wx] = UINT_MAX;
                clique_count[wx] = 0;
            }

            __syncwarp(partMask);
            T blockOffset = sm_id * CBPSM * (numPartitions * MAXDEG)
                + levelPtr * (numPartitions * MAXDEG);
            unsigned short* cl = &current_level[blockOffset + wx * MAXDEG /*srcStart[wx]*/];

            T blockImOffset = sm_id * CBPSM * (numPartitions * MAXLEVEL* MAXDEG)
                + levelPtr * (numPartitions * MAXLEVEL*MAXDEG);
            T* im = &im_level[blockImOffset + wx * MAXLEVEL*  MAXDEG];


            __syncwarp(partMask);

            uint64 warpCount = 0;
            if ( subgraph_counters[j] >= KCCOUNT - l[wx])
            {
                
                if (l[wx] + 1 == KCCOUNT && lx == 0)
                    clique_count[wx] +=  subgraph_counters[j];
                else if (l[wx] + 1 < KCCOUNT)
                {
                    if (lx == 0)
                    {
                        (l[wx])++;
                        (new_level[wx])++;
                        level_count[wx][l[wx] - 2] =  subgraph_counters[j];
                        level_index[wx][l[wx] - 2] = 0;
                        level_prev_index[wx][l[wx] - 2] = 0;
                    }
                }
            }

            __syncwarp(partMask);

            for (T k = lx; k < subgraph_counters[j]; k += CPARTSIZE)
            {
                im[(l[wx] - 2)*srcLen +  k] = encode[j*srcLen + k];
            }

            __syncwarp(partMask);
            while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
            {

                T startIndex = level_prev_index[wx][l[wx]- 2];
                if (lx == 0)
                {
                    //current_node_index[0] = finalIndex;
                    current_node_index[wx] = startIndex;
                    level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
                    level_index[wx][l[wx] - 2]++;
                    new_level[wx] = l[wx];
                    level_count[wx][l[wx]-1] = 0;
                }

                __syncwarp(partMask);

                //assert(current_node_index[wx] < subgraph_counters[j]);
                uint64 warpCountIn = 0;
                const T dst = im[(l[wx] - 2)*srcLen + current_node_index[wx]];  //encode[srcLen*j +  current_node_index[wx]];
                const T dstStart = 0;
                const T dstLen = subgraph_counters[dst];

                bool limit = ((l[wx] - 1 + level_count[wx][l[wx] - 2]) >= KCCOUNT) && (dstLen >= KCCOUNT - l[wx]);

                if (limit /*dstLen >= KCCOUNT - l[wx]*/)
                {
                    warpCountIn = graph::warp_sorted_count_and_subgraph_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&im[(l[wx] - 2) * srcLen ],  level_count[wx][l[wx]-2],
                        &encode[dst * srcLen], dstLen, &im[(l[wx] - 1) * srcLen ], &(level_count[wx][l[wx]-1]), l[wx] + 1, KCCOUNT, partMask );
                    
                    __syncwarp(partMask);

                    if (lx == 0 && level_count[wx][l[wx]-1] > 0)
                    {
                        if (l[wx] + 1 == KCCOUNT)
                            clique_count[wx] += level_count[wx][l[wx]-1];
                        else if ((l[wx] + 1 < KCCOUNT) /*&& ((l[wx] + warpCountIn) >= KCCOUNT)*/)
                        {
                            //if(warpCountIn >= KCCOUNT - l[wx])
                            {
                                (l[wx])++;
                                (new_level[wx])++;
                                level_index[wx][l[wx] - 2] = 0;
                                level_prev_index[wx][l[wx] - 2] = 0;
                            }
                        }
                    }
                }
                __syncwarp(partMask);
                if (lx == 0)
                {
                    while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
                    {
                        (new_level[wx])--;
                    }
                }

                if (lx == 0)
                {
                    l[wx] = new_level[wx];
                    current_node_index[wx] = UINT_MAX;
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}




//come back
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE=32>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    unsigned short* current_level,
    T* levelStats,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ T level_index[numPartitions][10];
    __shared__ T level_count[numPartitions][10];
    __shared__ T level_prev_index[numPartitions][10];

    __shared__ T current_node_index[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ char l[numPartitions];
    __shared__ char new_level[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, secNodeStart, secNodeLen, tri_offset, *tri, scounter;
    __shared__ T src2Start[numPartitions], src2Len[numPartitions];
    __shared__ T *refIndex[numPartitions], refLen[numPartitions], srcLenBlocks[numPartitions];

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
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;

            T secNode = g.colInd[current.queue[i]];
            secNodeStart = g.rowPtr[secNode];
            secNodeLen = g.rowPtr[secNode + 1] - secNodeStart;

            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
            scounter = 0;

        }

        T partMask = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);

        __syncthreads();
        for(T j = threadIdx.x; j< srcLen; j+=BLOCK_DIM_X)
            tri[j] = 0xFFFFFFFF;

        __syncthreads();

        // //get tri list: by block :!!
        graph::block_sorted_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[secNodeStart], secNodeLen,
            tri);

        __syncthreads();

        block_filter_pivot<T, BLOCK_DIM_X, T>(srcLen, tri, &scounter);
        __syncthreads();


        if(KCCOUNT == 3 && threadIdx.x == 0)
            atomicAdd(counter, scounter);

        //warp loop
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for(T k = lx; k < 10; k+= CPARTSIZE)
            {
                level_count[wx][k] = 0;
                level_index[wx][k] = 0;
                level_prev_index[wx][k] = 0;
            }
            
            
            if (lx == 0)
            {
                T src2 = tri[j];

                //reinitialize with respect to the triangle now
                src2Start[wx] = g.rowPtr[src2];
                src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];

                if(scounter < src2Len[wx])
                {
                    refIndex[wx] = tri;
                    refLen[wx] = scounter;
                }
                else
                {
                    refIndex[wx] = &(g.colInd[src2Start[wx]]);
                    refLen[wx] =  src2Len[wx];
                }

                srcLenBlocks[wx] = (refLen[wx] + CPARTSIZE - 1) / CPARTSIZE;
        
                l[wx] = 3;
                new_level[wx] = 3;
                current_node_index[wx] = UINT_MAX;
                clique_count[wx] = 0;
            }

            __syncwarp(partMask);

            T blockOffset = sm_id * CBPSM * (numPartitions * MAXDEG)
                + levelPtr * (numPartitions * MAXDEG);
            unsigned short* cl = &current_level[blockOffset + wx * MAXDEG /*srcStart[wx]*/];
            for (T k = lx; k < refLen[wx]; k += CPARTSIZE)
            {
                cl[k] = 0x01;
            }
            __syncwarp(partMask);

            uint64 warpCount = 0;
            if (src2Len[wx] >= KCCOUNT - l[wx])
            {
                if (scounter < src2Len[wx])
                {
                    warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE, unsigned short>(tri, scounter,
                        &g.colInd[src2Start[wx]], src2Len[wx], true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 4);
                }
                else {
                    warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE, unsigned short>(&g.colInd[src2Start[wx]], src2Len[wx],
                        tri, scounter, true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 4);
                }

                __syncwarp(partMask);
                if (warpCount > 0)
                {
                    if (l[wx] + 1 == KCCOUNT && lx == 0)
                        clique_count[wx] += warpCount;
                    else if (l[wx] + 1 < KCCOUNT)
                    {
                        if (lx == 0)
                        {
                            (l[wx])++;
                            (new_level[wx])++;
                            level_count[wx][l[wx] - 3] = warpCount;
                            level_index[wx][l[wx] - 3] = 0;
                            level_prev_index[wx][l[wx] - 3] = 0;
                        }
                    }

                }
            }

            __syncwarp(partMask);
            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                // for (T k = 0; k < srcLenBlocks[wx]; k++)
                // {
                // 	T index = level_prev_index[wx][l[wx] - 3] + k * 32 + lx;
                // 	int condition = index < refLen[wx] && (cl[index] & (0x01 << (l[wx] - 2)));
                // 	unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
                // 	if (newmask != 0)
                // 	{
                // 		uint elected_lane_deq = __ffs(newmask) - 1;
                // 		current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
                // 		break;
                // 	}
                // }


                T startIndex = level_prev_index[wx][l[wx]- 3];
                T newIndex = cl[startIndex] & (0x01 << (l[wx] - 3));
                while(newIndex == 0)
                {
                    startIndex++;
                    newIndex = cl[startIndex] & (0x01 << (l[wx] - 3));
                }

                if (lx == 0)
                {
                    //current_node_index[0] = finalIndex;
                    current_node_index[wx] = startIndex;
                    level_prev_index[wx][l[wx] - 3] = current_node_index[wx] + 1;
                    level_index[wx][l[wx] - 3]++;
                    new_level[wx] = l[wx];
                }

                __syncwarp(partMask);

                uint64 warpCountIn = 0;
                const T dst = (refIndex[wx])[current_node_index[wx]]; //g.colInd[current_node_index[wx] + refIndex[wx]];
                const T dstStart = g.rowPtr[dst];
                const T dstLen = g.rowPtr[dst + 1] - dstStart;

                //bool limit = ((l[wx] - 3 + level_count[wx][l[wx] - 4]) >= KCCOUNT) && (dstLen >= KCCOUNT - l[wx]);

                if (dstLen >= KCCOUNT - l[wx])
                {
                    if (dstLen > refLen[wx])
                    {
                        warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE>(refIndex[wx], refLen[wx],
                            &g.colInd[dstStart], dstLen, true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 4);
                    }
                    else {
                        warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[dstStart], dstLen,
                            refIndex[wx], refLen[wx], false, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 4);

                    }
                    __syncwarp(partMask);
                    if (lx == 0 && warpCountIn > 0)
                    {
                        if (l[wx] + 1 == KCCOUNT)
                            clique_count[wx] += warpCountIn;
                        else if ((l[wx] + 1 < KCCOUNT) /*&& ((l[wx] + warpCountIn) >= KCCOUNT)*/)
                        {
                            //if(warpCountIn >= KCCOUNT - l[wx])
                            {
                                (l[wx])++;
                                (new_level[wx])++;
                                level_count[wx][l[wx] - 3] = warpCountIn;
                                level_index[wx][l[wx] - 3] = 0;
                                level_prev_index[wx][l[wx] - 3] = 0;
                            }
                        }
                    }
                }


                __syncwarp(partMask);
                if (lx == 0)
                {
                    while (new_level[wx] > 4 && level_index[wx][new_level[wx] - 3] >= level_count[wx][new_level[wx] - 3])
                    {
                        (new_level[wx])--;
                    }
                }

                __syncwarp(partMask);
                if (new_level[wx] < l[wx])
                {
                    char clearMask = ~((1 << (l[wx] - 2)) - (1 << (new_level[wx] - 2)));
                    for (auto k = 0; k < srcLenBlocks[wx]; k++)
                    {
                        T index = k * CPARTSIZE + lx;
                        if (index < refLen[wx])
                            cl[index] = cl[index] & clearMask;
                    }

                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    l[wx] = new_level[wx];
                    current_node_index[wx] = UINT_MAX;
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }
        }

        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE=32>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_subgraph_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    unsigned short* current_level,
    T* levelStats,
    T* adj_enc,
    T* adj_tri
)
{

    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ T level_index[numPartitions][10];
    __shared__ T level_count[numPartitions][10];
    __shared__ T level_prev_index[numPartitions][10];

    __shared__ T current_node_index[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ char l[numPartitions];
    __shared__ char new_level[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2Start, secNode, src2Len, tri_offset, *tri, scounter;

    __shared__ T encode_offset, *encode;
    __shared__ T subgraph_counters[1024];
    __shared__ T blockOffset;
    

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
            src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;

            secNode = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[secNode];
            src2Len = g.rowPtr[secNode + 1] - src2Start;

            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * MAXDEG) + levelPtr * (MAXDEG * MAXDEG);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

        }

        T partMask = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);

        __syncthreads();

        // //get tri list: by block :!!
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        __syncthreads();


        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < scounter; k += CPARTSIZE)
            {
                encode[j * scounter + k] = 0xFFFFFFFF;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_subgraph<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                j, scounter, encode);
        }

        __syncthreads();
        //compact: simpler than cub
        for (T j = wx; j < scounter; j += numPartitions)
        {
            if(lx == 0)
            {
                subgraph_counters[j] = 0;
                for(T k = 0; k< scounter; k++)
                {
                    if(encode[j*scounter + k] != 0xFFFFFFFF)
                    {
                        encode[j*scounter + subgraph_counters[j]] = encode[j*scounter + k];
                        subgraph_counters[j]++;
                    }
                }
            }
        }

        __syncthreads(); //Done encoding


        if(KCCOUNT == 3 && threadIdx.x == 0)
            atomicAdd(counter, scounter);

        //warp loop
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for(T k = lx; k < 10; k+= CPARTSIZE)
            {
                level_count[wx][k] = 0;
                level_index[wx][k] = 0;
                level_prev_index[wx][k] = 0;
            }
            
            
            if (lx == 0)
            {
                l[wx] = 3;
                new_level[wx] = 3;
                current_node_index[wx] = UINT_MAX;
                clique_count[wx] = 0;
            }

            __syncwarp(partMask);

            blockOffset = sm_id * CBPSM * (numPartitions * MAXDEG)
                + levelPtr * (numPartitions * MAXDEG);
            unsigned short *cl = &current_level[blockOffset + wx * MAXDEG /*srcStart[wx]*/];
            
            for (T k = lx; k < scounter; k += CPARTSIZE)
            {
                cl[k] = 4;
            }
            __syncwarp(partMask);

            if (subgraph_counters[j] >= KCCOUNT - l[wx])
            {
                if (l[wx] + 1 == KCCOUNT && lx == 0)
                    clique_count[wx] += subgraph_counters[j];
                else if (l[wx] + 1 < KCCOUNT)
                {
                    if (lx == 0)
                    {
                        (l[wx])++;
                        (new_level[wx])++;
                        level_count[wx][l[wx] - 3] = subgraph_counters[j];
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                }
            }

            __syncwarp(partMask);
            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                // for (T k = 0; k < srcLenBlocks[wx]; k++)
                // {
                // 	T index = level_prev_index[wx][l[wx] - 3] + k * 32 + lx;
                // 	int condition = index < refLen[wx] && (cl[index] & (0x01 << (l[wx] - 2)));
                // 	unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
                // 	if (newmask != 0)
                // 	{
                // 		uint elected_lane_deq = __ffs(newmask) - 1;
                // 		current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
                // 		break;
                // 	}
                // }


                T startIndex = level_prev_index[wx][l[wx]- 3];
                T newIndex = cl[startIndex]  == l[wx];
                while(newIndex == 0)
                {
                    startIndex++;
                    newIndex = cl[startIndex] == l[wx];
                }

                if (lx == 0)
                {
                    //current_node_index[0] = finalIndex;
                    current_node_index[wx] = startIndex;
                    level_prev_index[wx][l[wx] - 3] = current_node_index[wx] + 1;
                    level_index[wx][l[wx] - 3]++;
                    new_level[wx] = l[wx];
                }

                __syncwarp(partMask);

                uint64 warpCountIn = 0;
                const T dst = encode[scounter*j + current_node_index[wx]];
        
                //bool limit = ((l[wx] - 3 + level_count[wx][l[wx] - 4]) >= KCCOUNT) && (dstLen >= KCCOUNT - l[wx]);

                if (subgraph_counters[dst] >= KCCOUNT - l[wx])
                {
                    // if (subgraph_counters[dst] > subgraph_counters[j])
                    // {
                    // 	warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&encode[j * scounter],  subgraph_counters[j],
                    // 		&encode[dst * scounter], subgraph_counters[dst], true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 4);
                    // }
                    // else {
                        warpCountIn = graph::warp_sorted_count_and_set_binary3<T,  CPARTSIZE>(&encode[dst * scounter], subgraph_counters[dst],
                            &encode[j * scounter],  subgraph_counters[j], cl, l[wx] + 1, KCCOUNT, partMask, 4);

                    //}
                    __syncwarp(partMask);
                    if (lx == 0 && warpCountIn > 0)
                    {
                        if (l[wx] + 1 == KCCOUNT)
                            clique_count[wx] += warpCountIn;
                        else if ((l[wx] + 1 < KCCOUNT) /*&& ((l[wx] + warpCountIn) >= KCCOUNT)*/)
                        {
                            //if(warpCountIn >= KCCOUNT - l[wx])
                            {
                                (l[wx])++;
                                (new_level[wx])++;
                                level_count[wx][l[wx] - 3] = warpCountIn;
                                level_index[wx][l[wx] - 3] = 0;
                                level_prev_index[wx][l[wx] - 3] = 0;
                            }
                        }
                    }
                }


                __syncwarp(partMask);
                if (lx == 0)
                {
                    while (new_level[wx] > 4 && level_index[wx][new_level[wx] - 3] >= level_count[wx][new_level[wx] - 3])
                    {
                        (new_level[wx])--;
                    }
                }

                __syncwarp(partMask);
                if (new_level[wx] < l[wx])
                {
                    for (auto k = lx; k < subgraph_counters[j]; k+= CPARTSIZE)
                    {
                        if(cl[k] > new_level[wx])
                            cl[k] = new_level[wx];
                    }	

                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    l[wx] = new_level[wx];
                    current_node_index[wx] = UINT_MAX;
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }
        }

        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}








//No partition
template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_warp_binary_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    T kclique,
    T maxDeg,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T conc_blocks_per_SM,
    T* levelStats,
    T* adj_enc,
    T* adj_tri
)
{
    //will be removed later
    const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
    const int wx = threadIdx.x / 32; // which warp in thread block
    const size_t lx = threadIdx.x % 32;
    __shared__ T level_index[warpsPerBlock][7];
    __shared__ T level_count[warpsPerBlock][7];
    __shared__ T level_prev_index[warpsPerBlock][7];

    __shared__ T level_offset[warpsPerBlock];
    __shared__ uint64 clique_count[warpsPerBlock];
    __shared__ T l[warpsPerBlock];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock];
    __shared__ T src2Start[warpsPerBlock], src2Len[warpsPerBlock];

    __shared__ T num_divs[warpsPerBlock], num_divs_local[warpsPerBlock], 
    encode_offset[warpsPerBlock], *encode[warpsPerBlock], tri_offset[warpsPerBlock], 
    *tri[warpsPerBlock], scounter[warpsPerBlock];
    


    //__shared__ T scl[896];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x)
    {
        if (lx == 0)
        {
            T src = g.rowInd[current.queue[i]];
            srcStart[wx] = g.rowPtr[src];
            srcLen[wx] = g.rowPtr[src + 1] - srcStart[wx];
            //printf("src = %u, srcLen = %u\n", src, srcLen);
        }
        else if (lx == 1)
        {
            T src2 = g.colInd[current.queue[i]];
            src2Start[wx] = g.rowPtr[src2];
            src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];
        }
        else if(lx == 2)
        {
            tri_offset[wx] = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg) + levelPtr * (warpsPerBlock * maxDeg);
            tri[wx] = &adj_tri[tri_offset[wx] + wx*maxDeg];
            scounter[wx] = 0;
        }

        // //get tri list: by block :!!
        __syncwarp();
        graph::warp_sorted_count_and_set_tri<WARPS_PER_BLOCK, T>(&g.colInd[srcStart[wx]], srcLen[wx], &g.colInd[src2Start[wx]], src2Len[wx],
            tri[wx], &(scounter[wx]));
        
            __syncwarp();
        T mm = (1 << scounter[wx]) - 1;
        if (lx == 0)
            num_divs_local[wx] = (scounter[wx] + 32 - 1) / 32; // 32 here is for div
        else if (lx == 1)
        {
            num_divs[wx] = (maxDeg + 32 - 1) / 32;
            encode_offset[wx] = sm_id * conc_blocks_per_SM * (warpsPerBlock* maxDeg * num_divs[wx]) + levelPtr * (warpsPerBlock *maxDeg * num_divs[wx]);
            encode[wx] = &adj_enc[encode_offset[wx] + wx * maxDeg * num_divs[wx]];
        }

        if(kclique == 3 && lx == 0)
            atomicAdd(counter, scounter[wx]);

    
        __syncwarp(mm);
        //Encode
        for (unsigned long long j = 0; j < scounter[wx]; j++)
        {
            for (unsigned long long k = lx; k < num_divs_local[wx]; k += 32)
            {
                encode[wx][j * num_divs_local[wx] + k] = 0x00;
            }
            __syncwarp(mm);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true>(tri[wx], scounter[wx],
                &g.colInd[g.rowPtr[tri[wx][j]]], g.rowPtr[tri[wx][j] + 1] - g.rowPtr[tri[wx][j]],
                 &encode[wx][j * num_divs_local[wx]]);
        }

        __syncwarp(mm); //Done encoding
        level_offset[wx] = sm_id * conc_blocks_per_SM * (warpsPerBlock * num_divs[wx] * 7) + levelPtr * (warpsPerBlock * num_divs[wx] * 7);
        T* cl = &current_level[level_offset[wx] + wx * (num_divs[wx] * 7)];
    
        for (unsigned long long j = 0; j < scounter[wx]; j++)
        {
            if (lx < 7)
            {
                level_count[wx][lx] = 0;
                level_index[wx][lx] = 0;
                level_prev_index[wx][lx] = 0;
            }
            else if (lx == 7 + 1)
            {
                l[wx] = 4;
                clique_count[wx] = 0;
            }
            for (unsigned long long k = lx; k < num_divs_local[wx] * 7; k += 32)
            {
                cl[k] = 0x00;
            }
            //get warp count ??
            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local[wx]; k += 32)
            {
                warpCount += __popc(encode[wx][j * num_divs_local[wx] + k]);
            }
            reduce_part<T, 32>(mm, warpCount);

            if (lx == 0 && l[wx] == kclique)
                clique_count[wx] += warpCount;
            else if (lx == 0 && kclique > 4 && warpCount >= kclique - 3)
            {
                level_count[wx][l[wx] - 4] = warpCount;
                level_index[wx][l[wx] - 4] = 0;
                level_prev_index[wx][l[wx] - 4] = 0;
            }
            __syncwarp(mm);
            while (level_count[wx][l[wx] - 4] > level_index[wx][l[wx] - 4])
            {
                //First Index
                T* from = l[wx] == 4 ? &(encode[wx][num_divs_local[wx] * j]) : &(cl[num_divs_local[wx] * (l[wx] - 4)]);
                T* to = &(cl[num_divs_local[wx] * (l[wx] - 3)]);
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
                    level_index[wx][l[wx] - 4]++;
                }

                __syncwarp(mm);
                //Intersect
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local[wx]; k += 32)
                {
                    to[k] = from[k] & encode[wx][newIndex * num_divs_local[wx] + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, 32>(mm, warpCount);

                if (lx == 0)
                {
                    if (l[wx] + 1 == kclique)
                        clique_count[wx] += warpCount;
                    else if (l[wx] + 1 < kclique && warpCount >= kclique - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        level_index[wx][l[wx] - 4] = 0;
                        level_prev_index[wx][l[wx] - 4] = 0;
                    }
                
                    //Readjust
                    while (l[wx] > 4 && level_index[wx][l[wx] - 4] >= level_count[wx][l[wx] - 4])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(mm);
            }
            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }

            __syncwarp();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_o(
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
    
    //__shared__ unsigned short level_index[numPartitions][8];
    __shared__ short level_count[numPartitions][8];
    __shared__ unsigned short level_prev_index[numPartitions][8];

    //__shared__ T level_offset[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T srcStart, srcLen;
    __shared__ T src2Start, src2Len;
    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;
    
    //__shared__ T scl[896];

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
            T src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            T src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
            scounter = 0;
            tc = 0;
        }

        //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();
        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
        }

        if(KCCOUNT == 3 && threadIdx.x == 0)
            atomicAdd(counter, scounter);

    
        __syncthreads();
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

        while(wtc[wx] < scounter)
        //for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            T j = wtc[wx];
            /*level_offset[wx]*/ T aa = sm_id * CBPSM * (numPartitions * NUMDIVS * 8) + levelPtr * (numPartitions * NUMDIVS * 8);
            T* cl = &current_level[/*level_offset[wx]*/ aa + wx * (NUMDIVS * 8)];
            if (lx < 8)
            {
                level_count[wx][lx] = 0;
                //level_index[wx][lx] = 0;
                level_prev_index[wx][lx] = 0;
            }
            if (lx == 0)
            {
                l[wx] = 4;
                clique_count[wx] = 0;
            }

            //get warp count ??
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0 && l[wx] == KCCOUNT)
                clique_count[wx] += warpCount;
            else if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
            {
                level_count[wx][l[wx] - 4] = warpCount;
                //level_index[wx][l[wx] - 4] = 0;
                level_prev_index[wx][l[wx] - 4] = 0;
            }
             __syncwarp(partMask);
            while (level_count[wx][l[wx] - 4] > 0 /*level_index[wx][l[wx] - 4]*/)
            {
                 //First Index
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
                    //level_index[wx][l[wx] - 4]++;
                    level_count[wx][l[wx] - 4]--; 
                }

                 //Intersect
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                        clique_count[wx] += warpCount;
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        //level_index[wx][l[wx] - 4] = 0;
                        level_prev_index[wx][l[wx] - 4] = 0;
                    }
                
                    //Readjust
                    while (l[wx] > 4 &&  level_count[wx][l[wx] - 4] <= 0)//level_index[wx][l[wx] - 4] >= level_count[wx][l[wx] - 4])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }
            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }

            __syncwarp(partMask);

            if(lx == 0)
            wtc[wx] = atomicAdd(&(tc), 1);
            __syncwarp(partMask);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}






//Not good, so much global memory
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_o_global(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,
    T* level_index_global,
    T* level_count_global,
    T* level_prev_index_global
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int lx = threadIdx.x % CPARTSIZE;
    
    const int aa = numPartitions * 8;
    //__shared__ unsigned short level_index[aa];
    __shared__ unsigned short level_count[aa];
    __shared__ unsigned short level_prev_index[aa];

    __shared__ T level_offset[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T srcStart, srcLen;
    __shared__ T src2Start, src2Len;

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;

    __shared__ T stack_start, spp;
    __shared__ T *level_index;
    //__shared__ T *level_count;
    //__shared__ T *level_prev_index;
    


    //__shared__ T scl[896];

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
            T src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            T src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset  /*srcStart[wx]*/];

            spp = 8;
            stack_start = sm_id * CBPSM * (spp*numPartitions) + levelPtr * (spp*numPartitions);
            level_index = &level_index_global[stack_start];
            // level_count = &level_count_global[stack_start];
            // level_prev_index = &level_prev_index_global[stack_start];

            scounter = 0;
            tc = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
        }

        if(KCCOUNT == 3 && threadIdx.x == 0)
            atomicAdd(counter, scounter);

    
        __syncthreads();
        //Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        // T mm = (1 << scounter) - 1;
        // mm = mm << ((wx/numPartitions) * CPARTSIZE);
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

        while(wtc[wx] < scounter)
        //for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            T j = wtc[wx];
            level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * 8) + levelPtr * (numPartitions * NUMDIVS * 8);
            T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * 8)];
            if (lx < 8)
            {
                //level_count[wx][lx] = 0;
                level_count[spp*wx + lx] = 0;

                //level_index[wx][lx] = 0;
                level_index[spp*wx + lx] = 0;


                //level_prev_index[wx][lx] = 0;
                level_prev_index[spp*wx + lx] = 0;
            }
            if (lx == 0)
            {
                l[wx] = 4;
                clique_count[wx] = 0;
            }

            //get warp count ??
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0 && l[wx] == KCCOUNT)
                clique_count[wx] += warpCount;
            else if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
            {
                level_count[spp*wx + l[wx] - 4] = warpCount;
                level_index[spp*wx + l[wx] - 4] = 0;
                level_prev_index[spp*wx + l[wx] - 4] = 0;
            }
             __syncwarp(partMask);
            while (level_count[spp*wx + l[wx] - 4] > level_index[spp*wx + l[wx] - 4])
            {
                 //First Index
                T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
                T* to = &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[spp*wx + l[wx] - 4] / 32;
                T maskIndex = ~((1 << (level_prev_index[spp*wx + l[wx] - 4] & 0x1F)) -1);
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
                    level_prev_index[spp*wx + l[wx] - 4] = newIndex + 1;
                    level_index[spp*wx + l[wx] - 4]++;
                }

                 //Intersect
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);
                // warpCount += __shfl_down_sync(partMask, warpCount, 4);
                // warpCount += __shfl_down_sync(partMask, warpCount, 2);
                // warpCount += __shfl_down_sync(partMask, warpCount, 1);

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                        clique_count[wx] += warpCount;
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[spp*wx + l[wx] - 4] = warpCount;
                        level_index[spp*wx + l[wx] - 4] = 0;
                        level_prev_index[spp*wx + l[wx] - 4] = 0;
                    }
                
                    //Readjust
                    while (l[wx] > 4 && level_index[spp*wx + l[wx] - 4] >= level_count[spp*wx + l[wx] - 4])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }
            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                //cpn[current.queue[i]] = clique_count[wx];
            }

            __syncwarp(partMask);

            if(lx == 0)
            wtc[wx] = atomicAdd(&(tc), 1);
            __syncwarp(partMask);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}




template<typename T>
struct KCTask
{
    T queue_encode_index;
    T level;
    T level_count;
};


template<typename T, uint CPARTSIZE, uint numPartitions>
__device__ __forceinline__ uint64 explore_branch_o(uint start_level, uint num_divs_local, uint *l, uint partMask,
    T startEncodeIndex, T* encode, T* cl, T* level_count, T* level_index, T *level_prev_index)
{
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    uint64 globalCount = 0;
    

    while (level_count[l[wx] - start_level] > level_index[l[wx] - start_level])
    {
        T l_mi_sl = l[wx] - start_level;
        T* from = (l_mi_sl==0) ? &(encode[num_divs_local * startEncodeIndex]) : &(cl[num_divs_local * l_mi_sl]);
        T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
        T maskBlock = level_prev_index[l_mi_sl] / 32;
        T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) -1);
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
            level_prev_index[l_mi_sl] = newIndex + 1;
            level_index[l_mi_sl]++;
        }

        uint64 warpCount = 0;
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {
            to[k] = from[k] & encode[newIndex * num_divs_local + k];
            warpCount += __popc(to[k]);
        }
        reduce_part<T, CPARTSIZE>(partMask, warpCount);

        if(lx == 0)
        {
            if (l[wx] + 1 == KCCOUNT)
                globalCount += warpCount;
            else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
            {
                (l[wx])++;
                level_count[l[wx] - start_level] = warpCount;
                level_index[l[wx] - start_level] = 0;
                level_prev_index[l[wx] - start_level] = 0;
            }

        
            //Readjust
            while (l[wx] > start_level && level_index[l[wx] - start_level] >= level_count[l[wx] - start_level])
            {
                (l[wx])--;
            }
            
        }

        __syncwarp(partMask);

    }

    return globalCount;
}



//very bad: sync_shfl is very low performance !!
template<typename T, uint CPARTSIZE, uint numPartitions>
__device__ __forceinline__ uint64 explore_branch_sync(uint start_level, uint num_divs_local, uint *l, uint partMask,
    T startEncodeIndex, T* encode, T* cl, T* level_count, T* level_index, T *level_prev_index)
{
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    uint64 globalCount = 0;

    T ll = l[wx];
    
    

    while (level_count[ll- start_level] > level_index[ll - start_level])
    {
        T l_mi_sl = ll - start_level;
        T* from = (l_mi_sl==0) ? &(encode[num_divs_local * startEncodeIndex]) : &(cl[num_divs_local * l_mi_sl]);
        T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
        T maskBlock = level_prev_index[l_mi_sl] / 32;
        T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) -1);
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
            level_prev_index[l_mi_sl] = newIndex + 1;
            level_index[l_mi_sl]++;
        }

        uint64 warpCount = 0;
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {
            to[k] = from[k] & encode[newIndex * num_divs_local + k];
            warpCount += __popc(to[k]);
        }
        reduce_part<T, CPARTSIZE>(partMask, warpCount);

        warpCount = __shfl_sync(partMask,  warpCount, 0, CPARTSIZE);
    
        if (ll + 1 == KCCOUNT)
            globalCount += warpCount;
        else if (ll + 1 < KCCOUNT && warpCount >= KCCOUNT - ll)
        {
            (ll)++;
            if(lx == 0)
            {
                level_count[ll - start_level] = warpCount;
                level_index[ll - start_level] = 0;
                level_prev_index[ll - start_level] = 0;
            }
        }

        __syncwarp(partMask);
        
        //Readjust
        while (ll > start_level && level_index[ll - start_level] >= level_count[ll - start_level])
        {
            (ll)--;
        }
    }

    return globalCount;
}


template<typename T, uint CPARTSIZE, uint numPartitions>
__device__ __forceinline__ uint64 explore_branch(KCTask<T> *task, T* queue_encode, uint num_divs_local, uint partMask,
    T* encode, T* cl, T* level_count, T* level_index, T *level_prev_index, int j)
{
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    uint64 globalCount = 0;


    __shared__ T ll[numPartitions];

    T l = task->level;
    ll[wx] = task->level;
    level_count[0] = task->level_count;
    level_index[0] = 0;
    level_prev_index[0] = 0;
    __syncwarp(partMask);

    while (level_count[ll[wx]- l] > level_index[ll[wx] - l])
    {
        T l_mi_sl = ll[wx] - l;
        T* from = (l_mi_sl==0) ?  /*&(encode[num_divs_local * j])*/ &(queue_encode[task->queue_encode_index]) : &(cl[num_divs_local * l_mi_sl]);
        T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
        T maskBlock = level_prev_index[l_mi_sl] / 32;
        T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) -1);
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
            level_prev_index[l_mi_sl] = newIndex + 1;
            level_index[l_mi_sl]++;
        }

        uint64 warpCount = 0;
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {
            to[k] = from[k] & encode[newIndex * num_divs_local + k];
            warpCount += __popc(to[k]);
        }
        reduce_part<T, CPARTSIZE>(partMask, warpCount);

        if(lx == 0)
        {
            if (ll[wx] + 1 == KCCOUNT)
                globalCount += warpCount;
            else if (ll[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - ll[wx])
            {
                (ll[wx])++;
                level_count[ll[wx] - l] = warpCount;
                level_index[ll[wx] - l] = 0;
                level_prev_index[ll[wx] - l] = 0;
            }

        
            //Readjust
            while (ll[wx] > l && level_index[ll[wx] - l] >= level_count[ll[wx] - l])
            {
                (ll[wx])--;
            }
            
        }

        __syncwarp(partMask);
    }

    return globalCount;
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,
    //simt::atomic<KCTask<T>, simt::thread_scope_device> *queue_data
    KCTask<T> *queue_data,
    T* queue_encode
)
{
    __shared__ T queue_count;
     

    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ T level_index[numPartitions][7];
    __shared__ T level_count[numPartitions][7];
    __shared__ T level_prev_index[numPartitions][7];

    __shared__ T level_offset[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions], tc, wtc[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T srcStart, srcLen;
    __shared__ T src2Start, src2Len;

    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;


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
            T src = g.rowInd[current.queue[i]];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            //printf("src = %u, srcLen = %u\n", src, srcLen);
        }
        else if (threadIdx.x == 1)
        {
            T src2 = g.colInd[current.queue[i]];
            src2Start = g.rowPtr[src2];
            src2Len = g.rowPtr[src2 + 1] - src2Start;
        }
        else if(threadIdx.x == 2)
        {
            tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
            scounter = 0;
            tc = 0;
        }
        else if(threadIdx.x == 3)
        {
            queue_count = 0;
        }

        // //get tri list: by block :!!
        __syncthreads();
        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (scounter + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
        }

        if(KCCOUNT == 3 && threadIdx.x == 0)
            atomicAdd(counter, scounter);

    
        __syncthreads();
        //Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
        
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
                 &encode[j * num_divs_local]);
        }

        __syncthreads(); //Done encoding
        level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * 7) + levelPtr * (numPartitions * NUMDIVS * 7);
        T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * 7)];
        KCTask<T>* queue_data_block = &(queue_data[sm_id * CBPSM * (QUEUE_SIZE) + levelPtr * (QUEUE_SIZE)]);
        T* queue_encode_block = &(queue_encode[sm_id * CBPSM * (QUEUE_SIZE*NUMDIVS) + levelPtr * (QUEUE_SIZE*NUMDIVS)]);
        // if(lx == 0)
        // 	wtc[wx] = atomicAdd(&(tc), 1);
        // __syncwarp(partMask);

        //while(wtc[wx] < scounter)
        for (unsigned long long j = wx; j < scounter; j += numPartitions)
        {
            //T j = wtc[wx];
            if (lx < 7)
            {
                level_count[wx][lx] = 0;
                level_index[wx][lx] = 0;
                level_prev_index[wx][lx] = 0;
            }
            if (lx == 0)
            {
                l[wx] = 4;
                clique_count[wx] = 0;
            }

            //get warp count ??
            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0 && l[wx] == KCCOUNT)
                atomicAdd(counter, warpCount);
            else if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
            {
                level_count[wx][l[wx] - 4] = warpCount;
                level_index[wx][l[wx] - 4] = 0;
                level_prev_index[wx][l[wx] - 4] = 0;
            }
            __syncwarp(partMask);
            // uint64 wc = explore_branch_o<T, CPARTSIZE, numPartitions>(4, num_divs_local, l, partMask,
            // 		j, encode, cl, level_count[wx], level_index[wx], level_prev_index[wx]);

            // if(lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
            // 	atomicAdd(counter, wc); //clique_count[wx] = wc;
            while (level_count[wx][l[wx] - 4] > level_index[wx][l[wx] - 4])
            {
                 //First Index
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
                    level_index[wx][l[wx] - 4]++;
                }

                 //Intersect
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                        clique_count[wx] = 1; //+= warpCount;
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 4] = warpCount;
                        level_index[wx][l[wx] - 4] = 0;
                        level_prev_index[wx][l[wx] - 4] = 0;

                        // if(warpCount > 100)
                        // {
                        // 	if(queue_count < QUEUE_SIZE)
                        // 	{
                        // 		T index = atomicAdd(&queue_count, 1);
                        // 		if(index < QUEUE_SIZE)
                        // 		{
                        // 			level_count[wx][l[wx] - 4] = 0;

                        // 			queue_data_block[index].level = l[wx];
                        // 			queue_data_block[index].level_count = warpCount;
                        // 			queue_data_block[index].queue_encode_index = index * num_divs_local;
                        // 			for (unsigned long long k = 0; k < num_divs_local; k++)
                        // 			{
                        // 				queue_encode_block[queue_data_block[index].queue_encode_index + k] = to[k];
                        // 			}
                        // 		}
                        // 	}
        
                        // }
                    }
                
                    //Readjust
                    while (l[wx] > 4 && level_index[wx][l[wx] - 4] >= level_count[wx][l[wx] - 4])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }
            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx] == 1? 0: 1);
                //cpn[current.queue[i]] = clique_count[wx];
            }

            __syncwarp(partMask);

            // if(lx == 0)
            // 	wtc[wx] = atomicAdd(&(tc), 1);
            // __syncwarp(partMask);
        }
        __syncthreads();
        //dequeue simply

        T limit = (queue_count > QUEUE_SIZE) ? QUEUE_SIZE: queue_count;
        for (unsigned long long j = wx; j < limit; j += numPartitions)
        {
            KCTask<T> t = queue_data_block[j];
            uint64 wc = explore_branch<T, CPARTSIZE, numPartitions>(&t, queue_encode_block, num_divs_local, partMask,
                encode, cl, level_count[wx], level_index[wx], level_prev_index[wx], j);
            if(lx == 0)
                atomicAdd(counter, wc); //clique_count[wx] = wc;
        }

        __syncthreads();
    }


    //search in the queue

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}




template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    T* adj_tri,

    T* possible,
    T* level_index_g,
    T* level_count_g,
    T* level_prev_g,
    T* level_r,
    T* level_d,
    T* level_tmp,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[1120];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ uint64 path_more_explore;
    __shared__ T l;
    __shared__ uint64 maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
    __shared__ bool  partition_set[numPartitions];
    __shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_index, *level_prev_index, *rsize, *drop;
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

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
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
            tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

            lo = sm_id * CBPSM * (/*numPartitions **/ NUMDIVS * MAXDEG) + levelPtr * (/*numPartitions **/ NUMDIVS * MAXDEG);
            cl = &current_level[lo/*level_offset[wx]/* + wx * (NUMDIVS * MAXDEG)*/];
            pl = &possible[lo/*level_offset[wx] /*+ wx * (NUMDIVS * MAXDEG)*/];

            level_item_offset = sm_id * CBPSM * (/*numPartitions **/ MAXDEG) + levelPtr * (/*numPartitions **/ MAXDEG);
            level_count = &level_count_g[level_item_offset /*+ wx*MAXDEG*/];
            level_index = &level_index_g[level_item_offset /*+ wx*MAXDEG*/];
            level_prev_index = &level_prev_g[level_item_offset /*+ wx*MAXDEG*/];
            rsize = &level_r[level_item_offset /*+ wx*MAXDEG*/]; // will be removed
            drop = &level_d[level_item_offset /*+ wx*MAXDEG*/];  //will be removed

            level_count[0] = 0;
            level_prev_index[0] = 0;
            level_index[0] = 0;
            l = 3;
            rsize[0] = 1;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

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
        
        // if(KCCOUNT == 3 && threadIdx.x == 0)
        // 	atomicAdd(counter, scounter);

    
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

        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
            partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        //Find the first pivot
        for (T j = wx; j < scounter; j += numPartitions)
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
            if(maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
            {
                atomicMin(&(level_pivot[0]),maxIndex[wx]);
            }
        }
        __syncthreads();

        //Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = 0xFFFFFFFF;
            warpCount += __popc(pl[j]);
        }
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();
        while((level_count[l - 3] > level_index[l - 3]))
        {
            T maskBlock = level_prev_index[l- 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            }
            newIndex =  32*maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();
            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_index[l - 3]++;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                rsize[l-2] = rsize[l-3] + 1;
                drop[l-2] = drop[l-3];
                if(newIndex == level_pivot[l-3])
                    drop[l-2] = drop[l-3] + 1;
            }
            __syncthreads();
            //assert(level_prev_index[l - 2] == newIndex + 1);

            if(rsize[l-2] - drop[l-2] > KCCOUNT)
            {	
                __syncthreads();
                //printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
                if(threadIdx.x == 0)
                { 
                    //printf, go back
                    while (l > 3 && level_index[l - 3] >= level_count[l - 3])
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
                    //remove previous pivots from here
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l-3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF:  sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = scounter + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                //////////////////////////////////////////////////////////////////////
                //Now new pivot generation, then check to extend to new level or not

                //T limit = (srcLen + numPartitions -1)/numPartitions;
                for (T j = wx; j < /*numPartitions*limit*/scounter; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1<<ii)) != 0)
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
                        if(rsize[l-2] >= KCCOUNT)
                        {
                            T c = rsize[l-2] - KCCOUNT;
                            unsigned long long ncr = nCR[ drop[l-2] * 401 + c  ];
                            atomicAdd(counter, ncr/*rsize[l-1]*/);
                        }
                        //printf, go back
                        while (l > 3 && level_index[l - 3] >= level_count[l - 3])
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
                            while (l > 3 && level_index[l - 3] >= level_count[l - 3])
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
                            pl[(l-2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
                            warpCount += __popc(pl[(l-2)*num_divs_local + j]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        __syncthreads(); // Need this for degeneracy > 1024

                        if(threadIdx.x == 0)
                        {
                            l++;
                            level_count[l-3] = 0;
                            level_prev_index[l-3] = 0;
                            level_index[l-3] = 0;
                        }

                        __syncthreads();
                        if(lx == 0 && threadIdx.x < num_divs_local)
                        {
                            atomicAdd(&(level_count[l-3]), warpCount);
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



