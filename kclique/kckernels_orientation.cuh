#pragma once

template<typename T, uint CPARTSIZE>
__device__ void build_induced_subgraph(
    const int wx, const size_t lx,
    graph::COOCSRGraph_d<T> g, T srcStart, T srcLen, 
    T numPartitions, T num_divs_local, T partMask,
    T *encode
    )
{
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
}


template<typename T, uint CPARTSIZE>
__device__ void build_induced_subgraph(
    const int wx, const size_t lx,
    graph::COOCSRGraph_d<T> g, T srcLen, 
    T numPartitions, T num_divs_local, T partMask,
    T* tri, T *encode
    )
{
    for (T j = wx; j < srcLen; j += numPartitions)
    {
    	for (T k = lx; k < num_divs_local; k += CPARTSIZE)
    	{
    		encode[j * num_divs_local + k] = 0x00;
    	}
    	__syncwarp(partMask);
    	graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, srcLen,
    		&g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
    		 &encode[j * num_divs_local]);
    }
}



template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE, uint MAXDEPTH, uint SL>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_subgraph_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    unsigned short* current_level,
    T* levelStats,
    T* adj_enc
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
    __shared__ T encode_offset, *encode;
    __shared__ unsigned short subgraph_counters[1024];
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


            
            for (T k = lx; k < srcLen; k += CPARTSIZE)
            {
                cl[k] = 0x03;
            }
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

                //assert(current_node_index[wx] < subgraph_counters[j]);
                uint64 warpCountIn = 0;
                const T dst = encode[srcLen*j + current_node_index[wx]];
                const T dstStart = 0;
                const T dstLen = subgraph_counters[dst];

                bool limit = ((l[wx] - 1 + level_count[wx][l[wx] - 2]) >= KCCOUNT) && (dstLen >= KCCOUNT - l[wx]);

                if (limit /*dstLen >= KCCOUNT - l[wx]*/)
                {
                    if (dstLen >  subgraph_counters[j])
                    {
                        warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&encode[j * srcLen],  subgraph_counters[j],
                            &encode[dst * srcLen], dstLen, true, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 3);
                    }
                    else {
                        warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&encode[dst * srcLen], dstLen,
                            &encode[j * srcLen],  subgraph_counters[j], false, srcStart, cl, l[wx] + 1, KCCOUNT, partMask, 3);

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
                        if (index <  subgraph_counters[j])
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



template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE, uint MAXDEPTH, uint SL/*Start Level*/>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    uint64* globalCounter)
{



    //Variables
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    
    __shared__ unsigned short level_count[numPartitions][MAXDEPTH];
    __shared__ unsigned short level_prev_index[numPartitions][MAXDEPTH];

    __shared__ T  level_offset[numPartitions];
    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;


    #define LA l[wx] - SL

    reserve_space<T>(sm_id, levelPtr, levelStats);
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //Read Node Info
        if (threadIdx.x == 0)
        {
            T src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        //Mask for each thread group in the warp
        T partMask = get_thread_group_mask<T, CPARTSIZE>(wx);
        level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * MAXDEPTH) + levelPtr * (numPartitions * NUMDIVS * MAXDEPTH);
        T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * MAXDEPTH)];

        //Build Induced Subgraph
        build_induced_subgraph<T, CPARTSIZE>(wx, lx, g, srcStart, srcLen, numPartitions, num_divs_local, partMask, encode);
        __syncthreads(); //Done encoding

        //Explore each subtree at the second level
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            //Init Stack per thread group
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = SL;
                clique_count[wx] = 0;
            }

            //Get number of elements for the next level (3rd)
            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

         
            if (lx == 0)
            {
                //For each subtree, check if we reached the level we are looking for
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > SL && warpCount >= KCCOUNT - SL + 1)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][LA] > 0)
            {

                if(lx == 0)
                    atomicAdd(&cpn[sm_id], 1);

                //Current and Next Level Lists
                T* from = l[wx] == SL ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (LA)]);
                T* to = &(cl[num_divs_local * (LA + 1)]);

                T maskBlock = level_prev_index[wx][LA] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][LA] & 0x1F)) -1);
                T newIndex = get_next_sibling_index<T>(from, maskIndex, maskBlock);

                if (lx == 0)
                {
                    level_prev_index[wx][LA] = newIndex + 1;
                    level_count[wx][LA]--;
                }

                //Intersect
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                //Decide Next Step: Count, Go Deeper, Go back
                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                        clique_count[wx] += warpCount;
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][LA] = warpCount;
                        level_prev_index[wx][LA] = 0;
                    }
                
                    //Go back, if needed
                    while (l[wx] > SL && level_count[wx][LA] == 0 )
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
    }

    __syncthreads();
    release_space<T>(sm_id, levelPtr, levelStats);
}



template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE, uint MAXDEPTH, uint SL/*Start Level*/>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_o2(
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
	//Variables
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const int lx = threadIdx.x % CPARTSIZE;
	
	__shared__ unsigned short level_count[numPartitions][MAXDEPTH];
	__shared__ unsigned short level_prev_index[numPartitions][MAXDEPTH];

	__shared__ T level_offset[numPartitions];
	__shared__ uint64 clique_count[numPartitions];
	__shared__ T l[numPartitions], tc, wtc[numPartitions];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart, srcLen;
	__shared__ T src2Start, src2Len;
	__shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;

    #define LA l[wx] - SL

    reserve_space<T>(sm_id, levelPtr, levelStats);
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

        //Triangle Count
		if(KCCOUNT == 3 && threadIdx.x == 0)
			atomicAdd(counter, scounter);
	
		__syncthreads();
		//Encode
        T partMask = get_thread_group_mask<T, CPARTSIZE>(wx);
        level_offset[wx]= sm_id * CBPSM * (numPartitions * NUMDIVS * MAXDEPTH) + levelPtr * (numPartitions * NUMDIVS * MAXDEPTH);
		T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * MAXDEPTH)];

        build_induced_subgraph<T, CPARTSIZE>(wx, lx, g, scounter, numPartitions, num_divs_local, partMask, tri, encode);
		__syncthreads(); //Done encoding

		if(lx == 0)
			wtc[wx] = atomicAdd(&(tc), 1);
		__syncwarp(partMask);

		while(wtc[wx] < scounter)
		//for (unsigned long long j = wx; j < scounter; j += numPartitions)
		{
			T j = wtc[wx];
			if (lx == 0)
			{
                level_count[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
				l[wx] = SL;
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
			else if (lx == 0 && KCCOUNT > SL && warpCount >= KCCOUNT - SL + 1)
			{
				level_count[wx][LA] = warpCount;
				level_prev_index[wx][LA] = 0;
			}
		 	__syncwarp(partMask);
			while (level_count[wx][LA] > 0)
			{
			 	//Current and Next Level Lists
				T* from = l[wx] == SL ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (LA)]);
				T* to = &(cl[num_divs_local * (LA + 1)]);

                T maskBlock = level_prev_index[wx][LA] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][LA] & 0x1F)) -1);
                T newIndex = get_next_sibling_index<T>(from, maskIndex, maskBlock);
				
				if (lx == 0)
				{
					level_prev_index[wx][LA] = newIndex + 1;
					level_count[wx][LA]--;
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
						level_count[wx][LA] = warpCount;
						level_prev_index[wx][LA] = 0;
					}
				
					//Readjust
					while (l[wx] > SL &&  level_count[wx][LA] == 0)
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
    release_space<T>(sm_id, levelPtr, levelStats);
}

