
// //For PL, yuo cannot mark it only you need previous PL copies !!
// template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
// __launch_bounds__(BLOCK_DIM_X, 16)
// __global__ void
// kckernel_node_block_warp_pivot_count_wrong(
// 	uint64* counter,
// 	graph::COOCSRGraph_d<T> g,
// 	const  graph::GraphQueue_d<T, bool>  current,
// 	T* current_level,
// 	uint64* cpn,
// 	T* levelStats,
// 	T* adj_enc,

// 	T* possible,
// 	T* level_index_g,
// 	T* level_count_g,
// 	T* level_prev_g,
// 	T* level_r,
// 	T* level_d,
// 	T* level_tmp,
// 	unsigned long long* nCR
// )
// {
// 	//will be removed later
// 	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
// 	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
// 	const size_t lx = threadIdx.x % CPARTSIZE;

// 	//__shared__ T  level_offset[numPartitions], level_item_offset[numPartitions]; //for l and p
// 	__shared__ T level_pivot[512];
// 	__shared__ uint64 clique_count[numPartitions];
// 	__shared__ uint64 path_more_explore;
// 	__shared__ T l, new_l;
// 	__shared__ uint64 maxIntersection;
// 	__shared__ uint32_t  sm_id, levelPtr;
// 	__shared__ T src, srcStart, srcLen, srcLenBlocks;
// 	__shared__ bool  partition_set[numPartitions];
// 	__shared__ T encode_offset, *encode;
// 	__shared__ T *pl, *cl;
// 	__shared__ T *level_count, *level_index, *level_prev_index, *rsize, *drop;
// 	__shared__ T lo, po, level_item_offset;
// 	__shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
// 	__shared__ 	T lastMask_i, lastMask_ii;
// 	// __shared__ T pl_counter[512];
// 	__shared__ T cl_counter[512];

// 	__shared__ T subgraph_counters[512];

// 	if (threadIdx.x == 0)
// 	{
// 		sm_id = __mysmid();
// 		T temp = 0;
// 		while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
// 		{
// 			temp++;
// 		}
// 		levelPtr = temp;
// 	}
// 	__syncthreads();

// 	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0] && current.queue[i] == 40; i += gridDim.x)
// 	{
// 		__syncthreads();
// 		//block things
// 		if (threadIdx.x == 0)
// 		{
// 			src = current.queue[i];
// 			srcStart = g.rowPtr[src];
// 			srcLen = g.rowPtr[src + 1] - srcStart;
// 			srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

// 			printf("SRC = %u, SRCLEN = %u, %llu\n", src, srcLen, *counter);

// 			//printf("src = %u, srcLen = %u\n", src, srcLen);
// 			encode_offset = sm_id * CBPSM * (MAXDEG * MAXDEG) + levelPtr * (MAXDEG * MAXDEG);
// 			encode = &adj_enc[encode_offset];

// 			lo = sm_id * CBPSM * (MAXDEG* MAXDEG) + levelPtr * (MAXDEG* MAXDEG);
// 			level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
// 			cl = &current_level[lo];
// 			pl = &possible[level_item_offset];

			
// 			level_count = &level_count_g[level_item_offset];
// 			level_index = &level_index_g[level_item_offset];
// 			level_prev_index = &level_prev_g[level_item_offset];
// 			rsize = &level_r[level_item_offset ]; // will be removed
// 			drop = &level_d[level_item_offset];  //will be removed

// 			level_count[0] = 0;
// 			level_prev_index[0] = 0;
// 			level_index[0] = 0;
// 			l = 2;
// 			rsize[0] = 1;
// 			drop[0] = 0;

// 			level_pivot[0] = 0xFFFFFFFF;

// 			maxIntersection = 0;

// 			lastMask_i = srcLen / 32;
// 			lastMask_ii = (1 << (srcLen & 0x1F)) - 1;

// 			cl_counter[0] = 0;
// 		}
// 		__syncthreads();
// 		//Encode Clear
	
// 		for (T j = wx; j < srcLen; j += numPartitions)
// 		{
// 			for (T k = lx; k < srcLen; k += CPARTSIZE)
// 			{
// 				encode[j * srcLen + k] = 0xFFFFFFFF;
// 			}
// 		}
// 		__syncthreads();
// 		//Full Subgraph
// 		for (T j = wx; j < srcLen; j += numPartitions)
// 		{
// 			graph::warp_sorted_count_and_subgraph_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
// 				&g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
// 				j, srcLen, encode);
// 		}
// 		__syncthreads(); //Done encoding

// 		//compact
// 		for (T j = wx; j < srcLen; j += numPartitions)
// 		{
// 			if(lx == 0)
// 			{
// 				subgraph_counters[j] = 0;
// 				for(T k = 0; k< srcLen; k++)
// 				{
// 					if(encode[j*srcLen + k] != 0xFFFFFFFF)
// 					{
// 						encode[j*srcLen + subgraph_counters[j]] = encode[j*srcLen + k];
// 						subgraph_counters[j]++;
// 					}
// 				}
// 			}
// 		}

// 		__syncthreads();

// 		if(threadIdx.x == 0)
// 		{
// 			for(T k = 0; k< srcLen; k++)
// 			{
// 				printf("%u: ", k);
// 				for(T kk = 0; kk< subgraph_counters[k]; kk++)
// 				{
// 					printf("%u,", encode[k*srcLen + kk]);
// 				}
// 				printf("\n");
// 			}
// 		}

// 		//Find the first pivot
// 		if(lx == 0)
// 		{
// 			maxCount[wx] = 0;
// 			maxIndex[wx] = 0xFFFFFFFF;
// 			partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
// 			partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
// 		}
// 		__syncthreads();
	
// 		for (T j = wx; j < srcLen; j += numPartitions)
// 		{
// 			if(lx == 0 && maxCount[wx] < subgraph_counters[j])
// 			{
// 				maxCount[wx] = subgraph_counters[j];
// 				maxIndex[wx] = j;
// 			}	
// 		}
// 		__syncthreads();
// 		if(lx == 0)
// 		{
// 			atomicMax(&(maxIntersection), maxCount[wx]);
// 		}
// 		__syncthreads();
// 		if(lx == 0)
// 		{
// 			if(maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
// 			{
// 				atomicMin(&(level_pivot[0]),maxIndex[wx]);
// 			}
// 		}
// 		__syncthreads();

// 		//Prepare the Possible and Intersection Encode Lists
// 		uint64 warpCount = 0;
// 		for (T j = threadIdx.x; j < srcLen && maxIntersection > 0; j += BLOCK_DIM_X)
// 		{
// 			bool found = false;
// 			const T searchVal =  j;
// 			const T lb = graph::binary_search<T>(&encode[level_pivot[0] * srcLen], 0, subgraph_counters[level_pivot[0]], searchVal, found);
// 			if(!found)
// 			{
// 				pl[j] = 2;
// 				warpCount++;
// 			}
// 			else
// 			{
// 				pl[j] = 1;
// 			}
// 		}
// 		reduce_part<T>(partMask[wx], warpCount);
// 		if(lx == 0 && threadIdx.x < srcLen)
// 		{
// 			atomicAdd(&(level_count[0]), (T)warpCount);
// 		}
// 		__syncthreads();

// 		if(threadIdx.x == 0)
// 		{printf("Before While:\n");
// 		printf("Pivot = %u, Pivot List Len=%u\n", level_pivot[0], level_count[0]);
	
// 		for(T k = 0; k<srcLen; k++)
// 			printf("%u, ", pl[k]);
	
// 		printf("\n");
// 		}


// 		// // //Explore the tree
// 		while((level_count[l - 2] > level_index[l - 2]))
// 		{
// 			T startIndex = level_prev_index[l- 2];
// 			T newIndex = pl[startIndex];
// 			while(newIndex != l)
// 			{
// 				startIndex++;
// 				newIndex = pl[startIndex];
// 			}
// 			__syncthreads();
// 			if (threadIdx.x == 0)
// 			{
// 				level_prev_index[l - 2] = startIndex + 1;
// 				level_index[l - 2]++;
// 				level_pivot[l - 1] = 0xFFFFFFFF;
// 				new_l = l;
// 				path_more_explore = false;
// 				maxIntersection = 0;
// 				rsize[l-1] = rsize[l-2] + 1;
// 				drop[l-1] = drop[l-2];
// 				if(startIndex == level_pivot[l-2])
// 					drop[l-1] = drop[l-2] + 1;
// 			}
// 			__syncthreads();

// 			if(threadIdx.x == 0)
// 				printf("L=%u, Pivot = %u, newIndex = %u, startInex = %u, levelPrev=%u\n", l, level_pivot[l-2], newIndex, startIndex, level_prev_index[l-2]);


// 			if(threadIdx.x == 0)
// 			{printf("---------Inside While:\n");
// 			printf("---------Pivot = %u, Pivot List Len=%u\n---------PL: ", level_pivot[l-2], level_count[l-2]);
		
// 			for(T k = 0; k<srcLen; k++)
// 				printf("%u, ", pl[ k]);
		
// 			printf("\n");

// 			}


// 			if(rsize[l-1] - drop[l-1] > KCCOUNT)
// 			{	
// 				__syncthreads();
// 				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
// 				if(threadIdx.x == 0)
// 				{
// 					T c = rsize[l-1] - KCCOUNT;
// 					unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
// 					atomicAdd(counter, ncr/*rsize[l-1]*/);
					
// 					//printf, go back
// 					while (new_l > 2 && level_index[new_l - 2] >= level_count[new_l - 2])
// 					{
// 						(new_l)--;
// 					}
// 				}
// 				__syncthreads();
// 				if (new_l < l)
// 				{
// 					for (auto k = threadIdx.x; k < srcLen; k+=BLOCK_DIM_X)
// 					{
// 						if (pl[k] > new_l)
// 							pl[k] = new_l;
// 					}
// 					__syncthreads();
// 				}
// 				if (threadIdx.x == 0)
// 				{
// 					l = new_l;
// 				}
// 				__syncthreads();
// 			}
// 			else
// 			{
// 				__syncthreads();
// 				for (T j = threadIdx.x; j < srcLen; j += BLOCK_DIM_X)
// 				{
// 					cl[(l-1)*srcLen + j] = 0xFFFFFFFF;
// 				}
// 				__syncthreads();
// 				// Now prepare intersection list
// 				T* to =  &(cl[srcLen * (l - 1)]);
// 				T len = l == 2? subgraph_counters[startIndex] : cl_counter[l-2];
// 				if(l == 2)
// 				{
// 					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
// 					{
// 						T dest = encode[startIndex* srcLen + j];
// 						if(dest > startIndex || (dest < startIndex && pl[dest] != l))
// 							to[j] = dest;
// 					}
// 					__syncthreads();
// 				}
// 				else
// 				{
// 					T* from = &(cl[srcLen * (l - 2)]);
// 					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
// 					{
// 						bool found = false;
// 						const T searchVal =  from[j];
// 						const T lb = graph::binary_search<T>(&encode[startIndex * srcLen], 0, subgraph_counters[startIndex], searchVal, found);
// 						if(found &&  (searchVal > startIndex || (searchVal < startIndex && pl[searchVal] != l)))
// 						{
// 							to[j] = searchVal;
// 						}
// 						else
// 							to[j] = 0xFFFFFFFF;
// 					}
// 					__syncthreads();
// 				}
// 				if(threadIdx.x == 0)
// 				{
// 					cl_counter[l-1] = 0;
// 					for(T k = 0; k< len; k++)
// 					{
// 						if(to[k] != 0xFFFFFFFF)
// 						{
// 							to[cl_counter[l-1]] = to[k];
// 							cl_counter[l-1]++;
// 						}
// 					}
// 				}
// 				__syncthreads();


// 				if(lx == 0)
// 				{	
// 					partition_set[wx] = false;
// 					maxCount[wx] = srcLen + 1; //make it shared !!
// 					maxIndex[wx] = 0;
// 				}
// 				__syncthreads();
// 				//////////////////////////////////////////////////////////////////////
// 				//Now new pivot generation, then check to extend to new level or not
// 				for (T j = wx; j < cl_counter[l-1]; j += numPartitions)
// 				{
// 					uint64 warpCount = 0;
// 					T searchIndex = to[j];
// 					for (T k = lx; k < cl_counter[l-1]; k += CPARTSIZE)
// 					{
// 						bool found = false;
// 						const T searchVal =  to[k];
// 						const T lb = graph::binary_search<T>(&encode[searchIndex * srcLen], 0, subgraph_counters[searchIndex], searchVal, found);
// 						if(found)
// 						{
// 							warpCount++;
// 						}
// 					}
// 					reduce_part<T>(partMask[wx], warpCount);
// 					if(lx == 0 && maxCount[wx] == srcLen + 1)
// 					{
// 						partition_set[wx] = true;
// 						path_more_explore = true; //shared, unsafe, but okay
// 						maxCount[wx] = warpCount;
// 						maxIndex[wx] = to[j];
// 					}
// 					else if(lx == 0 && maxCount[wx] < warpCount)
// 					{
// 						maxCount[wx] = warpCount;
// 						maxIndex[wx] = to[j];
// 					}	
					
// 				}

// 				__syncthreads();

// 				if(threadIdx.x == 0){
// 					printf("\n---------NEw CL: ");

// 					for(T k = 0; k<cl_counter[l-1]; k++)
// 						printf("%u, ", cl[(l-1)*srcLen + k]);
// 					printf("\n");
// 				}
				
// 				if(!path_more_explore)
// 				{
// 					__syncthreads();
// 					if(threadIdx.x == 0)
// 					{	
// 						if(rsize[l-1] >= KCCOUNT)
// 						{
// 							T c = rsize[l-1] - KCCOUNT;
// 							unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
// 							atomicAdd(counter, ncr/*rsize[l-1]*/);
// 						}
// 						//printf, go back
// 						while (new_l > 2 && level_index[new_l - 2] >= level_count[new_l - 2])
// 						{
// 							(new_l)--;
// 						}
// 					}
// 					__syncthreads();
// 					if (new_l < l)
// 					{
// 						for (auto k = threadIdx.x; k < srcLen; k+=BLOCK_DIM_X)
// 						{
// 							if (pl[k] > new_l)
// 								pl[k] = new_l;
// 						}
// 						__syncthreads();
// 					}

// 					if (threadIdx.x == 0)
// 					{
// 						l = new_l;
// 					}
// 					__syncthreads();
// 				}
// 				else
// 				{
// 					__syncthreads();
// 					if(lx == 0 && partition_set[wx])
// 					{
// 						atomicMax(&(maxIntersection), maxCount[wx]);
// 					}
// 					__syncthreads();

// 					if(lx == 0 && maxIntersection == maxCount[wx])
// 					{	
// 						atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
// 					}
				
// 					__syncthreads();
// 					if(threadIdx.x == 0)
// 					{
// 						printf("---------New Pivot = %u\n", level_pivot[l-1]);
// 					}

// 					uint64 warpCount = 0;
// 					for (T j = threadIdx.x; j < cl_counter[l-1]; j += BLOCK_DIM_X)
// 					{

// 						bool found = false;
// 						const T searchVal =  to[j];
// 						const T lb = graph::binary_search<T>(&encode[level_pivot[l - 1] * srcLen], 0, subgraph_counters[level_pivot[l - 1]], searchVal, found);
// 						if(!found)
// 						{
// 							pl[searchVal] = l + 1;
// 							warpCount++;
// 						}
// 					}
// 					reduce_part<T>(partMask[wx], warpCount);


// 					__syncthreads();
// 					if(threadIdx.x == 0)
// 					{
// 						printf("\n---------NEw PL: ");
	
// 						for(T k = 0; k<srcLen; k++)
// 							printf("%u, ", pl[k]);
// 						printf("\n");
// 					}

// 					if(threadIdx.x == 0)
// 					{
// 						l++;
// 						new_l++;
// 						level_count[l-2] = 0;
// 						level_prev_index[l-2] = 0;
// 						level_index[l-2] = 0;
// 					}

// 					__syncthreads();
// 					if(lx == 0 && threadIdx.x < cl_counter[l-2])
// 					{
// 						atomicAdd(&(level_count[l-2]), warpCount);
// 					}

					
// 					__syncthreads();


// 				}
				
// 			}
// 			__syncthreads();
// 			/////////////////////////////////////////////////////////////////////////
// 		}
// 	}

// 	__syncthreads();
// 	if (threadIdx.x == 0)
// 	{
// 		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
// 	}
// }