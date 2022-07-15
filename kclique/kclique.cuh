


#pragma once
#define QUEUE_SIZE 1024

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "../include/utils.cuh"
#include "../include/Logger.cuh"
#include "../include/CGArray.cuh"


// #include "../triangle_counting/TcBase.cuh"
// #include "../triangle_counting/TcSerial.cuh"
// #include "../triangle_counting/TcBinary.cuh"
// #include "../triangle_counting/TcVariablehash.cuh"
// #include "../triangle_counting/testHashing.cuh"
// #include "../triangle_counting/TcBmp.cuh"

#include "../include/GraphQueue.cuh"


#include "common.cuh"
#include "kckernels_orientation.cuh"
#include "kckernels_pivoting.cuh"
#include "kckernels.cuh"


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_pivot_count_nocub(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc,

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

	//__shared__ T  level_offset[numPartitions], level_item_offset[numPartitions]; //for l and p
	__shared__ T level_pivot[512];
	__shared__ uint64 path_more_explore;
	__shared__ T l;
	__shared__ uint64 maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen, srcLenBlocks;
	__shared__ bool  partition_set[numPartitions];
	__shared__ T encode_offset, *encode;
	__shared__ T *pl, *cl;
	__shared__ T *level_count, *level_index, *level_prev_index, *rsize, *drop;
	__shared__ T lo, level_item_offset;
	__shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
	__shared__ T cl_counter[512];
	__shared__ T subgraph_counters[512];

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
		__syncthreads();
		//block things
		if (threadIdx.x == 0)
		{
			src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
			srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

			//printf("SRC = %u, SRCLEN = %u, %llu\n", src, srcLen, *counter);

			//printf("src = %u, srcLen = %u\n", src, srcLen);
			encode_offset = sm_id * CBPSM * (MAXDEG * MAXDEG) + levelPtr * (MAXDEG * MAXDEG);
			encode = &adj_enc[encode_offset];

			lo = sm_id * CBPSM * (MAXDEG* MAXDEG) + levelPtr * (MAXDEG* MAXDEG);
			cl = &current_level[lo];
			pl = &possible[lo];

			level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
			level_count = &level_count_g[level_item_offset];
			level_index = &level_index_g[level_item_offset];
			level_prev_index = &level_prev_g[level_item_offset];
			rsize = &level_r[level_item_offset ]; // will be removed
			drop = &level_d[level_item_offset];  //will be removed

			level_count[0] = 0;
			level_prev_index[0] = 0;
			level_index[0] = 0;
			l = 2;
			rsize[0] = 1;
			drop[0] = 0;

			level_pivot[0] = 0xFFFFFFFF;
			maxIntersection = 0;
			cl_counter[0] = 0;
		}
		__syncthreads();

		//Encode Clear
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			for (T k = lx; k < srcLen; k += CPARTSIZE)
			{
				encode[j * srcLen + k] = 0xFFFFFFFF;
			}
		}
		__syncthreads();
		//Full Subgraph
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			graph::warp_sorted_count_and_subgraph_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
				&g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
				j, srcLen, encode);
		}
		__syncthreads(); //Done encoding

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

		__syncthreads();

		//Find the first pivot
		if(lx == 0)
		{
			maxCount[wx] = 0;
			maxIndex[wx] = 0xFFFFFFFF;
			partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
			partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		}
		__syncthreads();
	
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			if(lx == 0 && maxCount[wx] < subgraph_counters[j])
			{
				maxCount[wx] = subgraph_counters[j];
				maxIndex[wx] = j;
			}	
		}
		__syncthreads();
		if(lx == 0)
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

		//Creat the first/top Possible (Pivot list)
		uint64 warpCount = 0;
		for (T j = threadIdx.x; j < srcLen && maxIntersection > 0; j += BLOCK_DIM_X)
		{
			bool found = false;
			const T searchVal =  j;
			const T lb = graph::binary_search<T>(&encode[level_pivot[0] * srcLen], 0, subgraph_counters[level_pivot[0]], searchVal, found);
			if(!found)
			{
				pl[j] = j;
				warpCount++;
			}
			else
			{
				pl[j] = 0xFFFFFFFF;
			}
		}
		reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
		if(lx == 0 && threadIdx.x < srcLen)
		{
			atomicAdd(&(level_count[0]), (T)warpCount);
		}
		__syncthreads();

		//Explore the tree
		while((level_count[l - 2] > level_index[l - 2]))
		{
			T startIndex = level_prev_index[l- 2];
			T newIndex = pl[(l-2)*srcLen + startIndex];
			while(newIndex == 0xFFFFFFFF)
			{
				startIndex++;
				newIndex = pl[(l-2)*srcLen + startIndex];
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				level_prev_index[l - 2] = newIndex + 1;
				level_index[l - 2]++;
				level_pivot[l - 1] = 0xFFFFFFFF;
				path_more_explore = false;
				maxIntersection = 0;
				rsize[l-1] = rsize[l-2] + 1;
				drop[l-1] = drop[l-2];
				if(newIndex == level_pivot[l-2])
					drop[l-1] = drop[l-2] + 1;
			}
			__syncthreads();
			
			//Stop condition based on k
			if(rsize[l-1] - drop[l-1] > KCCOUNT)
			{	
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if(threadIdx.x == 0)
				{
					T c = rsize[l-1] - KCCOUNT;
					unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
					atomicAdd(counter, ncr/*rsize[l-1]*/);
					
					//printf, go back
					while (l > 2 && level_index[l - 2] >= level_count[l - 2])
					{
						(l)--;
					}
				}
				__syncthreads();
			}
			else
			{
				__syncthreads();

				//Clear CL and PL of the new level
				for (T j = threadIdx.x; j < srcLen; j += BLOCK_DIM_X)
				{
					pl[(l-1)*srcLen + j] = 0xFFFFFFFF;
					cl[(l-1)*srcLen + j] = 0xFFFFFFFF;
				}
				__syncthreads();
				// Now prepare intersection list
				T* to =  &(cl[srcLen * (l - 1)]);
				T len = l == 2? subgraph_counters[newIndex] : cl_counter[l-2];
				if(l == 2)
				{
					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
					{
						T dest = encode[newIndex* srcLen + j];
						if(dest > newIndex || (dest < newIndex && pl[dest] == 0xFFFFFFFF))
							to[j] = dest;
					}
					__syncthreads();
				}
				else
				{
					T* from = &(cl[srcLen * (l - 2)]);
					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
					{
						bool found = false;
						const T searchVal =  from[j];
						const T lb = graph::binary_search<T>(&encode[newIndex * srcLen], 0, subgraph_counters[newIndex], searchVal, found);
						if(found &&  (searchVal > newIndex || (searchVal < newIndex && pl[(l-2)*srcLen + searchVal] == 0xFFFFFFFF)))
						{
							to[j] = searchVal;
						}
						else
							to[j] = 0xFFFFFFFF;
					}
					__syncthreads();
				}

				//Please use cub here 
				if(threadIdx.x == 0)
				{
					cl_counter[l-1] = 0;
					for(T k = 0; k< len; k++)
					{
						if(to[k] != 0xFFFFFFFF)
						{
							to[cl_counter[l-1]] = to[k];
							cl_counter[l-1]++;
						}
					}
				}
				__syncthreads();


				if(lx == 0)
				{	
					partition_set[wx] = false;
					maxCount[wx] = srcLen + 1; //make it shared !!
					maxIndex[wx] = 0;
				}
				__syncthreads();
				//////////////////////////////////////////////////////////////////////
				//Now new pivot generation, then check to extend to new level or not
				for (T j = wx; j < cl_counter[l-1]; j += numPartitions)
				{
					uint64 warpCount = 0;
					T searchIndex = to[j];
					for (T k = lx; k < cl_counter[l-1]; k += CPARTSIZE)
					{
						bool found = false;
						const T searchVal =  to[k];
						const T lb = graph::binary_search<T>(&encode[searchIndex * srcLen], 0, subgraph_counters[searchIndex], searchVal, found);
						if(found)
						{
							warpCount++;
						}
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
					if(lx == 0 && maxCount[wx] == srcLen + 1)
					{
						partition_set[wx] = true;
						path_more_explore = true; //shared, unsafe, but okay
						maxCount[wx] = warpCount;
						maxIndex[wx] = to[j];
					}
					else if(lx == 0 && maxCount[wx] < warpCount)
					{
						maxCount[wx] = warpCount;
						maxIndex[wx] = to[j];
					}	
					
				}

				__syncthreads(); //All wait here to check to go further
				
				if(!path_more_explore)
				{
					__syncthreads();
					if(threadIdx.x == 0)
					{	
						if(rsize[l-1] >= KCCOUNT)
						{
							T c = rsize[l-1] - KCCOUNT;
							unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
							atomicAdd(counter, ncr/*rsize[l-1]*/);
						}
						//printf, go back
						while (l > 2 && level_index[l - 2] >= level_count[l - 2])
						{
							(l)--;
						}
					}
					__syncthreads();
				}
				else
				{
					__syncthreads();
					if(lx == 0 && partition_set[wx])
					{
						atomicMax(&(maxIntersection), maxCount[wx]);
					}
					__syncthreads();

					if(lx == 0 && maxIntersection == maxCount[wx])
					{	
						atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
					}
				
					__syncthreads();
					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < cl_counter[l-1]; j += BLOCK_DIM_X)
					{

						bool found = false;
						const T searchVal =  to[j];
						const T lb = graph::binary_search<T>(&encode[level_pivot[l - 1] * srcLen], 0, subgraph_counters[level_pivot[l - 1]], searchVal, found);
						if(!found)
						{
							pl[(l-1)*srcLen + searchVal] = searchVal;
							warpCount++;
						}
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

					__syncthreads();
					if(threadIdx.x == 0)
					{
						l++;
						level_count[l-2] = 0;
						level_prev_index[l-2] = 0;
						level_index[l-2] = 0;
					}

					__syncthreads();
					if(lx == 0 && threadIdx.x < cl_counter[l-2])
					{
						atomicAdd(&(level_count[l-2]), warpCount);
					}

					
					__syncthreads();


				}
				
			}
			__syncthreads();
			/////////////////////////////////////////////////////////////////////////
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
kckernel_node_block_warp_pivot_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc,

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

	//__shared__ T  level_offset[numPartitions], level_item_offset[numPartitions]; //for l and p
	__shared__ T level_pivot[512];
	__shared__ uint64 path_more_explore;
	__shared__ T l;
	__shared__ uint64 maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen, srcLenBlocks;
	__shared__ bool  partition_set[numPartitions];
	__shared__ T encode_offset, *encode;
	__shared__ T *pl, *cl;
	__shared__ T *level_count, /**level_index,*/ *level_prev_index, *rsize, *drop;
	__shared__ T lo, level_item_offset;
	__shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
	__shared__ unsigned short cl_counter[512];
	__shared__ unsigned short subgraph_counters[512];

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
		__syncthreads();
		//block things
		if (threadIdx.x == 0)
		{
			src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
			srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

			//printf("SRC = %u, SRCLEN = %u, %llu\n", src, srcLen, *counter);

			//printf("src = %u, srcLen = %u\n", src, srcLen);
			encode_offset = sm_id * CBPSM * (MAXDEG * MAXDEG) + levelPtr * (MAXDEG * MAXDEG);
			encode = &adj_enc[encode_offset];

			lo = sm_id * CBPSM * (MAXDEG* MAXDEG) + levelPtr * (MAXDEG* MAXDEG);
			cl = &current_level[lo];
			pl = &possible[lo];

			level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
			level_count = &level_count_g[level_item_offset];
			//level_index = &level_index_g[level_item_offset];
			level_prev_index = &level_prev_g[level_item_offset];
			rsize = &level_r[level_item_offset ]; // will be removed
			drop = &level_d[level_item_offset];  //will be removed

			level_count[0] = 0;
			level_prev_index[0] = 0;
			//level_index[0] = 0;
			l = 2;
			rsize[0] = 1;
			drop[0] = 0;

			level_pivot[0] = 0xFFFFFFFF;
			maxIntersection = 0;
			cl_counter[0] = 0;
		}
		__syncthreads();

		//Encode Clear
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			for (T k = lx; k < srcLen; k += CPARTSIZE)
			{
				encode[j * srcLen + k] = 0xFFFFFFFF;
			}
		}
		__syncthreads();
		//Full Subgraph
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			graph::warp_sorted_count_and_subgraph_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
				&g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
				j, srcLen, encode);
		}
		__syncthreads(); //Done encoding

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

		__syncthreads();

		//Find the first pivot
		if(lx == 0)
		{
			maxCount[wx] = 0;
			maxIndex[wx] = 0xFFFFFFFF;
			partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
			partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		}
		__syncthreads();
	
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			if(lx == 0 && maxCount[wx] < subgraph_counters[j])
			{
				maxCount[wx] = subgraph_counters[j];
				maxIndex[wx] = j;
			}	
		}
		__syncthreads();
		if(lx == 0)
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

		//Creat the first/top Possible (Pivot list)
		uint64 warpCount = 0;
		for (T j = threadIdx.x; j < srcLen && maxIntersection > 0; j += BLOCK_DIM_X)
		{
			bool found = false;
			const T searchVal =  j;
			const T lb = graph::binary_search<T>(&encode[level_pivot[0] * srcLen], 0, subgraph_counters[level_pivot[0]], searchVal, found);
			if(!found)
			{
				pl[j] = j;
				warpCount++;
			}
			else
			{
				pl[j] = 0xFFFFFFFF;
			}
		}
		reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
		if(lx == 0 && threadIdx.x < srcLen)
		{
			atomicAdd(&(level_count[0]), (T)warpCount);
		}
		__syncthreads();

		//Explore the tree
		while((level_count[l - 2] > 0 /*level_index[l - 2]*/))
		{
			T startIndex = level_prev_index[l- 2];
			T newIndex = pl[(l-2)*srcLen + startIndex];
			while(newIndex == 0xFFFFFFFF)
			{
				startIndex++;
				newIndex = pl[(l-2)*srcLen + startIndex];
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				level_prev_index[l - 2] = newIndex + 1;
				//level_index[l - 2]++;

				level_count[l - 2]--;

				level_pivot[l - 1] = 0xFFFFFFFF;
				path_more_explore = false;
				maxIntersection = 0;
				rsize[l-1] = rsize[l-2] + 1;
				drop[l-1] = drop[l-2];
				if(newIndex == level_pivot[l-2])
					drop[l-1] = drop[l-2] + 1;
			}
			__syncthreads();
			
			//Stop condition based on k
			if(rsize[l-1] - drop[l-1] > KCCOUNT)
			{	
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if(threadIdx.x == 0)
				{
					T c = rsize[l-1] - KCCOUNT;
					unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
					atomicAdd(counter, ncr/*rsize[l-1]*/);
					
					//printf, go back
					while (l > 2 && level_count[l - 2] <= 0 /*level_index[l - 2] >= level_count[l - 2]*/)
					{
						(l)--;
					}
				}
				__syncthreads();
			}
			else
			{
				__syncthreads();

				//Clear CL and PL of the new level
				for (T j = threadIdx.x; j < srcLen; j += BLOCK_DIM_X)
				{
					pl[(l-1)*srcLen + j] = 0xFFFFFFFF;
					cl[(l-1)*srcLen + j] = 0xFFFFFFFF;
				}
				__syncthreads();
				// Now prepare intersection list
				T* to =  &(cl[srcLen * (l - 1)]);
				T len = l == 2? subgraph_counters[newIndex] : cl_counter[l-2];
				if(l == 2)
				{
					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
					{
						T dest = encode[newIndex* srcLen + j];
						if(dest > newIndex || (dest < newIndex && pl[dest] == 0xFFFFFFFF))
							to[j] = dest;
					}
					__syncthreads();
				}
				else
				{
					T* from = &(cl[srcLen * (l - 2)]);
					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
					{
						bool found = false;
						const T searchVal =  from[j];
						const T lb = graph::binary_search<T>(&encode[newIndex * srcLen], 0, subgraph_counters[newIndex], searchVal, found);
						if(found &&  (searchVal > newIndex || (searchVal < newIndex && pl[(l-2)*srcLen + searchVal] == 0xFFFFFFFF)))
						{
							to[j] = searchVal;
						}
						else
							to[j] = 0xFFFFFFFF;
					}
					__syncthreads();
				}

				//Please use cub here 
				// if(threadIdx.x == 0)
				// {
				// 	cl_counter[l-1] = 0;
				// 	for(T k = 0; k< len; k++)
				// 	{
				// 		if(to[k] != 0xFFFFFFFF)
				// 		{
				// 			to[cl_counter[l-1]] = to[k];
				// 			cl_counter[l-1]++;
				// 		}
				// 	}
				// }

				block_filter_pivot<T, BLOCK_DIM_X>(len, to, &(cl_counter[l-1]));
				__syncthreads();


				if(lx == 0)
				{	
					partition_set[wx] = false;
					maxCount[wx] = srcLen + 1; //make it shared !!
					maxIndex[wx] = 0;
				}
				__syncthreads();
				//////////////////////////////////////////////////////////////////////
				//Now new pivot generation, then check to extend to new level or not
				for (T j = wx; j < cl_counter[l-1]; j += numPartitions)
				{
					uint64 warpCount = 0;
					T searchIndex = to[j];
					for (T k = lx; k < cl_counter[l-1]; k += CPARTSIZE)
					{
						bool found = false;
						const T searchVal =  to[k];
						const T lb = graph::binary_search<T>(&encode[searchIndex * srcLen], 0, subgraph_counters[searchIndex], searchVal, found);
						if(found)
						{
							warpCount++;
						}
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
					if(lx == 0 && maxCount[wx] == srcLen + 1)
					{
						partition_set[wx] = true;
						path_more_explore = true; //shared, unsafe, but okay
						maxCount[wx] = warpCount;
						maxIndex[wx] = to[j];
					}
					else if(lx == 0 && maxCount[wx] < warpCount)
					{
						maxCount[wx] = warpCount;
						maxIndex[wx] = to[j];
					}	
					
				}

				__syncthreads(); //All wait here to check to go further
				
				if(!path_more_explore)
				{
					__syncthreads();
					if(threadIdx.x == 0)
					{	
						if(rsize[l-1] >= KCCOUNT)
						{
							T c = rsize[l-1] - KCCOUNT;
							unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
							atomicAdd(counter, ncr/*rsize[l-1]*/);
						}
						//printf, go back
						while (l > 2 && level_count[l - 2] <= 0 /*level_index[l - 2] >= level_count[l - 2]*/)
						{
							(l)--;
						}
					}
					__syncthreads();
				}
				else
				{
					__syncthreads();
					if(lx == 0 && partition_set[wx])
					{
						atomicMax(&(maxIntersection), maxCount[wx]);
					}
					__syncthreads();

					if(lx == 0 && maxIntersection == maxCount[wx])
					{	
						atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
					}
				
					__syncthreads();
					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < cl_counter[l-1]; j += BLOCK_DIM_X)
					{

						bool found = false;
						const T searchVal =  to[j];
						const T lb = graph::binary_search<T>(&encode[level_pivot[l - 1] * srcLen], 0, subgraph_counters[level_pivot[l - 1]], searchVal, found);
						if(!found)
						{
							pl[(l-1)*srcLen + searchVal] = searchVal;
							warpCount++;
						}
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

					__syncthreads();
					if(threadIdx.x == 0)
					{
						l++;
						level_count[l-2] = 0;
						level_prev_index[l-2] = 0;
						//level_index[l-2] = 0;
					}

					__syncthreads();
					if(lx == 0 && threadIdx.x < cl_counter[l-2])
					{
						atomicAdd(&(level_count[l-2]), warpCount);
					}

					
					__syncthreads();


				}
				
			}
			__syncthreads();
			/////////////////////////////////////////////////////////////////////////
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
kckernel_edge_block_warp_pivot_count(
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

	//__shared__ T  level_offset[numPartitions], level_item_offset[numPartitions]; //for l and p
	__shared__ T level_pivot[512];
	__shared__ uint64 path_more_explore;
	__shared__ T l;
	__shared__ uint64 maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen, srcLenBlocks, src2, src2Start, src2Len, scounter;
	__shared__ bool  partition_set[numPartitions];
	__shared__ T encode_offset, *encode, tri_offset, *tri;
	__shared__ T *pl, *cl;
	__shared__ T *level_count, /**level_index,*/ *level_prev_index, *rsize, *drop;
	__shared__ T lo, level_item_offset;
	__shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
	__shared__ unsigned short cl_counter[512];
	__shared__ unsigned short subgraph_counters[512];

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
			tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
			scounter = 0;
			//srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

			//printf("SRC = %u, SRCLEN = %u, %llu\n", src, srcLen, *counter);

			//printf("src = %u, srcLen = %u\n", src, srcLen);
			encode_offset = sm_id * CBPSM * (MAXDEG * MAXDEG) + levelPtr * (MAXDEG * MAXDEG);
			encode = &adj_enc[encode_offset];

			lo = sm_id * CBPSM * (MAXDEG* MAXDEG) + levelPtr * (MAXDEG* MAXDEG);
			cl = &current_level[lo];
			pl = &possible[lo];

			level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
			level_count = &level_count_g[level_item_offset];
			//level_index = &level_index_g[level_item_offset];
			level_prev_index = &level_prev_g[level_item_offset];
			rsize = &level_r[level_item_offset ]; // will be removed
			drop = &level_d[level_item_offset];  //will be removed

			level_count[0] = 0;
			level_prev_index[0] = 0;
			//level_index[0] = 0;
			l = 3;
			rsize[0] = 1;
			drop[0] = 0;

			level_pivot[0] = 0xFFFFFFFF;
			maxIntersection = 0;
			cl_counter[0] = 0;
		}
		__syncthreads();
		graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
			tri, &scounter);
		__syncthreads();

		//Encode Clear
		for (T j = wx; j < scounter; j += numPartitions)
		{
			for (T k = lx; k < scounter; k += CPARTSIZE)
			{
				encode[j * scounter + k] = 0xFFFFFFFF;
			}
		}
		__syncthreads();
		//Full Subgraph
		for (T j = wx; j < scounter; j += numPartitions)
		{
			graph::warp_sorted_count_and_subgraph_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
				&g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
				j, scounter, encode);
		}
		__syncthreads(); //Done encoding

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

		__syncthreads();

		//Find the first pivot
		if(lx == 0)
		{
			maxCount[wx] = 0;
			maxIndex[wx] = 0xFFFFFFFF;
			partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
			partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		}
		__syncthreads();
	
		for (T j = wx; j < scounter; j += numPartitions)
		{
			if(lx == 0 && maxCount[wx] < subgraph_counters[j])
			{
				maxCount[wx] = subgraph_counters[j];
				maxIndex[wx] = j;
			}	
		}
		__syncthreads();
		if(lx == 0)
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

		//Creat the first/top Possible (Pivot list)
		uint64 warpCount = 0;
		for (T j = threadIdx.x; j < scounter && maxIntersection > 0; j += BLOCK_DIM_X)
		{
			bool found = false;
			const T searchVal =  j;
			const T lb = graph::binary_search<T>(&encode[level_pivot[0] * scounter], 0, subgraph_counters[level_pivot[0]], searchVal, found);
			if(!found)
			{
				pl[j] = j;
				warpCount++;
			}
			else
			{
				pl[j] = 0xFFFFFFFF;
			}
		}
		reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
		if(lx == 0 && threadIdx.x < scounter)
		{
			atomicAdd(&(level_count[0]), (T)warpCount);
		}
		__syncthreads();

		//Explore the tree
		while(level_count[l - 3] > 0 /*level_index[l - 3]*/)
		{
			T startIndex = level_prev_index[l- 3];
			T newIndex = pl[(l-3)*scounter + startIndex];
			while(newIndex == 0xFFFFFFFF)
			{
				startIndex++;
				newIndex = pl[(l-3)*scounter + startIndex];
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				level_prev_index[l - 3] = newIndex + 1;
				//level_index[l - 3]++;

				level_count[l - 3]--;

				level_pivot[l - 2] = 0xFFFFFFFF;
				path_more_explore = false;
				maxIntersection = 0;
				rsize[l-2] = rsize[l-3] + 1;
				drop[l-2] = drop[l-3];
				if(newIndex == level_pivot[l-3])
					drop[l-2] = drop[l-3] + 1;
			}
			__syncthreads();
			
			//Stop condition based on k
			if(rsize[l-2] - drop[l-2] > KCCOUNT)
			{	
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if(threadIdx.x == 0)
				{
					T c = rsize[l-2] - KCCOUNT;
					unsigned long long ncr = nCR[ drop[l-2] * 401 + c  ];
					atomicAdd(counter, ncr/*rsize[l-1]*/);
					
					//printf, go back
					while (l > 3 && level_count[l - 3] <= 0/*level_index[l - 3] >= level_count[l - 3]*/)
					{
						(l)--;
					}
				}
				__syncthreads();
			}
			else
			{
				__syncthreads();

				//Clear CL and PL of the new level
				for (T j = threadIdx.x; j < scounter; j += BLOCK_DIM_X)
				{
					pl[(l-2)*scounter + j] = 0xFFFFFFFF;
					cl[(l-2)*scounter + j] = 0xFFFFFFFF;
				}
				__syncthreads();
				// Now prepare intersection list
				T* to =  &(cl[scounter * (l - 2)]);
				T len = l == 3? subgraph_counters[newIndex] : cl_counter[l-3];
				if(l == 3)
				{
					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
					{
						T dest = encode[newIndex* scounter + j];
						if(dest > newIndex || (dest < newIndex && pl[dest] == 0xFFFFFFFF))
							to[j] = dest;
					}
					__syncthreads();
				}
				else
				{
					T* from = &(cl[scounter * (l - 3)]);
					for (T j = threadIdx.x; j < len; j += BLOCK_DIM_X)
					{
						bool found = false;
						const T searchVal =  from[j];
						const T lb = graph::binary_search<T>(&encode[newIndex * scounter], 0, subgraph_counters[newIndex], searchVal, found);
						if(found &&  (searchVal > newIndex || (searchVal < newIndex && pl[(l-3)*scounter + searchVal] == 0xFFFFFFFF)))
						{
							to[j] = searchVal;
						}
						else
							to[j] = 0xFFFFFFFF;
					}
					__syncthreads();
				}

				//Please use cub here 
				// if(threadIdx.x == 0)
				// {
				// 	cl_counter[l-2] = 0;
				// 	for(T k = 0; k< len; k++)
				// 	{
				// 		if(to[k] != 0xFFFFFFFF)
				// 		{
				// 			to[cl_counter[l-2]] = to[k];
				// 			cl_counter[l-2]++;
				// 		}
				// 	}
				// }

				block_filter_pivot<T, BLOCK_DIM_X>(len, to, &(cl_counter[l-2]));
				__syncthreads();


				if(lx == 0)
				{	
					partition_set[wx] = false;
					maxCount[wx] = scounter + 1; //make it shared !!
					maxIndex[wx] = 0;
				}
				__syncthreads();
				//////////////////////////////////////////////////////////////////////
				//Now new pivot generation, then check to extend to new level or not
				for (T j = wx; j < cl_counter[l-2]; j += numPartitions)
				{
					uint64 warpCount = 0;
					T searchIndex = to[j];
					for (T k = lx; k < cl_counter[l-2]; k += CPARTSIZE)
					{
						bool found = false;
						const T searchVal =  to[k];
						const T lb = graph::binary_search<T>(&encode[searchIndex * scounter], 0, subgraph_counters[searchIndex], searchVal, found);
						if(found)
						{
							warpCount++;
						}
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
					if(lx == 0 && maxCount[wx] == scounter + 1)
					{
						partition_set[wx] = true;
						path_more_explore = true; //shared, unsafe, but okay
						maxCount[wx] = warpCount;
						maxIndex[wx] = to[j];
					}
					else if(lx == 0 && maxCount[wx] < warpCount)
					{
						maxCount[wx] = warpCount;
						maxIndex[wx] = to[j];
					}	
					
				}

				__syncthreads(); //All wait here to check to go further
				
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
						while (l > 3 && level_count[l - 3] <=0 /*level_index[l - 3] >= level_count[l - 3]*/)
						{
							(l)--;
						}
					}
					__syncthreads();
				}
				else
				{
					__syncthreads();
					if(lx == 0 && partition_set[wx])
					{
						atomicMax(&(maxIntersection), maxCount[wx]);
					}
					__syncthreads();

					if(lx == 0 && maxIntersection == maxCount[wx])
					{	
						atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
					}
				
					__syncthreads();
					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < cl_counter[l-2]; j += BLOCK_DIM_X)
					{

						bool found = false;
						const T searchVal =  to[j];
						const T lb = graph::binary_search<T>(&encode[level_pivot[l - 2] * scounter], 0, subgraph_counters[level_pivot[l - 2]], searchVal, found);
						if(!found)
						{
							pl[(l-2)*scounter + searchVal] = searchVal;
							warpCount++;
						}
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

					__syncthreads();
					if(threadIdx.x == 0)
					{
						l++;
						level_count[l-3] = 0;
						level_prev_index[l-3] = 0;
						//level_index[l-3] = 0;
					}

					__syncthreads();
					if(lx == 0 && threadIdx.x < cl_counter[l-3])
					{
						atomicAdd(&(level_count[l-3]), warpCount);
					}

					
					__syncthreads();


				}
				
			}
			__syncthreads();
			/////////////////////////////////////////////////////////////////////////
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}

namespace graph
{
	template<typename T>
	class SingleGPU_Kclique
	{
	private:
		int dev_;
		cudaStream_t stream_;

		//Outputs:
		//Max k of a complete ktruss kernel
		int k;


		//Percentage of deleted edges for a specific k
		float percentage_deleted_k;

		//Same Function for any comutation
		void bucket_scan(
			GPUArray<T> nodeDegree, T node_num, T level, T span,
			GraphQueue<T, bool>& current,
			GPUArray<T> asc,
			GraphQueue<T, bool>& bucket,
			T& bucket_level_end_)
		{
			static bool is_first = true;
			static int multi = 1;
			if (is_first)
			{
				current.mark.setAll(false, true);
				bucket.mark.setAll(false, true);
				is_first = false;
			}

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator


				long grid_size = (node_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_window<T, T>), grid_size, BLOCK_SIZE, dev_, false,
					nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + KCL_NODE_LEVEL_SKIP_SIZE);

				multi++;

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num, dev_);
				bucket_level_end_ += KCL_NODE_LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_with_random_append<T, T>), grid_size, BLOCK_SIZE, dev_, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, span);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			//Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
		}


		//Same Function for any comutation
		void bucket_edge_scan(
			GPUArray<T> nodeDegree, T node_num, T level, T span,
			GraphQueue<T, bool>& current,
			GPUArray<T> asc,
			GraphQueue<T, bool>& bucket,
			T& bucket_level_end_)
		{
			static bool is_first = true;
			static int multi = 1;
			if (is_first)
			{
				current.mark.setAll(false, true);
				bucket.mark.setAll(false, true);
				is_first = false;
			}

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator
				long grid_size = (node_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel(filter_window, grid_size, BLOCK_SIZE, dev_, false,
					nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + KCL_EDGE_LEVEL_SKIP_SIZE);

				multi++;

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num, dev_);
				bucket_level_end_ += KCL_EDGE_LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_with_random_append<T>), grid_size, BLOCK_SIZE, dev_, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, span);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			//Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
		}

		void AscendingGpu(T n, GPUArray<T>& identity_arr_asc)
		{
			long grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
			identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::gpu, n, dev_);
			execKernel(init_asc, grid_size, BLOCK_SIZE, dev_, false, identity_arr_asc.gdata(), n);
		}

	public:
		GPUArray<T> nodeDegree;
		GPUArray<T> edgePtr;
		graph::GraphQueue<T, bool> bucket_q;
		graph::GraphQueue<T, bool> current_q;
		GPUArray<T> identity_arr_asc;

		SingleGPU_Kclique(int dev, COOCSRGraph_d<T>& g) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			bucket_q.Create(unified, g.numEdges, dev_);
			current_q.Create(unified, g.numEdges, dev_);
			AscendingGpu(g.numEdges, identity_arr_asc);

			edgePtr.initialize("Edge Support", unified, g.numEdges, dev_);
		}

		SingleGPU_Kclique() : SingleGPU_Kclique(0) {}


		void getNodeDegree(COOCSRGraph_d<T>& g, T* maxDegree,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 128;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
		}


		////Node
		template<const int PSIZE>
		void findKclqueIncremental_node_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;
			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numNodes;
			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::unified, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);


			cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;

			/*level = 32;
			bucket_level_end_ = level;*/
			while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, g.numNodes, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{

					// std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					// current_q.count.gdata()[0] = current_q.count.gdata()[0]< 128? current_q.count.gdata()[0]: 128;
					//current_q.count.gdata()[0] = 1; 
					if (pe == BlockWarp)
					{
						factor = (block_size / PSIZE);
						const uint max_level = 10;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0];
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * maxDegree.gdata()[0];
						const uint64 intermediate_size = num_SMs * conc_blocks_per_SM * factor * max_level * maxDegree.gdata()[0];

						GPUArray<unsigned short> current_level2("Temp level Counter", gpu, level_size, dev_);
						current_level2.setAll(0, true);

						
						GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);
						
			
						//printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);

						
						cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
						cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
						cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
						cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

						const T partitionSize = PSIZE;
						cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_node_block_warp_subgraph_count<T, block_size, partitionSize, 9, NodeStartLevelOr>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(),
							d_bitmap_states.gdata(),
							node_be.gdata()
						);



						//GPUArray<T> im("Intermediate level Counter", gpu, intermediate_size, dev_);
							// 	execKernel((kckernel_node_block_warp_subgraph_im_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						// 	counter.gdata(),
						// 	g,
						// 	current_q.device_queue->gdata()[0],
						// 	current_level2.gdata(),
						// 	d_bitmap_states.gdata(),
						// 	node_be.gdata(),
						// 	im.gdata()
						// );

						// execKernel((kckernel_node_block_warp_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						// 	counter.gdata(),
						// 	g,
						// 	current_q.device_queue->gdata()[0],
						// 	current_level2.gdata(),
						// 	d_bitmap_states.gdata()
						// 	//,node_be.gdata()
						// );


						current_level2.freeGPU();
						node_be.freeGPU();
					}
					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}
				level += span;
			}

			counter.freeGPU();
			cpn.freeGPU();

			d_bitmap_states.freeGPU();
			k = level;
			//printf("Max Degree (+span) = %d\n", k - 1);
		}


		template<const int PSIZE>
		void findKclqueIncremental_node_binary_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			printf("Test Test \n");

			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;
			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numNodes;
			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);

			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;


			const T partitionSize = PSIZE; //PART_SIZE;
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = 9;
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
			//printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);
			current_level2.setAll(0, true);
			node_be.setAll(0, true);

			GPUArray<uint64> globalCounter("Temp level Counter", gpu, 1, dev_);
			globalCounter.setSingle(0, 0, true);

			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			/*level = 32;
			bucket_level_end_ = level;*/
			while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, g.numNodes, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{

					// std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					// current_q.count.gdata()[0] = current_q.count.gdata()[0]< 128? current_q.count.gdata()[0]: 128;
					//current_q.count.gdata()[0] = 1; 
				
					auto grid_block_size = current_q.count.gdata()[0]; //num_SMs * conc_blocks_per_SM; //
					execKernel((kckernel_node_block_warp_binary_count<T, block_size, partitionSize, 6, NodeStartLevelOr>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), cpn.gdata(),
						d_bitmap_states.gdata(), node_be.gdata(), globalCounter.gdata());

					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}
				level += span;
			}

			// for(int i = 0; i<80; i++)
			// 	printf("%lu\n", cpn.gdata()[i]);

			current_level2.freeGPU();
			counter.freeGPU();
			cpn.freeGPU();
			node_be.freeGPU();
			d_bitmap_states.freeGPU();
			k = level;
			//printf("Max Degree (+span) = %d\n", k - 1);
		}

		template<const int PSIZE>
		void findKclqueIncremental_node_pivot_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			FILE *infile;
			GPUArray<unsigned long long> nCr("bmp bitmap stats", AllocationTypeEnum::gpu, 1001*401, dev_);
			infile = fopen("/home/almasri3/mewcp-gpu/kclique/nCr.txt","r"); //change it
			double d=0;
			if(infile==NULL)
			{
				printf("file could not be opened\n");
				exit(1);
			}
			
			for(int row = 0; row < 1001; ++row)
			{
				for (int col = 0; col < 401; ++col)
				{
					if (!fscanf(infile,"%lf,",&d)) 
						fprintf(stderr, "Error\n");
					// fprintf(stderr, "%lf\n", d);
					nCr.cdata()[row*401 + col] = (unsigned long long)d;
				}
			}
			fclose(infile);
			//printf("Test, 5c4 = %llu\n", nCr.cdata()[7*401 + 4]);

			nCr.switch_to_gpu();


			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;
			T level = 0;
			T span = 2048;
			T bucket_level_end_ = level;
			T todo = g.numNodes;
			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);

			cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;



			//printf("%d\n", maxDegree.gdata()[0]++);
			const T partitionSize = PSIZE; //Defined
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = maxDegree.gdata()[0];
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs; //per block
			GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);

			const uint64 level_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level * num_divs; //per partition
			const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level; //per partition
			const uint64 level_partition_size = num_SMs * conc_blocks_per_SM * /*factor **/ num_divs; //per partition

			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> possible("Temp level Counter", gpu, level_size, dev_);
			
			GPUArray<T> level_index("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_count("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_prev("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_r("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_d("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_temp("Temp level Counter", unified, level_partition_size, dev_);

			//printf("Level Size = %llu, Encode Size = %llu\n", 2 *level_size + 5*level_item_size + 1*level_partition_size, encode_size);

			// current_level2.setAll(0, true);
			// node_be.setAll(0, true);
			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			//while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, g.numNodes, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					auto grid_block_size =  current_q.count.gdata()[0];
					execKernel((kckernel_node_block_warp_binary_pivot_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), cpn.gdata(),
						d_bitmap_states.gdata(), node_be.gdata(),
					
						possible.gdata(),
						level_index.gdata(),
						level_count.gdata(),
						level_prev.gdata(),
						level_r.gdata(),
						level_d.gdata(),
						level_temp.gdata(),
						nCr.gdata()
					);

					
			
					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}
				level += span;
			}


			// for(int i = 0; i<80; i++)
			// 	printf("%lu\n", cpn.gdata()[i]);

			counter.freeGPU();
			cpn.freeGPU();
			current_level2.freeGPU();
			d_bitmap_states.freeGPU();
			node_be.freeGPU();
			possible.freeGPU();
			level_index.freeGPU();
			level_count.freeGPU();
			level_prev.freeGPU();
			level_r.freeGPU();
			level_d.freeGPU();
			nCr.freeGPU();
			d_bitmap_states.freeGPU();

			k = level;
			//printf("Max Degree (+span) = %d\n", k - 1);
		}

		template<const int PSIZE>
		void findKclqueIncremental_node_nobin_pivot_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			FILE *infile;
			GPUArray<unsigned long long> nCr("bmp bitmap stats", AllocationTypeEnum::gpu, 1001*401, dev_);
			infile = fopen("/home/almasri3/mewcp-gpu/kclique/nCr.txt","r"); //change it
			double d=0;
			if(infile==NULL)
			{
				printf("file could not be opened\n");
				exit(1);
			}
			
			for(int row = 0; row < 1001; ++row)
			{
				for (int col = 0; col < 401; ++col)
				{
					if (!fscanf(infile,"%lf,",&d)) 
						fprintf(stderr, "Error\n");
					// fprintf(stderr, "%lf\n", d);
					nCr.cdata()[row*401 + col] = (unsigned long long)d;
				}
			}
			fclose(infile);
			//printf("Test, 5c4 = %llu\n", nCr.cdata()[7*401 + 4]);

			nCr.switch_to_gpu();


			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;
			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numNodes;
			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);

			// cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;


			const T partitionSize = PSIZE; //Defined
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = maxDegree.gdata()[0];
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * maxDegree.gdata()[0]; //per block
			GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);

			const uint64 level_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * maxDegree.gdata()[0]; //per partition
			const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level; //per partition
			const uint64 level_partition_size = num_SMs * conc_blocks_per_SM * /*factor **/ num_divs; //per partition

			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> possible("Temp level Counter", gpu, level_size, dev_);
			
			GPUArray<T> level_index("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_count("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_prev("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_r("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_d("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_temp("Temp level Counter", unified, level_partition_size, dev_);

			//printf("Level Size = %llu, Encode Size = %llu\n", 2 *level_size + 5*level_item_size + 1*level_partition_size, encode_size);

			// current_level2.setAll(0, true);
			// node_be.setAll(0, true);
			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, g.numNodes, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					auto grid_block_size =  current_q.count.gdata()[0];
					execKernel((kckernel_node_block_warp_pivot_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), cpn.gdata(),
						d_bitmap_states.gdata(), node_be.gdata(),
					
						possible.gdata(),
						level_index.gdata(),
						level_count.gdata(),
						level_prev.gdata(),
						level_r.gdata(),
						level_d.gdata(),
						level_temp.gdata(),
						nCr.gdata()
					);

					
			
					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}
				level += span;
			}

			counter.freeGPU();
			cpn.freeGPU();
			current_level2.freeGPU();
			d_bitmap_states.freeGPU();
			node_be.freeGPU();
			possible.freeGPU();
			level_index.freeGPU();
			level_count.freeGPU();
			level_prev.freeGPU();
			level_r.freeGPU();
			level_d.freeGPU();
			nCr.freeGPU();


			k = level;
			//printf("Max Degree (+span) = %d\n", k - 1);
		}



		////Edge
		template<const int PSIZE>
		void findKclqueIncremental_edge_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));

			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numEdges;

			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);



			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			execKernel((getEdgeDegree_kernel<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());
			GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);


			//printf("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;
			CUDA_RUNTIME(cudaGetLastError());
			cudaDeviceSynchronize();

			/*	GPUArray <uint64> cpn("Temp Degree", unified, g.numEdges, dev_);
				cpn.setAll(0, true);*/

			//while (todo > 0)
			{
				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;

				
					if (pe == BlockWarp)
					{
						const T partitionSize = PSIZE;
						factor = (block_size / partitionSize);

						const uint dv = 32;
						const uint max_level = 10;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0];
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * maxDegree.gdata()[0];
						const uint64 tri_size = num_SMs * conc_blocks_per_SM *  maxDegree.gdata()[0];
						//printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<unsigned short> current_level2("Temp level Counter", gpu, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);
						GPUArray<T> tri_list("Temp level Counter", gpu, tri_size, dev_);

						// simt::atomic<KCTask<T>, simt::thread_scope_device> *queue_data;
						// CUDA_RUNTIME(cudaMalloc((void **)&queue_data, (num_SMs * conc_blocks_per_SM * QUEUE_SIZE) * sizeof(simt::atomic<KCTask<T>, simt::thread_scope_device>)));

						// GPUArray<KCTask<T>> queue_data("test", unified, num_SMs * conc_blocks_per_SM * QUEUE_SIZE, dev_);
						// GPUArray<T> queue_encode("test", unified, num_SMs * conc_blocks_per_SM * QUEUE_SIZE * num_divs, dev_);
					
						// current_level2.setAll(0, true);
						// node_be.setAll(0, true);
						// tri_list.setAll(0, true);

						const T numPartitions = block_size/partitionSize;
						cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
						cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
						cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
						cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
						cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
						cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
						cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));
						
						auto grid_block_size = current_q.count.gdata()[0];
						// execKernel((kckernel_edge_block_warp_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						// 	counter.gdata(),
						// 	g,
						// 	current_q.device_queue->gdata()[0],
						// 	current_level2.gdata(), 
						// 	d_bitmap_states.gdata(), tri_list.gdata()
						
						// );




						execKernel((kckernel_edge_block_warp_subgraph_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), 
						d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata());



						current_level2.freeGPU();
						node_be.freeGPU();
						tri_list.freeGPU();
					}

				}
				level += span;

				std::cout.imbue(std::locale(""));
				std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
			}

			counter.freeGPU();
			//cpn.freeGPU();
			current_level.freeGPU();

			d_bitmap_states.freeGPU();
			maxDegree.freeGPU();
			k = level;

			//printf("Max Edge Min Degree = %d\n", k - 1);

		}

		template<const int PSIZE>
		void findKclqueIncremental_edge_binary_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			//printf("HERE EDGE BINARY \n");
			CUDA_RUNTIME(cudaSetDevice(dev_));

			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numEdges;

			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);



			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			execKernel((getEdgeDegree_kernel<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());
			GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);


			//("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;

			const T partitionSize = PSIZE;
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = 8;
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
			const uint64 tri_size = num_SMs * conc_blocks_per_SM *  maxDegree.gdata()[0];
			const uint64 stack_size = num_SMs * conc_blocks_per_SM * max_level * factor;


			//printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);
			GPUArray<T> tri_list("Temp level Counter", gpu, tri_size, dev_);


			// GPUArray<T> count_list("Temp level Counter", gpu, stack_size, dev_);
			// GPUArray<T> index_list("Temp level Counter", gpu, stack_size, dev_);
			// GPUArray<T> prev_list("Temp level Counter", gpu, stack_size, dev_);


			

			// simt::atomic<KCTask<T>, simt::thread_scope_device> *queue_data;
			// CUDA_RUNTIME(cudaMalloc((void **)&queue_data, (num_SMs * conc_blocks_per_SM * QUEUE_SIZE) * sizeof(simt::atomic<KCTask<T>, simt::thread_scope_device>)));

			// GPUArray<KCTask<T>> queue_data("test", unified, num_SMs * conc_blocks_per_SM * QUEUE_SIZE, dev_);
			// GPUArray<T> queue_encode("test", unified, num_SMs * conc_blocks_per_SM * QUEUE_SIZE * num_divs, dev_);
			
			current_level2.setAll(0, false);
			node_be.setAll(0, false);
			tri_list.setAll(0, false);

			// count_list.setAll(0, false);
			// index_list.setAll(0, false);
			// prev_list.setAll(0, false);


			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			/*	GPUArray <uint64> cpn("Temp Degree", unified, g.numEdges, dev_);
				cpn.setAll(0, true);*/

			while (todo > 0)
			{
				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;
					auto grid_block_size = current_q.count.gdata()[0];
					// execKernel((kckernel_edge_block_warp_binary_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
					// 	counter.gdata(),
					// 	g,
					// 	current_q.device_queue->gdata()[0],
					// 	current_level2.gdata(), NULL,
					// 	d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata(),
					// 	queue_data.gdata(),
					// 	queue_encode.gdata()
					
					// );


					execKernel((kckernel_edge_block_warp_binary_count_o2<T, block_size, partitionSize, 8, EdgeStartLevelOr>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), NULL,
						d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata()
					
					);

				}
				level += span;

				std::cout.imbue(std::locale(""));
				std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
			}

			current_level2.freeGPU();
			node_be.freeGPU();
			tri_list.freeGPU();
			counter.freeGPU();
			//cpn.freeGPU();
			current_level.freeGPU();

			d_bitmap_states.freeGPU();
			maxDegree.freeGPU();
			k = level;

			//printf("Max Edge Min Degree = %d\n", k - 1);

		}

		template<const int PSIZE>
		void findKclqueIncremental_edge_pivot_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{



			Log(info, "HERE\n");
			//Please be carful
			kcount = kcount - 1;

			FILE *infile;
			GPUArray<unsigned long long> nCr("bmp bitmap stats", AllocationTypeEnum::gpu, 1001*401, dev_);
			infile = fopen("/home/almasri3/mewcp-gpu/kclique/nCr.txt","r"); //change it
			double d=0;
			if(infile==NULL)
			{
				printf("file could not be opened\n");
				exit(1);
			}
			for(int row = 0; row < 1001; ++row)
			{
				for (int col = 0; col < 401; ++col)
				{
					if (!fscanf(infile,"%lf,",&d)) 
						fprintf(stderr, "Error\n");
					// fprintf(stderr, "%lf\n", d);
					nCr.cdata()[row*401 + col] = (unsigned long long)d;
				}
			}
			fclose(infile);
			//printf("Test, 5c4 = %llu\n", nCr.cdata()[7*401 + 4]);
			nCr.switch_to_gpu();

			CUDA_RUNTIME(cudaSetDevice(dev_));

			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;

			T level = 0;
			T span = 2048;
			T bucket_level_end_ = level;
			T todo = g.numEdges;

			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			execKernel((getEdgeDegree_kernel<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());
			GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);


			//printf("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;

			const T partitionSize = PSIZE;
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = maxDegree.gdata()[0];
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs; //per block
			const uint64 tri_size = num_SMs * conc_blocks_per_SM *  maxDegree.gdata()[0]; //per block
			GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);
			GPUArray<T> tri_list("Temp level Counter", gpu, tri_size, dev_);

			const uint64 level_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level * num_divs; //per partition
			const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /** factor **/ max_level; //per partition
			const uint64 level_partition_size = num_SMs * conc_blocks_per_SM *  /** factor **/ num_divs; //per partition

			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> possible("Temp level Counter", gpu, level_size, dev_);
			
			GPUArray<T> level_index("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_count("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_prev("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_r("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_d("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_temp("Temp level Counter", unified, level_partition_size, dev_);

			
			//printf("Level Size = %llu, Encode Size = %llu, Tri size = %llu\n", 2 *level_size + 5*level_item_size + 1*level_partition_size, encode_size, tri_size);
	
			// current_level2.setAll(0, true);
			// node_be.setAll(0, true);
			// tri_list.setAll(0, true);

			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			/*	GPUArray <uint64> cpn("Temp Degree", unified, g.numEdges, dev_);
				cpn.setAll(0, true);*/

			//while (todo > 0)
			//{
				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;

						auto grid_block_size = current_q.count.gdata()[0];

						execKernel((kckernel_edge_block_warp_binary_pivot_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), NULL,
							d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata(),

							possible.gdata(),
							level_index.gdata(),
							level_count.gdata(),
							level_prev.gdata(),
							level_r.gdata(),
							level_d.gdata(),
							level_temp.gdata(),
							nCr.gdata()
						);


						current_level2.freeGPU();
				}
				level += span;

				std::cout.imbue(std::locale(""));
				std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
			//}

			counter.freeGPU();
			//cpn.freeGPU();
			current_level2.freeGPU();
			d_bitmap_states.freeGPU();
			node_be.freeGPU();
			possible.freeGPU();
			level_index.freeGPU();
			level_count.freeGPU();
			level_prev.freeGPU();
			level_r.freeGPU();
			level_d.freeGPU();
			nCr.freeGPU();


			d_bitmap_states.freeGPU();
			maxDegree.freeGPU();
			k = level;

			//printf("Max Edge Min Degree = %d\n", k - 1);

		}

		template<const int PSIZE>
		void findKclqueIncremental_edge_nobin_pivot_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{


			//Please be carful
			kcount = kcount - 1;

			FILE *infile;
			GPUArray<unsigned long long> nCr("bmp bitmap stats", AllocationTypeEnum::gpu, 1001*401, dev_);
			infile = fopen("/home/almasri3/mewcp-gpu/kclique/nCr.txt","r"); //change it
			double d=0;
			if(infile==NULL)
			{
				printf("file could not be opened\n");
				exit(1);
			}
			for(int row = 0; row < 1001; ++row)
			{
				for (int col = 0; col < 401; ++col)
				{
					if (!fscanf(infile,"%lf,",&d)) 
						fprintf(stderr, "Error\n");
					// fprintf(stderr, "%lf\n", d);
					nCr.cdata()[row*401 + col] = (unsigned long long)d;
				}
			}
			fclose(infile);
			//printf("Test, 5c4 = %llu\n", nCr.cdata()[7*401 + 4]);
			nCr.switch_to_gpu();

			CUDA_RUNTIME(cudaSetDevice(dev_));

			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numEdges;

			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			execKernel((getEdgeDegree_kernel<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());

			//printf("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;

			const T partitionSize = PSIZE;
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = maxDegree.gdata()[0];
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * maxDegree.gdata()[0]; //per block
			const uint64 tri_size = num_SMs * conc_blocks_per_SM *  maxDegree.gdata()[0]; //per block
			GPUArray<T> node_be("Temp level Counter", gpu, encode_size, dev_);
			GPUArray<T> tri_list("Temp level Counter", gpu, tri_size, dev_);

			const uint64 level_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level * maxDegree.gdata()[0]; //per partition
			const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /** factor **/ max_level; //per partition
			const uint64 level_partition_size = num_SMs * conc_blocks_per_SM *  /** factor **/ num_divs; //per partition

			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> possible("Temp level Counter", gpu, level_size, dev_);
			
			GPUArray<T> level_index("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_count("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_prev("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_r("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_d("Temp level Counter", gpu, level_item_size, dev_);
			GPUArray<T> level_temp("Temp level Counter", unified, level_partition_size, dev_);

			//printf("Level Size = %llu, Encode Size = %llu, Tri size = %llu\n", 2 *level_size + 5*level_item_size + 1*level_partition_size, encode_size, tri_size);
	
			// current_level2.setAll(0, true);
			// node_be.setAll(0, true);
			// tri_list.setAll(0, true);

			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			/*	GPUArray <uint64> cpn("Temp Degree", unified, g.numEdges, dev_);
				cpn.setAll(0, true);*/

			while (todo > 0)
			{
				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;

						auto grid_block_size = current_q.count.gdata()[0];

						execKernel((kckernel_edge_block_warp_pivot_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), NULL,
							d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata(),

							possible.gdata(),
							level_index.gdata(),
							level_count.gdata(),
							level_prev.gdata(),
							level_r.gdata(),
							level_d.gdata(),
							level_temp.gdata(),
							nCr.gdata()
						);


						current_level2.freeGPU();
				}
				level += span;

				std::cout.imbue(std::locale(""));
				std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
			}

			counter.freeGPU();
			//cpn.freeGPU();
			current_level2.freeGPU();
			d_bitmap_states.freeGPU();
			node_be.freeGPU();
			possible.freeGPU();
			level_index.freeGPU();
			level_count.freeGPU();
			level_prev.freeGPU();
			level_r.freeGPU();
			level_d.freeGPU();
			nCr.freeGPU();

			d_bitmap_states.freeGPU();
			maxDegree.freeGPU();
			k = level;

			//printf("Max Edge Min Degree = %d\n", k - 1);

		}




		void free()
		{
			current_q.free();
			bucket_q.free();
			identity_arr_asc.freeGPU();

			nodeDegree.freeGPU();
			edgePtr.freeGPU();

		}
		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

		uint count() const { return k - 1; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

} // namespace pangolin
