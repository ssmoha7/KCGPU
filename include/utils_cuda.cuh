/*---- 1. CUDA kernel check functions ----*/
#pragma once
#include <driver_types.h>
#include "utils.cuh"
#include "Logger.cuh"
#define WARP_BITS   (5)
#define WARP_SIZE   (1<<WARP_BITS)
#define WARP_MASK   (WARP_SIZE-1)
#define BLOCK_SIZE  (128)  /*default block size*/
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)
#define GRID_SIZE   (1024) /*default grid size*/





/*! Binary search

 \tparam arr  			Pointer to the array
 \tparam l 					Left boundary of arr
 \tparam r   				Right boundary of arr
     \tparam x				 	Value to search for
*/
__device__ inline uint binarySearch_b(const uint* arr, uint l, uint r, uint x)
{
    size_t left = l;
    size_t right = r;
    while (left < right) {
        const size_t mid = (left + right) / 2;
        uint val = arr[mid];
        bool pred = val < x;
        if (pred) {
            left = mid + 1;
        }
        else {
            right = mid;
        }
    }
    return left;
}

/*! Obtain the index of an edge using its two nodes

 \tparam mat  			Graph view represented in COO+CSR format
 \tparam sn 				Source node
 \tparam dn   			Destination node
*/

__device__ inline uint getEdgeId(uint* rowPtr, uint* colInd, uint sn, const uint dn)
{
    uint index = 0;

    uint start = rowPtr[sn];
    uint end2 = rowPtr[sn + 1];
    index = binarySearch_b(colInd, start, end2, dn); // pangolin::binary_search(p, length, dn);
    return index;
}


/*
 * cudaPeekAtLastError(): get the code of last error, no resetting
 * udaGetLastError(): get the code of last error, resetting to cudaSuccess
 * */
#define CHECK_KERNEL(func)          if (cudaSuccess != cudaPeekAtLastError()) { \
                                        cudaError_t error = cudaGetLastError(); \
                                        Log(LogPriorityEnum::critical ,"Kernel %s: %s.", func, \
                                        cudaGetErrorString(error)); \
                                        exit(1); \
                                    }

/*---- 1. CUDA kernel check functions end ----*/

/*---- 2. CUDA kernel execution wrapper ----*/
/*normal execution without dynamic shared memory allocation*/
#define execKernel(kernel, gridSize, blockSize, deviceId, verbose, ...) \
{ \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    CUDA_RUNTIME(cudaSetDevice(deviceId)); \
    kernel<<<grid,block>>>(__VA_ARGS__); \
    CUDA_RUNTIME(cudaDeviceSynchronize()); \
}


#define execKernel2(kernel, gridSize, blockSize, deviceId, verbose, ...) \
{ \
    float singleKernelTime;\
    cudaEvent_t start, end; \
    CUDA_RUNTIME(cudaEventCreate(&start)); \
    CUDA_RUNTIME(cudaEventCreate(&end)); \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    CUDA_RUNTIME(cudaSetDevice(deviceId)); \
    CUDA_RUNTIME(cudaEventRecord(start)); \
    kernel<<<grid,block>>>(__VA_ARGS__); \
    CHECK_KERNEL(#kernel)\
    CUDA_RUNTIME(cudaPeekAtLastError());\
    CUDA_RUNTIME(cudaEventRecord(end));\
    \
    CUDA_RUNTIME(cudaEventSynchronize(start)); \
    CUDA_RUNTIME(cudaEventSynchronize(end)); \
    CUDA_RUNTIME(cudaDeviceSynchronize()); \
    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end)); \
    \
    {\
    }\
}

//if (false) Log(LogPriorityEnum::info, "Kernel: %s, time: %.2f ms.", #kernel, singleKernelTime); \
/*execution with dynamic shared memory allocation*/
#define execKernelDynamicAllocation(kernel, gridSize, blockSize, sharedSize, deviceId, verbose, ...) \
{ \
    float singleKernelTime;\
    cudaEvent_t start, end; \
    CUDA_RUNTIME(cudaEventCreate(&start)); \
    CUDA_RUNTIME(cudaEventCreate(&end)); \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    CUDA_RUNTIME(cudaSetDevice(deviceId)); \
    CUDA_RUNTIME(cudaEventRecord(start)); \
    kernel<<<grid,block,sharedSize>>>(__VA_ARGS__); \
    CHECK_KERNEL(#kernel)\
    CUDA_RUNTIME(cudaPeekAtLastError());\
    CUDA_RUNTIME(cudaEventRecord(end));\
    \
    CUDA_RUNTIME(cudaEventSynchronize(start)); \
    CUDA_RUNTIME(cudaEventSynchronize(end)); \
    CUDA_RUNTIME(cudaDeviceSynchronize()); \
    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end)); \
    if(verbose) printf("%f", singleKernelTime/1000.0); \
    \
    {\
    }\
}

//if (false) Log(LogPriorityEnum::info, "Kernel: %s, time: %.2f ms.", #kernel, singleKernelTime); \
/*---- 2. CUDA kernel execution wrapper end ----*/

/*---- 3. CUDA function macros ---- */

#define WARP_REDUCE(var)    { \
                                var += __shfl_down_sync(0xFFFFFFFF, var, 16);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 8);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 4);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 2);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 1);\
                            }



struct CUDAContext {
    uint32_t max_threads_per_SM;
    uint32_t num_SMs;
    uint32_t shared_mem_size_per_block;
    uint32_t shared_mem_size_per_sm;

    CUDAContext() {
        /*get the maximal number of threads in an SM*/
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); /*currently 0th device*/
        max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
        //Log(LogPriorityEnum::info, "Shared MemPerBlock: %zu, PerSM: %zu", prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor);
        shared_mem_size_per_block = prop.sharedMemPerBlock;
        shared_mem_size_per_sm = prop.sharedMemPerMultiprocessor;
        num_SMs = prop.multiProcessorCount;
    }

    uint32_t GetConCBlocks(uint32_t block_size) {
        auto conc_blocks_per_SM = max_threads_per_SM / block_size; /*assume regs are not limited*/
        //Log(LogPriorityEnum::info, "#SMs: %d, con blocks/SM: %d", num_SMs, conc_blocks_per_SM);
        return conc_blocks_per_SM;
    }
};

template<typename T>
T getVal(T* arr, T index, AllocationTypeEnum at)
{

    if (at == AllocationTypeEnum::unified)
        return (arr[index]);

    T val = 0;
    CUDA_RUNTIME(cudaMemcpy(&val, &(arr[index]), sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return val;
}