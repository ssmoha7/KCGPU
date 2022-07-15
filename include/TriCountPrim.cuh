#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"

//#define PART_SIZE 8
// template<typename T, uint CPARTSIZE>
// __device__ __forceinline__ void reduce_part(T mask, uint64 &count)
// {
// 	// count += __shfl_down_sync(mask, count, 16);
// 	// count += __shfl_down_sync(mask, count, 8);
// 	// count += __shfl_down_sync(mask, count, 4);
// 	// count += __shfl_down_sync(mask, count, 2);
// 	// count += __shfl_down_sync(mask, count, 1);

// }


template<typename T, uint CPARTSIZE>
__device__ __forceinline__ void reduce_part(T partMask, uint64& warpCount) {
    for (int i = CPARTSIZE / 2; i >= 1; i /= 2) 
        warpCount += __shfl_down_sync(partMask, warpCount, i);
}

template<typename T, uint CPARTSIZE>
__device__ __forceinline__ void reduce_partT(T partMask, T& warpCount) {
	for (int i = CPARTSIZE / 2; i >= 1; i /= 2) 
		warpCount += __shfl_down_sync(partMask, warpCount, i);
}



template <typename T, int BLOCK_DIM_X, typename OUTTYPE=unsigned short>
__device__ __forceinline__ void block_filter_pivot(T count, T* output, OUTTYPE* gl)
{
    typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    auto tid = threadIdx.x;
    T srcLenBlocks = (count + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
    T threadData = 0;
    T aggreagtedData = 0;
    T total = 0;

    for (T k = 0; k < srcLenBlocks; k++)
    {
        T index = k * BLOCK_DIM_X + tid;
        T dataread = 0xFFFFFFFF;
        if(index < count)
            dataread = output[index];
        threadData = 0;
        aggreagtedData = 0;

        if (index < count && dataread != 0xFFFFFFFF)
        {
            threadData = 1;
        }

        __syncthreads();
        BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
        __syncthreads();


        if (index < count && dataread != 0xFFFFFFFF)
            output[threadData + total] = dataread;

        total += aggreagtedData;
        __syncthreads();
    }

    if(threadIdx.x == 0)
        *gl = total;
        
}


#define WARP_REDUCE_MASK(var, mask)    { \
                                var += __shfl_down_sync(mask, var, 16);\
                                var += __shfl_down_sync(mask, var, 8);\
                                var += __shfl_down_sync(mask, var, 4);\
                                var += __shfl_down_sync(mask, var, 2);\
                                var += __shfl_down_sync(mask, var, 1);\
                            }

namespace graph
{
    template <typename T>
    __host__ __device__ T binary_search(const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal   //!< [in] value to search for
    ) {
        T left = lt;
        T right = rt;
        while (left < right) {
            const T mid = (left + right) / 2;
            T val = arr[mid];
            if (val == searchVal)
                return mid;
            bool pred = val < searchVal;
            if (pred) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }
        return left;
    }

    template <typename T>
    __host__ __device__ T binary_search(const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal,   //!< [in] value to search for
        bool& found
    ) {
        T left = lt;
        T right = rt;
        found = false;
        while (left < right) {
            const T mid = (left + right) / 2;
            T val = arr[mid];
            if (val == searchVal)
            {
                found = true;
                return mid;
            }
            bool pred = val < searchVal;
            if (pred) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }
        return left;
    }


    template <typename T>
    __host__ __device__ T binary_search_left(const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal,   //!< [in] value to search for
        bool& found
    ) {
        T left = lt;
        T right = rt;
        found = false;
        while (left < right) {
            const T mid = (left + right) / 2;
            T val = arr[mid];
            if (val == searchVal)
            {
                found = true;
                return mid;
            }
            bool pred = val < searchVal;
            if (pred) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }
        return left > 0 ? left - 1 : 0;
    }




    template <typename T>
    __host__ __device__ T binary_search_full_bst(const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal   //!< [in] value to search for
    ) {
        T n = lt;
        while (n < rt)
        {
            if (arr[n] == searchVal)
                break;
            else if (arr[n] > searchVal)
                n = 2 * n + 1;
            else n = 2 * n + 2;
        }

        return n;

    }

    template <typename T>
    __global__ void binary_search_g(T* result,
        const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal   //!< [in] value to search for
    ) {
        T left = lt;
        T right = rt;
        while (left < right) {
            const T mid = (left + right) / 2;
            T val = arr[mid];
            bool pred = val < searchVal;
            if (pred) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }

        *result = left;
    }


    template <typename T>
    __device__ bool hash_search(
        const T* arr,         //!< [in] array to search
        const T binSize,
        const T size, //!< [in] size of array
        const T stashSize,
        const T searchVal   //!< [in] value to search for
    ) {
        const int numBins = (size + binSize - 1) / binSize;
        const int stashStart = binSize * numBins;
        T b = (searchVal / 11) % numBins;


        for (int i = 0; i < binSize; i++)
        {
            T val = arr[b * binSize + i];
            if (searchVal == arr[b * binSize + i])
            {
                return true;
            }
            if (val == 0xFFFFFFFF)
            {
                return false;
            }
        }
        //for (int i = 0; i < stashSize; i++)
        //{
        //    if (arr[i + stashStart] == searchVal)
        //    {
        //        //printf("Hash - Bin: %u\n", searchVal);
        //        return true;
        //    }
        //}


       /*T left = graph::binary_search<T>(&arr[b*binSize], 0, binSize, searchVal);
       if (arr[b*binSize + left] == searchVal)
       {
           return true;
       }*/

        T left = graph::binary_search<T>(&arr[stashStart], 0, stashSize, searchVal);
        if (arr[stashStart + left] == searchVal)
        {
            return true;
        }

        return false;
    }

    template<typename T, size_t BLOCK_DIM_X>
    __global__ void hash_search_g(T* count,
        T* A, T sizeA, T* B, T sizeB, const T binSize, const T stashSize)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int threadCount = 0;
        for (int i = tid; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            threadCount += graph::hash_search<T>(B, binSize, sizeB, stashSize, searchVal) ? 1 : 0;
        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }

    }

    template <typename T>
    __device__ bool hash_nostash_search(
        const T* arrPointer,
        const T* arr,         //!< [in] array to search
        const T numBins,
        const T searchVal   //!< [in] value to search for
    ) {
        T b = (searchVal / 11) % numBins;
        T start = arrPointer[b];
        T end = arrPointer[b + 1];

        if (end - start == 0)//empty bin
            return false;

        /* if (end - start < 32)
         {
             for (T i = start; i < end; i++)
             {
                 if (searchVal == arr[i])
                 {
                     return true;
                 }
             }
         }
        else*/
        {
            T left = graph::binary_search<T>(&arr[start], 0, end - start, searchVal);
            if (arr[start + left] == searchVal)
            {
                return true;
            }

        }
        return false;
    }


    template<typename T, size_t BLOCK_DIM_X>
    __device__ int hash_search_nostash_thread_d(T* A, T sizeA, T* BP, T* BD, T numBins)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int threadCount = 0;
        for (int i = 0; i < sizeA; i++)
        {
            T searchVal = A[i];

            //printf("%d, %u\n", i, searchVal);
            threadCount += graph::hash_nostash_search<T>(BP, BD, numBins, searchVal) ? 1 : 0;
        }
        return threadCount;
    }

    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ int hash_search_nostash_warp_d(T* A, T sizeA, T* BP, T* BD, T numBins)
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < sizeA; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            threadCount += graph::hash_nostash_search<T>(BP, BD, numBins, searchVal) ? 1 : 0;
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else
        {
            return threadCount;
        }
    }


    template<typename T, size_t BLOCK_DIM_X>
    __global__ void hash_search_nostash_g(T* count,
        T* A, T sizeA, T* BP, T* BD, T numBins)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int threadCount = 0;
        for (int i = tid; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            threadCount += graph::hash_nostash_search<T>(BP, BD, numBins, searchVal) ? 1 : 0;
        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }

    }



    template<typename T, size_t BLOCK_DIM_X>
    __global__ void binary_search_2arr_g(T* count,
        T* A, T sizeA, T* B, T sizeB)
    {
        size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
        uint64_t threadCount = 0;

        for (size_t i = gx; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            const T lb = graph::binary_search<T>(B, 0, sizeB, searchVal);
            if (lb < sizeB)
            {
                threadCount += (B[lb] == searchVal ? 1 : 0);
            }

        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }
    }

    template<typename T, size_t BLOCK_DIM_X>
    __global__ void binary_search_bst_g(T* count,
        T* A, T sizeA, T* B, T sizeB)
    {
        size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
        uint64_t threadCount = 0;

        for (size_t i = gx; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            const T lb = graph::binary_search_full_bst<T>(B, 0, sizeB, searchVal);
            if (lb < sizeB)
            {
                threadCount += (B[lb] == searchVal ? 1 : 0);
            }

        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }
    }



    // Count per thread
    template <typename T>
    __device__ __forceinline__ uint64_t thread_sorted_count_binary(const T* A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T* B, //!< [in] array B
        const T bSz  //!< [in] the number of elements in B
    ) {
        uint64_t threadCount = 0;
        T lb = 0;
        // cover entirety of A with warp
        for (size_t i = 0; i < aSz; i++) {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            lb = graph::binary_search<T>(B, lb, bSz, searchVal);
            if (lb < bSz)
            {
                threadCount += (B[lb] == searchVal ? 1 : 0);
            }
            else
            {
                break;
            }
        }
        return threadCount;
    }


    template <typename T>
    __device__ __forceinline__ uint64_t thread_sorted_count_upto_binary(int upto, bool* mask, T srcStart, T dstStart, const T* arr, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T bSz  //!< [in] the number of elements in B
    ) {
        uint64_t threadCount = 0;
        T lb = 0;
        // cover entirety of A with warp
        for (size_t i = 0; i < aSz; i++)
        {

            // one element of A per thread, just search for A into B
            const T searchVal = arr[srcStart + i];

            lb = graph::binary_search<T>(&arr[dstStart], lb, bSz, searchVal);

            if (lb < bSz)
            {
                if (mask[dstStart + lb] && mask[srcStart + i])
                {
                    threadCount += (arr[dstStart + lb] == searchVal ? 1 : 0);
                }
                if (threadCount == upto)
                    return threadCount;
            }
            else
            {
                break;
            }
        }

        return threadCount;
    }


    template <typename T>
    __device__ __forceinline__ uint64_t thread_sorted_set_binary(T* arr, const T* A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T* B, //!< [in] array B
        const T bSz  //!< [in] the number of elements in B
    ) {
        uint64_t threadCount = 0;
        T lb = 0;
        // cover entirety of A with warp
        for (size_t i = 0; i < aSz; i++) {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            lb = graph::binary_search<T>(B, lb, bSz, searchVal);
            if (lb < bSz)
            {
                if (searchVal == B[lb])
                {
                    arr[threadCount] = searchVal;
                    threadCount++;
                }
            }
            else
            {
                break;
            }
        }
        return threadCount;
    }



    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_binary(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz  //!< [in] the number of elements in B
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {

                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        //printf("At %u, SearchVal = %u\n", lb, searchVal);
                        threadCount++;
                    }


                }

                lastIndex = lb;
            }

            // unsigned int writemask_deq = __activemask();
            // lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else
        {
            return threadCount;
        }
    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_binary(int startCount, const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz  //!< [in] the number of elements in B
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = startCount;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {

                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        //printf("At %u, SearchVal = %u\n", lb, searchVal);
                        threadCount++;
                    }


                }

                lastIndex = lb;
            }

            // unsigned int writemask_deq = __activemask();
            // lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64_t> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            // return aggregate;

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);
        }
        else
        {
            return threadCount;
        }
    }

    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_binary_s(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B
        T* first_level,
        T par,
        int numElements,
        int pwMaxSize
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;
        T fl = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];

            bool found = false;
            fl = graph::binary_search_left<T>(first_level, fl, numElements, searchVal, found);
            lastIndex = par * fl;
            T right = 0;

            if (bSz < pwMaxSize)
            {
                if (found)
                {
                    threadCount++;
                    //printf("At %u, searchVal = %u, direct\n", fl * par, searchVal);
                }
                continue;
            }
            else if (fl == numElements - 1)
                right = bSz;
            else
                right = (fl + 1) * par;

            const T lb = graph::binary_search_left<T>(B, lastIndex, right, searchVal, found);
            if (found)
            {
                threadCount++;
            }

            //lastIndex = lb;


           /* unsigned int writemask_deq = __activemask();
            fl = __shfl_sync(writemask_deq, fl, 31);*/
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64_t> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_mask_binary(bool* mask, T srcStart, T dstStart, const T* const arr, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T bSz  //!< [in] the number of elements in B
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = arr[srcStart + i];
            const T leftValue = arr[dstStart + lastIndex];

            if (searchVal >= leftValue && mask[srcStart + i])
            {

                const T lb = graph::binary_search<T>(&arr[dstStart], lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    threadCount += ((arr[dstStart + lb] == searchVal) && (mask[dstStart + lb]) ? 1 : 0);
                }

                lastIndex = lb;
            }

            unsigned int writemask_deq = __activemask();
            lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else
        {
            return threadCount;
        }
    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_binary_upto(int k, bool* keep,
        const T* const A, //!< [in] array A
        const T aStart,
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        const T bStart,
        T bSz  //!< [in] the number of elements in B
    )
    {
        uint64_t  lastCount = 0;
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        uint64_t warpCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        int round = 0;
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            if (keep[i + aStart])
            {
                const T searchVal = A[i];
                const T leftValue = B[lastIndex];
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal && keep[lb + bStart])
                    {
                        //printf("At %u, SearchVal = %u\n", lb, searchVal);
                        threadCount++;
                    }
                }
                lastIndex = lb;

            }

            round++;

            if (round * 32 > k)
            {
                unsigned int writemask_deq = __activemask();
                warpCount = threadCount;
                WARP_REDUCE_MASK(warpCount, writemask_deq);
                warpCount = __shfl_sync(writemask_deq, warpCount, 0);
                if (warpCount >= k)
                    break;
            }
        }

        return warpCount;
    }




    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_set_binary(T* indecies, T* arr, const T* A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T* B, //!< [in] array B
        const T bSz  //!< [in] the number of elements in B

    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {

                const T lb = graph::binary_search<T>(B, 0, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        T index = atomicAdd(indecies, 1);
                        arr[index] = searchVal;
                        threadCount++;
                    }

                }

                lastIndex = lb;
            }

            unsigned int writemask_deq = __activemask();
            lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }


        return threadCount;

    }


    template <typename T>
    __host__ __device__ static uint8_t serial_sorted_count_binary(const T* const array, //!< [in] array to search through
        size_t left,          //!< [in] lower bound of search
        size_t right,         //!< [in] upper bound of search
        const T search_val    //!< [in] value to search for
    ) {
        while (left < right) {
            size_t mid = (left + right) / 2;
            T val = array[mid];
            if (val < search_val) {
                left = mid + 1;
            }
            else if (val > search_val) {
                right = mid;
            }
            else { // val == search_val
                return 1;
            }
        }
        return 0;
    }
    template <size_t BLOCK_DIM_X, typename T, bool reduce = true>
    __device__ uint64_t block_sorted_count_binary(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz  //!< [in] the number of elements in B
    ) {


        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with block
        for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        threadCount++;
                    }

                }

                lastIndex = lb;
            }
        }

        if (reduce)
        {
            __syncthreads();

            typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
            __shared__ typename BlockReduce::TempStorage tempStorage;
            uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);
            return aggregate;
        }
        else return threadCount;



    }

    template <size_t BLOCK_DIM_X, typename T, bool reduce = true>
    __device__ uint64_t block_sorted_count_and_set_binary(const T* const A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const T bSz,  //!< [in] the number of elements in B

        bool AisMaster,
        T startIndex,
        T endIndex,
        char* current_level,
        T* filter_scan,
        T new_level,
        T clique_number


    ) {

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with block
        for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        threadCount++;

                        //////////////////////////////Device function ///////////////////////
                        if (new_level < clique_number)
                        {
                            T level_index = filter_scan[startIndex + (AisMaster ? i : lb)];
                            current_level[level_index] = new_level;
                        }
                        /////////////////////////////////////////////////////////////////////
                    }
                }

                lastIndex = lb;
            }
        }

        if (reduce)
        {
            __syncthreads();

            typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
            __shared__ typename BlockReduce::TempStorage tempStorage;
            uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);
            return aggregate;
        }
        else return threadCount;



    }



    template <size_t BLOCK_DIM_X, typename T, bool reduce = true>
    __device__ uint64_t block_sorted_count_and_set_binary2(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz,  //!< [in] the number of elements in B

        bool AisMaster,
        T startIndex,
        char* current_level,
        T new_level,
        T clique_number


    ) {

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with block
        for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
        {
            bool a = !AisMaster || (AisMaster && current_level[i] == new_level - 1);
            if (a)
            {
                // one element of A per thread, just search for A into B
                const T searchVal = A[i];
                const T leftValue = B[lastIndex];
                if (searchVal >= leftValue)
                {
                    bool found = false;
                    const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal, found);

                    if (found)
                    {
                        bool b = AisMaster || (!AisMaster && current_level[lb] == new_level - 1);
                        if (b)
                        {
                            threadCount++;

                            //////////////////////////////Device function ///////////////////////
                            if (new_level < clique_number)
                            {
                                T level_index = (AisMaster ? i : lb);
                                current_level[level_index] = new_level;
                            }
                            /////////////////////////////////////////////////////////////////////
                        }
                    }

                    lastIndex = lb;
                }
            }
        }

        if (reduce)
        {
            __syncthreads();

            typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
            __shared__ typename BlockReduce::TempStorage tempStorage;
            uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);
            return aggregate;
        }
        else return threadCount;
    }



    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64 warp_sorted_count_and_set_binary(uint64 startCount, const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        bool AisMaster,
        T startIndex,
        char* current_level,
        T new_level,
        T clique_number
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64 threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += 32)
        {
            bool a = !AisMaster || (AisMaster && current_level[i] == new_level - 1);
            if (a)
            {
                const T searchVal = A[i];
                const T leftValue = B[lastIndex];
                if (searchVal >= leftValue)
                {
                    bool found = false;
                    const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal, found);

                    if (found)
                    {
                        bool b = AisMaster || (!AisMaster && current_level[lb] == new_level - 1);
                        if (b)
                        {
                            //printf("At %u, SearchVal = %u\n", lb, searchVal);
                            threadCount++;
                            //////////////////////////////Device function ///////////////////////
                            if (new_level < clique_number)
                            {
                                T level_index = (AisMaster ? i : lb);
                                current_level[level_index] = new_level;
                            }
                            /////////////////////////////////////////////////////////////////////
                        }


                    }

                    lastIndex = lb;
                }

            }
            // one element of A per thread, just search for A into B


            // unsigned int writemask_deq = __activemask();
            // lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64 aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            // return aggregate;

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }



    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_set_binary2(uint64 startCount, const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        bool AisMaster,
        T startIndex,
        char* current_level,
        T new_level,
        T clique_number
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp

        uint64 threadCount = startCount;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            // one element of A per thread, just search for A into B

            bool a = !AisMaster || (AisMaster && (current_level[i] & (0x01 << (new_level - 3))));
            if (a)
            {
                const T searchVal = A[i];
                const T leftValue = B[lastIndex];
                bool found = false;
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal, found);
                if (found)
                {
                    bool b = AisMaster || (!AisMaster && (current_level[lb] & (0x01 << (new_level - 3))));   //current_level[lb] == new_level - 1);
                    if (b)
                    {
                        //printf("At %u, SearchVal = %u\n", lb, searchVal);
                        threadCount++;
                        //////////////////////////////Device function ///////////////////////
                        if (new_level < clique_number)
                        {
                            T level_index = (AisMaster ? i : lb);
                            current_level[level_index] |= (0x01 << (new_level - 2));
                        }
                        /////////////////////////////////////////////////////////////////////
                    }


                }

                lastIndex = lb;
            }

            // unsigned int writemask_deq = __activemask();
            // lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64 aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            // return aggregate;

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);
            //reduce_part<T>(partMask, threadCount);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }



     template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32, typename K=char>
    __device__ __forceinline__ uint64 warp_sorted_count_and_set_binary2(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        bool AisMaster,
        T startIndex,
        K* current_level,
        T new_level,
        T clique_number,
        T partMask,
        T origin
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp

        uint64 threadCount = 0;
        //T lastIndex = 0;

        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            // one element of A per thread, just search for A into B

            bool a = !AisMaster || (AisMaster && (current_level[i] & (0x01 << (new_level - origin))));
            if (a)
            {
                const T searchVal = A[i];
                //const T leftValue = B[lastIndex];
                bool found = false;
                const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
                if (found)
                {
                    bool b = AisMaster || (!AisMaster && (current_level[lb] & (0x01 << (new_level - origin))));   //current_level[lb] == new_level - 1);
                    if (b)
                    {
                        //printf("At %u, SearchVal = %u\n", lb, searchVal);
                        threadCount++;
                        //////////////////////////////Device function ///////////////////////
                        if (new_level < clique_number)
                        {
                            T level_index = (AisMaster ? i : lb);
                            current_level[level_index] |= (0x01 << (new_level - origin + 1));
                        }
                        /////////////////////////////////////////////////////////////////////
                    }


                }

               // lastIndex = lb;
            }

            // unsigned int writemask_deq = __activemask();
            // lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64 aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            // return aggregate;

            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            reduce_part<T,CPARTSIZE>(partMask, threadCount);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }



    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_subgraph_binary2( T* A, //!< [in] array A
        T aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T* current_level,
        T* counter,
        T new_level,
        T clique_number,
        T partMask
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp

        uint64 threadCount = 0;
        //T lastIndex = 0;

        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            //const T leftValue = B[lastIndex];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                    //printf("At %u, SearchVal = %u\n", lb, searchVal);
                    //threadCount++;
                    //////////////////////////////Device function ///////////////////////
                    //if (new_level < clique_number)
                    {
                        //current_level[i] = searchVal;

                        T old = atomicAdd(counter, 1);
                        current_level[old] = searchVal;
                    }
                    /////////////////////////////////////////////////////////////////////
                


            }

               // lastIndex = lb;
            

            // unsigned int writemask_deq = __activemask();
            // lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64 aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            // return aggregate;

            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            // threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            //reduce_part<T>(partMask, threadCount);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }





    template <typename T, uint CPARTSIZE = 32, typename K=char>
    __device__ __forceinline__ uint64 warp_sorted_count_and_set_binary3(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B
        K* current_level,
        T new_level,
        T clique_number,
        T partMask,
        T origin
    )
    {
        const T warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const T laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp

        uint64 threadCount = 0;
        //T lastIndex = 0;

        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            //const T leftValue = B[lastIndex];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                if (current_level[lb] == new_level - 1)
                {
                    //printf("At %u, SearchVal = %u\n", lb, searchVal);
                    threadCount++;
                    //////////////////////////////Device function ///////////////////////
                    if (new_level < clique_number)
                    {
                        current_level[lb] = new_level;
                    }
                    /////////////////////////////////////////////////////////////////////
                }


            }

        }

        reduce_part<T, CPARTSIZE>(partMask, threadCount);
        return threadCount;
    }




    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_set_binary_s(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B
        T* first_level,
        T par,
        T numElements,
        T pwMaxSize,

        bool AisMaster,
        char* current_level,
        T new_level,
        T clique_number
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp
        uint64_t threadCount = 0;
        T lastIndex = 0;
        T fl = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            bool a = !AisMaster || (AisMaster && current_level[i] == new_level - 1);
            if (a)
            {
                bool found = false;
                fl = graph::binary_search_left<T>(first_level, fl, numElements, searchVal, found);
                lastIndex = par * fl;
                T right = 0;

                if (bSz < pwMaxSize)
                {
                    bool b = AisMaster || (!AisMaster && current_level[lastIndex] == new_level - 1);
                    if (found && b)
                    {
                        threadCount++;

                        /////////////////////////////////////////
                        if (new_level < clique_number)
                        {
                            T level_index = (AisMaster ? i : lastIndex);
                            current_level[level_index] = new_level;
                        }
                        ////////////////////////////////////////////////

                        //printf("At %u, searchVal = %u, direct\n", fl * par, searchVal);
                    }
                    continue;
                }
                else if (fl == numElements - 1)
                    right = bSz;
                else
                    right = (fl + 1) * par;

                found = false;
                const T lb = graph::binary_search_left<T>(B, lastIndex, right, searchVal, found);
                if (found)
                {
                    bool b = AisMaster || (!AisMaster && current_level[lb] == new_level - 1);
                    if (b)
                    {
                        //printf("At %u, searchVal = %u\n", lb, searchVal);
                        threadCount++;
                        if (new_level < clique_number)
                        {
                            T level_index = (AisMaster ? i : lb);
                            current_level[level_index] = new_level;
                        }
                    }

                }

            }

            //lastIndex = lb;


           /* unsigned int writemask_deq = __activemask();
            fl = __shfl_sync(writemask_deq, fl, 31);*/
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64_t> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }







    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_set_binary_sbs(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B
        T* first_level,
        T par,
        T numElements,
        T pwMaxSize,

        char* current_level,
        T new_level,
        T clique_number
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp
        uint64_t threadCount = 0;
        T lastIndex = 0;
        T fl = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            if (current_level[i] & (0x01 << (new_level - 3)))
            {
                const T searchVal = A[i];
                bool found = false;
                fl = graph::binary_search_left<T>(first_level, fl, numElements, searchVal, found);
                lastIndex = par * fl;
                T right = 0;

                if (bSz < pwMaxSize)
                {
                    if (found)
                    {
                        threadCount++;
                        /////////////////////////////////////////
                        if (new_level < clique_number)
                        {
                            current_level[i] |= (0x01 << (new_level - 2));
                        }
                        ////////////////////////////////////////////////
                    }
                    continue;
                }
                else if (fl == numElements - 1)
                    right = bSz;
                else
                    right = (fl + 1) * par;

                found = false;
                const T lb = graph::binary_search_left<T>(B, lastIndex, right, searchVal, found);
                if (found)
                {
                    threadCount++;
                    if (new_level < clique_number)
                    {
                        current_level[i] |= (0x01 << (new_level - 2));
                    }

                }
            }
        }

        if (reduce)
        {
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_set_binary_sbd(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B
        T* first_level,
        T par,
        T numElements,
        T pwMaxSize,
        char* current_level,
        T new_level,
        T clique_number
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp
        uint64_t threadCount = 0;
        T lastIndex = 0;
        T fl = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            const T searchVal = A[i];
            bool found = false;
            fl = graph::binary_search_left<T>(first_level, fl, numElements, searchVal, found);
            lastIndex = par * fl;
            T right = 0;

            if (bSz < pwMaxSize)
            {
                bool b = (current_level[lastIndex] & (0x01 << (new_level - 3)));
                if (found && b)
                {
                    threadCount++;

                    /////////////////////////////////////////
                    if (new_level < clique_number)
                    {
                        current_level[lastIndex] |= (0x01 << (new_level - 2));
                    }
                    ////////////////////////////////////////////////
                }
                continue;
            }
            else if (fl == numElements - 1)
                right = bSz;
            else
                right = (fl + 1) * par;

            found = false;
            const T lb = graph::binary_search_left<T>(B, lastIndex, right, searchVal, found);
            if (found)
            {
                bool b = (current_level[lb] & (0x01 << (new_level - 3)));
                if (b)
                {
                    //printf("At %u, searchVal = %u\n", lb, searchVal);
                    threadCount++;
                    if (new_level < clique_number)
                    {
                        current_level[lb] |= (0x01 << (new_level - 2));
                    }
                }
            }
        }

        if (reduce)
        {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64_t> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_encode(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T* encode
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            const T searchVal = A[i];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                //////////////////////////////Device function ///////////////////////
                T chunk_index = i / 32; // 32 here is the division size of the encode
                T inChunkIndex = i % 32;
                atomicOr(&encode[chunk_index], 1 << inChunkIndex);
                /////////////////////////////////////////////////////////////////////
            }
        }


        return 0;

    }



    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_encode_full_hyb(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T j,
        T num_divs_local,
        T* encode
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
        // cover entirety of A with warp
        if(bSz < aSz)
            for (T i = laneIdx; i < bSz; i += CPARTSIZE)
            {
                const T searchVal = B[i];
                bool found = false;
                const T lb = graph::binary_search<T>(A, 0, aSz, searchVal, found);

                if (found)
                {
                    //////////////////////////////Device function ///////////////////////
                    T chunk_index = lb / 32; // 32 here is the division size of the encode
                    T inChunkIndex = lb % 32;
                    atomicOr(&encode[j*num_divs_local + chunk_index], 1 << inChunkIndex);

                    T chunk_index1 = j / 32; // 32 here is the division size of the encode
                    T inChunkIndex1 = j % 32;
                    atomicOr(&encode[lb*num_divs_local + chunk_index1], 1 << inChunkIndex1);

                    /////////////////////////////////////////////////////////////////////
                }
            }
        else
            for (T i = laneIdx; i < aSz; i += CPARTSIZE)
            {
                const T searchVal = A[i];
                bool found = false;
                const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
                if (found)
                {
                    //////////////////////////////Device function ///////////////////////
                    T chunk_index = i / 32; // 32 here is the division size of the encode
                    T inChunkIndex = i % 32;
                    atomicOr(&encode[j*num_divs_local + chunk_index], 1 << inChunkIndex);

                    T chunk_index1 = j / 32; // 32 here is the division size of the encode
                    T inChunkIndex1 = j % 32;
                    atomicOr(&encode[i*num_divs_local + chunk_index1], 1 << inChunkIndex1);

                    /////////////////////////////////////////////////////////////////////
                }
            }


        return 0;

    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_encode_full(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T j,
        T num_divs_local,
        T* encode
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            const T searchVal = A[i];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                //////////////////////////////Device function ///////////////////////
                T chunk_index = i / 32; // 32 here is the division size of the encode
                T inChunkIndex = i % 32;
                atomicOr(&encode[j*num_divs_local + chunk_index], 1 << inChunkIndex);

                T chunk_index1 = j / 32; // 32 here is the division size of the encode
                T inChunkIndex1 = j % 32;
                atomicOr(&encode[i*num_divs_local + chunk_index1], 1 << inChunkIndex1);

                /////////////////////////////////////////////////////////////////////
            }
        }


        return 0;

    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_encode_full_mclique(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T j,
        T num_divs_local,
        T* encode,
        T base
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            const T searchVal = A[i];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                //////////////////////////////Device function ///////////////////////
                T chunk_index = i / 32; // 32 here is the division size of the encode
                T inChunkIndex = i % 32;
                atomicOr(&encode[(j >= base ? j - base : j + aSz)*num_divs_local + chunk_index], 1 << inChunkIndex);

                if (j >= base)
                {
                    T chunk_index1 = (j - base) / 32; // 32 here is the division size of the encode
                    T inChunkIndex1 = (j - base) % 32;
                    atomicOr(&encode[i*num_divs_local + chunk_index1], 1 << inChunkIndex1);
                }
                /////////////////////////////////////////////////////////////////////
            }
        }
        return 0;
    }

    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_subgraph_full(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T j,
        T maxdeg,
        T* encode
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            const T searchVal = A[i];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                encode[j*maxdeg + i] = i;
                encode[i*maxdeg + j] = j;
            }
        }


        return 0;

    }

    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true, uint CPARTSIZE = 32>
    __device__ __forceinline__ uint64 warp_sorted_count_and_subgraph(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        T j,
        T maxdeg,
        T* encode
    )
    {
        const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
        const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += CPARTSIZE)
        {
            const T searchVal = A[i];
            bool found = false;
            const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
            if (found)
            {
                encode[j*maxdeg + i] = i;
                //encode[i*maxdeg + j] = j;
            }
        }


        return 0;

    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64 warp_sorted_count_and_encode_old(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B

        bool AisMaster,
        T* encode,
        T new_level,
        T clique_number
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64 threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (T i = laneIdx; i < aSz; i += 32)
        {
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];
            bool found = false;
            const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal, found);
            if (found)
            {
                threadCount++;
                //////////////////////////////Device function ///////////////////////
                //if (new_level < clique_number)
                {
                    T intr_index = (AisMaster ? i : lb);
                    T chunk_index = intr_index / 32;
                    T inChunkIndex = intr_index % 32;
                    atomicOr(&encode[chunk_index], 1 << inChunkIndex);
                }
                /////////////////////////////////////////////////////////////////////
            }

            lastIndex = lb;
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            // typedef cub::WarpReduce<uint64> WarpReduce;
            // __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            // uint64 aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            // return aggregate;

            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 16);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 8);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 4);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 2);
            threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, 1);

            return threadCount;
        }
        else
        {
            return threadCount;
        }
    }



    template <size_t BLOCK_DIM_X, typename T>
    __device__ __forceinline__ uint64_t block_count_and_set_tri(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz,  //!< [in] the number of elements in B
        T *tri,
        T *counter
    ) {
        //T lastIndex = 0;
        // cover entirety of A with block
        for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            //const T leftValue = B[lastIndex];
            //if (searchVal >= leftValue)
            {
                bool found = false;
                graph::binary_search<T>(B, 0, bSz, searchVal, found);
                if (found)
                {
                    T old = atomicAdd(counter, 1);
                    tri[old] = searchVal;
                }

                //lastIndex = lb;
            }
            
        }

        return 0;
    }



    template <size_t BLOCK_DIM_X, typename T>
    __device__ __forceinline__ uint64_t block_sorted_count_and_set_tri(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz,  //!< [in] the number of elements in B
        T *tri
    ) {
        T lastIndex = 0;
        // cover entirety of A with block
        for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];
            if (searchVal >= leftValue)
            {
                bool found = false;
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal, found);
                if (found)
                {
                   
                    tri[i] = searchVal;
                }

                lastIndex = lb;
            }
            
        }

        return 0;
    }


    template <size_t WARPS_PER_BLOCK, typename T>
    __device__ __forceinline__ uint64_t warp_sorted_count_and_set_tri(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz,  //!< [in] the number of elements in B
        T *tri,
        T *counter
    ) {
        T lastIndex = 0;
        const int laneIdx = threadIdx.x % 32;
        // cover entirety of A with block
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];
            if (searchVal >= leftValue)
            {
                bool found = false;
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal, found);
                if (found)
                {
                    T old = atomicAdd(counter, 1);
                    tri[old] = searchVal;
                }

                lastIndex = lb;
            }
            
        }

        return 0;
    }

    ///////////////////////////////////////SERIAL INTERSECTION //////////////////////////////////////////////
    template <typename T>
    __host__ __device__ static size_t serial_sorted_count_linear(T min, const T* const A, //!< beginning of a
        const size_t aSz,
        const T* const B, //!< beginning of b
        const size_t bSz) {
        return serial_sorted_count_linear(min, A, &A[aSz], B, &B[bSz]);
    }


    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_count_linear(const T min, const T* const aBegin, //!< beginning of a
        const T* const aEnd,   //!< end of a
        const T* const bBegin, //!< beginning of b
        const T* const bEnd    //!< end of b
    ) {
        uint64_t count = 0;
        const T* ap = aBegin;
        const T* bp = bBegin;

        bool loadA = true;
        bool loadB = true;

        T a, b;


        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = *ap;
                loadA = false;
            }
            if (loadB) {
                b = *bp;
                loadB = false;
            }

            if (a == b) {
                ++count;
                ++ap;
                ++bp;
                loadA = true;
                loadB = true;
            }
            else if (a < b) {
                ++ap;
                loadA = true;
            }
            else {
                ++bp;
                loadB = true;
            }
        }
        return count;
    }




    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_count_upto_linear(int upto, bool* mask, const T min, T* arr,
        const T const aBegin, //!< beginning of a
        const T const aEnd,   //!< end of a
        const T const bBegin, //!< beginning of b
        const T const bEnd    //!< end of b
    ) {
        uint64_t count = 0;
        T ap = aBegin;
        T bp = bBegin;

        bool loadA = true;
        bool loadB = true;

        T a, b;


        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = arr[ap];
                loadA = false;
            }
            if (loadB) {
                b = arr[bp];
                loadB = false;
            }

            if (a == b) {
                if (!mask[ap] && !mask[bp])
                    ++count;
                if (count == upto)
                {
                    return count;
                }
                ++ap;
                ++bp;
                loadA = true;
                loadB = true;
            }
            else if (a < b)
            {
                ++ap;
                loadA = true;
            }
            else {
                ++bp;
                loadB = true;
            }
        }
        return count;
    }


    //min is used to enforce increasing order !!
    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_set_linear(T* arr, const T* aBegin, //!< beginning of a
        const T* aEnd,   //!< end of a
        const T* bBegin, //!< beginning of b
        const T* bEnd    //!< end of b
    ) {
        uint64_t count = 0;
        const T* ap = aBegin;
        const T* bp = bBegin;

        bool loadA = true;
        bool loadB = true;

        T a, b;


        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = *ap;
                loadA = false;
            }
            if (loadB) {
                b = *bp;
                loadB = false;
            }

            if (a == b) {
                arr[count] = a;
                ++count;
                ++ap;
                ++bp;
                loadA = true;
                loadB = true;
            }
            else if (a < b) {
                ++ap;
                loadA = true;
            }
            else {
                ++bp;
                loadB = true;
            }
        }
        return count;
    }

}
