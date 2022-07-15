#pragma once

#include "utils.cuh"



#pragma once

#include "cuda_runtime.h"
#include "../include/CGArray.cuh"
#include "../include/GraphDataStructure.cuh"
#include "../include/GraphQueue.cuh"


template <typename T, int BLOCK_DIM_X>
__global__ void
kernel_per_block_reduce(
    T* input,
    T count,
    T* blockData,
    T* totalCount
)
{
    //CUB reduce
    typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    T threadData = 0;
    T aggreagtedData = 0;

    auto tid = threadIdx.x;
    const size_t gtx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    threadData = 0;
    aggreagtedData = 0;
    if (gtx < count)
    {
        threadData = input[gtx];
    }
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
    __syncthreads();

    if (tid == 0)
    {
        blockData[blockIdx.x] = aggreagtedData;

        if (blockIdx.x == gridDim.x - 1)
        {
            *totalCount = aggreagtedData;
        }
    }
}


template <typename T, int BLOCK_DIM_X, int LOOP=1>
__global__ void
kernel_per_block_reduce(
    bool* keep,
    T count,
    T* blockData,
    T* totalCount
)
{
    //CUB reduce
    typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    T threadData[LOOP] = {0};
    T aggreagtedData = 0;

    auto tid = threadIdx.x;
    const size_t gtx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    aggreagtedData = 0;
    auto stride = 1; //BLOCK_DIM_X * gridDim.x;
    for (int i = 0; i < LOOP; i++) {
        size_t idx = gtx*LOOP + i;
        if (idx < count)
        {   
            threadData[i] = keep[idx]? 1 : 0;
        }
    }

    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
    __syncthreads();

    if (tid == 0)
    {
        blockData[blockIdx.x] = aggreagtedData;

        if (blockIdx.x == gridDim.x - 1)
        {
            *totalCount = aggreagtedData;
        }
    }

}




template <typename T, int BLOCK_DIM_X>
__global__ void
kernel_per_block_scatter(
    T* input,
    T count,
    T* blockData, //prefix sum is done to it
    T* output
)
{
    //CUB reduce
    typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    T threadData = 0;
    T aggreagtedData = 0;

    auto tid = threadIdx.x;
    const size_t gtx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    threadData = 0;
    aggreagtedData = 0;
    if (gtx < count)
    {
        threadData = input[gtx];
    }
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
    __syncthreads();

    if (gtx < count)
        output[gtx] = threadData + blockData[blockIdx.x];
}




template <typename T, int BLOCK_DIM_X, int LOOP_CNT = 1>
__global__ void
kernel_per_block_scatter(
    T* input,
    bool* keep,
    T count,
    T* blockData, //prefix sum is done to it
    T* output
)
{
    //CUB reduce
    typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    T threadData[LOOP_CNT] = {0};
    T inData[LOOP_CNT] = {0};
    T aggreagtedData = 0;

    auto tid = threadIdx.x;
    auto offset = blockData[blockIdx.x];
    const size_t gtx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    auto stride = 1; //BLOCK_DIM_X * gridDim.x;

    for (int i = 0; i < LOOP_CNT; i++) {
        size_t idx = gtx * LOOP_CNT + i;
        if (idx < count)
        {   
            threadData[i] = keep[idx];
            inData[i] = input[idx];
        }
    }

    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
    __syncthreads();

    for (int i = 0; i < LOOP_CNT; i++) {
        size_t idx = gtx*LOOP_CNT + i;
        if (idx < count && keep[idx])
            output[threadData[i] + offset] = inData[i];
    }
}


template <typename T, int BLOCK_DIM_X, int LOOP_CNT = 1>
__global__ void
kernel_per_block_scatter2(
    T* input1,
    T* input2,
    bool* keep,
    T count,
    T* blockData, //prefix sum is done to it
    T* output1,
    T* output2
)
{
    //CUB reduce
    typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    T threadData[LOOP_CNT] = {0};
    T inData1[LOOP_CNT] = {0};
    T inData2[LOOP_CNT] = {0};
    T aggreagtedData = 0;

    auto tid = threadIdx.x;
    auto offset = blockData[blockIdx.x];
    const size_t gtx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    auto stride = 1; //BLOCK_DIM_X * gridDim.x;

    for (int i = 0; i < LOOP_CNT; i++) {
        size_t idx = gtx * LOOP_CNT + i;
        if (idx < count)
        {   
            threadData[i] = keep[idx];
            inData1[i] = input1[idx];
            inData2[i] = input2[idx];
        }
    }

    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
    __syncthreads();

    for (int i = 0; i < LOOP_CNT; i++) {
        size_t idx = gtx*LOOP_CNT + i;
        if (idx < count && keep[idx])
            output1[threadData[i] + offset] = inData1[i];
            output2[threadData[i] + offset] = inData2[i];

    }
}



namespace graph {
    template<typename T>
    class CubLarge {
    public:
        int dev_;
        cudaStream_t stream_;

        // events for measuring time
        cudaEvent_t kernelStart_;
        cudaEvent_t kernelStop_;

    public:
        /*! Device constructor
            Create a counter on device dev
        */
        CubLarge(int dev, cudaStream_t stream = 0) : dev_(dev), stream_(stream) {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            CUDA_RUNTIME(cudaGetLastError());

            CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
            CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
        }

        /*! copy ctor - create a new counter on the same device
        All fields are reset
         */
        CubLarge(const CubLarge& other) : CubLarge(other.dev_, other.stream_) {}

        ~CubLarge() {
            CUDA_RUNTIME(cudaEventDestroy(kernelStart_));
            CUDA_RUNTIME(cudaEventDestroy(kernelStop_));
        }

        CubLarge& operator=(CubLarge&& other) noexcept {

            /* We just swap other and this, which has the following benefits:
               Don't call delete on other (maybe faster)
               Opportunity for data to be reused since it was not deleted
               No exceptions thrown.
            */

            other.swap(*this);
            return *this;
        }

        void swap(CubLarge& other) noexcept {
            std::swap(other.dev_, dev_);
            std::swap(other.kernelStart_, kernelStart_);
            std::swap(other.kernelStop_, kernelStop_);
            std::swap(other.stream_, stream_);
        }

        T ExclusiveSum(T* input, T* output, const T count)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            const auto blockSize = 512;
            const auto gridSize = (count + blockSize - 1) / blockSize;
            graph::GPUArray<T> blockReduction("Block Reduction", AllocationTypeEnum::unified, gridSize, dev_);
            graph::GPUArray<T> totalCount("Block Reduction", AllocationTypeEnum::unified, 1, dev_);

            kernel_per_block_reduce<T, blockSize> << <gridSize, blockSize >> > (input, count, blockReduction.gdata(), totalCount.gdata());
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockReduction.gdata(), blockReduction.gdata(), gridSize);
            CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockReduction.gdata(), blockReduction.gdata(), gridSize);
            kernel_per_block_scatter<T, blockSize> << <gridSize, blockSize >> > (input, count, blockReduction.gdata(), output);
            sync();
            cudaGetLastError();
            CUDA_RUNTIME(cudaFree(d_temp_storage));
            blockReduction.freeGPU();

            return totalCount.gdata()[0] + blockReduction.gdata()[gridSize - 1];
        }

        T Select(T* input, T* output, bool* keep, const T count)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            const auto blockSize = 128;
            const auto loop = 10;
            const auto gridSize = (count - 1) / (blockSize * loop) + 1;
            graph::GPUArray<T> blockReduction("Block Reduction", AllocationTypeEnum::unified, gridSize, dev_);
            graph::GPUArray<T> totalCount("Block Reduction", AllocationTypeEnum::unified, 1, dev_);
            kernel_per_block_reduce<T, blockSize, loop> << <gridSize, blockSize, 0 >> > (keep, count, blockReduction.gdata(), totalCount.gdata());
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockReduction.gdata(), blockReduction.gdata(), gridSize);
            CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockReduction.gdata(), blockReduction.gdata(), gridSize);


            kernel_per_block_scatter<T, blockSize, loop> << <gridSize, blockSize, 0 >> > (input, keep, count, blockReduction.gdata(), output);
            sync();
            cudaGetLastError();
            CUDA_RUNTIME(cudaFree(d_temp_storage));


            return totalCount.gdata()[0] + blockReduction.gdata()[gridSize - 1];
        }


        T Select2(T* input1, T* input2, T* output1, T* output2, bool* keep, const T count)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            const auto blockSize = 512;
            const auto loop = 10;
            const auto gridSize = (count - 1) / (blockSize * loop) + 1;
            graph::GPUArray<T> blockReduction("Block Reduction", AllocationTypeEnum::unified, gridSize, dev_);
            graph::GPUArray<T> totalCount("Block Reduction", AllocationTypeEnum::unified, 1, dev_);
            kernel_per_block_reduce<T, blockSize, loop> << <gridSize, blockSize, 0 >> > (keep, count, blockReduction.gdata(), totalCount.gdata());
            cudaDeviceSynchronize();

            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockReduction.gdata(), blockReduction.gdata(), gridSize);
            CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, blockReduction.gdata(), blockReduction.gdata(), gridSize);
            cudaDeviceSynchronize();

            //kernel_per_block_scatter2<T, blockSize, loop> << <gridSize, blockSize, 0 >> > (input1, input2, keep, count, blockReduction.gdata(), output1, output2);

            kernel_per_block_scatter<T, blockSize, loop> << <gridSize, blockSize, 0 >> > (input1, keep, count, blockReduction.gdata(), output1);
            kernel_per_block_scatter<T, blockSize, loop> << <gridSize, blockSize, 0 >> > (input2, keep, count, blockReduction.gdata(), output2);
            cudaDeviceSynchronize();
            cudaGetLastError();
            CUDA_RUNTIME(cudaFree(d_temp_storage));


            return totalCount.gdata()[0] + blockReduction.gdata()[gridSize - 1];
        }


        void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
        int device() const { return dev_; }
        float kernel_time() {
            float ms;
            CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));
            CUDA_RUNTIME(cudaEventElapsedTime(&ms, kernelStart_, kernelStop_));
            return ms / 1e3;
        }
    };
}