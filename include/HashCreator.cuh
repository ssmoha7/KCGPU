#pragma once
#include "utils.cuh"

namespace graph 
{
    template<typename T>
    class HashCreator
    {

    protected:
        int dev_;
        cudaStream_t stream_;
        uint64_t numEdges;
        uint64_t numNodes;

        // events for measuring time
        cudaEvent_t kernelStart_;
        cudaEvent_t kernelStop_;

    public:

        int numEdges;
        int numRows;

        HashCreator(int dev, uint64_t ne, uint64_t nn, cudaStream_t stream = 0) : dev_(dev), numEdges(ne), numNodes(nn), stream_(stream)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            CUDA_RUNTIME(cudaGetLastError());

            CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
            CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
        }

        void create(GPUArray<T>& rowHashTablePointer, GPUArray<T>& hashPointer, GPUArray<T>& hashData,
            GPUArray<T> rowPtr, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0)
        {

            //Create rowHashTable, and create 
          
        }

       


    };
}