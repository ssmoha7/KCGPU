#pragma once
#include "utils.cuh"
#include "utils_cuda.cuh"

using namespace std;





template<typename InputType, typename OutputType>
OutputType CUBScanExclusive(
    InputType* input,
    OutputType* output,
    const int count,
    int devId,
    cudaStream_t 	stream = 0,
    AllocationTypeEnum at = unified)
{
    CUDA_RUNTIME(cudaSetDevice(devId));

    float singleKernelTime, elaspedTime = 0;
    cudaEvent_t start, end;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    /*record the last input item in case it is an in-place scan*/
    auto last_input = getVal<InputType>(input, count - 1, at);// input[count - 1];

   /* CUDA_RUNTIME(cudaEventCreate(&start));
    CUDA_RUNTIME(cudaEventCreate(&end));*/

    //CUDA_RUNTIME(cudaEventRecord(start));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count);
   /* CUDA_RUNTIME(cudaEventRecord(end));
    CUDA_RUNTIME(cudaEventSynchronize(start));
    CUDA_RUNTIME(cudaEventSynchronize(end));
    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));
    elaspedTime += singleKernelTime;
    CHECK_KERNEL("CUB 1st scan");*/

    CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    /*Run exclusive prefix sum*/
    //CUDA_RUNTIME(cudaEventRecord(start));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count);
//    CUDA_RUNTIME(cudaEventRecord(end));
//    CUDA_RUNTIME(cudaEventSynchronize(start));
//    CUDA_RUNTIME(cudaEventSynchronize(end));
//    CUDA_RUNTIME(cudaDeviceSynchronize());
//
//    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));
//    elaspedTime += singleKernelTime;
//
//    CHECK_KERNEL("CUB 2nd scan");
//
//#ifdef __VERBOSE__
//   // Log(LogPriorityEnum::info ,"Kernel: %s, count: %d, time: %.2f ms.", "CUB scan", count, elaspedTime);
//#endif // __VERBOSE__

    CUDA_RUNTIME(cudaFree(d_temp_storage));

    return getVal<OutputType>(output, count - 1, at) + (OutputType)last_input;
}

template<typename DataType, typename SumType, typename CntType>
SumType CUBSum(
    DataType* input,
    CntType count)
{
    cudaEvent_t start, end;
    float elaspedTime = 0, singleKernelTime = 0;

    CUDA_RUNTIME(cudaEventCreate(&start));
    CUDA_RUNTIME(cudaEventCreate(&end));

    SumType* sum_value = nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&sum_value, sizeof(SumType)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_RUNTIME(cudaEventRecord(start));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, sum_value, count);
    CUDA_RUNTIME(cudaEventRecord(end));
    CUDA_RUNTIME(cudaEventSynchronize(start));
    CUDA_RUNTIME(cudaEventSynchronize(end));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));
    elaspedTime += singleKernelTime;

    CHECK_KERNEL("CUB 1st sum");

    CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CUDA_RUNTIME(cudaEventRecord(start));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, sum_value, count);
    CUDA_RUNTIME(cudaEventRecord(end));
    CUDA_RUNTIME(cudaEventSynchronize(start));
    CUDA_RUNTIME(cudaEventSynchronize(end));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));
    elaspedTime += singleKernelTime;

    CHECK_KERNEL("CUB 2nd sum");


#ifdef __VERBOSE__
    //Log(LogPriorityEnum::info, "Kernel: %s, count: %d, time: %.2f ms.", "CUB Sum", count, elaspedTime);
#endif // __VERBOSE__

    SumType res = *sum_value;
    CUDA_RUNTIME(cudaFree(d_temp_storage));
    CUDA_RUNTIME(cudaFree(sum_value));

    return res;
}

template<typename InputType, typename OutputType, typename CountType, typename FlagIterator>
uint32_t CUBSelect(
    InputType input, OutputType output,
    FlagIterator flags,
    const CountType countInput,
    int devId)
{

    CUDA_RUNTIME(cudaSetDevice(devId));

    cudaEvent_t start, end;
    float elaspedTime = 0, singleKernelTime = 0;

    uint32_t* countOutput = nullptr;
    CUDA_RUNTIME(cudaMallocManaged(&countOutput, sizeof(uint32_t)));

  /*  CUDA_RUNTIME(cudaEventCreate(&start));
    CUDA_RUNTIME(cudaEventCreate(&end));*/

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    //CUDA_RUNTIME(cudaEventRecord(start));
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, flags, output, countOutput, countInput);
    /*CUDA_RUNTIME(cudaEventRecord(end));
    CUDA_RUNTIME(cudaEventSynchronize(start));
    CUDA_RUNTIME(cudaEventSynchronize(end));*/
    CUDA_RUNTIME(cudaDeviceSynchronize());

  /*  CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));
    elaspedTime += singleKernelTime;

    CHECK_KERNEL("CUB 1st select");*/

    CUDA_RUNTIME(cudaMallocManaged(&d_temp_storage, temp_storage_bytes));

    //CUDA_RUNTIME(cudaEventRecord(start));
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, flags, output, countOutput, countInput);
   /* CUDA_RUNTIME(cudaEventRecord(end));
    CUDA_RUNTIME(cudaEventSynchronize(start));
    CUDA_RUNTIME(cudaEventSynchronize(end));*/
    CUDA_RUNTIME(cudaDeviceSynchronize());

    //CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));
    //elaspedTime += singleKernelTime;

    //CHECK_KERNEL("CUB 2nd select"); 
    
    uint32_t res = *countOutput;

#ifdef __VERBOSE__
    //Log(LogPriorityEnum::info, "Kernel: %s, count: %d, time: %.2f ms.", "CUB select", res, elaspedTime);
#endif // __VERBOSE__

    CUDA_RUNTIME(cudaFree(d_temp_storage));
    CUDA_RUNTIME(cudaFree(countOutput));

    return res;
}
