#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "utils.cuh"

template<typename T>
__global__ void setelements(T* arr, uint64 count, T val)
{
	uint64 gtx = threadIdx.x + blockDim.x * blockIdx.x;
	for (uint64 i = gtx; i < count; i += blockDim.x * gridDim.x)
	{
		arr[i] = val;
	}
}

namespace graph
{
	template<class T>
	class GPUArray 
	{
	public:


		GPUArray()
		{
			N = 0;
			name = "Unknown";
		}

		void initialize(std::string s, AllocationTypeEnum at, size_t size, int devId)
		{
			N = size;
			name = s;
			_at = at;
			_deviceId = devId;
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaStreamCreate(&_stream));

			switch (at)
			{
			case cpuonly:
				cpu_data = (T*)malloc(size * sizeof(T));
				break;

			case gpu:
				cpu_data = (T*)malloc(size * sizeof(T));
				CUDA_RUNTIME(cudaMalloc(&gpu_data, size * sizeof(T)));
				break;
			case unified:
				CUDA_RUNTIME(cudaMallocManaged(&gpu_data, size * sizeof(T)));
				break;
			case zerocopy:
				break;
			default:
				break;
			}
		}

		void initialize(std::string s, AllocationTypeEnum at)
		{
			name = s;
			_at = at;
		}

		GPUArray(std::string s, AllocationTypeEnum at, size_t size, int devId, bool pinned)
		{
			N = size;
			name = s;
			_at = at;
			_deviceId = devId;
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaStreamCreate(&_stream));

			// if(pinned)
			// {
			// 	cudaMallocHost((void**)&cpu_data, size * sizeof(T));
			// }
			// else
			// {
			// 	cpu_data = (T*)malloc(size * sizeof(T));
			// }
			

			// cpu_data = (T*)malloc(size * sizeof(T));
			// CUDA_RUNTIME(cudaMalloc(&gpu_data, size * sizeof(T)));
			
		}



		GPUArray(std::string s, AllocationTypeEnum at, size_t size, int devId)
		{
			initialize(s, at, size, devId);
		}

		GPUArray(std::string s, AllocationTypeEnum at) {
			initialize(s, at);
		}

		void freeGPU()
		{
			if (N != 0)
			{
				freed = true;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				if(_at == AllocationTypeEnum::gpu || _at == AllocationTypeEnum::unified)
					CUDA_RUNTIME(cudaFree(gpu_data));
				CUDA_RUNTIME(cudaStreamDestroy(_stream));
				N = 0;
			}
		}
		void freeCPU()
		{
			delete cpu_data;
		}

		void allocate_cpu(size_t size)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = size;
				cpu_data = (T*)malloc(size * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::unified)
			{
				N = size;
				CUDA_RUNTIME(cudaMallocManaged(&gpu_data, size * sizeof(T)));
			}
			else
			{
				Log(LogPriorityEnum::critical, "At allocate_cpu: Only CPU allocation\n");
			}
		}
		void setAlloc(AllocationTypeEnum at)
		{
			_at = at;
		}

		void switch_to_gpu(int devId=0, size_t size=0)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = (size==0) ? N : size;
				_at = AllocationTypeEnum::gpu;
				_deviceId = devId;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMalloc(&gpu_data, N * sizeof(T)));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else if (_at == AllocationTypeEnum::gpu) //memory is already allocated
			{
				if (size > N)
				{
					Log(LogPriorityEnum::critical, "Memory needed is more than allocated-Nothing is done\n");
					return;
				}

				N = (size == 0) ? N : size;
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}

		void switch_to_unified(int devId, size_t size = 0)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = (size == 0) ? N : size;
				_at = AllocationTypeEnum::unified;
				_deviceId = devId;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMallocManaged(&gpu_data, N * sizeof(T)));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else if (_at == AllocationTypeEnum::gpu) //memory is already allocated
			{
				if (size > N)
				{
					Log(LogPriorityEnum::critical, "Memory needed is more than allocated-Nothing is done\n");
					return;
				}

				N = (size == 0) ? N : size;
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}

		}

		void copyCPUtoGPU(int devId, size_t size = 0)
		{
			if (_at != cpuonly)
			{
				N = (size == 0) ? N : size;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}


		void setAll(T val, bool sync)
		{
			if (N < 1)
				return;

			if (_at == AllocationTypeEnum::cpuonly)
			{
				memset(cpu_data, val, N * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::gpu)
			{
				
				memset(cpu_data, val, N * sizeof(T));
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <(N+512-1)/512, 512, 0, _stream >> > (gpu_data, N, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
			else if (_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <(N + 512 - 1)/512, 512, 0, _stream >> > (gpu_data, N, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}

		void setSingle(uint64 index, T val, bool sync)
		{
			if (N < 1)
				return;

			if (_at == AllocationTypeEnum::cpuonly)
			{
				memset(&cpu_data[index], val, 1 * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::gpu)
			{
				memset(&cpu_data[index], val, 1 * sizeof(T));
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <1, 1, 0, _stream >> > (&gpu_data[index], 1, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
			else if(_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <1, 1, 0, _stream >> > (&gpu_data[index], 1, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}


		T getSingle(uint64 index)
		{
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			T val = 0;
			if (_at == AllocationTypeEnum::unified)
				return (gpu_data[index]);
			
			CUDA_RUNTIME(cudaMemcpy(&val, &(gpu_data[index]), sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			return val;
		}

		T* copytocpu(uint64 startIndex, size_t count=0, bool newAlloc=false)
		{
			size_t c = count == 0 ? N : count;


			if (_at == AllocationTypeEnum::unified)
				return &(gpu_data[startIndex]);
			
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			if (newAlloc)
			{
				T *temp = (T*)malloc(c * sizeof(T));
				CUDA_RUNTIME(cudaMemcpy(temp, &(gpu_data[startIndex]), c *sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
				return temp;
			}

			CUDA_RUNTIME(cudaMemcpy(cpu_data, &(gpu_data[startIndex]), c *sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			return cpu_data;
		}


		void advicePrefetch(bool sync)
		{
			if (_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));

		#ifndef __VS__
				CUDA_RUNTIME(cudaMemPrefetchAsync (gpu_data, N*sizeof(T), _deviceId, _stream));
		#endif // !__VS__

				
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}

		T*& gdata()
		{
			return gpu_data;
		}

		T*& cdata()
		{
			if (_at == unified)
				return gpu_data;

			return cpu_data;
		}

		uint64 N;
		std::string name;
	private:
		T* cpu_data;
		T* gpu_data;
		AllocationTypeEnum _at;
		cudaStream_t _stream;
		int _deviceId;
		bool freed = false;

	};
}