#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"

#define INT_INVALID  (INT32_MAX)

#define LEVEL_SKIP_SIZE (128)
#define KCL_NODE_LEVEL_SKIP_SIZE (1024)
#define KCL_EDGE_LEVEL_SKIP_SIZE (1024)

#define INBUCKET_BOOL
#ifndef INBUCKET_BOOL
using InBucketWinType = int;
#define InBucketTrue (1)
#define InBucketFalse (0)
#else
using InBucketWinType = bool;
#define InBucketTrue (true)
#define InBucketFalse (false)
#endif

template<typename DataType, typename CntType>
__global__
void init_asc(DataType* data, CntType count) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) data[gtid] = (DataType)gtid;
}


static __inline__ __device__ bool atomicCASBool(bool* address, bool compare, bool val) {
    unsigned long long addr = (unsigned long long) address;
    unsigned pos = addr & 3;  // byte position within the int
    int* int_addr = (int*)(addr - pos);  // int-aligned address
    int old = *int_addr, assumed, ival;

    do {
        assumed = old;
        if (val)
            ival = old | (1 << (8 * pos));
        else
            ival = old & (~((0xFFU) << (8 * pos)));
        old = atomicCAS(int_addr, assumed, ival);
    } while (assumed != old);

    return (bool)(old & ((0xFFU) << (8 * pos)));
}

template<typename NeedleType, typename HaystackType>
__device__
int binary_search(
    NeedleType needle, HaystackType* haystacks,
    int hay_begin, int hay_end) {
    while (hay_begin <= hay_end) {
        int middle = hay_begin + (hay_end - hay_begin) / 2;
        if (needle > haystacks[middle])
            hay_begin = middle + 1;
        else if (needle < haystacks[middle])
            hay_end = middle - 1;
        else
            return middle;
    }
    return INT_INVALID;  //not found
}

template<typename T>
__global__
void update_processed(T* curr, T curr_cnt, bool* inCurr, bool* processed) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < curr_cnt) {
        auto edge_off = curr[gtid];
        processed[edge_off] = true;
        inCurr[edge_off] = false;
    }
}


__global__
void output_edge_support(
    int* output, int* curr, uint32_t curr_cnt,
    uint* edge_off_origin, uint start_pos) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < curr_cnt) {
        output[gtid + start_pos] = edge_off_origin[curr[gtid]];
    }
}

template<typename T>
__global__
void warp_detect_deleted_edges(
    T* old_offsets, T old_offset_cnt,
    T* eid, bool* old_processed,
    T* histogram, bool* focus) 
{

    __shared__ uint32_t cnts[WARPS_PER_BLOCK];

    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gtnum = blockDim.x * gridDim.x;
    auto gwid = gtid >> WARP_BITS;
    auto gwnum = gtnum >> WARP_BITS;
    auto lane = threadIdx.x & WARP_MASK;
    auto lwid = threadIdx.x >> WARP_BITS;

    for (auto u = gwid; u < old_offset_cnt; u += gwnum) {
        if (0 == lane) cnts[lwid] = 0;
        __syncwarp();

        auto start = old_offsets[u];
        auto end = old_offsets[u + 1];
        for (auto v_idx = start + lane; v_idx < end; v_idx += WARP_SIZE) {
            auto target_edge_idx = eid[v_idx];
            focus[v_idx] = !old_processed[target_edge_idx];
            if (focus[v_idx])
                atomicAdd(&cnts[lwid], 1);
        }
        __syncwarp();

        if (0 == lane) histogram[u] = cnts[lwid];
    }
}


template<typename T, typename PeelT>
__global__
void filter_window(PeelT* edge_sup, T count, InBucketWinType* in_bucket, T low, T high) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto v = edge_sup[gtid];
        in_bucket[gtid] = (v >= low && v < high);
    }
}

template<typename T, typename PeelT>
__global__
void filter_pointer_window(PeelT* edge_sup, T count, InBucketWinType* in_bucket, T low, T high) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto v = edge_sup[gtid+1] - edge_sup[gtid];
        in_bucket[gtid] = (v >= low && v < high);
    }
}

template<typename T, typename PeelT>
__global__
void filter_with_random_append(T* bucket_buf, T count, PeelT* EdgeSupport, bool* in_curr, T* curr, T* curr_cnt,
    T ref) 
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto edge_off = bucket_buf[gtid];
        if (EdgeSupport[edge_off] == ref) {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}

template<typename T, typename PeelT>
__global__
void filter_with_random_append(T* bucket_buf, T count, PeelT* EdgeSupport, bool* in_curr, T* curr, T* curr_cnt,
    T ref, T span)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto edge_off = bucket_buf[gtid];
        if (EdgeSupport[edge_off] >= ref && EdgeSupport[edge_off] < ref + span) {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}

template<typename T, typename PeelT>
__global__
void filter_with_random_append_pointer(T* bucket_buf, T count, PeelT* EdgeSupport, bool* in_curr, T* curr, T* curr_cnt,
    int ref, int span)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto edge_off = bucket_buf[gtid];
        auto v = EdgeSupport[edge_off + 1] - EdgeSupport[edge_off];

        if (v >= ref && v < ref + span) {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}



template<typename T>
__device__ void add_to_queue_1(graph::GraphQueue_d<T, bool>& q, T element)
{
    auto insert_idx = atomicAdd(q.count, 1);
    q.queue[insert_idx] = element;
    q.mark[element] = true;
}

template<typename T>
__device__ void add_to_queue_1_no_dup(graph::GraphQueue_d<T, bool>& q, T element)
{
    auto old_token = atomicCASBool(q.mark + element, InBucketFalse, InBucketTrue);
    if (!old_token) {
        auto insert_idx = atomicAdd(q.count, 1);
        q.queue[insert_idx] = element;
    }
}







