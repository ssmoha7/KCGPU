#pragma once
#include "cub_wrappers.cuh"

namespace graph
{
    template<typename T>
    struct COOCSRGraph {
        T numNodes;
        T numEdges;
        T capacity;

        GPUArray<T>* rowPtr;
        GPUArray<T>* rowInd;
        GPUArray<T>* colInd;
    };

    template<typename T>
    struct COOCSRGraph_d 
    {
        T numNodes;
        T numEdges;
        T capacity;

        T* rowPtr;
        T* rowInd;
        T* colInd;
        T* splitPtr;     
    };

    template<typename T>
   void to_csrcoo_device(COOCSRGraph<T> g, COOCSRGraph_d<T>*& graph, int dev, AllocationTypeEnum at=unified) {
        graph = (COOCSRGraph_d<T>*)malloc(sizeof(COOCSRGraph_d<T>));
        graph->numNodes = g.numNodes;
        graph->numEdges = g.numEdges;
        graph->capacity = g.capacity;

        g.rowPtr->switch_to_gpu(dev, g.numNodes + 1);
        graph->rowPtr = g.rowPtr->gdata();

        if (at == AllocationTypeEnum::unified)
        {
            g.rowInd->switch_to_unified(dev, g.numEdges);
            g.colInd->switch_to_unified(dev, g.numEdges);
            graph->rowInd = g.rowInd->gdata();
            graph->colInd = g.colInd->gdata();
        }
        else if (at == AllocationTypeEnum::gpu)
        {
           
            g.rowInd->switch_to_gpu(dev, g.numEdges);
            g.colInd->switch_to_gpu(dev, g.numEdges);
            graph->rowInd = g.rowInd->gdata();
            graph->colInd = g.colInd->gdata();
        }
        else if(at == AllocationTypeEnum::zerocopy)
        {
            graph->rowInd = g.rowInd->cdata();
            graph->colInd = g.colInd->cdata();
        }

       
    }

   template<typename T>
   void free_csrcoo_device(COOCSRGraph<T>& g)
   {
       g.rowPtr->freeGPU();
       g.rowInd->freeGPU();
       g.colInd->freeGPU();
       g.capacity = 0; 
       g.numEdges = 0;
       g.numNodes = 0;
   }



    template<typename T>
    struct EidGraph_d
    {
        T numNodes;
        T numEdges;
        T capacity;

        T* rowPtr_csr;
        T* colInd_csr;

        T* eid;
        T* colInd;
        T* rowInd;
    };

    template<typename T>
    struct TiledCOOCSRGraph {
        T numNodes;
        T numEdges;
        T tilesPerDim;
        T tileSize;
        T capacity;
        GPUArray<T>* tileRowPtr;
        GPUArray<T>* rowInd;
        GPUArray<T>* colInd;
    };


    template<typename T>
    struct TiledCOOCSRGraph_d {
        unsigned int numNodes;
        unsigned int numEdges;
        unsigned int tilesPerDim;
        unsigned int tileSize;
        unsigned int capacity;
        T* tileRowPtr;
        T* rowInd;
        T* colInd;
    };

    template<typename T>
    TiledCOOCSRGraph<T> createEmptyTiledCOOCSROnDevice(unsigned int numNodes, unsigned int tilesPerDim, unsigned int capacity) {

        TiledCOOCSRGraph<T> g_shd;
        g_shd.numNodes = numNodes;
        g_shd.numEdges = 0;
        g_shd.capacity = capacity;
        g_shd.tilesPerDim = tilesPerDim;
        g_shd.tileSize = (numNodes + tilesPerDim - 1) / tilesPerDim;
        unsigned int numTileSrcPtrs = tilesPerDim * tilesPerDim * g_shd.tileSize + 1;

        g_shd.tileSrcPtr = new GPUArray<T>("Tiled Row Pointer", unified, numTileSrcPtrs, 0);
        g_shd.rowInd = new GPUArray<T>("Row Index", unified, capacity, 0);
        g_shd.colInd = new GPUArray<T>("Col Index", unified, capacity, 0);

        return g_shd;
    }

    template<typename T=uint>
    __global__ void histogram_tiled_kernel(T numEdges, T* tileSrcPtr, T tileSize, T tilesPerDim, T* rowInd, T* colInd) 
    {
        unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e < numEdges) {
            unsigned int src = rowInd[e];
            unsigned int dst = colInd[e];
            unsigned int tileSrc = (src / tileSize * tilesPerDim + dst / tileSize) * tileSize + src % tileSize;
            atomicAdd(&tileSrcPtr[tileSrc + 1], 1);
        }
    }

    template<typename T = uint>
    __global__ void binning_kernel(
        T* cooRowInd, T* cooColInd,
        T numEdges, T* tileSrcPtr, T tileSize, T tilesPerDim, T* rowInd, T* colInd) {
        unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e < numEdges) {
            unsigned int src = cooRowInd[e];
            unsigned int dst = cooColInd[e];
            unsigned int tileSrc = (src / tileSize * tilesPerDim + dst / tileSize) * tileSize + src % tileSize;
            unsigned int j = atomicAdd(&tileSrcPtr[tileSrc + 1], 1);
            rowInd[j] = src;
            colInd[j] = dst;
        }
    }


    template<typename T>
    void coo2tiledcoocsrOnDevice(COOCSRGraph<T> g, int tilesPerDim,  TiledCOOCSRGraph<T>*& g_shd, AllocationTypeEnum at = unified)
    {


        if (at == AllocationTypeEnum::unified)
        {
            g.rowPtr->switch_to_unified(0, g.numNodes + 1);
            g.rowInd->switch_to_unified(0, g.numEdges);
            g.colInd->switch_to_unified(0, g.numEdges);
        }
        else if (at == AllocationTypeEnum::gpu)
        {
            g.rowPtr->switch_to_gpu(0, g.numNodes + 1);
            g.rowInd->switch_to_gpu(0, g.numEdges);
            g.colInd->switch_to_gpu(0, g.numEdges);
        }

        unsigned int numNodes = g.numNodes;
        g_shd = new TiledCOOCSRGraph<T>;
        g_shd->numNodes = g.numNodes;
        g_shd->numEdges = g.numEdges;
        g_shd->capacity = g.capacity;
        g_shd->tilesPerDim = tilesPerDim;
        g_shd->tileSize = (numNodes + tilesPerDim - 1) / tilesPerDim;
        unsigned int numTileSrcPtrs = tilesPerDim * tilesPerDim * g_shd->tileSize + 1;

        g_shd->tileRowPtr = new GPUArray<T>("Tiled Row Pointer", unified, numTileSrcPtrs, 0);
        g_shd->tileRowPtr->setAll(0, true);
        g_shd->rowInd = new GPUArray<T>("Tiled Row Index", unified, g.numEdges, 0);
        g_shd->colInd = new GPUArray<T>("Tiled Col Index", unified, g.numEdges, 0);

        histogram_tiled_kernel<T> << < (g.numEdges + 512 - 1) / 512, 512 >> > (g.numEdges, g_shd->tileRowPtr->gdata(), g_shd->tileSize, tilesPerDim, g.rowInd->gdata(), g.colInd->gdata());
        cudaDeviceSynchronize();

        T lastInput = CUBScanExclusive(g_shd->tileRowPtr->gdata() + 1, g_shd->tileRowPtr->gdata() + 1, numTileSrcPtrs );
        
        for (int i = 0; i < numTileSrcPtrs; i++)
            if (g_shd->tileRowPtr->gdata()[i] > g_shd->tileRowPtr->gdata()[i + 1])
                printf("Shit\n");
        binning_kernel<T> << < (g.numEdges + 512 - 1) / 512, 512 >> > (g.rowInd->gdata(), g.colInd->gdata(), g.numEdges, g_shd->tileRowPtr->gdata(), g_shd->tileSize, tilesPerDim, g_shd->rowInd->gdata(), g_shd->colInd->gdata());
        cudaDeviceSynchronize();


        for (T tileSrc = 0; tileSrc < numTileSrcPtrs; ++tileSrc) {
            T start = g_shd->tileRowPtr->gdata()[tileSrc];
            T end = g_shd->tileRowPtr->gdata()[tileSrc + 1]>0? g_shd->tileRowPtr->gdata()[tileSrc + 1] - 1: 0;
            quicksort(g_shd->colInd->gdata(), start, end); // NOTE: No need to sort srcIdx because they are all the same
        }
    }

};