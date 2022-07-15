
#include <cuda_runtime.h>
#include <iostream>
#include<string>
#include <fstream>
#include <map>

#include "omp.h"
#include<vector>

#include "../include/Logger.cuh"
#include "../include/FIleReader.cuh"
#include "../include/CGArray.cuh"
#include "../include/TriCountPrim.cuh"

#include "../include/CSRCOO.cuh"






#include "../include/main_support.cuh"

#include "../kcore/kcore.cuh"
#include "../kclique/kclique.cuh"
#include "../kclique/kclique_local.cuh"


#include "../include/Config.h"
#include "../include/ScanLarge.cuh"

using namespace std;

int main(int argc, char** argv)
{

    //CUDA_RUNTIME(cudaDeviceReset());
    Config config = parseArgs(argc, argv);

    printf("\033[0m");
    printf("Welcome ---------------------\n");
    printConfig(config);

    graph::MtB_Writer mwriter;
    auto fileSrc = config.srcGraph;
    auto fileDst = config.dstGraph;
    if (config.mt == CONV_MTX_BEL) {
        mwriter.write_market_bel<uint, int>(fileSrc, fileDst, false);
        return;
    }

    if (config.mt == CONV_TSV_BEL) {
        mwriter.write_tsv_bel<uint64, uint64>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_TSV_MTX) {
        mwriter.write_tsv_market<uint, int>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_BEL_MTX) {
        mwriter.write_bel_market<uint, int>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_TXT_BEL) {
        mwriter.write_txt_bel<uint, uint>(fileSrc, fileDst, true, 2, 0);
        return;
    }


    Timer read_graph_timer;

    const char* matr = config.srcGraph;
    graph::EdgeListFile f(matr);
    std::vector<EdgeTy<uint>> edges;
    std::vector<EdgeTy<uint>> fileEdges;
    auto lowerTriangular = [](const Edge& e) { return e.first > e.second; };
    auto upperTriangular = [](const Edge& e) { return e.first < e.second; };
    auto full = [](const Edge& e) { return false; };


    while (f.get_edges(fileEdges, 100))
    {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
    }

    if (config.sortEdges)
    {
        f.sort_edges(edges);
    }

    graph::CSRCOO<uint> csrcoo;
    if (config.orient == Upper)
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, lowerTriangular);
    else if (config.orient == Lower)
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, upperTriangular);
    else
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, full);

    uint n = csrcoo.num_rows();
    uint m = csrcoo.nnz();


    graph::COOCSRGraph<uint> g;
    g.capacity = m;
    g.numEdges = m;
    g.numNodes = n;

    //No Allocation
    g.rowPtr = new graph::GPUArray<uint>("Row pointer", AllocationTypeEnum::noalloc, n+1, config.deviceId, true );
    g.rowInd = new graph::GPUArray<uint>("Src Index", AllocationTypeEnum::noalloc,  m, config.deviceId, true );
    g.colInd = new graph::GPUArray<uint>("Dst Index", AllocationTypeEnum::noalloc,  m, config.deviceId, true);

    uint *rp, *ri, *ci;
    cudaMallocHost((void**)&rp, (n+1)*sizeof(uint));
    cudaMallocHost((void**)&ri, (m)*sizeof(uint));
    cudaMallocHost((void**)&ci, (m)*sizeof(uint));

    CUDA_RUNTIME(cudaMemcpy(rp, csrcoo.row_ptr(), (n+1)*sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));
    CUDA_RUNTIME(cudaMemcpy(ri, csrcoo.row_ind(), (m)*sizeof(uint) , cudaMemcpyKind::cudaMemcpyHostToHost));
    CUDA_RUNTIME(cudaMemcpy(ci, csrcoo.col_ind(), (m)*sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));

    g.rowPtr->cdata() = rp; g.rowPtr->setAlloc(cpuonly);
    g.rowInd->cdata() = ri; g.rowInd->setAlloc(cpuonly);
    g.colInd->cdata() = ci; g.colInd->setAlloc(cpuonly);

    Log(info, "Read graph time: %f s", read_graph_timer.elapsed());

    ///Now we need to orient the graph
    Timer total_timer;

    graph::COOCSRGraph_d<uint>* gd = (graph::COOCSRGraph_d<uint>*)malloc(sizeof(graph::COOCSRGraph_d<uint>));
    g.rowPtr->switch_to_gpu(config.deviceId);

    gd->numNodes = g.numNodes;
    gd->numEdges = g.numEdges;
    gd->capacity = g.capacity;
    gd->rowPtr = g.rowPtr->gdata();

    //Rules
    if(config.mt == GRAPH_COUNT || config.mt == GRAPH_MATCH)
        config.orient = None;

    if((!config.isSmall || g.numEdges > 500000000) && config.mt != GRAPH_COUNT && config.mt != GRAPH_MATCH)
    {
        gd->rowInd = g.rowInd->cdata();
        gd->colInd = g.colInd->cdata();

    }
    else
    {
        g.rowInd->switch_to_gpu(config.deviceId);
        g.colInd->switch_to_gpu(config.deviceId);
        gd->rowInd = g.rowInd->gdata();
        gd->colInd = g.colInd->gdata();
    }
    double total = total_timer.elapsed();
    Log(info, "Transfer Time: %f s", total);

    Timer t;
    graph::SingleGPU_Kcore<uint, PeelType> mohacore(config.deviceId);
    if (config.orient == Degree || config.orient == Degeneracy)
    {
        if (config.orient == Degeneracy)
            mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);
        else if (config.orient == Degree)
            mohacore.getNodeDegree(*gd);

        graph::GPUArray<uint> rowInd_half("Half Row Index", config.allocation, m / 2, config.deviceId),
            colInd_half("Half Col Index", config.allocation, m / 2, config.deviceId),
            new_rowPtr("New Row Pointer", config.allocation, n + 1, config.deviceId),
            asc("ASC temp", AllocationTypeEnum::unified, m, config.deviceId);
        graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, config.deviceId);

        if (config.orient == Degree)
        {
            execKernel((init<uint, PeelType>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, asc.gdata(), keep.gdata(), mohacore.nodeDegree.gdata());
        }
        else if (config.orient == Degeneracy)
        {
            execKernel((init<uint, PeelType>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, asc.gdata(), keep.gdata(), mohacore.nodeDegree.gdata(), mohacore.nodePriority.gdata());
        }

        graph::CubLarge<uint> s(config.deviceId);
        uint newNumEdges;
        if (m < INT_MAX)
        {
            CUBSelect(gd->rowInd, rowInd_half.gdata(), keep.gdata(), m, config.deviceId);
            newNumEdges = CUBSelect(gd->colInd, colInd_half.gdata(), keep.gdata(), m, config.deviceId);
        }
        else
        {
            newNumEdges = s.Select2(gd->rowInd,  gd->colInd,  rowInd_half.gdata(), colInd_half.gdata(), keep.gdata(), m);
        }

        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, config.deviceId, false, gd->rowPtr, n, keep.gdata(), new_rowPtr.gdata());
        uint total = CUBScanExclusive<uint, uint>(new_rowPtr.gdata(), new_rowPtr.gdata(), n, config.deviceId, 0, config.allocation);
        new_rowPtr.setSingle(n, total, false);
        //assert(total == new_edge_num * 2);
        cudaDeviceSynchronize();
        asc.freeGPU();
        keep.freeGPU();
        free_csrcoo_device(g);

        m = m / 2;

        g.capacity = m;
        g.numEdges = m;
        g.numNodes = n;

        g.rowPtr = &new_rowPtr;
        g.rowInd = &rowInd_half;
        g.colInd = &colInd_half;

        // cudaFreeHost(rp);
        // cudaFreeHost(ri);
        // cudaFreeHost(ci);

        gd->numNodes = g.numNodes;
        gd->numEdges = g.numEdges;
        gd->capacity = g.capacity;
        gd->rowPtr = new_rowPtr.gdata();
        gd->rowInd = g.rowInd->gdata();
        gd->colInd = g.colInd->gdata();
    }

    double time_init = t.elapsed();
    if (config.orient == Degree || config.orient == Degeneracy)
    {
        Log(info, "HH Preprocess time: %f s", time_init);
    }



    if (config.printStats) {

        g.rowPtr->copytocpu(0);
        g.colInd->copytocpu(0);

        MatrixStats(m, n, n, g.rowPtr->cdata(), g.colInd->cdata());
        PrintMtarixStruct(m, n, n, g.rowPtr->cdata(), g.colInd->cdata());
    }

    if (config.mt == KCORE)
    {
        graph::COOCSRGraph_d<uint>* gd;
        to_csrcoo_device(g, gd, config.deviceId, config.allocation); //got to device !!
        cudaDeviceSynchronize();

        graph::SingleGPU_Kcore<uint, PeelType> mohacore(config.deviceId);
        Timer t;
        mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);
        mohacore.sync();
        double time = t.elapsed();
        Log(info, "count time %f s", time);
        Log(info, "MOHA %d kcore (%f teps)", mohacore.count(), m / time);
    }

    if (config.mt == KCLIQUE)
    {
        if (config.orient == None)
            Log(warn, "Redundunt K-cliques, Please orient the graph\n");



        // if(config.processElement == BlockWarp)
        // {
        // 	graph::SingleGPU_Kclique_NoOutQueue<uint> mohaclique(config.deviceId, *gd);
        // 	for (int i = 0; i < 3; i++)
        // 	{
        // 		Timer t;
        // 		if (config.processBy == ByNode)
        // 			mohaclique.findKclqueIncremental_node_async(config.k, *gd, config.processElement);
        // 		else if (config.processBy == ByEdge)
        // 			mohaclique.findKclqueIncremental_edge_async(config.k, *gd, config.processElement);
        // 		mohaclique.sync();
        // 		double time = t.elapsed();
        // 		Log(info, "count time %f s", time);
        // 		Log(info, "MOHA %d k-clique (%f teps)", mohaclique.count(), m / time);
        // 	}
        // 	mohaclique.free();
        // }
        // else
        {

            //read the nCr values, to be saved in (Constant or global or )

            graph::SingleGPU_Kclique<uint> mohaclique(config.deviceId, *gd);


            KcliqueConfig kcc = config.kcConfig;

            // int k = 4;
            // while ( k < 11)
            {
                //printf("------------------ K=%d ----------------------\n", k);
                for (int i = 0; i < 1; i++)
                {
                    Timer t;
                    if (config.processBy == ByNode)
                    {
                        if(kcc.Algo == GraphOrient)
                        {

                            if(kcc.BinaryEncode)
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_binary_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_binary_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_binary_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_binary_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_binary_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_binary_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }


                            }
                            else
                            {


                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_node_async(config.k, *gd, config.processElement);
                            }

                        }
                        else // Pivoting
                        {
                            if(kcc.BinaryEncode)
                            {


                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_node_pivot_async(config.k, *gd, config.processElement);
                            }
                            else
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_node_nobin_pivot_async(config.k, *gd, config.processElement);
                            }

                        }
                    }
                    else if (config.processBy == ByEdge)
                    {
                        if(kcc.Algo == GraphOrient)
                        {

                            if(kcc.BinaryEncode)
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_binary_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_binary_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_binary_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_binary_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_binary_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_binary_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_edge_binary_async(config.k, *gd, config.processElement);
                            }
                            else
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_edge_async(config.k, *gd, config.processElement);
                            }

                        }
                        else //Pivoting
                        {
                            if(kcc.BinaryEncode)
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_edge_pivot_async(config.k, *gd, config.processElement);
                            }
                            else
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");

                                }

                                //mohaclique.findKclqueIncremental_edge_nobin_pivot_async(config.k, *gd, config.processElement);
                            }

                        }
                    }
                    mohaclique.sync();
                    double time = t.elapsed();
                    Log(info, "count time %f s", time);
                    Log(info, "MOHA %d k-clique (%f teps)", mohaclique.count(), m / time);
                }

                //k++;
            }


        }
    }

    if (config.mt == KCLIQUE_LOCAL)
    {
        if (config.orient == None)
            Log(warn, "Redundunt K-cliques, Please orient the graph\n");

        // read the nCr values, to be saved in (Constant or global or )
        graph::SingleGPU_Kclique_Local<uint> localclique(config.deviceId, *gd);

        KcliqueConfig kcc = config.kcConfig;

        Timer t;
        if (config.processBy == ByNode)
        {
            if(kcc.Algo == GraphOrient)
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_node_binary_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_node_binary_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_node_binary_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_node_binary_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_node_binary_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_node_binary_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
            else // Pivoting
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_node_pivot_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_node_pivot_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_node_pivot_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_node_pivot_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_node_pivot_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_node_pivot_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
        }
        else if (config.processBy == ByEdge)
        {
            if(kcc.Algo == GraphOrient)
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_edge_binary_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_edge_binary_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_edge_binary_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_edge_binary_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_edge_binary_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_edge_binary_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
            else // Pivoting
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_edge_pivot_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_edge_pivot_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_edge_pivot_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_edge_pivot_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_edge_pivot_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_edge_pivot_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
        }
        localclique.sync();
        double time = t.elapsed();
        // (teps: traversed edges per second)
        Log(info, "count time %f s (%f teps)", time, m / time);
        localclique.show(n);
    }

    printf("Done ....\n");

    //A.freeGPU();
    //B.freeGPU();
    return 0;
}


