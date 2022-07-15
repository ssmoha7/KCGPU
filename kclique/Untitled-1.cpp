
template<typename T, int BLOCKSIZE, int GROUPSIZE, PARSCHEME PAR, PARLEVEL HL>
void HybridTreeTraversal(Task t, Graph G, Graph P)
{
    //Get GPU Info: number of SMs and number of concurrent blocks per SM
    cudaDeviceProp deviceProp;
    int numBlocksPerSm = 0;
    cudaGetDeviceProperties(&deviceProp, ...);
    int numSMs = deviceProp.multiProcessorCount;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel_problem<T, BLOCKSIZE, GROUPSIZE>, BLOCKSIZE, ...);

    //Graph Maximum Degree
    uint64 *maxDegree, *inducedSubgraphPointer;
    AllocateGPU<uint64>(maxDegree, 1);
    getMaxOutDegree(G, maxDegree);
   
    //Local Variables
    const uint WordSize = 32; //32 bits per word (binary encoding)
    const uint GroupsPerBlock = BLOCKSIZE / GROUPSIZE; //For three-level parallelism
   
    //Allocate on GPU
    uint64 *maxDegree, *inducedSubgraphPointer;
    AllocateGPU<uint64>(maxDegree, 1);
    getMaxOutDegree(G, maxDegree);

    numWords = maxDegree[0] / WordSize;
    const uint Depth = (t == Task::GPM || t == Task::KCGO)? P.K: maxDegree[0];

    //Induced Subgraph Allocation
    AllocateGPU<uint>(inducedSubgraphPointer,  maxDegree[0] * numWords * numSMs * numBlocksPerSm);

    //Stack Allocation
    uint numParSubtrees = 0;
    uint *levelVerticesPointer, ...
    if(HL == PARLEVEL::ThreeLevel)
    {
        numParSubtrees = GroupsPerBlock * numSMs * numBlocksPerSm;
    }
    else // Two-Level
    {
        numParSubtrees = numSMs * numBlocksPerSm;
    }
    AllocateGPU<uint>(levelVerticesPointer,  Depth * numWords * numParSubtrees);
    ...

    if(t == Task::GPM)
    {
        preprocess(P, &)
    }

    int StartTraversalLevel = (PAR == PARSCHEME.VerCentric)? 2 : 3/*EdgeCentric*/;
    //Based on the task: problem = {GPM, KCOR, KCPIV, MCE}
    (kernel_problem<T, BLOCKSIZE, GROUPSIZE>)<<<gridSize, BLOCKSIZE>>>(G, P, StartTraversalLevel, 
                maxDegree, Depth, inducedSubgraphPointer, stackData...);
}
