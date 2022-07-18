
**Parallel K-clique counting on GPUs**

################################################################################################

If you use this code, please reference our paper Parallel K-clique counting on GPUs which is 
accepted in ICS'22
https://dl.acm.org/doi/abs/10.1145/3524059.3532382

################################################################################################


################################################################################################

This code is suppoed to count all cliques of size k using graph orinetation and pivoting 
approaches on GPUs


(1) To compile:

  #Do not forget to change compute capability -arch=sm_XX -gencode=arch=compute_XX,code=sm_XX
  
  make

(2) To run:
  
  ./build/exe/src/main.cu.exe -g <graph_filename> -d <device_id> -m kc -o <graph_orient> -k 5 -p <process_unit>  -q <Q8b> -w 0


  Such that:

  - graph_filename: graph name, Matrix Market Format (.mtx) and edge list (.tsv), also we support custom binary format. 
  The also code can convert from MTX to binary, more clarification in progress

  - device_id: gpu id

  - graph_orient: [degree, degen]

  - process_unit: [node, edge]

  - Q8b: (Q:[o, p]   o--> graph orientation and p--> pivoting) / 8: the partition size [1, 2, 4, 8, 16, 32] / b is binary encoding

  - [-w 0] : optional, use device memory for graph preprocessing, remove it for large graphs (> 500K edges)
  
  - [-x 0] : optional, print graph stats
  
  - [-s 0] : optional, to sort edges read from file
  
  (3) Dataset:
    All the datasets used in the paper are downloaded from SNAP dataset.
    
    Download the graph file and run the code without any modification to graph file name or content
  
 

(3) Example:
  ./build/exe/src/main.cu.exe -g as-skitter.mtx -d 0 -m kc -o degree -k 2 -p edge  -q p8b -w 0 - s 0 - x 0
  
  * * For more help, use ./build/exe/src/main.cu.exe -h
