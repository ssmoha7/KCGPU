# KCLIEQUE counting on GPUs
This code is suppoed to count all cliques of size k using graph orinetation and pivoting apporach


To compile:
#Do not forget to change compute capability -arch=sm_XX -gencode=arch=compute_XX,code=sm_XX

make

To run:

#./build/exe/src/main.cu.exe -h --> for help

./build/exe/src/main.cu.exe -g <graph_filename> -d <device_id> -m kc -o <graph_orient> -k 5 -p <process_unit>  -q <Q8b> -w 0


Such that:

<graph_filename> --> graph name, Matrix Market Format (.mtx) and edge list (.tsv), also we support custom binary format. The code can convert from MTX to binary (let me know if you need this service)

<device_id> --> gpu id

<graph_orient> --> [degree, degen]

<process_unit> --> [node, edge]

<Q8b>--> (Q:[o, p]   o--> graph orientation and p--> pivoting) / 8: the partition size [1, 2, 4, 8, 16, 32] / b is binary encoding

-w 0 : use device memory for graph preprocessing, remove it for large graphs (> 500K edges)

Example:
./build/exe/src/main.cu.exe -g as-skitter.mtx -d 0 -m kc -o degree -k 2 -p edge  -q p8b -w 0




