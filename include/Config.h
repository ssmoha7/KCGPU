#pragma once

#include "utils.cuh"
#include "defs.cuh"

#ifndef __VS__
    #include <unistd.h>
#endif
#include <stdio.h>
#include "string.h"


struct KcliqueConfig
{
    KCAlgoEnum Algo;
    int PartSize;
    bool BinaryEncode;
    bool EncodeHalf;
};

struct Config {

    const char* srcGraph;
    const char* dstGraph; //for conversion
    const char* patGraph; //for subgraph matching
    MAINTASK mt;
    bool printStats;
    int deviceId;
    AllocationTypeEnum allocation;
    OrientGraphByEnum orient;
    bool verbosity;
    int k;
    bool sortEdges;
    ProcessBy processBy;
    ProcessingElementEnum processElement;
    KcliqueConfig kcConfig;
    bool isSmall;


};

static MAINTASK parseMainTask(const char* s)
{
    if (strcmp(s, "tsv-mtx") == 0)
        return CONV_TSV_MTX;

    if (strcmp(s, "tsv-bel") == 0)
        return CONV_TSV_BEL;

    if (strcmp(s, "txt-bel") == 0)
        return CONV_TXT_BEL;

    if (strcmp(s, "mtx-bel") == 0)
        return CONV_MTX_BEL;

    if (strcmp(s, "bel-mtx") == 0)
        return CONV_BEL_MTX;

    if (strcmp(s, "tc") == 0)
        return TC;

    if (strcmp(s, "kcore") == 0)
        return KCORE;


    if (strcmp(s, "ktruss") == 0)
        return KTRUSS;

    if (strcmp(s, "kc") == 0)
        return KCLIQUE;


    if (strcmp(s, "lkc") == 0)
        return KCLIQUE_LOCAL;
    
    if (strcmp(s, "mc") == 0)
        return MAXIMAL_CLIQUE;

    if (strcmp(s, "cd") == 0)
        return CROSSDECOMP;

    if (strcmp(s, "sgm") == 0)
        return GRAPH_MATCH;

    if (strcmp(s, "sgc") == 0)
        return GRAPH_COUNT;

    fprintf(stderr, "Unrecognized -mt option (Main TASK): %s\n", s);
    exit(0);
}

static const char* asString(MAINTASK mt) {
    switch (mt) {
    case CONV_TSV_MTX:            return "tsv-mtx";
    case CONV_TSV_BEL:            return "tsv-bel";
    case CONV_TXT_BEL:            return "txt-bel";
    case CONV_MTX_BEL:            return "mtx-bel";
    case CONV_BEL_MTX:            return "bel-mtx";
    case TC:                return "tc";
    case KCORE:            return "kcore";
    case KTRUSS:            return "ktruss";
    case KCLIQUE:            return "kc -> kclique";
    case KCLIQUE_LOCAL:            return "lkc -> kclique-local";
    case MAXIMAL_CLIQUE:            return "mc -> maximal-clique";
    case CROSSDECOMP:   return "cd";
    case GRAPH_MATCH:   return "sgm --> Subgraph matching Enumeration";
    case GRAPH_COUNT:   return "sgc --> Subgraph matching Counting";

    default:
        fprintf(stderr, "Unrecognized main task\n");
        exit(0);
    }
}

static OrientGraphByEnum parseOrient(const char* s)
{
    if (strcmp(s, "full") == 0)
        return None;
    if (strcmp(s, "upper") == 0)
        return Upper;
    if (strcmp(s, "lower") == 0)
        return Lower;
    if (strcmp(s, "degree") == 0)
        return Degree;
    if (strcmp(s, "degen") == 0)
        return Degeneracy;

    fprintf(stderr, "Unrecognized -o option (Graph Orient): %s\n", s);
    exit(0);
}




static const char* asString(OrientGraphByEnum mt) {
    switch (mt) {
    case None:            return "full";
    case Upper:            return "upper";
    case Lower:            return "lower";
    case Degree:            return "degree";
    case Degeneracy:            return "degen";
    default:
        fprintf(stderr, "Unrecognized orient\n");
        exit(0);
    }
}


static ProcessingElementEnum parseElement(const char* s)
{
    if (strcmp(s, "t") == 0)
        return Thread;
    if (strcmp(s, "w") == 0)
        return Warp;
    if (strcmp(s, "b") == 0)
        return Block;
    if (strcmp(s, "g") == 0)
        return Grid;
    if (strcmp(s, "bw") == 0)
        return BlockWarp;

    fprintf(stderr, "Unrecognized -e option (Processing Element): %s\n", s);
    exit(0);
}




static const char* asString(ProcessingElementEnum mt) {
    switch (mt) {
    case Thread:          return "t";
    case Warp:            return "w";
    case Block:           return "b";
    case Grid:            return "g";
    case BlockWarp:       return "bw";
    default:
        fprintf(stderr, "Unrecognized processing element\n");
        exit(0);
    }
}


static const char* asString(AllocationTypeEnum mt) {
    switch (mt) {
    case gpu:            return "gpu";
    case unified:            return "unified";

    default:
        fprintf(stderr, "Unrecognized allocation (cpu, gpu, unified)\n");
        exit(0);
    }
}




static ProcessBy parseProcessBy(const char* s)
{
    if (strcmp(s, "node") == 0)
        return ByNode;
    if (strcmp(s, "edge") == 0)
        return ByEdge;

    fprintf(stderr, "Unrecognized -p option (Process By Node or Edge): %s\n", s);
    exit(0);

}

static const char* asString(ProcessBy pb) {
    switch (pb) {
    case ByNode:            return "node";
    case ByEdge:            return "edge";
    default:
        fprintf(stderr, "Unrecognized process by\n");
        exit(0);
    }
}

static AllocationTypeEnum parseAllocation(const char* s)
{
    if (strcmp(s, "unified") == 0)
        return unified;
    if (strcmp(s, "gpu") == 0)
        return gpu;
    if (strcmp(s, "cpu") == 0)
        return cpuonly;


    fprintf(stderr, "Unrecognized -a option (Allocation): %s\n", s);
    exit(0);
}



static KcliqueConfig parseKcConfig(const char* s)
{
    // KCAlgoEnum Algo;
    // int PartSize;
    // bool BinaryEncode;

    KcliqueConfig kc;
    kc.Algo = KCAlgoEnum::GraphOrient;
    kc.PartSize = 8;
    kc.BinaryEncode = true;
    kc.EncodeHalf = false;

    if(strlen(s) > 2)
    {
        if(s[0] == 'p')
             kc.Algo = KCAlgoEnum::Pivoting;
        

        string sp(1, s[1]);
        int next = 2;
        while(isdigit(s[next]))
        {
            sp += s[next];
            next++;
        }

        kc.PartSize = stoi(sp);

        if(s[next] == 'n')
            kc.BinaryEncode = false;
        else if(s[next] == 'h')
            kc.EncodeHalf = true;
    }

    return kc;

}
static const char* asString(KcliqueConfig kc) 
{
    std::string s = "";

    if( kc.Algo == KCAlgoEnum::GraphOrient)
        s+= "GO-"; 
    else
        s+= "PIV-";   
    s += std::to_string(kc.PartSize);
    s+= "-";
    if(kc.BinaryEncode)
        s += "BE";
    else
        s += "NBE";

    return s.c_str();
}


static void usage() {
    fprintf(stderr,
        "\nUsage:  ./build/exe/src/main.cu.exe [options]"
        "\n"
        "\nOptions:"
        "\n    -g <Src graph FileName>       Name of file with input graph (default = )"
        "\n    -r <Dst graph FileName>       Name of file with dst graph only for conversion (default = )"
        "\n    -w <is small graph>         Use global memory to allocate the undirected graph, otherwise zerocopy memory"
        "\n    -t <Pattern graph filename>   Name of file with template/pattern graph only for subgraph matching"
        "\n    -d <Device Id>                      GPU Device Id (default = 0)"
        "\n    -m <MainTask>     Name of the task to perform (default = TC)"
        "\n    -x                   Print Graph Stats         "
        "\n    -o <orientGraph>       How to orient undirected to directed graph (default = full)"
        "\n    -a <allocation>        Data allocation on GPU (default = unified)"
        "\n    -v <verbosity>        Verbosity"
        "\n    -k <k>        k"
        "\n    -s <sort>        Sort Read Edges by src then dst (default = false)"
        "\n    -p <processBy>        Process by node or edge (default = node)"
        "\n    -e <process element>         Granulaity of element processor (default = t) <t: Thread, w: warp, b: block, g: grid>"
        "\n    -q <kclique specs>           Specify KC Specs: (o4b --> graph orient, Partition Size = 4, binary encoeding)  (p4n --> pivoting, Partition Size = 4, NO binary encoding) default: o8b"
        "\n    -h                       Help"
        "\n"
        "\n");
}


static Config parseArgs(int argc, char** argv) {
    Config config;
    config.srcGraph = "D:\\graphs\\as-Skitter2.bel";
    config.dstGraph = "D:\\graphs\\as-Skitter2.bel";
    config.patGraph = "D:\\graphs\\as-Skitter2.bel";
    config.deviceId = 0;
    config.mt = KCLIQUE;
    config.printStats = false;
    config.orient = Degree;
    config.allocation = gpu;
    config.k = 5;
    config.sortEdges = false;;
    config.processBy = ByNode;
    config.processElement = BlockWarp;
    config.kcConfig = parseKcConfig("");
    config.isSmall = false;

#ifndef __VS__
    int opt;

    //printf("parsing configuration .... \n");

    while ((opt = getopt(argc, argv, "g:r:d:m:x:o:a:k:h:v:s:p:e:q:w:t:")) >= 0) {
        switch (opt) {
        case 'g': config.srcGraph = optarg;                                 break;
        case 'r': config.dstGraph = optarg;                                 break;
        case 'w': config.isSmall = true;                                    break;                         
        case 't': config.patGraph = optarg;                                 break;
        case 'd': config.deviceId = atoi(optarg);                           break;
        case 'm': config.mt = parseMainTask(optarg);                        break;
        case 'x': config.printStats = true;                                 break;
        case 'o': config.orient = parseOrient(optarg);                      break;
        case 'a': config.allocation = parseAllocation(optarg);              break;
        case 'v': config.verbosity = atoi(optarg);                          break;
        case 'k': config.k = atoi(optarg);                                  break;
        case 's': config.sortEdges = true;                                  break;
        case 'p': config.processBy = parseProcessBy(optarg);                break;
        case 'e': config.processElement = parseElement(optarg);             break;
        case 'q': config.kcConfig = parseKcConfig(optarg);                  break;
        case 'h': usage(); exit(0);                                         break;
        default: fprintf(stderr, "\nUnrecognized option!\n");
            usage(); exit(0);
        }
    }
#endif
    return config;
}

static void printConfig(Config config)
{
    printf("    Graph: %s\n", config.srcGraph);
    printf("    DST Graph: %s\n", config.dstGraph);
    printf("    Pattern Graph: %s\n", config.patGraph);
    printf("    Device Id = %u\n", config.deviceId);
    printf("    Allocation = %s\n", asString(config.allocation));
    printf("    Small Graph = %s\n", config.isSmall? "Small Graph Allocation" : "Large Graph Allocation");
    printf("    Main Task = %s\n", asString(config.mt));
    printf("    Graph Orientation = %s\n", asString(config.orient));
    printf("    Process By = %s\n", asString(config.processBy));
    printf("    Process Element = %s\n", asString(config.processElement));
    printf("    k: %u\n", config.k);


    if (config.mt == KCLIQUE || config.mt == KCLIQUE_LOCAL || config.mt == MAXIMAL_CLIQUE)
         printf("    KC Config = %s\n", asString(config.kcConfig));


}