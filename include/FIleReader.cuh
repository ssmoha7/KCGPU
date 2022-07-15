#pragma once
#include "defs.cuh"
#include "utils.cuh"

namespace graph
{
    struct MatrixFileProp
    {
        bool symmetric = false;  // whether the graph is undirected
        bool skew = false;  // whether edge values are inverse for symmetric matrices
        bool array = false;  // whether the mtx file is in dense array format
        bool pattern = false;

        void Set(bool sy, bool sk, bool ar, bool pa)
        {
            symmetric = sy;
            skew = sk;
            array = ar;
            pattern = pa;
        }
    };

    bool endswith(const std::string& base,  //!< [in] the base string
        const std::string& suffix //!< [in] the suffix to check for
    ) {
        if (base.size() < suffix.size()) {
            return false;
        }
        return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
    }


    class EdgeListFile {

    private:
        enum class FileType { TSV, BEL, MTX };
        FILE* fp_;
        std::string path_;
        FileType type_;
        std::vector<char> belBuf_; // a buffer used by read_bel

        //For market format
        std::ifstream market_file;
        MatrixFileProp mfp;

        /*! read n edges into ptr

          \tparam T the Node type
          \return the number of edges read
        */
        template <typename T>
        size_t read_bel(EdgeTy<T>* ptr, //<! buffer for edges (allocated by caller)
            const size_t n  //<! number of edges to read
        ) {
            if (fp_ == nullptr) {
                Log(LogPriorityEnum::error, "error reading {} or file was already closed ", path_);
                return 0;
            }
            if (ptr == nullptr) {
                Log(LogPriorityEnum::error, "buffer is a nullptr");
                return 0;
            }
            belBuf_.resize(24 * n);
            const size_t numRead = fread(belBuf_.data(), 24, n, fp_);

            // end of file or error
            if (numRead != n) {
                // end of file
                if (feof(fp_)) {
                    // do nothing
                }
                // some error
                else if (ferror(fp_)) {
                    Log(LogPriorityEnum::error, "Error while reading {}: {}", path_, strerror(errno));
                    fclose(fp_);
                    fp_ = nullptr;
                    assert(0);
                }
                else {
                    Log(LogPriorityEnum::error, "Unexpected error while reading {}", path_);
                    assert(0);
                }
            }
            for (size_t i = 0; i < numRead; ++i) {
                uint64_t src, dst;
                memcpy(&src, &belBuf_[i * 24 + 8], 8);
                memcpy(&dst, &belBuf_[i * 24 + 0], 8);
                ptr[i].first = src;
                ptr[i].second = dst;
                //SPDLOG_TRACE(logger::console(), "read {} -> {}", ptr[i].first, ptr[i].second);
            }

            // no characters extracted or parsing error
            return numRead;
        }





        template <typename T> size_t read_tsv(EdgeTy<T>* ptr, const size_t n) {

            assert(ptr != nullptr);
            assert(fp_ != nullptr);

            size_t i = 0;
            for (; i < n; ++i) {
                long long unsigned dst, src, weight;
                const size_t numFilled = fscanf(fp_, "%llu %llu %llu", &dst, &src, &weight);
                if (numFilled != 3) {
                    if (feof(fp_)) {
                        return i;
                    }
                    else if (ferror(fp_)) {
                        Log(LogPriorityEnum::error, "Error while reading {}: {}", path_, strerror(errno));
                        return i;
                    }
                    else {
                        Log(LogPriorityEnum::critical, "Unexpected error while reading {}", path_);
                        exit(-1);
                    }
                }
                ptr[i].first = static_cast<T>(src);
                ptr[i].second = static_cast<T>(dst);
            }
            return i;
        }


        template <typename T, typename ValueT>
        size_t read_market(EdgeTy<T>* ptr, const size_t n, bool has_edge_val = true)
        {

            assert(ptr != nullptr);
           
            std::istream& input_stream = market_file;
            size_t i = 0;
            bool succeed = true;
            for (; i < n; ++i)
            {
                succeed = true;
                std::string line;
                std::getline(input_stream, line);

                long long ll_row, ll_col;
                ValueT ll_value;  // used for parse float / double
                double lf_value;  // used to sscanf value variable types
                int num_input;
                bool has_edge_val = !mfp.pattern;

                if (has_edge_val)
                {
                    num_input = sscanf(line.c_str(), "%lld %lld %lf", &ll_row, &ll_col, &lf_value);

                    if (mfp.array && (num_input == 1))
                    {
                        ll_value = ll_row;
                        ll_col = ll_row;
                        ll_row = ll_row;
                    }
                    else if (mfp.array || num_input < 2)
                    {
                        succeed = false;
                        break;
                    }
                    else if (num_input == 2)
                    {
                        ll_value = 1;
                    }
                    else if (num_input > 2)
                    {
                        if (typeid(ValueT) == typeid(float) ||
                            typeid(ValueT) == typeid(double) ||
                            typeid(ValueT) == typeid(long double))
                            ll_value = (ValueT)lf_value;
                        else
                            ll_value = (ValueT)(lf_value + 1e-10);

                    }

                }
                else
                { // if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
                    num_input = sscanf(line.c_str(), "%lld %lld", &ll_row, &ll_col);

                    if (mfp.array && (num_input == 1))
                    {
                        ll_value = ll_row;
                        ll_col = 0;
                        ll_row = 0;
                    }
                    else if (mfp.array || (num_input != 2)) {
                        succeed = false;
                        break;
                    }
                }

                if (succeed)
                {
                    ptr[i].first = static_cast<T>(ll_row);
                    ptr[i].second = static_cast<T>(ll_col);
                    if (mfp.symmetric)
                        if (ll_col != ll_row)
                        {
                            ptr[i].first = static_cast<T>(ll_col);
                            ptr[i].second = static_cast<T>(ll_row);
                            i++;
                        }
                }
            }
            return i;
        }


        template <typename T, typename WT> size_t read_tsv(WEdgeTy<T, WT>* ptr, const size_t n) {

            assert(ptr != nullptr);
            assert(fp_ != nullptr);

            size_t i = 0;
            for (; i < n; ++i) {
                long long unsigned dst, src, weight;
                const size_t numFilled = fscanf(fp_, "%llu %llu %llu", &dst, &src, &weight);
                if (numFilled != 3) {
                    if (feof(fp_)) {
                        return i;
                    }
                    else if (ferror(fp_)) {
                        Log(LogPriorityEnum::error, "Error while reading {}: {}", path_, strerror(errno));
                        return i;
                    }
                    else {
                        Log(LogPriorityEnum::critical, "Unexpected error while reading {}", path_);
                        exit(-1);
                    }
                }

                ptr[i] = std::make_tuple(static_cast<T>(src), static_cast<T>(dst), static_cast<WT>(weight));
            }
            return i;
        }


    public:
        /*! \brief Construct an EdgeListFile

          Supports GraphChallenge TSV or BEL files
        */
        EdgeListFile(const std::string& path //!< [in] the path of the file
        )
            : path_(path) {
            if (endswith(path, ".bel")) {
                type_ = FileType::BEL;
                fp_ = fopen(path_.c_str(), "rb");
                if (nullptr == fp_) {
                    Log(LogPriorityEnum::error, "unable to open \"{}\"", path_);
                }
            }
            else if (endswith(path, ".tsv")) {
                type_ = FileType::TSV;
                fp_ = fopen(path_.c_str(), "r");
                if (nullptr == fp_) {
                    Log(LogPriorityEnum::error, "unable to open \"{}\"", path_);
                }

            }
            else if (endswith(path, ".mtx"))
            {
                type_ = FileType::MTX;
                market_file.open(path_.c_str(), std::ios::in);
                AdvanceWithMarketFile();
            }
            else {
                Log(LogPriorityEnum::critical, "no reader for file {}", path);
                exit(-1);
            }

          
        }

        ~EdgeListFile() {
            if (fp_ && type_ != FileType::MTX) {
                fclose(fp_);
                fp_ = nullptr;
            }
            else if (market_file && type_ == FileType::MTX)
            {
                market_file.close();
            }

        }
        void sort_edges(std::vector<EdgeTy<uint>>& edges)
        {
            std::sort(edges.begin(), edges.end(), [](const EdgeTy<uint>& a, const EdgeTy<uint>& b) -> bool
                {
                    return a.first < b.first || (a.first == b.first && a.second < b.second);
                });
        }
        void AdvanceWithMarketFile()
        {
            std::istream& input_stream = market_file;
            uint64 nodes = 0;
            uint64 edges = 0;
            bool got_edge_values = false;
            bool symmetric = false;  // whether the graph is undirected
            bool skew = false;  // whether edge values are inverse for symmetric matrices
            bool array = false;  // whether the mtx file is in dense array format
            bool pattern = false;

            std::string line;
            while (true) {
                std::getline(input_stream, line);
                if (line[0] != '%')
                {
                    break;
                }
                else
                {
                    if (strlen(line.c_str()) >= 2 && line[1] == '%') {
                        symmetric = (strstr(line.c_str(), "symmetric") != NULL);
                        skew = (strstr(line.c_str(), "skew") != NULL);
                        array = (strstr(line.c_str(), "array") != NULL);
                        pattern = (strstr(line.c_str(), "pattern") != NULL);
                    }
                }
            }

            mfp.Set(symmetric, skew, array, pattern);

            long long ll_nodes_x, ll_nodes_y, ll_edges;
            unsigned long long items_scanned = sscanf(line.c_str(), "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges);

            if (array && items_scanned == 2) {
                ll_edges = ll_nodes_x * ll_nodes_y;
            }
            else if (!array && items_scanned == 3) {

                if (ll_nodes_x != ll_nodes_y) {
                    Log(LogPriorityEnum::critical, "Error parsing MARKET graph: not square %d, %d\n", ll_nodes_x, ll_nodes_y);
                    return;
                }
            }
            else
            {
                Log(LogPriorityEnum::critical, "Error parsing MARKET graph: invalid problem description\n");
            }

            nodes = ll_nodes_x;
            edges = ll_edges;


            Log(LogPriorityEnum::debug, "%llu nodes, %llu directed edges\n", ll_nodes_x, ll_edges);
        }

        /*! \brief attempt to read n edges from the file

          \tparam T the node ID type
          \return the number of edges read

        */
        template <typename T>
        size_t
            get_edges(std::vector<EdgeTy<T>>& edges, //!< [out] the read edges. Resized to the number of successfully read edges
                const size_t n                 //!< [in] the number of edges to try to read
            ) {
            //SPDLOG_TRACE(logger::console(), "requested {} edges", n);
            edges.resize(n);

            size_t numRead;
            switch (type_) {
            case FileType::BEL: {
                numRead = read_bel(edges.data(), n);
                break;
            }
            case FileType::TSV: {
                numRead = read_tsv(edges.data(), n);
                break;
            }
            case FileType::MTX:
                numRead = read_market<T, T>(edges.data(), n, false);
                break;
            default: {
                Log(LogPriorityEnum::critical, "unexpected file type");
                exit(-1);
            }
            }
            edges.resize(numRead);
            return numRead;
        }


        template <typename T, typename WT>
        size_t
            get_weighted_edges(std::vector<WEdgeTy<T, WT>>& edges, //!< [out] the read edges. Resized to the number of successfully read edges
                const size_t n                 //!< [in] the number of edges to try to read
            ) {
            //SPDLOG_TRACE(logger::console(), "requested {} edges", n);
            edges.resize(n);

            size_t numRead;
            switch (type_) {
            case FileType::TSV: {
                numRead = read_tsv(edges.data(), n);
                break;
            }
            default: {
                Log(LogPriorityEnum::critical, "unexpected file type");
                exit(-1);
            }
            }
            edges.resize(numRead);
            return numRead;
        }

        template <typename T>
        size_t write_tsv_bel(
            const std::string path,
            EdgeTy<T>* ptr, //<! buffer for edges (allocated by caller)
            const size_t n  //<! number of edges to read
        ) {

            FILE* writer = fopen(path.c_str(), "wb");

            if (writer == nullptr) {

                return 0;
            }
            if (ptr == nullptr) {

                return 0;
            }

            unsigned long long* l;
            l = (unsigned long long*)malloc(3 * n * sizeof(unsigned long long));
            uint64 elementCounter = 0;
            for (uint64 i = 0; i < n; i++)
            {
                EdgeTy<T> p = ptr[i];

                //in TSV (d, s, w)
                l[elementCounter++] = p.second;
                l[elementCounter++] = p.first;
                l[elementCounter++] = 0;
            }


            const size_t numWritten = fwrite(l, 8, 3 * n, writer);

            fclose(writer);
            free(l);
        }


    };




    class MtB_Writer
    {
    public:
        template <typename T, typename ValueT = int>
        void write_market_bel(
            const std::string path,
            const std::string outputPath,
            bool makeFull = true
        ) {
            //Read Market Matrix File
            std::ifstream file;
            file.open(path.c_str(), std::ios::in);
            std::istream& input_stream = file;
            T nodes = 0;
            T edges = 0;
            bool got_edge_values = false;
            bool symmetric = false;  // whether the graph is undirected
            bool skew = false;  // whether edge values are inverse for symmetric matrices
            bool array = false;  // whether the mtx file is in dense array format
            bool pattern = false;

            std::string line;
            while (true) {
                std::getline(input_stream, line);
                if (line[0] != '%')
                {
                    break;
                }
                else
                {
                    if (strlen(line.c_str()) >= 2 && line[0] == '%') {
                        symmetric = (strstr(line.c_str(), "symmetric") != NULL);
                        skew = (strstr(line.c_str(), "skew") != NULL);
                        array = (strstr(line.c_str(), "array") != NULL);
                        pattern = (strstr(line.c_str(), "pattern") != NULL);
                    }
                }
            }
            printf("%s, %s\n", symmetric?"sym":"", pattern?"pattern":"" );

            long long ll_nodes_x, ll_nodes_y, ll_edges;
            int items_scanned = sscanf(line.c_str(), "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges);

            if (array && items_scanned == 2) {
                ll_edges = ll_nodes_x * ll_nodes_y;
            }
            else if (!array && items_scanned == 3) {

                if (ll_nodes_x != ll_nodes_y) {
                    Log(LogPriorityEnum::critical, "Error parsing MARKET graph: not square %d, %d\n", ll_nodes_x, ll_nodes_y);
                    return;
                }
            }
            else
            {
                Log(LogPriorityEnum::critical, "Error parsing MARKET graph: invalid problem description\n");
            }

            nodes = ll_nodes_x;
            edges = ll_edges;


            Log(LogPriorityEnum::debug, "%d nodes, %d directed edges\n", ll_nodes_x, ll_edges);

            std::vector<EdgeTy<T>> ptr;
            bool succeed = true;
            for (unsigned long long i = 0; i < edges; ++i)
            {
                succeed = true;
                std::string line;
                std::getline(input_stream, line);

                long long ll_row, ll_col;
                ValueT ll_value;  // used for parse float / double
                double lf_value;  // used to sscanf value variable types
                int num_input;
                bool has_edge_val = !pattern;

                if (has_edge_val)
                {
                    num_input = sscanf(line.c_str(), "%lld %lld %lf", &ll_row, &ll_col, &lf_value);

                    if (array && (num_input == 1))
                    {
                        ll_value = ll_row;
                        ll_col = i / nodes;
                        ll_row = i - nodes * ll_col;
                    }
                    else if (array || num_input < 2)
                    {
                        Log(LogPriorityEnum::error, "Error parsing MARKET graph : badly formed edge: LINE %d\n", i);
                        succeed = false;
                        break;
                    }
                    else if (num_input == 2)
                    {
                        ll_value = 1;
                    }
                    else if (num_input > 2)
                    {
                        if (typeid(ValueT) == typeid(float) ||
                            typeid(ValueT) == typeid(double) ||
                            typeid(ValueT) == typeid(long double))
                            ll_value = (ValueT)lf_value;
                        else
                            ll_value = (ValueT)(lf_value + 1e-10);
                        got_edge_values = true;
                    }

                }
                else { // if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
                    num_input = sscanf(line.c_str(), "%lld %lld", &ll_row, &ll_col);

                    if (array && (num_input == 1)) {
                        ll_value = ll_row;
                        ll_col = i / nodes;
                        ll_row = i - nodes * ll_col;
                    }
                    else if (array || (num_input != 2)) {
                        Log(LogPriorityEnum::error,
                            "Error parsing MARKET graph: badly formed edge: Line %d", i);
                        succeed = false;
                        break;
                    }
                }

                if (succeed)
                {
                    ptr.push_back(std::make_pair(ll_row, ll_col));
                    if (symmetric || makeFull)
                        if (ll_col != ll_row)
                        {
                            ptr.push_back(std::make_pair(ll_col, ll_row));
                        }
                }
            } // endfor

            printf("File Edges = %d, Edges found = %llu\n", edges, ptr.size());
            edges = ptr.size();


            std::sort(ptr.begin(), ptr.end(), [](const EdgeTy<T>& a, const EdgeTy<T>& b) -> bool
                {
                    return a.first < b.first || (a.first == b.first && a.second < b.second);
                });


            printf("Done Sorting ...\n");

            FILE* writer = fopen(outputPath.c_str(), "wb");

            if (writer == nullptr) {

                return;
            }

            const T shard = 100000;
            unsigned long long* l;
            l = (unsigned long long*)malloc(3 * shard * sizeof(unsigned long long));
            printf("Done allocating total memory ...\n");

            unsigned long long blocks = (edges + shard - 1) / shard;
            for (unsigned long long i = 0; i < blocks; i++)
            {
                unsigned long long startEdge = i * shard;
                unsigned long long endEdge = (i + 1) * shard < edges ? (i + 1) * shard : edges;

                unsigned long long elementCounter = 0;
                for (unsigned long long j = startEdge; j < endEdge; j++)
                {
                    EdgeTy<T> p = ptr[j];

                    l[elementCounter++] = p.second;
                    l[elementCounter++] = p.first;
                    l[elementCounter++] = 0;
                }

                const unsigned long long numWritten = fwrite(l, 8, 3 * (endEdge - startEdge), writer);
            }

            fclose(writer);
            free(l);
        }


        template <typename T, typename ValueT = int>
        void write_tsv_bel(
            const std::string path,
            const std::string outputPath
        ) {

            FILE* fp_;
            fp_ = fopen(path.c_str(), "r");
            std::vector<EdgeTy<T>> fileEdges;

            int numread = 0;
            do {
                long long unsigned dst, src, weight;
                numread = fscanf(fp_, "%llu %llu %llu", &dst, &src, &weight);
                if (numread == 3)
                {
                    fileEdges.push_back(std::make_pair(src, dst));
                }
            } while (numread == 3);

            uint64 n = fileEdges.size();

            printf("Read the file: N=%llu\n", n);

            FILE* writer = fopen(outputPath.c_str(), "wb");

            if (writer == nullptr) {

                return;
            }

            const T shard = 100000;
            unsigned long long* l;
            l = (unsigned long long*)malloc(3 * shard * sizeof(unsigned long long));
            printf("Done allocating total memory ...\n");

            unsigned long long blocks = (n + shard - 1) / shard;
            for (unsigned long long i = 0; i < blocks; i++)
            {
                unsigned long long startEdge = i * shard;
                unsigned long long endEdge = (i + 1) * shard < n ? (i + 1) * shard : n;

                unsigned long long elementCounter = 0;
                for (unsigned long long j = startEdge; j < endEdge; j++)
                {
                    EdgeTy<T> p = fileEdges[j];

                    l[elementCounter++] = p.second;
                    l[elementCounter++] = p.first;
                    l[elementCounter++] = 0;
                }

                const unsigned long long numWritten = fwrite(l, 8, 3 * (endEdge - startEdge), writer);
            }

            fclose(writer);
            free(l);
        }



        template <typename T, typename ValueT = double>
        void write_tsv_market(
            const std::string path,
            const std::string outputPath
        ) {

            FILE* fp_;
            fp_ = fopen(path.c_str(), "r");
            std::vector<EdgeTy<T>> fileEdges;
            std::vector<ValueT> weights;

            int numread = 0;
            do {
                long long unsigned dst, src;
                ValueT weight;
                if (typeid(ValueT) != typeid(double))
                    numread = fscanf(fp_, "%llu %llu %llu", &dst, &src, &weight);
                else
                    numread = fscanf(fp_, "%llu %llu %llg", &dst, &src, &weight);

                if (numread == 3)
                {
                    fileEdges.push_back(std::make_pair(src, dst));
                    weights.push_back(weight);
                }
            } while (numread == 3);

            unsigned long long m = fileEdges.size();
            unsigned long long n = fileEdges[m - 1].first;





            std::fstream file;
            file.open(outputPath, std::ios::out);

            file << "%%MatrixMarket matrix coordinate general" << std::endl;
            file << n << " " << n << " " << m << std::endl;

            for (unsigned long long i = 0; i < m; i++)
            {
                file << fileEdges[i].first << " " << fileEdges[i].second << " " << weights[i] << std::endl;
            }

            //closing the file
            file.close();
        }



        template <typename T, typename ValueT = double>
        void write_bel_market(
            const std::string path,
            const std::string outputPath
        ) {

            FILE* fp_;
            std::vector<char> belBuf_;
            fp_ = fopen(path.c_str(), "rb");
            std::vector<EdgeTy<T>> fileEdges;
            std::vector<ValueT> weights;

            int numRead = 0;
            int blockSize = 1000;
            do {


                belBuf_.resize(24 * blockSize);
                numRead = fread(belBuf_.data(), 24, blockSize, fp_);

                // end of file or error
                if (numRead > 0)
                {
                    for (size_t i = 0; i < numRead; ++i) {
                        uint64_t src, dst;
                        memcpy(&src, &belBuf_[i * 24 + 8], 8);
                        memcpy(&dst, &belBuf_[i * 24 + 0], 8);
                        fileEdges.push_back(std::make_pair(src, dst));
                        ValueT weight;

                        memcpy(&weight, &belBuf_[i * 24 + 16], 8);
                        weights.push_back(weight);
                    }
                }
            } while (numRead > 0);

            unsigned long long m = fileEdges.size();
            unsigned long long n = fileEdges[m - 1].first;


            std::fstream file;
            file.open(outputPath, std::ios::out);

            file << "%%MatrixMarket matrix coordinate general" << std::endl;
            file << n << " " << n << " " << m << std::endl;

            for (unsigned long long i = 0; i < m; i++)
            {
                file << fileEdges[i].first << " " << fileEdges[i].second << " " << weights[i] << std::endl;
            }

            //closing the file
            file.close();
        }


        template <typename T, typename ValueT = int>
        void write_txt_bel(
            const std::string path,
            const std::string outputPath,
            bool makeFull,
            int numCol = 3,
            int srcIndex = 1
        ) {

            FILE* fp_;
            fp_ = fopen(path.c_str(), "r");
            std::vector<EdgeTy<T>> fileEdges;
            int numread = 0;
            long long unsigned a, b, weight;

            char line[100];
            while ((fscanf(fp_, "%[^\n]%*c", line))!= EOF)
            {
               

                if(line[0] == '%')
                {    
                    numread = numCol;
                    printf("%s\n", line);
                }
                else
                {
                    if (numCol == 3)
                        numread = sscanf(line, "%llu %llu %llu", &a, &b, &weight);
                    else if (numCol == 2)
                        numread =  sscanf(line, "%llu %llu", &a, &b);

                    if (numread == numCol && a != b)
                    {
                        long long unsigned src, dst;

                        if (srcIndex == 0)
                        {
                            src = a;
                            dst = b;
                        }
                        else
                        {
                            src = b;
                            dst = a;
                        }
                        fileEdges.push_back(std::make_pair(src, dst));
                        if (makeFull)
                            fileEdges.push_back(std::make_pair(dst, src));
                    }
                }
            } //while (numread == numCol);



            std::sort(fileEdges.begin(), fileEdges.end(), [](const EdgeTy<T>& a, const EdgeTy<T>& b) -> bool
            {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            });


            fileEdges.resize(std::unique(fileEdges.begin(), fileEdges.end()) - fileEdges.begin());

            size_t n = fileEdges.size();
            FILE* writer = fopen(outputPath.c_str(), "wb");

            if (writer == nullptr) {

                return;
            }

            const T shard = 100000;
            unsigned long long* l;
            l = (unsigned long long*)malloc(3 * shard * sizeof(unsigned long long));
            printf("Done allocating total memory n=%lu ...\n", n);

            unsigned long long blocks = (n + shard - 1) / shard;
            for (unsigned long long i = 0; i < blocks; i++)
            {
                unsigned long long startEdge = i * shard;
                unsigned long long endEdge = (i + 1) * shard < n ? (i + 1) * shard : n;

                unsigned long long elementCounter = 0;
                for (unsigned long long j = startEdge; j < endEdge; j++)
                {
                    EdgeTy<T> p = fileEdges[j];

                    l[elementCounter++] = p.second;
                    l[elementCounter++] = p.first;
                    l[elementCounter++] = 0;
                }

                const unsigned long long numWritten = fwrite(l, 8, 3 * (endEdge - startEdge), writer);
            }

            fclose(writer);
            free(l);
        }


    };


}
