#pragma once
#include "utils.cuh"

namespace graph {
    template <typename Index, typename Vector = std::vector<Index>> class CSRCOO {

    public:
        typedef Index index_type;
        typedef EdgeTy<Index> edge_type;
        CSRCOO() {}     //!< empty matrix
        Vector rowPtr_; //!< offset in colInd_/rowInd_ that each row starts at
        Vector colInd_; //!< non-zero column indices
        Vector rowInd_; //!< non-zero row indices
        __host__ __device__ uint64_t nnz() const {
            assert(colInd_.size() == rowInd_.size());
            return colInd_.size();
        }                           //!< number of non-zeros
        uint64_t num_nodes() const { return num_rows(); }

        /*! number of matrix rows

        0 if rowPtr_.size() is 0, otherwise rowPtr_.size() - 1
         */
        __host__ __device__  uint64_t num_rows() const {
            if (rowPtr_.size() == 0) {
                return 0;
            }
            else {
                return rowPtr_.size() - 1;
            }
        }

        /*! Build a CSRCOO from an EdgeList

        Do not include edges where edgeFilter(edge) returns true
        */
        static CSRCOO<Index, Vector> from_edgelist(const EdgeList& es, bool (*edgeFilter)(const Edge&) = nullptr)
        {
            CSRCOO csr;

            if (es.size() == 0) {
                Log(warn, "constructing from empty edge list");
                return csr;
            }

            for (const auto& edge : es) {

                const Index src = static_cast<Index>(edge.first);
                const Index dst = static_cast<Index>(edge.second);

                if (src == dst)
                    continue;

                // edge has a new src and should be in a new row
                // even if the edge is filtered out, we need to add empty rows
                while (csr.rowPtr_.size() != size_t(src + 1)) {
                    // expecting inputs to be sorted by src, so it should be at least
                    // as big as the current largest row we have recored
                    assert(src >= csr.rowPtr_.size());
                    // SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.src_,
                    // csr.edgeSrc_.size());
                    csr.rowPtr_.push_back(csr.colInd_.size());
                }

                // filter or add the edge
                if (nullptr != edgeFilter && edgeFilter(edge)) {
                    continue;
                }
                else {
                    csr.rowInd_.push_back(src);
                    csr.colInd_.push_back(dst);
                }
            }

            // add the final length of the non-zeros to the offset array
            csr.rowPtr_.push_back(csr.colInd_.size());

            assert(csr.rowInd_.size() == csr.colInd_.size());
            return csr;
        }

        /*! Call shrink_to_fit on the underlying containers
         */
        __host__ void shrink_to_fit() {
            rowPtr_.shrink_to_fit();
            rowInd_.shrink_to_fit();
            colInd_.shrink_to_fit();
        }

        /*! The total capacity of the underlying containers in bytes
         */
        __host__ uint64_t capacity_bytes() const noexcept {
            return rowPtr_.capacity() * sizeof(typename decltype(rowPtr_)::value_type) +
                rowInd_.capacity() * sizeof(typename decltype(rowInd_)::value_type) +
                colInd_.capacity() * sizeof(typename decltype(colInd_)::value_type);
        }
        /*! The total size of the underlying containers in bytes
         */
        __host__ uint64_t size_bytes() const noexcept {
            return rowPtr_.size() * sizeof(typename decltype(rowPtr_)::value_type) +
                rowInd_.size() * sizeof(typename decltype(rowInd_)::value_type) +
                colInd_.size() * sizeof(typename decltype(colInd_)::value_type);
        }

        Index* row_ptr() { return rowPtr_.data(); } //!< row offset array
        Index* col_ind() { return colInd_.data(); } //!< column index array
        Index* row_ind() { return rowInd_.data(); } //!< row index array
    };
}