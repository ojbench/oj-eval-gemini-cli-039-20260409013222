#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override {
        return "Size mismatch";
    }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override {
        return "Index out of range";
    }
};

template <typename T>
class CSRMatrix {

private:
    size_t n, m, count;
    std::vector<size_t> indptr;
    std::vector<size_t> indices;
    std::vector<T> data;
    
public:
    // Assignment operators are deleted
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    // Constructor for empty matrix with dimensions
    CSRMatrix(size_t n, size_t m) : n(n), m(m), count(0), indptr(n + 1, 0), indices(), data() {}

    // Constructor with pre-built CSR components
    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr, 
        const std::vector<size_t> &indices,
        const std::vector<T> &data) : n(n), m(m), count(count), indptr(indptr), indices(indices), data(data) {
        if (indptr.size() != n + 1) throw size_mismatch();
        if (indices.size() != count) throw size_mismatch();
        if (data.size() != count) throw size_mismatch();
        if (indptr[0] != 0) throw invalid_index();
        if (indptr[n] != count) throw size_mismatch();
        for (size_t i = 0; i < n; ++i) {
            if (indptr[i] > indptr[i+1]) throw invalid_index();
            for (size_t j = indptr[i]; j < indptr[i+1]; ++j) {
                if (indices[j] >= m) throw invalid_index();
                if (j > indptr[i] && indices[j] <= indices[j-1]) throw invalid_index();
            }
        }
    }

    // Copy constructor
    CSRMatrix(const CSRMatrix &other) = default;

    // Move constructor
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor from dense matrix format (given as vector of vectors)
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &dense_data) : n(n), m(m), count(0) {
        if (dense_data.size() != n) throw size_mismatch();
        indptr.push_back(0);
        for (size_t i = 0; i < n; ++i) {
            if (dense_data[i].size() != m) throw size_mismatch();
            for (size_t j = 0; j < m; ++j) {
                if (!(dense_data[i][j] == T{})) {
                    indices.push_back(j);
                    data.push_back(dense_data[i][j]);
                    count++;
                }
            }
            indptr.push_back(count);
        }
    }

    // Destructor
    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    size_t getRowSize() const { return n; }

    size_t getColSize() const { return m; }

    size_t getNonZeroCount() const { return count; }

    // Element access
    T get(size_t i, size_t j) const {
        if (i >= n || j >= m) throw invalid_index();
        size_t start = indptr[i];
        size_t end = indptr[i+1];
        size_t l = start, r = end;
        while (l < r) {
            size_t mid = l + (r - l) / 2;
            if (indices[mid] == j) return data[mid];
            if (indices[mid] < j) l = mid + 1;
            else r = mid;
        }
        return T{};
    }

    // Set element at position (i,j), updating CSR structure as needed
    void set(size_t i, size_t j, const T &value) {
        if (i >= n || j >= m) throw invalid_index();
        size_t start = indptr[i];
        size_t end = indptr[i+1];
        size_t l = start, r = end;
        while (l < r) {
            size_t mid = l + (r - l) / 2;
            if (indices[mid] == j) {
                data[mid] = value;
                return;
            }
            if (indices[mid] < j) l = mid + 1;
            else r = mid;
        }
        
        if (value == T{}) return;
        
        indices.insert(indices.begin() + l, j);
        data.insert(data.begin() + l, value);
        for (size_t r_idx = i + 1; r_idx <= n; ++r_idx) {
            indptr[r_idx]++;
        }
        count++;
    }

    // Access CSR components
    const std::vector<size_t> &getIndptr() const { return indptr; }

    const std::vector<size_t> &getIndices() const { return indices; }

    const std::vector<T> &getData() const { return data; }

    // Convert to dense matrix format
    std::vector<std::vector<T>> getMatrix() const {
        std::vector<std::vector<T>> res(n, std::vector<T>(m, T{}));
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = indptr[i]; k < indptr[i+1]; ++k) {
                res[i][indices[k]] = data[k];
            }
        }
        return res;
    }

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (vec.size() != m) throw size_mismatch();
        std::vector<T> res(n, T{});
        for (size_t i = 0; i < n; ++i) {
            T sum = T{};
            for (size_t k = indptr[i]; k < indptr[i+1]; ++k) {
                sum = sum + data[k] * vec[indices[k]];
            }
            res[i] = sum;
        }
        return res;
    }

    // Row slicing
    CSRMatrix getRowSlice(size_t l, size_t r) const {
        if (l > r || r > n) throw invalid_index();
        size_t new_n = r - l;
        size_t new_count = indptr[r] - indptr[l];
        std::vector<size_t> new_indptr(new_n + 1, 0);
        
        for (size_t i = 0; i < new_n; ++i) {
            new_indptr[i+1] = indptr[l + i + 1] - indptr[l];
        }
        
        std::vector<size_t> new_indices(indices.begin() + indptr[l], indices.begin() + indptr[r]);
        std::vector<T> new_data(data.begin() + indptr[l], data.begin() + indptr[r]);
        
        return CSRMatrix(new_n, m, new_count, new_indptr, new_indices, new_data);
    }
};

}

#endif // CSR_MATRIX_HPP