#pragma once
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

// stride computation
inline std::vector<size_t> compute_contiguous_stride_from_shape(const std::vector<int> &shape) {
    std::vector<size_t> strides(shape.size());
    size_t temp_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = temp_stride;
        temp_stride *= shape[i];
    }

    return strides;
}

// numel computation
inline size_t shape_to_numel(const std::vector<int> &shape) {
    size_t numel =
        std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());

    return numel;
}

struct Coalesced {
    std::vector<int> shape;
    std::vector<std::vector<size_t>> strides;
};

// for each/iterator over
inline Coalesced coalesce(const std::vector<int> &shape,
                          const std::vector<std::vector<size_t>> &strides) {
    std::vector<int> temp_shape = shape;
    std::vector<std::vector<size_t>> out_strides = strides;

    for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
        bool valid_merge = true;

        for (const auto &stride : out_strides) {
            if (stride[k] != (stride[k + 1] * temp_shape[k + 1])) {
                valid_merge = false;
                break;
            }
        }

        if (valid_merge) {
            // do this, merge temp_shape
            temp_shape[k] *= temp_shape[k + 1];
            temp_shape.erase(temp_shape.begin() + (k + 1));

            for (auto &stride : out_strides) {
                stride[k] = stride[k + 1];
                stride.erase(stride.begin() + (k + 1));
            }
        }
    }

    return {.shape = temp_shape, .strides = out_strides};
}

template <typename F>
void for_each(const std::vector<int> &shape_param,
              const std::vector<std::vector<size_t>> &strides_param, F callback) {
    size_t numel = shape_to_numel(shape_param);
    if (numel == 0) {
        throw std::runtime_error("Invalid shape.");
    }

    Coalesced new_shape_strides = coalesce(shape_param, strides_param);
    std::vector<int> shape = new_shape_strides.shape;
    std::vector<std::vector<size_t>> strides = new_shape_strides.strides;

    int rank = static_cast<int>(shape.size());

    // last dim
    int run_len = shape[rank - 1];

    std::vector<size_t> inner_strides(strides.size());
    for (size_t j = 0; j < strides.size(); j++) {
        inner_strides[j] = strides[j][rank - 1];
    }

    std::vector<int> counts(rank - 1, 0);
    std::vector<size_t> offsets(strides.size(), 0);

    size_t num_runs = numel / run_len;

    // go through each element
    for (size_t i = 0; i < num_runs; i++) {
        // go through each stride
        for (size_t j = 0; j < strides.size(); j++) {
            offsets[j] = 0;
            // go through each layer to get to the right place
            for (size_t k = 0; k < counts.size(); k++) {
                offsets[j] += strides[j][k] * counts[k];
            }
        }

        callback(offsets, run_len, inner_strides);

        // advance
        for (int k = static_cast<int>(counts.size()) - 1; k >= 0; k--) {
            counts[k] += 1;
            if (counts[k] >= shape[k]) {
                counts[k] = 0;
            } else {
                break;
            }
        }
    }
}
