#include "tensor.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

Tensor::Tensor(std::vector<int> shape, DTYPE dtype, bool requires_grad, DTYPE grad_dtype)
    : shape_(std::move(shape)), dtype_(dtype), grad_dtype_(grad_dtype),
      requires_grad_(requires_grad) {
    // check shape dimensions
    for (int &elem : shape_) {
        if (elem <= 0)
            throw std::out_of_range(std::format("Attempting to add dimension of size {}", elem));
    }

    // build strides
    strides_.resize(shape_.size());
    size_t temp_stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        strides_[i] = temp_stride;
        temp_stride *= shape_[i];
    }

    // initialize data
    numel_ = std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                             std::multiplies<std::size_t>());
    data_ = std::make_shared<std::vector<std::byte>>(numel_ * byte_width(dtype_));

    if (requires_grad_) {
        grad_ = std::make_shared<std::vector<std::byte>>(numel_ * byte_width(grad_dtype_));
    }
}

Tensor::Tensor(std::vector<int> shape, std::vector<size_t> strides,
               std::shared_ptr<std::vector<std::byte>> data,
               std::shared_ptr<std::vector<std::byte>> grad, DTYPE dtype, bool requires_grad,
               DTYPE grad_dtype)
    : shape_(std::move(shape)), strides_(std::move(strides)), data_(std::move(data)),
      grad_(std::move(grad)), dtype_(dtype), grad_dtype_(grad_dtype),
      requires_grad_(requires_grad) {
    if (static_cast<bool>(grad_) != requires_grad)
        throw std::runtime_error("Grad and requires grad must be consistent.");

    numel_ = std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                             std::multiplies<std::size_t>());
}

// getters
const std::vector<int> &Tensor::shape() const { return shape_; }
const std::vector<size_t> &Tensor::strides() const { return strides_; }

DTYPE Tensor::dtype() const { return dtype_; }
DTYPE Tensor::grad_dtype() const { return grad_dtype_; }
bool Tensor::requires_grad() const { return requires_grad_; }
std::size_t Tensor::numel() const { return numel_; }

// ops
void Tensor::zero_grad() {
    if (!requires_grad_) {
        throw std::runtime_error("Called zero_grad() on a tensor that does not support gradients.");
    } else
        std::fill(grad_->begin(), grad_->end(), std::byte{0});
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    int rank = static_cast<int>(shape_.size());
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
        throw std::out_of_range(std::format("Dim out of range for rank {}", rank));
    }

    std::vector<int> temp_shape = shape_;
    std::swap(temp_shape[dim0], temp_shape[dim1]);
    std::vector<size_t> temp_strides = strides_;
    std::swap(temp_strides[dim0], temp_strides[dim1]);

    return Tensor(temp_shape, temp_strides, data_, grad_, dtype_, requires_grad_, grad_dtype_);
}