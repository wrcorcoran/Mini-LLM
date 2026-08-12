#include "tensor.h"
#include <algorithm>
#include <cstring>
#include <memory>
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

    strides_ = compute_contiguous_stride_from_shape(shape_);

    // initialize data
    numel_ = shape_to_numel(shape_);
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

    numel_ = shape_to_numel(shape_);
}

// getters
const std::vector<int> &Tensor::shape() const { return shape_; }
const std::vector<size_t> &Tensor::strides() const { return strides_; }

bool Tensor::is_contiguous() const {
    size_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        if (shape_[i] != 1 && strides_[i] != expected) {
            return false;
        }
        expected *= static_cast<size_t>(shape_[i]);
    }
    return true;
}

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

    return {temp_shape, temp_strides, data_, grad_, dtype_, requires_grad_, grad_dtype_};
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return {shape_, strides_, data_, grad_, dtype_, requires_grad_, grad_dtype_};
    }

    // not contiguous
    std::shared_ptr<std::vector<std::byte>> new_data =
        std::make_shared<std::vector<std::byte>>(numel_ * byte_width(dtype_));
    std::shared_ptr<std::vector<std::byte>> new_grad;

    std::vector<size_t> new_strides = compute_contiguous_stride_from_shape(shape_);

    // need src and dst pointers
    const std::byte *src = data_->data();
    std::byte *dst = new_data->data();
    std::size_t width = byte_width(dtype_);

    // need to use for_each to make contiguous (data)
    for_each(shape_, {strides_, new_strides},
             [&](const std::vector<size_t> &offsets, int run_len,
                 const std::vector<size_t> &inner_strides) {
                 if (inner_strides[0] == 1 && inner_strides[1] == 1) {
                     std::memcpy(dst + offsets[1] * width, src + offsets[0] * width,
                                 run_len * width);
                 } else {
                     for (int i = 0; i < run_len; i++)
                         std::memcpy(dst + (offsets[1] + (i * inner_strides[1])) * width,
                                     src + (offsets[0] + (i * inner_strides[0])) * width, width);
                 }
             });

    if (requires_grad_) {
        new_grad = std::make_shared<std::vector<std::byte>>(numel_ * byte_width(grad_dtype_));
        const std::byte *src_grad = grad_->data();
        std::byte *dst_grad = new_grad->data();
        std::size_t width_grad = byte_width(grad_dtype_);

        for_each(shape_, {strides_, new_strides},
                 [&](const std::vector<size_t> &offsets, int run_len,
                     const std::vector<size_t> &inner_strides) {
                     if (inner_strides[0] == 1 && inner_strides[1] == 1) {
                         std::memcpy(dst_grad + offsets[1] * width_grad,
                                     src_grad + offsets[0] * width_grad, run_len * width_grad);
                     } else {
                         for (int i = 0; i < run_len; i++)
                             std::memcpy(
                                 dst_grad + (offsets[1] + (i * inner_strides[1])) * width_grad,
                                 src_grad + (offsets[0] + (i * inner_strides[0])) * width_grad,
                                 width_grad);
                     }
                 });
    }

    return {shape_, new_strides, new_data, new_grad, dtype_, requires_grad_, grad_dtype_};
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const {
    // check all dims are positive
    for (const auto &dim : new_shape) {
        if (dim <= 0) {
            throw std::out_of_range("Invalid dimensions for reshape. All must be positive.");
        }
    }

    // check if not possible
    size_t new_numel = shape_to_numel(new_shape);
    if (new_numel != numel_) {
        throw std::runtime_error(std::format(
            "Total number of elements do not match. Expected {}, received {}", numel_, new_numel));
    }

    // check if not contiguous
    if (!is_contiguous()) {
        throw std::runtime_error(
            "A non-contiguous tensor cannot be reshaped. Please convert to contiguous.");
    }

    // reshape
    return {new_shape,  compute_contiguous_stride_from_shape(new_shape),
            data_,      grad_,
            dtype_,     requires_grad_,
            grad_dtype_};
}