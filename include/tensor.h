#pragma once
#include "dtype.h"
#include "shape.h"
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <vector>

class Tensor {
  public:
    Tensor(std::vector<int> shape, DTYPE dtype = DTYPE::FP32, bool requires_grad = false,
           DTYPE grad_dtype = DTYPE::FP32);

    // delete copy constructor
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    // add move back
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    // getters
    [[nodiscard]] const std::vector<int> &shape() const;
    [[nodiscard]] const std::vector<size_t> &strides() const;
    [[nodiscard]] bool is_contiguous() const;

    template <typename T> [[nodiscard]] T *data_as();
    template <typename T> [[nodiscard]] const T *data_as() const;
    template <typename T> [[nodiscard]] T *grad_as();
    template <typename T> [[nodiscard]] const T *grad_as() const;

    [[nodiscard]] DTYPE dtype() const;
    [[nodiscard]] DTYPE grad_dtype() const;
    [[nodiscard]] bool requires_grad() const;
    [[nodiscard]] std::size_t numel() const;

    // ops
    void zero_grad();
    [[nodiscard]] Tensor transpose(int dim0, int dim1) const;
    [[nodiscard]] Tensor contiguous() const;
    [[nodiscard]] Tensor reshape(const std::vector<int> &new_shape) const;

  private:
    // constructor for transposes, views, contiguous, etc.
    Tensor(std::vector<int> shape, std::vector<size_t> strides,
           std::shared_ptr<std::vector<std::byte>> data,
           std::shared_ptr<std::vector<std::byte>> grad, DTYPE dtype, bool requires_grad,
           DTYPE grad_dtype);

    // shape info
    std::vector<int> shape_;      // keeps shape per dimension
    std::vector<size_t> strides_; // keeps stride per dim for data management

    // data
    std::shared_ptr<std::vector<std::byte>> data_;
    std::shared_ptr<std::vector<std::byte>> grad_;

    // metadata
    DTYPE dtype_;
    DTYPE grad_dtype_;

    size_t numel_;

    bool requires_grad_;
};

/*
 * Casts data from bytes to the expected data type.
 */
template <typename T> T *Tensor::data_as() {
    if (dtype_of<T>() != dtype_)
        throw std::runtime_error(std::format(
            "Expected {}, received {}", static_cast<int>(dtype_of<T>()), static_cast<int>(dtype_)));
    return reinterpret_cast<T *>(data_->data());
}

template <typename T> const T *Tensor::data_as() const {
    if (dtype_of<T>() != dtype_)
        throw std::runtime_error(std::format(
            "Expected {}, received {}", static_cast<int>(dtype_of<T>()), static_cast<int>(dtype_)));
    return reinterpret_cast<const T *>(data_->data());
}

template <typename T> T *Tensor::grad_as() {
    if (!grad_)
        throw std::runtime_error(
            std::format("Gradient is null. Requires grad is {}", requires_grad_));

    if (dtype_of<T>() != grad_dtype_)
        throw std::runtime_error(std::format("Expected {}, received {}",
                                             static_cast<int>(dtype_of<T>()),
                                             static_cast<int>(grad_dtype_)));
    return reinterpret_cast<T *>(grad_->data());
}

template <typename T> const T *Tensor::grad_as() const {
    if (!grad_)
        throw std::runtime_error(
            std::format("Gradient is null. Requires grad is {}", requires_grad_));

    if (dtype_of<T>() != grad_dtype_)
        throw std::runtime_error(std::format("Expected {}, received {}",
                                             static_cast<int>(dtype_of<T>()),
                                             static_cast<int>(grad_dtype_)));
    return reinterpret_cast<const T *>(grad_->data());
}