#include "primitives.h"
#include "autograd.h"
#include "shape.h"
#include "tensor.h"
#include <catch2/catch_message.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

// HELPERS

std::string shape_to_string(const std::vector<int> &s) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < s.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << s[i];
    }

    oss << "]";
    return oss.str();
}

void verify_compatible_operations(const std::shared_ptr<Tensor> &a,
                                  const std::shared_ptr<Tensor> &b, std::string op) {
    // check dimensions
    if (a->shape().size() != b->shape().size()) {
        throw std::runtime_error(
            std::format("[{}]: Not possible for Tensors with different number of dimensions.", op));
    }

    for (size_t i = 0; i < a->shape().size(); i++) {
        if (a->shape()[i] != b->shape()[i]) {
            throw std::runtime_error(
                std::format("[{}]: Not possible for Tensors with shape {} and {}", op,
                            shape_to_string(a->shape()), shape_to_string(b->shape())));
        }
    }

    // check dtypes
    if (a->dtype() != b->dtype()) {
        throw std::runtime_error(
            std::format("[{}]: Not possible for Tensors of different types.", op));
    }

    // check grad types
    if (a->grad_dtype() != b->grad_dtype()) {
        throw std::runtime_error(
            std::format("[{}]: Not possible for Tensors of different grad types.", op));
    }
}

void accumulate_grad_into(const std::shared_ptr<Tensor> &dst, const std::shared_ptr<Tensor> &src,
                          float scalar) {
    // check dimensions and shape
    if (dst->shape().size() != src->shape().size()) {
        throw std::runtime_error("Cannot accumulate Tensors with different number of dimensions.");
    }

    for (size_t i = 0; i < dst->shape().size(); i++) {
        if (dst->shape()[i] != src->shape()[i]) {
            throw std::runtime_error(std::format("Cannot accumulate Tensors with shape {} and {}",
                                                 shape_to_string(dst->shape()),
                                                 shape_to_string(src->shape())));
        }
    }

    elementwise(use_grad(dst), {use_grad(dst), use_grad(src)},
                [scalar](std::span<const float> v) { return v[0] + (scalar * v[1]); });
}

// FORWARD

std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    verify_compatible_operations(a, b, "Add");

    // create result c
    bool requires_grad = a->requires_grad() || b->requires_grad();
    std::shared_ptr<Tensor> c =
        std::make_shared<Tensor>(a->shape(), a->dtype(), requires_grad, a->grad_dtype());

    // compute c
    elementwise(use_data(c), {use_data(a), use_data(b)},
                [](std::span<const float> v) { return v[0] + v[1]; });

    // add to autograd
    if (requires_grad) {
        autograd().push([a, b, c]() { add_backward(a, b, c); });
    }

    // return ptr to tensor C
    return c;
}

std::shared_ptr<Tensor> subtract(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b) {
    verify_compatible_operations(a, b, "Subtract");

    // create result c
    bool requires_grad = a->requires_grad() || b->requires_grad();
    std::shared_ptr<Tensor> c =
        std::make_shared<Tensor>(a->shape(), a->dtype(), requires_grad, a->grad_dtype());

    // compute c
    elementwise(use_data(c), {use_data(a), use_data(b)},
                [](std::span<const float> v) { return v[0] - v[1]; });

    // add to autograd
    if (requires_grad) {
        autograd().push([a, b, c]() { subtract_backward(a, b, c); });
    }

    // return ptr to tensor C
    return c;
}

std::shared_ptr<Tensor> exponential(const std::shared_ptr<Tensor> &a) {
    // create result c
    std::shared_ptr<Tensor> c =
        std::make_shared<Tensor>(a->shape(), a->dtype(), a->requires_grad(), a->grad_dtype());

    // compute c
    elementwise(use_data(c), {use_data(a)},
                [](std::span<const float> v) { return std::exp(v[0]); });

    // add to autograd
    if (a->requires_grad()) {
        autograd().push([a, c]() { exponential_backward(a, c); });
    }

    // return ptr to tensor C
    return c;
}

std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor> &a) {
    // create result c
    std::shared_ptr<Tensor> c =
        std::make_shared<Tensor>(a->shape(), a->dtype(), a->requires_grad(), a->grad_dtype());

    // compute c
    elementwise(use_data(c), {use_data(a)},
                [](std::span<const float> v) { return std::sqrt(v[0]); });

    // add to autograd
    if (a->requires_grad()) {
        autograd().push([a, c]() { sqrt_backward(a, c); });
    }

    // return ptr to tensor C
    return c;
}

std::shared_ptr<Tensor> hadamard(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b) {
    verify_compatible_operations(a, b, "Hadamard");

    // create result c
    bool requires_grad = a->requires_grad() || b->requires_grad();
    std::shared_ptr<Tensor> c =
        std::make_shared<Tensor>(a->shape(), a->dtype(), requires_grad, a->grad_dtype());

    // compute c
    elementwise(use_data(c), {use_data(a), use_data(b)},
                [](std::span<const float> v) { return v[0] * v[1]; });

    // add to autograd
    if (requires_grad) {
        autograd().push([a, b, c]() { hadamard_backward(a, b, c); });
    }

    // return ptr to tensor C
    return c;
}

std::shared_ptr<Tensor> divide(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    verify_compatible_operations(a, b, "Divide");

    // create result c
    bool requires_grad = a->requires_grad() || b->requires_grad();
    std::shared_ptr<Tensor> c =
        std::make_shared<Tensor>(a->shape(), a->dtype(), requires_grad, a->grad_dtype());

    // compute c
    elementwise(use_data(c), {use_data(a), use_data(b)},
                [](std::span<const float> v) { return v[0] / v[1]; });

    // add to autograd
    if (requires_grad) {
        autograd().push([a, b, c]() { divide_backward(a, b, c); });
    }

    // return ptr to tensor C
    return c;
}

////////
//////// BACKWARD
////////
void add_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                  const std::shared_ptr<Tensor> &c) {
    if (p1->requires_grad()) {
        accumulate_grad_into(p1, c);
    }

    if (p2->requires_grad()) {
        accumulate_grad_into(p2, c);
    }
}

void subtract_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                       const std::shared_ptr<Tensor> &c) {
    if (p1->requires_grad()) {
        accumulate_grad_into(p1, c);
    }

    if (p2->requires_grad()) {
        accumulate_grad_into(p2, c, -1.0f);
    }
}

void exponential_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &c) {
    if (p1->requires_grad()) {
        elementwise(use_grad(p1), {use_grad(p1), use_grad(c), use_data(c)},
                    [](std::span<const float> v) { return v[0] + v[1] * v[2]; });
    }
}

void sqrt_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &c) {
    if (p1->requires_grad()) {
        elementwise(use_grad(p1), {use_grad(p1), use_grad(c), use_data(p1)},
                    [](std::span<const float> v) {
                        return v[0] + (v[1] * (1.0f / (2.0f * std::sqrt(v[2]))));
                    });
    }
}

void hadamard_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                       const std::shared_ptr<Tensor> &c) {
    if (p1->requires_grad()) {
        // p1 += dL / dc * p2->data()
        elementwise(use_grad(p1), {use_grad(p1), use_grad(c), use_data(p2)},
                    [](std::span<const float> v) { return v[0] + v[1] * v[2]; });
    }

    if (p2->requires_grad()) {
        elementwise(use_grad(p2), {use_grad(p2), use_grad(c), use_data(p1)},
                    [](std::span<const float> v) { return v[0] + v[1] * v[2]; });
    }
}

void divide_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                     const std::shared_ptr<Tensor> &c) {
    if (p1->requires_grad()) {
        // p1 += dL / dc * (1 / p2->data())
        elementwise(use_grad(p1), {use_grad(p1), use_grad(c), use_data(p2)},
                    [](std::span<const float> v) { return v[0] + v[1] / v[2]; });
    }

    if (p2->requires_grad()) {
        // p2 += dL / dc * (-p1->data() / (p2->data())^2)
        elementwise(use_grad(p2), {use_grad(p2), use_grad(c), use_data(p1), use_data(p2)},
                    [](std::span<const float> v) { return v[0] - v[1] * v[2] / (v[3] * v[3]); });
    }
}