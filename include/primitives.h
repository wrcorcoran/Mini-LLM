#pragma once
#include "tensor.h"
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// enum to show which kind of buffer to use
constexpr size_t MAX_ELEMENTWISE_ARGS = 4;

enum class Buf : std::uint8_t { Data, Grad };

struct Arg {
    std::shared_ptr<Tensor> tensor;
    Buf buf;
};

inline Arg use_grad(std::shared_ptr<Tensor> t) {
    return {.tensor = std::move(t), .buf = Buf::Grad};
}
inline Arg use_data(std::shared_ptr<Tensor> t) {
    return {.tensor = std::move(t), .buf = Buf::Data};
}

inline float *buf_ptr(const std::shared_ptr<Tensor> &t, Buf b) {
    return (b == Buf::Data) ? t->data_as<float>() : t->grad_as<float>();
}

inline DTYPE buf_dtype(const std::shared_ptr<Tensor> &t, Buf b) {
    return (b == Buf::Data) ? t->dtype() : t->grad_dtype();
}

// helpers
std::string shape_to_string(const std::vector<int> &s);
void verify_compatible_operations(const std::shared_ptr<Tensor> &a,
                                  const std::shared_ptr<Tensor> &b, std::string op);

template <typename Op> void elementwise(const Arg &dst, const std::vector<Arg> &srcs, Op op);

void accumulate_grad_into(const std::shared_ptr<Tensor> &dst, const std::shared_ptr<Tensor> &src,
                          float scalar = 1.0f);

// IMPLEMENTATIONS

template <typename Op> void elementwise(const Arg &dst, const std::vector<Arg> &srcs, Op op) {
    if (srcs.size() > MAX_ELEMENTWISE_ARGS) {
        throw std::runtime_error("Elementwise can only handle up to 4 args.");
    }

    for (auto &elem : srcs) {
        assert(buf_dtype(elem.tensor, elem.buf) == buf_dtype(dst.tensor, dst.buf) &&
               "elementwise: src dtype mismatch with dst");
    }

    switch (buf_dtype(dst.tensor, dst.buf)) {
    case DTYPE::FP32: {
        auto *dst_ptr = buf_ptr(dst.tensor, dst.buf);
        std::vector<float *> src_ptrs;
        std::vector<std::vector<size_t>> all_strides;
        for (const auto &s : srcs) {
            src_ptrs.push_back(buf_ptr(s.tensor, s.buf));
            all_strides.push_back(s.tensor->strides());
        }
        all_strides.push_back(dst.tensor->strides());

        std::array<float, MAX_ELEMENTWISE_ARGS> vals;

        for_each(dst.tensor->shape(), all_strides,
                 [&](const std::vector<size_t> &offsets, int run_len,
                     const std::vector<size_t> &inner_strides) {
                     for (int i = 0; i < run_len; i++) {
                         for (size_t j = 0; j < srcs.size(); j++)
                             vals[j] = src_ptrs[j][offsets[j] + i * inner_strides[j]];
                         dst_ptr[offsets[srcs.size()] + (i * inner_strides[srcs.size()])] =
                             op(std::span<const float>(vals.data(), srcs.size()));
                     }
                 });
        break;
    }
    default:
        throw std::logic_error("Unimplemented data type for elementwise operation.");
    }
}
//////
// ###################################
/////
// forward
std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);
std::shared_ptr<Tensor> subtract(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b);
std::shared_ptr<Tensor> hadamard(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b);
std::shared_ptr<Tensor> divide(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);

std::shared_ptr<Tensor> exponential(const std::shared_ptr<Tensor> &a);
std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor> &a);

// backward
void add_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                  const std::shared_ptr<Tensor> &c);
void subtract_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                       const std::shared_ptr<Tensor> &c);
void hadamard_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                       const std::shared_ptr<Tensor> &c);
void divide_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &p2,
                     const std::shared_ptr<Tensor> &c);
void exponential_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &c);
void sqrt_backward(const std::shared_ptr<Tensor> &p1, const std::shared_ptr<Tensor> &c);
