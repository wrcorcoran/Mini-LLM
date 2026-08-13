#include "primitives.h"
#include "tensor.h"
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>

// BASIC ADD
TEST_CASE("Basic subtract.", "[primitives][subtract]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));

    auto *a_data = a->data_as<float>();
    auto *b_data = b->data_as<float>();
    a_data[0] = 0.0f;
    a_data[1] = 1.0f;
    a_data[2] = 2.0f;
    b_data[0] = 0.1f;
    b_data[1] = 0.2f;
    b_data[2] = 0.3f;

    std::shared_ptr<Tensor> c = subtract(a, b);
    auto *c_data = c->data_as<float>();

    REQUIRE(c_data[0] == -0.1f);
    REQUIRE(c_data[1] == 0.8f);
    REQUIRE(c_data[2] == 1.7f);
}

// TEST GRAD
TEST_CASE("Basic subtract backward.", "[primitives][subtract]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));

    auto *a_grad = a->grad_as<float>();
    auto *b_grad = b->grad_as<float>();

    std::shared_ptr<Tensor> c = subtract(a, b);
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 2.0f;
    c_grad[1] = 1.0f;
    c_grad[2] = 0.0f;

    subtract_backward(a, b, c);

    REQUIRE(c_grad[0] == a_grad[0]);
    REQUIRE(c_grad[1] == a_grad[1]);
    REQUIRE(c_grad[2] == a_grad[2]);

    REQUIRE(c_grad[0] == (-1.0f * b_grad[0]));
    REQUIRE(c_grad[1] == (-1.0f * b_grad[1]));
    REQUIRE(c_grad[2] == (-1.0f * b_grad[2]));
}