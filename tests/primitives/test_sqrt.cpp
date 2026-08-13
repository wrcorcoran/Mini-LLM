#include "primitives.h"
#include "tensor.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>

// BASIC ADD
TEST_CASE("Basic sqrt.", "[primitives][sqrt]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));
    auto *a_data = a->data_as<float>();

    a_data[0] = 0.0f;
    a_data[1] = 1.0f;
    a_data[2] = 4.0f;

    std::shared_ptr<Tensor> c = sqrt(a);
    auto *c_data = c->data_as<float>();

    REQUIRE(c_data[0] == Catch::Approx(0.0f));
    REQUIRE(c_data[1] == Catch::Approx(1.0f));
    REQUIRE(c_data[2] == Catch::Approx(2.0f));
}

// TEST GRAD
TEST_CASE("Basic sqrt backwards.", "[primitives][sqrt]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    auto *a_data = a->data_as<float>();
    auto *a_grad = a->grad_as<float>();

    a_data[0] = 4.0f;
    a_data[1] = 9.0f;
    a_data[2] = 16.0f;

    std::shared_ptr<Tensor> c = sqrt(a);
    auto *c_data = c->data_as<float>();
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 576.0f;
    c_grad[1] = 576.0f;
    c_grad[2] = 576.0f;

    sqrt_backward(a, c);

    REQUIRE(a_grad[0] == Catch::Approx(144.0f));
    REQUIRE(a_grad[1] == Catch::Approx(96.0f));
    REQUIRE(a_grad[2] == Catch::Approx(72.0f));
}