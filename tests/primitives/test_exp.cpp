#include "primitives.h"
#include "tensor.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>

// BASIC ADD
TEST_CASE("Basic exp.", "[primitives][exp]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));
    auto *a_data = a->data_as<float>();

    a_data[0] = 0.0f;
    a_data[1] = 1.0f;
    a_data[2] = 2.0f;

    std::shared_ptr<Tensor> c = exponential(a);
    auto *c_data = c->data_as<float>();

    REQUIRE(c_data[0] == Catch::Approx(1.0f));
    REQUIRE(c_data[1] == Catch::Approx(std::exp(1.0f)));
    REQUIRE(c_data[2] == Catch::Approx(std::exp(2.0f)));
}

// TEST GRAD
TEST_CASE("Basic exp backwards.", "[primitives][exp]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    auto *a_data = a->data_as<float>();
    auto *a_grad = a->grad_as<float>();

    a_data[0] = 0.0f;
    a_data[1] = 1.0f;
    a_data[2] = 2.0f;

    std::shared_ptr<Tensor> c = exponential(a);
    auto *c_data = c->data_as<float>();
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 2.0f;
    c_grad[1] = 1.0f;
    c_grad[2] = 0.0f;

    exponential_backward(a, c);

    REQUIRE(a_grad[0] == Catch::Approx(2.0f));
    REQUIRE(a_grad[1] == Catch::Approx(std::exp(1.0f)));
    REQUIRE(a_grad[2] == Catch::Approx(0.0f));
}