#include "primitives.h"
#include "tensor.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>

// BASIC ADD
TEST_CASE("Basic hadamard.", "[primitives][hadamard]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));
    auto *a_data = a->data_as<float>();
    auto *b_data = b->data_as<float>();

    a_data[0] = 0.0f;
    a_data[1] = 1.0f;
    a_data[2] = 2.0f;
    b_data[0] = 0.0f;
    b_data[1] = 1.0f;
    b_data[2] = 2.0f;

    std::shared_ptr<Tensor> c = hadamard(a, b);
    auto *c_data = c->data_as<float>();

    REQUIRE(c_data[0] == 0.0f);
    REQUIRE(c_data[1] == 1.0f);
    REQUIRE(c_data[2] == 4.0f);
}

// TEST GRAD
TEST_CASE("Basic hadamard backwards.", "[primitives][hadamard]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    auto *a_data = a->data_as<float>();
    auto *b_data = b->data_as<float>();
    auto *a_grad = a->grad_as<float>();
    auto *b_grad = b->grad_as<float>();

    a_data[0] = 0.0f;
    a_data[1] = 1.0f;
    a_data[2] = 2.0f;
    b_data[0] = 10.0f;
    b_data[1] = 20.0f;
    b_data[2] = 30.0f;

    std::shared_ptr<Tensor> c = hadamard(a, b);
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 3.0f;
    c_grad[1] = 3.0f;
    c_grad[2] = 3.0f;

    hadamard_backward(a, b, c);

    REQUIRE(a_grad[0] == Catch::Approx(30.0f));
    REQUIRE(a_grad[1] == Catch::Approx(60.0f));
    REQUIRE(a_grad[2] == Catch::Approx(90.0f));
    REQUIRE(b_grad[0] == Catch::Approx(0.0f));
    REQUIRE(b_grad[1] == Catch::Approx(3.0f));
    REQUIRE(b_grad[2] == Catch::Approx(6.0f));
}