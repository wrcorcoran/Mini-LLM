#include "autograd.h"
#include "primitives.h"
#include "tensor.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>

// BASIC ADD
TEST_CASE("Test autograd 1.", "[autograd]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({2, 1}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({2, 1}, DTYPE::FP32, true));

    // a = [0, 1]
    auto *a_data = a->data_as<float>();
    a_data[0] = 0.0f;
    a_data[1] = 1.0f;

    // b = [2, 3]
    auto *b_data = b->data_as<float>();
    b_data[0] = 2.0f;
    b_data[1] = 3.0f;

    // forward
    std::shared_ptr<Tensor> c = hadamard(exponential(a), b);
    auto *c_data = c->data_as<float>();

    REQUIRE(c_data[0] == Catch::Approx(2.0f));
    REQUIRE(c_data[1] == Catch::Approx(3.0f * std::exp(1.0f)));

    // dL / dc = [1, 1]
    auto *c_grad = c->grad_as<float>();
    c_grad[0] = 1.0f;
    c_grad[1] = 1.0f;

    autograd().backward();

    // test autograd
    auto *a_grad = a->grad_as<float>();
    auto *b_grad = b->grad_as<float>();

    REQUIRE(a_grad[0] == Catch::Approx(2.0f));
    REQUIRE(a_grad[1] == Catch::Approx(3.0f * std::exp(1.0f)));
    REQUIRE(b_grad[0] == Catch::Approx(1.0f));
    REQUIRE(b_grad[1] == Catch::Approx(std::exp(1.0f)));
}

// Test case provided by Claude!
TEST_CASE("Test autograd 2 — all primitives, multi-path accumulation.", "[autograd]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<int>{2, 1}, DTYPE::FP32, true);
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<int>{2, 1}, DTYPE::FP32, true);

    auto *a_data = a->data_as<float>();
    a_data[0] = 4.0f;
    a_data[1] = 9.0f;

    auto *b_data = b->data_as<float>();
    b_data[0] = 1.0f;
    b_data[1] = 2.0f;

    // c = (sqrt(a * b) + exp(b)) / (a - b)
    std::shared_ptr<Tensor> c = divide(add(sqrt(hadamard(a, b)), exponential(b)), subtract(a, b));

    // forward check
    for (int i = 0; i < 2; i++) {
        const float av = a_data[i];
        const float bv = b_data[i];
        const float expected = (std::sqrt(av * bv) + std::exp(bv)) / (av - bv);
        REQUIRE(c->data_as<float>()[i] == Catch::Approx(expected));
    }

    // seed dL / dc = 1
    auto *c_grad = c->grad_as<float>();
    c_grad[0] = 1.0f;
    c_grad[1] = 1.0f;

    autograd().backward();

    auto *a_grad = a->grad_as<float>();
    auto *b_grad = b->grad_as<float>();

    for (int i = 0; i < 2; i++) {
        const float av = a_data[i];
        const float bv = b_data[i];

        const float u = av * bv;      // hadamard
        const float v = std::sqrt(u); // sqrt
        const float w = std::exp(bv); // exponential
        const float x = v + w;        // add
        const float y = av - bv;      // subtract

        const float da = (1.0f / y) * (bv / (2.0f * v)) - x / (y * y);

        const float db = (1.0f / y) * (av / (2.0f * v) + w) + x / (y * y);

        REQUIRE(a_grad[i] == Catch::Approx(da));
        REQUIRE(b_grad[i] == Catch::Approx(db));
    }
}