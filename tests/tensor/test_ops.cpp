#include "tensor.h"
#include <catch2/catch_test_macros.hpp>

// ZERO GRAD
TEST_CASE("Testing grad zeroing", "[tensor][ops]") {
    Tensor t({3}, DTYPE::FP32, true);
    float *p = t.grad_as<float>();
    p[0] = 1.2f;
    p[1] = -1.0f;
    p[2] = 0.0f;

    REQUIRE(t.grad_as<float>()[0] == 1.2f);
    REQUIRE(t.grad_as<float>()[1] == -1.0f);
    REQUIRE(t.grad_as<float>()[2] == 0.0f);

    for (int i = 0; i < t.numel(); i++) {
        REQUIRE(t.grad_as<float>()[i] == p[i]);
    }

    t.zero_grad();
    for (int i = 0; i < t.numel(); i++) {
        REQUIRE(t.grad_as<float>()[i] == 0.0f);
    }
}

// TRANSPOSE
TEST_CASE("Testing transpose", "[tensor][ops]") {
    Tensor t({1, 3}, DTYPE::FP32);

    Tensor ct = t.transpose(0, 1);
    REQUIRE(ct.shape() == std::vector<int>{3, 1});
    REQUIRE(ct.strides() == std::vector<size_t>{1, 3});

    Tensor ct2 = ct.transpose(0, 1);
    REQUIRE(ct2.shape() == t.shape());
    REQUIRE(ct2.strides() == t.strides());
}

// TRANSPOSE
TEST_CASE("Testing transpose on 4d and other", "[tensor][ops]") {
    Tensor t({2, 3, 4, 5}, DTYPE::FP32, true);

    Tensor ct = t.transpose(1, 2);
    REQUIRE(ct.shape() == std::vector<int>{2, 4, 3, 5});
    REQUIRE(ct.strides() == std::vector<size_t>{60, 5, 20, 1});

    REQUIRE_THROWS_AS(t.transpose(0, 4), std::out_of_range);

    Tensor ct2 = t.transpose(0, 3);
    REQUIRE(ct2.shape() == std::vector<int>{5, 3, 4, 2});
    REQUIRE(ct2.strides() == std::vector<size_t>{1, 20, 5, 60});

    REQUIRE(ct2.numel() == t.numel());

    Tensor ct3 = t.transpose(1, 1);
    REQUIRE(ct3.shape() == t.shape());
    REQUIRE(ct3.strides() == t.strides());
}