#include "tensor.h"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

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

TEST_CASE("Testing transpose and is contiguous", "[tensor][ops]") {
    Tensor t({2, 3, 4, 5}, DTYPE::FP32, true);

    Tensor ct = t.transpose(2, 3);
    REQUIRE(!ct.is_contiguous());

    Tensor ct2 = ct.transpose(2, 3);
    REQUIRE(ct2.is_contiguous());
}

TEST_CASE("Testing transpose and is contiguous, multiple transpose", "[tensor][ops]") {
    Tensor t({2, 3, 4, 5}, DTYPE::FP32, true);
    REQUIRE(t.is_contiguous());

    Tensor ct = t.transpose(1, 3);
    REQUIRE(!ct.is_contiguous());

    Tensor ct2 = ct.transpose(2, 3);
    REQUIRE(!ct2.is_contiguous());

    Tensor ct3 = ct2.transpose(3, 2);
    REQUIRE(!ct3.is_contiguous());

    Tensor ct4 = ct3.transpose(3, 1);
    REQUIRE(ct4.is_contiguous());
}

TEST_CASE("Testing transpose and is contiguous, simple transpose", "[tensor][ops]") {
    Tensor t({1, 1}, DTYPE::FP32, true);

    Tensor ct = t.transpose(0, 1);
    REQUIRE(ct.is_contiguous());
    REQUIRE(ct.transpose(1, 0).is_contiguous());
}

// RESHAPE
TEST_CASE("Testing reshape", "[tensor][ops]") {
    Tensor t({10, 3}, DTYPE::FP32, true);

    Tensor ct = t.reshape({5, 6});
    REQUIRE(ct.shape() == std::vector<int>({5, 6}));

    Tensor ct2 = t.reshape({3, 5, 2});
    REQUIRE(ct2.shape() == std::vector<int>({3, 5, 2}));
}

TEST_CASE("Testing reshape, errors!", "[tensor][ops]") {
    Tensor t({10, 3}, DTYPE::FP32, true);

    REQUIRE_THROWS_AS(t.reshape({7, 4}), std::runtime_error);

    Tensor ct = t.transpose(0, 1);
    REQUIRE_THROWS_AS(ct.reshape({15, 2}), std::runtime_error);
}

// CONTIGUOUS
TEST_CASE("Testing contiguous, 1.", "[tensor][ops]") {
    Tensor t({2, 3}, DTYPE::FP32, true);

    Tensor c = t.contiguous();
    REQUIRE(c.data_as<float>() == t.data_as<float>()); // should be the same data object

    Tensor ct = t.transpose(0, 1);
    REQUIRE(ct.contiguous().data_as<float>() != t.data_as<float>()); // different object
}

TEST_CASE("Testing contiguous, 2.", "[tensor][ops]") {
    Tensor t({2, 3}, DTYPE::FP32, true);
    auto *p = t.grad_as<float>();
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;
    p[4] = 5.0f;
    p[5] = 6.0f;

    Tensor ct = t.transpose(0, 1);
    Tensor ct2 = ct.contiguous();
    auto *q = ct2.grad_as<float>();

    REQUIRE(q[0] == 1.0f);
    REQUIRE(q[1] == 4.0f);
    REQUIRE(q[2] == 2.0f);
    REQUIRE(q[3] == 5.0f);
    REQUIRE(q[4] == 3.0f);
    REQUIRE(q[5] == 6.0f);

    REQUIRE(!ct.is_contiguous());
    REQUIRE(ct2.is_contiguous());
}