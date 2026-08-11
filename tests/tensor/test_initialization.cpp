#include "tensor.h"
#include <catch2/catch_test_macros.hpp>

// FIELDS
TEST_CASE("Shapes and strides computed correctly, 3d", "[tensor][init]") {
    Tensor t({2, 3, 4}, DTYPE::FP32);

    REQUIRE(t.strides() == std::vector<size_t>{12, 4, 1});
    REQUIRE(t.shape() == std::vector<int>{2, 3, 4});
    REQUIRE(t.numel() == 24);
}

TEST_CASE("Strides computed correctly, 4d", "[tensor][init]") {
    Tensor t({2, 3, 4, 4}, DTYPE::FP32);

    REQUIRE(t.shape() == std::vector<int>{2, 3, 4, 4});
    REQUIRE(t.strides() == std::vector<size_t>{48, 16, 4, 1});
    REQUIRE(t.numel() == 96);
}

TEST_CASE("Initialized fields using defaults", "[tensor][init]") {
    Tensor t({1, 1}, DTYPE::FP32);
    REQUIRE(t.dtype() == DTYPE::FP32);
    REQUIRE(t.grad_dtype() == DTYPE::FP32);
    REQUIRE(t.requires_grad() == false);
    REQUIRE(t.numel() == 1);
}

TEST_CASE("Initialized fields without defaults", "[tensor][init]") {
    Tensor t({20, 10}, DTYPE::INT8, true, DTYPE::INT8);
    REQUIRE(t.dtype() == DTYPE::INT8);
    REQUIRE(t.grad_dtype() == DTYPE::INT8);
    REQUIRE(t.requires_grad() == true);
    REQUIRE(t.numel() == 200);
}

// DATA
TEST_CASE("Testing data allocation with overwrite", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32);
    float *p = t.data_as<float>();
    p[0] = 1.2f;
    p[1] = -1.0f;
    p[2] = 0.0f;

    REQUIRE(t.data_as<float>()[0] == 1.2f);
    REQUIRE(t.data_as<float>()[1] == -1.0f);
    REQUIRE(t.data_as<float>()[2] == 0.0f);
}

TEST_CASE("Testing data allocation with just 0", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32);
    for (int i = 0; i < t.numel(); i++) {
        REQUIRE(t.data_as<float>()[i] == 0.0);
    }
}

TEST_CASE("Testing calling INT on FP throw", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32);
    REQUIRE_THROWS_AS(t.data_as<std::int8_t>(), std::runtime_error);
}

TEST_CASE("Testing const on data call", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32);
    const Tensor &ct = t;
    const float *p = ct.data_as<float>();
    static_assert(std::is_same_v<decltype(ct.data_as<float>()), const float *>);
}

// GRAD
TEST_CASE("Testing grad allocation with overwrite", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32, true);
    float *p = t.grad_as<float>();
    p[0] = 1.2f;
    p[1] = -1.0f;
    p[2] = 0.0f;

    REQUIRE(t.grad_as<float>()[0] == 1.2f);
    REQUIRE(t.grad_as<float>()[1] == -1.0f);
    REQUIRE(t.grad_as<float>()[2] == 0.0f);
}

TEST_CASE("Testing grad allocation with just 0", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32, true);
    for (int i = 0; i < t.numel(); i++) {
        REQUIRE(t.grad_as<float>()[i] == 0.0);
    }
}

TEST_CASE("Testing calling grad INT on FP throw", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32, true);
    REQUIRE_THROWS_AS(t.grad_as<std::int8_t>(), std::runtime_error);
}

TEST_CASE("Testing calling grad on no requires grad", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32, false);
    REQUIRE_THROWS_AS(t.grad_as<float>(), std::runtime_error);
}

TEST_CASE("Testing const on grad call", "[tensor][init]") {
    Tensor t({3}, DTYPE::FP32, true);
    const Tensor &ct = t;
    const float *p = ct.grad_as<float>();
    static_assert(std::is_same_v<decltype(ct.grad_as<float>()), const float *>);
}