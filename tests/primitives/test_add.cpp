#include "primitives.h"
#include "tensor.h"
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>

// BASIC ADD
TEST_CASE("Basic add.", "[primitives][add]") {
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

    std::shared_ptr<Tensor> c = add(a, b);
    auto *c_data = c->data_as<float>();

    REQUIRE(c_data[0] == 0.1f);
    REQUIRE(c_data[1] == 1.2f);
    REQUIRE(c_data[2] == 2.3f);
}

TEST_CASE("Add -- throws shapes don't match", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32));

    REQUIRE_THROWS_AS(add(a, b), std::runtime_error);
}

TEST_CASE("Add -- throws dimensions don't match", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1, 2}, DTYPE::FP32));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32));

    REQUIRE_THROWS_AS(add(a, b), std::runtime_error);
}

TEST_CASE("Add -- throws data types don't match", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::INT8));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32));

    REQUIRE_THROWS_AS(add(a, b), std::runtime_error);
}

TEST_CASE("Add -- throws grad types don't match", "[primitives][add]") {
    std::shared_ptr<Tensor> a =
        std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true, DTYPE::FP32));
    std::shared_ptr<Tensor> b =
        std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true, DTYPE::INT8));

    REQUIRE_THROWS_AS(add(a, b), std::runtime_error);
}

TEST_CASE("Add -- check requires grad", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true));

    std::shared_ptr<Tensor> c = add(a, b);
    REQUIRE(c->requires_grad());

    std::shared_ptr<Tensor> a1 = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b1 = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32));

    std::shared_ptr<Tensor> c1 = add(a1, b1);
    REQUIRE(c1->requires_grad());

    std::shared_ptr<Tensor> a2 = std::make_shared<Tensor>(Tensor({3, 2}));
    std::shared_ptr<Tensor> b2 = std::make_shared<Tensor>(Tensor({3, 2}));

    std::shared_ptr<Tensor> c2 = add(a2, b2);
    REQUIRE(!c2->requires_grad());
}

// TEST GRAD
TEST_CASE("Basic add backward.", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));

    auto *a_grad = a->grad_as<float>();
    auto *b_grad = b->grad_as<float>();

    std::shared_ptr<Tensor> c = add(a, b);
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 2.0f;
    c_grad[1] = 1.0f;
    c_grad[2] = 0.0f;

    add_backward(a, b, c);

    REQUIRE(c_grad[0] == a_grad[0]);
    REQUIRE(c_grad[1] == a_grad[1]);
    REQUIRE(c_grad[2] == a_grad[2]);

    REQUIRE(c_grad[0] == b_grad[0]);
    REQUIRE(c_grad[1] == b_grad[1]);
    REQUIRE(c_grad[2] == b_grad[2]);
}

TEST_CASE("Basic add backward, including leaf node.", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 1}, DTYPE::FP32, false));

    auto *a_grad = a->grad_as<float>();

    std::shared_ptr<Tensor> c = add(a, b);
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 2.0f;
    c_grad[1] = 1.0f;
    c_grad[2] = 0.0f;

    add_backward(a, b, c);

    REQUIRE(c_grad[0] == a_grad[0]);
    REQUIRE(c_grad[1] == a_grad[1]);
    REQUIRE(c_grad[2] == a_grad[2]);

    REQUIRE_THROWS(b->grad_as<float>()[0]);
    REQUIRE_THROWS(b->grad_as<float>()[1]);
    REQUIRE_THROWS(b->grad_as<float>()[2]);
}

TEST_CASE("Basic add backward, larger shape.", "[primitives][add]") {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true));
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(Tensor({3, 2}, DTYPE::FP32, true));

    auto *a_grad = a->grad_as<float>();
    auto *b_grad = b->grad_as<float>();

    std::shared_ptr<Tensor> c = add(a, b);
    auto *c_grad = c->grad_as<float>();

    c_grad[0] = 2.0f;
    c_grad[1] = 1.0f;
    c_grad[2] = 0.0f;
    c_grad[3] = 2.0f;
    c_grad[4] = 1.0f;
    c_grad[5] = 0.0f;

    add_backward(a, b, c);

    REQUIRE(c_grad[0] == a_grad[0]);
    REQUIRE(c_grad[1] == a_grad[1]);
    REQUIRE(c_grad[2] == a_grad[2]);
    REQUIRE(c_grad[3] == a_grad[3]);
    REQUIRE(c_grad[4] == a_grad[4]);
    REQUIRE(c_grad[5] == a_grad[5]);

    REQUIRE(c_grad[0] == b_grad[0]);
    REQUIRE(c_grad[1] == b_grad[1]);
    REQUIRE(c_grad[2] == b_grad[2]);
    REQUIRE(c_grad[3] == b_grad[3]);
    REQUIRE(c_grad[4] == b_grad[4]);
    REQUIRE(c_grad[5] == b_grad[5]);
}