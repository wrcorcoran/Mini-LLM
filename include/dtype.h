#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

enum class DTYPE : std::uint8_t { FP32, INT8 };

// returns the bytewidth of each datatype
constexpr std::size_t byte_width(DTYPE dtype) {
    switch (dtype) {
    case DTYPE::FP32:
        return 4;
    case DTYPE::INT8:
        return 1;
    }
    throw std::invalid_argument("Unsupported dtype.");
}

// used for type conversions between cpp and our enum
template <typename T> constexpr DTYPE dtype_of();
template <> constexpr DTYPE dtype_of<float>() { return DTYPE::FP32; }
template <> constexpr DTYPE dtype_of<std::int8_t>() { return DTYPE::INT8; }
