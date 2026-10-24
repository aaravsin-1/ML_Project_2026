#pragma once

#include <immintrin.h>
#include <array>
#include <cstdint>
#include <cmath>

namespace hft {

// Struct-of-Arrays (SoA) layout aligned to 32 bytes (AVX register width)
// Padded to 12 levels (multiple of 4) for seamless 256-bit SIMD loads
constexpr size_t SIMD_LOB_LEVELS = 12;

struct alignas(64) LimitOrderBookSoA {
    alignas(32) double bid_prices[SIMD_LOB_LEVELS]{0.0};
    alignas(32) double bid_quantities[SIMD_LOB_LEVELS]{0.0};
    alignas(32) double ask_prices[SIMD_LOB_LEVELS]{0.0};
    alignas(32) double ask_quantities[SIMD_LOB_LEVELS]{0.0};
    uint64_t timestamp_ns{0};

    inline double best_bid_price() const noexcept { return bid_prices[0]; }
    inline double best_ask_price() const noexcept { return ask_prices[0]; }
    inline double best_bid_qty() const noexcept { return bid_quantities[0]; }
    inline double best_ask_qty() const noexcept { return ask_quantities[0]; }

    inline double mid_price() const noexcept {
        return (bid_prices[0] + ask_prices[0]) * 0.5;
    }

    inline double spread() const noexcept {
        return ask_prices[0] - bid_prices[0];
    }
};

} // namespace hft
