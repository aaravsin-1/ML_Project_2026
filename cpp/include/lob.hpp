#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace hft {

constexpr size_t MAX_LOB_LEVELS = 10;

struct alignas(16) Level {
    double price{0.0};
    double quantity{0.0};
};

struct alignas(64) LimitOrderBook {
    std::array<Level, MAX_LOB_LEVELS> bids{};
    std::array<Level, MAX_LOB_LEVELS> asks{};
    uint64_t timestamp_ns{0};
    size_t active_levels{MAX_LOB_LEVELS};

    inline double best_bid_price() const noexcept { return bids[0].price; }
    inline double best_ask_price() const noexcept { return asks[0].price; }
    inline double best_bid_qty() const noexcept { return bids[0].quantity; }
    inline double best_ask_qty() const noexcept { return asks[0].quantity; }

    inline double mid_price() const noexcept {
        return (bids[0].price + asks[0].price) * 0.5;
    }

    inline double spread() const noexcept {
        return asks[0].price - bids[0].price;
    }

    inline double micro_price() const noexcept {
        const double qb = bids[0].quantity;
        const double qa = asks[0].quantity;
        const double denom = qb + qa;
        if (denom <= 1e-9) return mid_price();
        return (qb * asks[0].price + qa * bids[0].price) / denom;
    }

    inline double top_imbalance() const noexcept {
        const double qb = bids[0].quantity;
        const double qa = asks[0].quantity;
        const double denom = qb + qa;
        return denom > 1e-9 ? (qb - qa) / denom : 0.0;
    }
};

} // namespace hft
