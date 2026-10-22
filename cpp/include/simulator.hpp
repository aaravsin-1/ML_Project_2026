#pragma once

#include "lob.hpp"
#include <vector>
#include <cstdint>

namespace hft {

struct ExecutionResult {
    uint64_t timestamp_ns;
    int side;            // +1 Buy, -1 Sell
    double fill_price;
    double slippage;
    double fee;
    bool adverse_selected;
};

class MicrosecondSimulator {
private:
    double cash_{1000000.0};
    double position_{0.0};
    double taker_fee_bps_{1.5};
    double maker_fee_bps_{-0.5};
    double adverse_penalty_bps_{2.0};
    double max_inventory_{500.0};

public:
    MicrosecondSimulator(double initial_cash = 1000000.0)
        : cash_(initial_cash) {}

    void reset() {
        cash_ = 1000000.0;
        position_ = 0.0;
    }

    ExecutionResult execute_aggressive(int side, double size, const LimitOrderBook& book) noexcept {
        ExecutionResult res{};
        res.timestamp_ns = book.timestamp_ns;
        res.side = side;

        const double mid = book.mid_price();
        double remaining = size;
        double cost = 0.0;

        if (side > 0) {
            // Sweep asks
            for (size_t i = 0; i < MAX_LOB_LEVELS; ++i) {
                double take = std::min(remaining, book.asks[i].quantity);
                cost += take * book.asks[i].price;
                remaining -= take;
                if (remaining <= 0.0) break;
            }
            if (remaining > 0.0) cost += remaining * (book.asks[MAX_LOB_LEVELS - 1].price + 0.05);
            res.fill_price = cost / size;
            res.fee = res.fill_price * size * (taker_fee_bps_ / 10000.0);
            res.slippage = (res.fill_price - mid) * size;

            cash_ -= (cost + res.fee);
            position_ += size;
        } else {
            // Hit bids
            for (size_t i = 0; i < MAX_LOB_LEVELS; ++i) {
                double take = std::min(remaining, book.bids[i].quantity);
                cost += take * book.bids[i].price;
                remaining -= take;
                if (remaining <= 0.0) break;
            }
            if (remaining > 0.0) cost += remaining * std::max(0.01, book.bids[MAX_LOB_LEVELS - 1].price - 0.05);
            res.fill_price = cost / size;
            res.fee = res.fill_price * size * (taker_fee_bps_ / 10000.0);
            res.slippage = (mid - res.fill_price) * size;

            cash_ += (cost - res.fee);
            position_ -= size;
        }

        return res;
    }

    inline double equity(double current_mid) const noexcept {
        return cash_ + position_ * current_mid;
    }

    inline double position() const noexcept { return position_; }
    inline double cash() const noexcept { return cash_; }
};

} // namespace hft
