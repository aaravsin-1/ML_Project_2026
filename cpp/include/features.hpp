#pragma once

#include "lob.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace hft {

struct alignas(32) MicrostructureFeatures {
    double ofi_l1{0.0};
    std::array<double, MAX_LOB_LEVELS> mlofi{};
    double micro_deviation{0.0};
    double relative_spread{0.0};
    double imbalance_l1{0.0};
    double imbalance_l3{0.0};
    double imbalance_l10{0.0};
    double vpin{0.0};
};

class FeatureEngine {
private:
    LimitOrderBook prev_book_{};
    bool initialized_{false};

    // VPIN State
    double bucket_volume_{1000.0};
    size_t window_buckets_{50};
    double curr_buy_vol_{0.0};
    double curr_sell_vol_{0.0};
    double curr_accum_vol_{0.0};
    std::vector<double> vpin_history_{};
    size_t hist_idx_{0};
    double rolling_vpin_sum_{0.0};

public:
    explicit FeatureEngine(double bucket_vol = 1000.0, size_t window = 50)
        : bucket_volume_(bucket_vol), window_buckets_(window),
          vpin_history_(window, 0.0) {}

    void reset() {
        initialized_ = false;
        curr_buy_vol_ = 0.0;
        curr_sell_vol_ = 0.0;
        curr_accum_vol_ = 0.0;
        std::fill(vpin_history_.begin(), vpin_history_.end(), 0.0);
        hist_idx_ = 0;
        rolling_vpin_sum_ = 0.0;
    }

    void update_trade(double size, int side) noexcept {
        double rem = size;
        while (rem > 0.0) {
            double needed = bucket_volume_ - curr_accum_vol_;
            double take = std::min(rem, needed);
            if (side > 0) {
                curr_buy_vol_ += take;
            } else {
                curr_sell_vol_ += take;
            }
            curr_accum_vol_ += take;
            rem -= take;

            if (curr_accum_vol_ >= bucket_volume_ - 1e-6) {
                double imb = std::abs(curr_buy_vol_ - curr_sell_vol_);
                // Update ring buffer
                rolling_vpin_sum_ -= vpin_history_[hist_idx_];
                vpin_history_[hist_idx_] = imb;
                rolling_vpin_sum_ += imb;
                hist_idx_ = (hist_idx_ + 1) % window_buckets_;

                curr_buy_vol_ = 0.0;
                curr_sell_vol_ = 0.0;
                curr_accum_vol_ = 0.0;
            }
        }
    }

    inline double current_vpin() const noexcept {
        return rolling_vpin_sum_ / (window_buckets_ * bucket_volume_);
    }

    inline MicrostructureFeatures compute(const LimitOrderBook& current_book) noexcept {
        MicrostructureFeatures feat{};
        const double mid = current_book.mid_price();
        const double spread = current_book.spread();

        feat.relative_spread = spread / (mid + 1e-9);
        feat.micro_deviation = current_book.micro_price() - mid;
        feat.imbalance_l1 = current_book.top_imbalance();
        feat.vpin = current_vpin();

        // Multi-level depth imbalances
        double cum_bid3 = 0.0, cum_ask3 = 0.0;
        double cum_bid10 = 0.0, cum_ask10 = 0.0;

        #pragma GCC unroll 10
        for (size_t i = 0; i < MAX_LOB_LEVELS; ++i) {
            cum_bid10 += current_book.bids[i].quantity;
            cum_ask10 += current_book.asks[i].quantity;
            if (i < 3) {
                cum_bid3 += current_book.bids[i].quantity;
                cum_ask3 += current_book.asks[i].quantity;
            }
        }

        feat.imbalance_l3 = (cum_bid3 - cum_ask3) / (cum_bid3 + cum_ask3 + 1e-9);
        feat.imbalance_l10 = (cum_bid10 - cum_ask10) / (cum_bid10 + cum_ask10 + 1e-9);

        if (!initialized_) {
            prev_book_ = current_book;
            initialized_ = true;
            return feat;
        }

        // SIMD-friendly loop for MLOFI across 10 levels
        #pragma GCC unroll 10
        for (size_t i = 0; i < MAX_LOB_LEVELS; ++i) {
            const double bp = current_book.bids[i].price;
            const double bq = current_book.bids[i].quantity;
            const double prev_bp = prev_book_.bids[i].price;
            const double prev_bq = prev_book_.bids[i].quantity;

            double dw = 0.0;
            if (bp > prev_bp) {
                dw = bq;
            } else if (bp == prev_bp) {
                dw = bq - prev_bq;
            } else {
                dw = -prev_bq;
            }

            const double ap = current_book.asks[i].price;
            const double aq = current_book.asks[i].quantity;
            const double prev_ap = prev_book_.asks[i].price;
            const double prev_aq = prev_book_.asks[i].quantity;

            double dv = 0.0;
            if (ap < prev_ap) {
                dv = aq;
            } else if (ap == prev_ap) {
                dv = aq - prev_aq;
            } else {
                dv = -prev_aq;
            }

            feat.mlofi[i] = dw - dv;
        }

        feat.ofi_l1 = feat.mlofi[0];
        prev_book_ = current_book;
        return feat;
    }
};

} // namespace hft
