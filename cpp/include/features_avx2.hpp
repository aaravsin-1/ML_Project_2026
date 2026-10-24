#pragma once

#include "lob_soa.hpp"
#include <immintrin.h>
#include <array>

namespace hft {

struct alignas(32) AVX2MicrostructureFeatures {
    double ofi_l1{0.0};
    alignas(32) double mlofi[12]{0.0};
    double imbalance_l1{0.0};
    double relative_spread{0.0};
    double micro_deviation{0.0};
};

class AVX2FeatureEngine {
private:
    LimitOrderBookSoA prev_book_{};
    bool initialized_{false};

public:
    inline AVX2MicrostructureFeatures compute(const LimitOrderBookSoA& book) noexcept {
        AVX2MicrostructureFeatures feat{};

        const double best_bp = book.bid_prices[0];
        const double best_bq = book.bid_quantities[0];
        const double best_ap = book.ask_prices[0];
        const double best_aq = book.ask_quantities[0];

        const double mid = (best_bp + best_ap) * 0.5;
        const double spread = best_ap - best_bp;
        const double denom = best_bq + best_aq;

        feat.relative_spread = spread / (mid + 1e-9);
        feat.imbalance_l1 = (denom > 1e-9) ? (best_bq - best_aq) / denom : 0.0;
        const double micro = (denom > 1e-9) ? (best_bq * best_ap + best_aq * best_bp) / denom : mid;
        feat.micro_deviation = micro - mid;

        if (!initialized_) {
            prev_book_ = book;
            initialized_ = true;
            return feat;
        }

        // =========================================================================
        // HAND-CODED AVX2 BRANCHLESS MLOFI (Zero Branches, 4 levels per cycle)
        // =========================================================================
        const __m256d zero = _mm256_setzero_pd();

        // Process 12 levels in 3 blocks of 4 levels (256-bit SIMD registers)
        #pragma GCC unroll 3
        for (size_t offset = 0; offset < SIMD_LOB_LEVELS; offset += 4) {
            // 1. Aligned Vector Loads from SoA memory
            __m256d curr_bp = _mm256_load_pd(&book.bid_prices[offset]);
            __m256d prev_bp = _mm256_load_pd(&prev_book_.bid_prices[offset]);
            __m256d curr_bq = _mm256_load_pd(&book.bid_quantities[offset]);
            __m256d prev_bq = _mm256_load_pd(&prev_book_.bid_quantities[offset]);

            // 2. Parallel comparisons for Bid side
            __m256d mask_b_gt = _mm256_cmp_pd(curr_bp, prev_bp, _CMP_GT_OQ);
            __m256d mask_b_eq = _mm256_cmp_pd(curr_bp, prev_bp, _CMP_EQ_OQ);

            __m256d diff_bq = _mm256_sub_pd(curr_bq, prev_bq);
            __m256d neg_prev_bq = _mm256_sub_pd(zero, prev_bq);

            // Branchless multiplexing using hardware vector blend:
            // dw = (curr_bp > prev_bp) ? curr_bq : ((curr_bp == prev_bp) ? diff_bq : -prev_bq)
            __m256d dw_eq_or_lt = _mm256_blendv_pd(neg_prev_bq, diff_bq, mask_b_eq);
            __m256d dw = _mm256_blendv_pd(dw_eq_or_lt, curr_bq, mask_b_gt);

            // 3. Aligned Vector Loads for Ask side
            __m256d curr_ap = _mm256_load_pd(&book.ask_prices[offset]);
            __m256d prev_ap = _mm256_load_pd(&prev_book_.ask_prices[offset]);
            __m256d curr_aq = _mm256_load_pd(&book.ask_quantities[offset]);
            __m256d prev_aq = _mm256_load_pd(&prev_book_.ask_quantities[offset]);

            // 4. Parallel comparisons for Ask side
            __m256d mask_a_lt = _mm256_cmp_pd(curr_ap, prev_ap, _CMP_LT_OQ);
            __m256d mask_a_eq = _mm256_cmp_pd(curr_ap, prev_ap, _CMP_EQ_OQ);

            __m256d diff_aq = _mm256_sub_pd(curr_aq, prev_aq);
            __m256d neg_prev_aq = _mm256_sub_pd(zero, prev_aq);

            // dv = (curr_ap < prev_ap) ? curr_aq : ((curr_ap == prev_ap) ? diff_aq : -prev_aq)
            __m256d dv_eq_or_gt = _mm256_blendv_pd(neg_prev_aq, diff_aq, mask_a_eq);
            __m256d dv = _mm256_blendv_pd(dv_eq_or_gt, curr_aq, mask_a_lt);

            // 5. MLOFI = dw - dv
            __m256d mlofi_vec = _mm256_sub_pd(dw, dv);
            _mm256_store_pd(&feat.mlofi[offset], mlofi_vec);
        }

        feat.ofi_l1 = feat.mlofi[0];
        prev_book_ = book;
        return feat;
    }
};

} // namespace hft
