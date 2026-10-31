// Lab 06: SIMD on a real feature. Multi-level order flow imbalance (OFI), three ways.
//
//   scalar_branchy   : the textbook if / else-if / else per level
//   scalar_select    : same maths written with ternaries on plain arrays (can the compiler vectorize it?)
//   avx2 (repo)      : hft::AVX2FeatureEngine from cpp/include/features_avx2.hpp
//
// Step 1 checks all three agree on random books AND on nasty edge cases (equal prices, zero
// padding, NaN). Step 2 times them.
//
//   ./build/lab06_simd_ofi
//   ./build/lab06_simd_ofi --tick=0             # prices never change between books (very predictable)
//   ./build/lab06_simd_ofi --tick=2             # prices move up to 2 ticks at random (the default, worst case)
//   ./build/lab06_simd_ofi_fastmath           # same code built with -ffast-math: does the NaN check still agree?
//   g++ -std=c++20 -O3 -march=native -fopt-info-vec-optimized -c lab06_simd_ofi.cpp -o /dev/null
//                                             # which loops did the compiler vectorize?

#include <cmath>
#include <random>

#include "../include/features_avx2.hpp"
#include "common.hpp"

using hft::LimitOrderBookSoA;
using hft::SIMD_LOB_LEVELS;

// OFI per level (Cont, Kukanov & Stoikov): bid side change minus ask side change.
__attribute__((noinline)) void ofi_branchy(const LimitOrderBookSoA& cur, const LimitOrderBookSoA& prev, double* out) {
    for (size_t i = 0; i < SIMD_LOB_LEVELS; ++i) {
        double dw, dv;
        if (cur.bid_prices[i] > prev.bid_prices[i]) dw = cur.bid_quantities[i];
        else if (cur.bid_prices[i] == prev.bid_prices[i]) dw = cur.bid_quantities[i] - prev.bid_quantities[i];
        else dw = -prev.bid_quantities[i];
        if (cur.ask_prices[i] < prev.ask_prices[i]) dv = cur.ask_quantities[i];
        else if (cur.ask_prices[i] == prev.ask_prices[i]) dv = cur.ask_quantities[i] - prev.ask_quantities[i];
        else dv = -prev.ask_quantities[i];
        out[i] = dw - dv;
    }
}

__attribute__((noinline)) void ofi_select(const LimitOrderBookSoA& cur, const LimitOrderBookSoA& prev, double* out) {
    for (size_t i = 0; i < SIMD_LOB_LEVELS; ++i) {
        const double bq_diff = cur.bid_quantities[i] - prev.bid_quantities[i];
        const double aq_diff = cur.ask_quantities[i] - prev.ask_quantities[i];
        const double dw_le = cur.bid_prices[i] == prev.bid_prices[i] ? bq_diff : -prev.bid_quantities[i];
        const double dw = cur.bid_prices[i] > prev.bid_prices[i] ? cur.bid_quantities[i] : dw_le;
        const double dv_ge = cur.ask_prices[i] == prev.ask_prices[i] ? aq_diff : -prev.ask_quantities[i];
        const double dv = cur.ask_prices[i] < prev.ask_prices[i] ? cur.ask_quantities[i] : dv_ge;
        out[i] = dw - dv;
    }
}

// The repo's engine keeps the previous book internally; this wrapper feeds it (prev, cur).
void ofi_avx2(const LimitOrderBookSoA& cur, const LimitOrderBookSoA& prev, double* out) {
    hft::AVX2FeatureEngine e;
    e.compute(prev);
    const auto f = e.compute(cur);
    for (size_t i = 0; i < SIMD_LOB_LEVELS; ++i) out[i] = f.mlofi[i];
}

static bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

int g_tick = 2;  // largest random price move per level, in ticks (--tick)

LimitOrderBookSoA random_book(std::mt19937& rng) {
    std::uniform_int_distribution<int> tick(-g_tick, g_tick);
    std::uniform_real_distribution<double> qty(1.0, 50.0);
    LimitOrderBookSoA b{};
    for (size_t i = 0; i < 10; ++i) {  // levels 10 and 11 stay zero: that is the padding
        b.bid_prices[i] = 100.00 - 0.01 * static_cast<double>(i) + 0.01 * tick(rng);
        b.ask_prices[i] = 100.01 + 0.01 * static_cast<double>(i) + 0.01 * tick(rng);
        b.bid_quantities[i] = qty(rng);
        b.ask_quantities[i] = qty(rng);
    }
    return b;
}

int main(int argc, char** argv) {
    const size_t n = static_cast<size_t>(lab::arg_long(argc, argv, "n", 2'000'000));
    g_tick = static_cast<int>(lab::arg_long(argc, argv, "tick", 2));
#ifdef __FAST_MATH__
    std::printf("built WITH -ffast-math\n");
#else
    std::printf("built without -ffast-math\n");
#endif

    // ---------- Step 1: correctness ----------
    std::mt19937 rng(3);
    size_t mismatches = 0;
    double a[SIMD_LOB_LEVELS], b[SIMD_LOB_LEVELS], c[SIMD_LOB_LEVELS];
    for (size_t t = 0; t < 100000; ++t) {
        const auto prev = random_book(rng), cur = random_book(rng);
        ofi_branchy(cur, prev, a);
        ofi_select(cur, prev, b);
        ofi_avx2(cur, prev, c);
        for (size_t i = 0; i < SIMD_LOB_LEVELS; ++i) mismatches += !same(a[i], b[i]) || !same(a[i], c[i]);
    }
    std::printf("random books: %zu mismatching values out of %zu\n", mismatches, 100000 * SIMD_LOB_LEVELS);

    // Edge case: a NaN price on level 0 (a data glitch). What should OFI be, and do all three agree?
    auto prev = random_book(rng), cur = prev;
    cur.bid_prices[0] = std::nan("");
    ofi_branchy(cur, prev, a);
    ofi_select(cur, prev, b);
    ofi_avx2(cur, prev, c);
    std::printf("NaN bid price, level 0:  branchy %g | select %g | avx2 %g\n", a[0], b[0], c[0]);

    // ---------- Step 2: speed ----------
    std::vector<LimitOrderBookSoA> books(4096);
    for (auto& bk : books) bk = random_book(rng);
    double sink = 0.0;
    auto bench = [&](const char* name, auto fn) {
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            const uint64_t t0 = lab::now_ns();
            for (size_t t = 1; t < n; ++t) {
                double out[SIMD_LOB_LEVELS];
                fn(books[t % books.size()], books[(t - 1) % books.size()], out);
                sink += out[t % SIMD_LOB_LEVELS];
            }
            best = std::min(best, static_cast<double>(lab::now_ns() - t0) / static_cast<double>(n));
        }
        std::printf("%-16s %6.2f ns per book  (tick=%d)\n", name, best, g_tick);
    };
    // The repo engine stores the previous book itself, so time it the way it is meant to be used
    hft::AVX2FeatureEngine engine;
    bench("scalar_branchy", ofi_branchy);
    bench("scalar_select", ofi_select);
    bench("avx2 (repo)", [&](const LimitOrderBookSoA& cur, const LimitOrderBookSoA&, double* out) {
        const auto f = engine.compute(cur);
        out[0] = f.mlofi[0];
        for (size_t i = 1; i < SIMD_LOB_LEVELS; ++i) out[i] = f.mlofi[i];
    });
    lab::do_not_optimize(sink);
    return 0;
}
