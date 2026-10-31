// Lab 02: data layout. Array of structs vs struct of arrays, using the repo's own order books.
//
// Two tasks over many snapshots:
//   "depth":  sum all 10 bid quantities of every book (touches one field across all levels)
//   "top":    compute the level-1 mid price of every book (touches 2 fields of level 1 only)
//
// Which layout wins depends on the task. Predict before you run.
//
//   ./build/lab02_data_layout
//   ./build/lab02_data_layout --books=20000          # small enough to fit in cache
//   ./build/lab02_data_layout --books=2000000        # much bigger than cache
//   ./build/lab02_data_layout --layout_report=1      # print sizes and field offsets

#include <cstddef>
#include <random>

#include "../include/lob.hpp"
#include "../include/lob_soa.hpp"
#include "common.hpp"

using namespace hft;

template <typename F>
double time_ms(F&& f, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const uint64_t t0 = lab::now_ns();
        f();
        best = std::min(best, static_cast<double>(lab::now_ns() - t0) / 1e6);
    }
    return best;  // best-of-N hides noise; ask yourself when that is and isn't honest
}

int main(int argc, char** argv) {
    const size_t n = static_cast<size_t>(lab::arg_long(argc, argv, "books", 500000));
    const int reps = static_cast<int>(lab::arg_long(argc, argv, "reps", 10));

    if (lab::arg_long(argc, argv, "layout_report", 0)) {
        std::printf("sizeof(Level)             = %zu\n", sizeof(Level));
        std::printf("sizeof(LimitOrderBook)    = %zu  (alignof %zu)\n", sizeof(LimitOrderBook), alignof(LimitOrderBook));
        std::printf("  offsetof bids           = %zu\n", offsetof(LimitOrderBook, bids));
        std::printf("  offsetof asks           = %zu\n", offsetof(LimitOrderBook, asks));
        std::printf("  offsetof timestamp_ns   = %zu\n", offsetof(LimitOrderBook, timestamp_ns));
        std::printf("sizeof(LimitOrderBookSoA) = %zu  (alignof %zu)\n", sizeof(LimitOrderBookSoA), alignof(LimitOrderBookSoA));
        std::printf("  offsetof bid_prices     = %zu\n", offsetof(LimitOrderBookSoA, bid_prices));
        std::printf("  offsetof bid_quantities = %zu\n", offsetof(LimitOrderBookSoA, bid_quantities));
        std::printf("  offsetof ask_prices     = %zu\n", offsetof(LimitOrderBookSoA, ask_prices));
        std::printf("  offsetof ask_quantities = %zu\n", offsetof(LimitOrderBookSoA, ask_quantities));
        std::printf("  offsetof timestamp_ns   = %zu\n", offsetof(LimitOrderBookSoA, timestamp_ns));
        std::printf("Cache lines per book: AoS %zu, SoA %zu\n\n", (sizeof(LimitOrderBook) + 63) / 64,
                    (sizeof(LimitOrderBookSoA) + 63) / 64);
    }

    std::vector<LimitOrderBook> aos(n);
    std::vector<LimitOrderBookSoA> soa(n);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> q(1.0, 100.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t l = 0; l < MAX_LOB_LEVELS; ++l) {
            aos[i].bids[l] = {100.0 - 0.01 * static_cast<double>(l), q(rng)};
            aos[i].asks[l] = {100.01 + 0.01 * static_cast<double>(l), q(rng)};
            soa[i].bid_prices[l] = aos[i].bids[l].price;
            soa[i].bid_quantities[l] = aos[i].bids[l].quantity;
            soa[i].ask_prices[l] = aos[i].asks[l].price;
            soa[i].ask_quantities[l] = aos[i].asks[l].quantity;
        }

    double sink = 0.0;
    const double depth_aos = time_ms([&] {
        double s = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t l = 0; l < MAX_LOB_LEVELS; ++l) s += aos[i].bids[l].quantity;
        sink += s;
    }, reps);
    const double depth_soa = time_ms([&] {
        double s = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t l = 0; l < MAX_LOB_LEVELS; ++l) s += soa[i].bid_quantities[l];
        sink += s;
    }, reps);
    const double top_aos = time_ms([&] {
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += aos[i].mid_price();
        sink += s;
    }, reps);
    const double top_soa = time_ms([&] {
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += soa[i].mid_price();
        sink += s;
    }, reps);
    lab::do_not_optimize(sink);

    const double mb = static_cast<double>(n) * static_cast<double>(sizeof(LimitOrderBook)) / 1e6;
    std::printf("%zu books (~%.0f MB per layout), best of %d runs\n", n, mb, reps);
    std::printf("%-30s %10s %10s\n", "task", "AoS ms", "SoA ms");
    std::printf("%-30s %10.2f %10.2f\n", "depth: sum 10 bid quantities", depth_aos, depth_soa);
    std::printf("%-30s %10.2f %10.2f\n", "top: level-1 mid price", top_aos, top_soa);
    return 0;
}
