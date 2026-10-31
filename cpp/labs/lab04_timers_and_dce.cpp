// Lab 04: what does it cost to read a clock, and can the compiler delete your benchmark?
//
// Part A measures the overhead of each clock by timing an empty region many times.
// Part B times a small computation three ways, two of which are subtly broken.
//
//   ./build/lab04_timers_and_dce
//   ./build/lab04_timers_and_dce --core=2
//   objdump -d -C --no-show-raw-insn build/lab04_timers_and_dce | less   # find part B's loops

#include <cmath>

#include "common.hpp"

// Something small and realistic: level-1 imbalance of a book
__attribute__((noinline)) double imbalance(double bid_qty, double ask_qty) {
    return (bid_qty - ask_qty) / (bid_qty + ask_qty);
}

int main(int argc, char** argv) {
    const long core = lab::arg_long(argc, argv, "core", -1);
    if (core >= 0) lab::pin_to_core(static_cast<int>(core));
    const size_t n = 2'000'000;

    // ---------------- Part A: clock overhead ----------------
    std::printf("Part A: timing an EMPTY region %zu times\n", n);
    const double ticks_per_ns = lab::tsc_per_ns();
    std::printf("TSC runs at %.3f ticks per ns\n", ticks_per_ns);
    std::vector<double> chrono_ns(n), tsc_ns(n), tscp_ns(n);
    for (size_t i = 0; i < n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto t1 = std::chrono::steady_clock::now();
        chrono_ns[i] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    for (size_t i = 0; i < n; ++i) {
        const uint64_t c0 = lab::rdtsc(), c1 = lab::rdtsc();
        tsc_ns[i] = static_cast<double>(c1 - c0) / ticks_per_ns;
    }
    for (size_t i = 0; i < n; ++i) {
        const uint64_t c0 = lab::rdtscp(), c1 = lab::rdtscp();
        tscp_ns[i] = static_cast<double>(c1 - c0) / ticks_per_ns;
    }
    lab::print_percentiles("steady_clock (empty)", chrono_ns);
    lab::print_percentiles("rdtsc (empty)", tsc_ns);
    lab::print_percentiles("rdtscp (empty)", tscp_ns);

    // ---------------- Part B: dead-code elimination ----------------
    std::printf("\nPart B: %zu imbalance computations\n", n);
    std::vector<double> bq(n), aq(n);
    for (size_t i = 0; i < n; ++i) {
        bq[i] = 100.0 + static_cast<double>(i % 97);
        aq[i] = 100.0 + static_cast<double>(i % 89);
    }

    // B1: result thrown away. Is anything left to time?
    uint64_t t0 = lab::now_ns();
    for (size_t i = 0; i < n; ++i) {
        double r = (bq[i] - aq[i]) / (bq[i] + aq[i]);
        (void)r;
    }
    const double b1 = static_cast<double>(lab::now_ns() - t0) / static_cast<double>(n);

    // B2: result kept with do_not_optimize
    t0 = lab::now_ns();
    for (size_t i = 0; i < n; ++i) {
        double r = (bq[i] - aq[i]) / (bq[i] + aq[i]);
        lab::do_not_optimize(r);
    }
    const double b2 = static_cast<double>(lab::now_ns() - t0) / static_cast<double>(n);

    // B3: through a noinline function, summed (a real dependency on every result)
    double sum = 0.0;
    t0 = lab::now_ns();
    for (size_t i = 0; i < n; ++i) sum += imbalance(bq[i], aq[i]);
    const double b3 = static_cast<double>(lab::now_ns() - t0) / static_cast<double>(n);
    lab::do_not_optimize(sum);

    std::printf("B1 result discarded         %6.2f ns per item\n", b1);
    std::printf("B2 do_not_optimize(result)  %6.2f ns per item\n", b2);
    std::printf("B3 noinline call, summed    %6.2f ns per item\n", b3);
    std::printf("\nNow compare these with Part A. What could a per-item timer around one imbalance() call tell you?\n");
    return 0;
}
