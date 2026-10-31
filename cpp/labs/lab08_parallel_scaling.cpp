// Lab 08: parallel scaling. When do more cores help, and when do they stop helping?
//
// Computes a per-book feature over many order books with OpenMP, sweeping the thread count,
// for two kinds of work:
//   light: level-1 imbalance per book (a few flops per 448-byte book: limited by memory bandwidth)
//   heavy: the same plus 200 rounds of arithmetic per book (limited by the cores)
// and prints time, speedup and parallel efficiency for each thread count.
//
//   ./build/lab08_parallel_scaling --work=heavy
//   ./build/lab08_parallel_scaling --work=light
//   ./build/lab08_parallel_scaling --work=heavy --schedule=dynamic
//   ./build/lab08_parallel_scaling --mode=false_sharing          # per-thread counters, packed vs padded
//   ./build/lab08_parallel_scaling --mode=latency_vs_batch       # one book at a time vs a whole day at once
//   OMP_PROC_BIND=close OMP_PLACES=cores ./build/lab08_parallel_scaling --work=heavy   # pinning

#include <omp.h>

#include <cmath>
#include <random>

#include "../include/lob_soa.hpp"
#include "common.hpp"

using hft::LimitOrderBookSoA;

static inline double light_feature(const LimitOrderBookSoA& b) {
    const double q = b.bid_quantities[0] + b.ask_quantities[0];
    return (b.bid_quantities[0] - b.ask_quantities[0]) / q;
}

static inline double heavy_feature(const LimitOrderBookSoA& b) {
    double x = light_feature(b);
    for (int i = 0; i < 200; ++i) x = std::sin(x) * 0.5 + std::sqrt(std::fabs(x) + 1.0) * 0.25;
    return x;
}

static double run(const std::vector<LimitOrderBookSoA>& books, std::vector<double>& out, int threads, bool heavy,
                  bool dynamic) {
    const long n = static_cast<long>(books.size());
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const uint64_t t0 = lab::now_ns();
        if (dynamic) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1024)
            for (long i = 0; i < n; ++i) out[i] = heavy ? heavy_feature(books[i]) : light_feature(books[i]);
        } else {
#pragma omp parallel for num_threads(threads) schedule(static)
            for (long i = 0; i < n; ++i) out[i] = heavy ? heavy_feature(books[i]) : light_feature(books[i]);
        }
        best = std::min(best, static_cast<double>(lab::now_ns() - t0) / 1e6);
    }
    return best;
}

struct Packed { long count; };                    // 8 bytes: 8 threads share one cache line
struct alignas(64) Padded { long count; };        // one cache line per thread

template <typename Counter>
static double count_events(int threads, long per_thread) {
    std::vector<Counter> c(threads);
    const uint64_t t0 = lab::now_ns();
#pragma omp parallel num_threads(threads)
    {
        Counter& mine = c[omp_get_thread_num()];
        for (long i = 0; i < per_thread; ++i) {
            mine.count += i & 1;
            lab::clobber_memory();  // force a real store each iteration, as a live counter would
        }
    }
    const double ms = static_cast<double>(lab::now_ns() - t0) / 1e6;
    long total = 0;
    for (auto& x : c) total += x.count;
    lab::do_not_optimize(total);
    return ms;
}

int main(int argc, char** argv) {
    const std::string mode = lab::arg_str(argc, argv, "mode", "scaling");
    const bool heavy = lab::arg_str(argc, argv, "work", "heavy") == "heavy";
    const bool dynamic = lab::arg_str(argc, argv, "schedule", "static") == "dynamic";
    const size_t n = static_cast<size_t>(lab::arg_long(argc, argv, "books", heavy ? 400'000 : 4'000'000));
    const int max_threads = omp_get_max_threads();

    if (mode == "false_sharing") {
        const long per_thread = 50'000'000;
        std::printf("%8s %14s %14s\n", "threads", "packed ms", "padded ms");
        for (int t = 1; t <= max_threads; t *= 2)
            std::printf("%8d %14.1f %14.1f\n", t, count_events<Packed>(t, per_thread), count_events<Padded>(t, per_thread));
        return 0;
    }

    std::vector<LimitOrderBookSoA> books(n);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> q(1.0, 100.0);
    for (auto& b : books)
        for (size_t l = 0; l < 10; ++l) {
            b.bid_quantities[l] = q(rng);
            b.ask_quantities[l] = q(rng);
        }
    std::vector<double> out(n);

    if (mode == "latency_vs_batch") {
        // Live: each book arrives on its own and is processed immediately on one core.
        std::vector<double> lat(std::min<size_t>(n, 200'000));
        for (size_t i = 0; i < lat.size(); ++i) {
            const uint64_t c0 = lab::rdtscp();
            out[i] = heavy_feature(books[i]);
            lat[i] = static_cast<double>(lab::rdtscp() - c0);
        }
        const double tpn = lab::tsc_per_ns();
        for (auto& v : lat) v /= tpn;
        lab::print_percentiles("per book, 1 core (latency)", lat);
        // Research: the whole day at once on every core.
        const double ms = run(books, out, max_threads, true, false);
        std::printf("whole batch, %d threads: %.1f ms for %zu books = %.1f ns per book (throughput)\n",
                    max_threads, ms, n, ms * 1e6 / static_cast<double>(n));
        return 0;
    }

    std::printf("work=%s schedule=%s books=%zu (%.0f MB)\n", heavy ? "heavy" : "light", dynamic ? "dynamic" : "static",
                n, static_cast<double>(n * sizeof(LimitOrderBookSoA)) / 1e6);
    std::printf("%8s %10s %9s %11s\n", "threads", "ms", "speedup", "efficiency");
    const double t1 = run(books, out, 1, heavy, dynamic);
    for (int t = 1; t <= max_threads; ++t) {
        const double ms = t == 1 ? t1 : run(books, out, t, heavy, dynamic);
        std::printf("%8d %10.1f %9.2f %10.0f%%\n", t, ms, t1 / ms, 100.0 * t1 / ms / t);
    }
    lab::do_not_optimize(out[n / 2]);
    return 0;
}
