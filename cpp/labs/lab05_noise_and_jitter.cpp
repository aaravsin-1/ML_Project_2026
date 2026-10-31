// Lab 05: noise and jitter. Same work, different conditions, different tails.
//
// Part A ("work"): times the same small fixed computation many times and prints the
// latency distribution. The median barely moves between conditions; watch p99.9 and max.
// Part B ("gaps"): spins reading the TSC and records every gap longer than a threshold:
// each gap is a moment the OS or hardware took the CPU away from you.
//
//   ./build/lab05_noise_and_jitter                       # baseline
//   ./build/lab05_noise_and_jitter --core=3              # pinned
//   (in another terminal: for i in 1 2 3 4; do yes > /dev/null & done)   # add load, then rerun
//   ./build/lab05_noise_and_jitter --mode=gaps --seconds=5 --core=3
//   killall yes                                          # stop the load afterwards

#include <cmath>

#include "common.hpp"

__attribute__((noinline)) double fixed_work(double x) {
    for (int i = 0; i < 200; ++i) x = x * 1.0000001 + 0.5 / (x + 1.0);
    return x;
}

int main(int argc, char** argv) {
    const long core = lab::arg_long(argc, argv, "core", -1);
    const std::string mode = lab::arg_str(argc, argv, "mode", "work");
    if (core >= 0 && !lab::pin_to_core(static_cast<int>(core))) std::printf("could not pin to core %ld\n", core);
    const double tpn = lab::tsc_per_ns();

    if (mode == "gaps") {
        const double seconds = static_cast<double>(lab::arg_long(argc, argv, "seconds", 5));
        const uint64_t threshold_ticks = static_cast<uint64_t>(1000.0 * tpn);  // gaps over 1 us
        const uint64_t end = lab::rdtsc() + static_cast<uint64_t>(seconds * 1e9 * tpn);
        std::vector<double> gaps_us;
        uint64_t prev = lab::rdtsc();
        while (prev < end) {
            const uint64_t now = lab::rdtsc();
            if (now - prev > threshold_ticks) gaps_us.push_back(static_cast<double>(now - prev) / tpn / 1000.0);
            prev = now;
        }
        std::printf("%zu interruptions longer than 1 us in %.0f s (core %ld)\n", gaps_us.size(), seconds, core);
        double total = 0;
        for (double g : gaps_us) total += g;
        std::printf("time lost: %.3f ms (%.4f%% of the run)\n", total / 1000.0, total / (seconds * 1e6) * 100.0);
        lab::print_percentiles("gap length", gaps_us, "us");
        return 0;
    }

    const size_t n = static_cast<size_t>(lab::arg_long(argc, argv, "n", 200000));
    std::vector<double> ns(n);
    double x = 1.0;
    for (size_t i = 0; i < 10000; ++i) x = fixed_work(x);  // warm-up
    for (size_t i = 0; i < n; ++i) {
        const uint64_t c0 = lab::rdtscp();
        x = fixed_work(x);
        const uint64_t c1 = lab::rdtscp();
        ns[i] = static_cast<double>(c1 - c0) / tpn;
    }
    lab::do_not_optimize(x);
    std::printf("core=%ld, %zu samples of the same work\n", core, n);
    lab::print_percentiles("fixed_work", ns);
    return 0;
}
