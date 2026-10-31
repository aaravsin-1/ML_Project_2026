// Lab 03: branch prediction.
//
// Sums the elements above a threshold, where a fraction of the decisions are random coin flips
// and the rest follow a fixed pattern. The work is identical in every run; only how
// *predictable* the branch is changes.
//
//   ./build/lab03_branch_prediction                     # sweep randomness 0% .. 100%
//   ./build/lab03_branch_prediction --mode=branchless   # same sweep, no branch
//   perf stat -e branches,branch-misses ./build/lab03_branch_prediction --random_pct=50
//   perf stat -e branches,branch-misses ./build/lab03_branch_prediction --random_pct=0
//
// Compiled twice by the Makefile; compare them:
//   build/lab03_branch_prediction        -O2 with if-conversion and vectorization switched off: a real branch
//   build/lab03_branch_prediction_O3     a normal -O3 build: what does the compiler do to your branch?

#include <random>

#include "common.hpp"

__attribute__((noinline)) long sum_branchy(const std::vector<int>& v, int threshold) {
    long s = 0;
    for (int x : v)
        if (x >= threshold) s += x;
    return s;
}

__attribute__((noinline)) long sum_branchless(const std::vector<int>& v, int threshold) {
    long s = 0;
    for (int x : v) s += (x >= threshold) ? x : 0;  // usually compiles to a conditional move
    return s;
}

int main(int argc, char** argv) {
    const size_t n = static_cast<size_t>(lab::arg_long(argc, argv, "n", 10'000'000));
    const std::string mode = lab::arg_str(argc, argv, "mode", "branchy");
    const long only_pct = lab::arg_long(argc, argv, "random_pct", -1);
    const int threshold = 128;

    std::printf("mode=%s, n=%zu\n%12s %12s %16s\n", mode.c_str(), n, "random %", "ms", "result");
    for (int pct = 0; pct <= 100; pct += 10) {
        if (only_pct >= 0 && pct != only_pct) continue;
        // Pattern: alternate long runs above and below the threshold (easy to predict),
        // then overwrite pct% of positions with random values (impossible to predict).
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> any(0, 255), coin(0, 99);
        std::vector<int> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = ((i / 64) % 2) ? 200 : 50;
        for (size_t i = 0; i < n; ++i)
            if (coin(rng) < pct) v[i] = any(rng);

        long result = 0;
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            const uint64_t t0 = lab::now_ns();
            result = mode == "branchless" ? sum_branchless(v, threshold) : sum_branchy(v, threshold);
            best = std::min(best, static_cast<double>(lab::now_ns() - t0) / 1e6);
        }
        std::printf("%12d %12.2f %16ld\n", pct, best, result);
    }
    return 0;
}
