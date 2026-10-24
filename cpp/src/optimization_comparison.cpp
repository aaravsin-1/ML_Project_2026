#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <windows.h>

#include "../include/lob.hpp"
#include "../include/features.hpp"
#include "../include/tree_engine.hpp"

#include "../include/lob_soa.hpp"
#include "../include/features_avx2.hpp"
#include "../include/tree_engine_branchless.hpp"

using namespace hft;

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    // Pin process/thread to highest priority to eliminate OS scheduler jitter
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    std::cout << "====================================================================\n";
    std::cout << "    HFT C++ OPTIMIZATION AUDIT: BASELINE VS HAND-CODED HARDWARE     \n";
    std::cout << "====================================================================\n\n";

    constexpr size_t N_EVENTS = 500000;
    std::cout << "[+] Generating " << N_EVENTS << " test events for high-precision benchmarking...\n";

    std::vector<LimitOrderBook> books_baseline(N_EVENTS);
    std::vector<LimitOrderBookSoA> books_soa(N_EVENTS);

    std::mt19937 rng(42);
    std::normal_distribution<double> shock(0.0, 0.02);
    std::uniform_real_distribution<double> v_dist(100.0, 1500.0);

    double mid = 100.0;
    for (size_t t = 0; t < N_EVENTS; ++t) {
        mid += shock(rng);
        const double bid1 = std::round((mid - 0.005) * 100.0) / 100.0;
        const double ask1 = bid1 + 0.01;

        // Baseline AoS
        for (size_t lvl = 0; lvl < MAX_LOB_LEVELS; ++lvl) {
            books_baseline[t].bids[lvl].price = bid1 - (lvl * 0.01);
            books_baseline[t].bids[lvl].quantity = v_dist(rng);
            books_baseline[t].asks[lvl].price = ask1 + (lvl * 0.01);
            books_baseline[t].asks[lvl].quantity = v_dist(rng);
        }

        // Optimized SoA (32-byte aligned)
        for (size_t lvl = 0; lvl < SIMD_LOB_LEVELS; ++lvl) {
            books_soa[t].bid_prices[lvl] = bid1 - (lvl * 0.01);
            books_soa[t].bid_quantities[lvl] = v_dist(rng);
            books_soa[t].ask_prices[lvl] = ask1 + (lvl * 0.01);
            books_soa[t].ask_quantities[lvl] = v_dist(rng);
        }
    }

    // -------------------------------------------------------------
    // 1. BENCHMARK BASELINE ENGINE (AoS + 60 Branches + Pointer Tree)
    // -------------------------------------------------------------
    std::cout << "[+] Benchmarking 1: Baseline Architecture (Array-of-Structs + 60 Branches)...\n";
    FeatureEngine feat_baseline;
    FastGBDTEnsemble tree_baseline;
    for (int i = 0; i < 50; ++i) {
        DecisionTree dt;
        dt.nodes.resize(31);
        for (int n = 0; n < 31; ++n) {
            if (2 * n + 1 < 31) {
                dt.nodes[n].feature_idx = n % 10;
                dt.nodes[n].threshold = 0.0f;
                dt.nodes[n].left_child = 2 * n + 1;
                dt.nodes[n].right_child = 2 * n + 2;
            } else {
                dt.nodes[n].feature_idx = -1;
                dt.nodes[n].leaf_value = 0.02f;
            }
        }
        tree_baseline.trees_flat.push_back(dt);
        tree_baseline.trees_up.push_back(dt);
        tree_baseline.trees_down.push_back(dt);
    }

    // Warmup
    for (size_t t = 0; t < 10000; ++t) {
        auto f = feat_baseline.compute(books_baseline[t]);
        (void)f;
    }
    feat_baseline.reset();

    volatile int32_t sink_base = 0;
    auto t_start_base = std::chrono::high_resolution_clock::now();
    for (size_t t = 0; t < N_EVENTS; ++t) {
        auto f = feat_baseline.compute(books_baseline[t]);
        float arr[16] = {
            static_cast<float>(f.ofi_l1), static_cast<float>(f.imbalance_l1),
            static_cast<float>(f.micro_deviation), static_cast<float>(f.relative_spread),
            static_cast<float>(f.vpin), static_cast<float>(f.imbalance_l3),
            static_cast<float>(f.imbalance_l10), 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        sink_base += tree_baseline.predict_class(arr);
    }
    auto t_end_base = std::chrono::high_resolution_clock::now();
    double base_ns_total = std::chrono::duration<double, std::nano>(t_end_base - t_start_base).count();
    double base_ns_per_event = base_ns_total / N_EVENTS;
    double base_throughput = N_EVENTS / (base_ns_total * 1e-9);

    // -------------------------------------------------------------
    // 2. BENCHMARK HAND-CODED ENGINE (SoA + AVX2 + Bitwise Trees)
    // -------------------------------------------------------------
    std::cout << "[+] Benchmarking 2: Hand-Coded Architecture (Struct-of-Arrays + AVX2 + Bitwise Trees)...\n";
    AVX2FeatureEngine feat_avx2;
    FastBranchlessEnsemble tree_branchless;
    for (int i = 0; i < 50; ++i) {
        ObliviousTreeDepth4 ot;
        ot.feature_indices[0] = 0;
        ot.feature_indices[1] = 1;
        ot.feature_indices[2] = 2;
        ot.feature_indices[3] = 3;
        ot.thresholds[0] = 0.0f;
        ot.thresholds[1] = 0.0f;
        ot.thresholds[2] = 0.0f;
        ot.thresholds[3] = 0.0f;
        for (int l = 0; l < 16; ++l) ot.leaves[l] = 0.02f;
        tree_branchless.trees_flat.push_back(ot);
        tree_branchless.trees_up.push_back(ot);
        tree_branchless.trees_down.push_back(ot);
    }

    // Warmup
    for (size_t t = 0; t < 10000; ++t) {
        auto f = feat_avx2.compute(books_soa[t]);
        (void)f;
    }

    volatile int32_t sink_opt = 0;
    auto t_start_opt = std::chrono::high_resolution_clock::now();
    for (size_t t = 0; t < N_EVENTS; ++t) {
        auto f = feat_avx2.compute(books_soa[t]);
        float arr[16] = {
            static_cast<float>(f.ofi_l1), static_cast<float>(f.imbalance_l1),
            static_cast<float>(f.micro_deviation), static_cast<float>(f.relative_spread),
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        sink_opt += tree_branchless.predict_class(arr);
    }
    auto t_end_opt = std::chrono::high_resolution_clock::now();
    double opt_ns_total = std::chrono::duration<double, std::nano>(t_end_opt - t_start_opt).count();
    double opt_ns_per_event = opt_ns_total / N_EVENTS;
    double opt_throughput = N_EVENTS / (opt_ns_total * 1e-9);

    std::cout << "\n====================================================================\n";
    std::cout << "               HEAD-TO-HEAD LATENCY COMPARISON RESULTS              \n";
    std::cout << "====================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Metric                    Baseline Engine      Hand-Coded AVX2/Bitwise    Speedup\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Throughput (msgs/sec)     " << std::setw(12) << base_throughput << "         " << std::setw(12) << opt_throughput << "    " << std::setw(6) << (opt_throughput / base_throughput) << "x\n";
    std::cout << "Mean Processing Latency   " << std::setw(10) << base_ns_per_event << " ns        " << std::setw(10) << opt_ns_per_event << " ns   " << std::setw(6) << (base_ns_per_event / opt_ns_per_event) << "x\n";
    std::cout << "Total Wall Clock Time     " << std::setw(10) << (base_ns_total / 1e6) << " ms        " << std::setw(10) << (opt_ns_total / 1e6) << " ms   " << std::setw(6) << (base_ns_total / opt_ns_total) << "x\n";
    std::cout << "====================================================================\n";
    std::cout << "Verification sinks: " << sink_base << ", " << sink_opt << "\n";
    return 0;
}
