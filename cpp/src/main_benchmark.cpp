#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "../include/lob.hpp"
#include "../include/features.hpp"
#include "../include/tree_engine.hpp"
#include "../include/simulator.hpp"

using namespace hft;

// Helper to construct a synthetic GBDT model for benchmarking
FastGBDTEnsemble create_benchmark_tree_ensemble(int n_trees = 50, int depth = 6) {
    FastGBDTEnsemble ensemble;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_thresh(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_val(-0.05f, 0.05f);
    std::uniform_int_distribution<int32_t> dist_feat(0, 15);

    auto build_tree = [&](int current_depth, int max_d) -> DecisionTree {
        DecisionTree tree;
        int n_nodes = (1 << (max_d + 1)) - 1;
        tree.nodes.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            if (2 * i + 1 < n_nodes) {
                // Internal node
                tree.nodes[i].feature_idx = dist_feat(rng);
                tree.nodes[i].threshold = dist_thresh(rng);
                tree.nodes[i].left_child = 2 * i + 1;
                tree.nodes[i].right_child = 2 * i + 2;
            } else {
                // Leaf node
                tree.nodes[i].feature_idx = -1;
                tree.nodes[i].leaf_value = dist_val(rng);
            }
        }
        return tree;
    };

    for (int i = 0; i < n_trees; ++i) {
        ensemble.trees_flat.push_back(build_tree(0, depth));
        ensemble.trees_up.push_back(build_tree(0, depth));
        ensemble.trees_down.push_back(build_tree(0, depth));
    }
    return ensemble;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "   HIGH-FREQUENCY LOB LOW-LATENCY C++ BENCHMARK ENGINE   \n";
    std::cout << "========================================================\n\n";

    constexpr size_t N_EVENTS = 100000;
    std::cout << "[+] Generating " << N_EVENTS << " synthetic high-frequency LOB snapshots...\n";

    std::vector<LimitOrderBook> books(N_EVENTS);
    std::mt19937 rng(42);
    std::normal_distribution<double> price_shock(0.0, 0.02);
    std::uniform_real_distribution<double> vol_dist(100.0, 1500.0);

    double current_mid = 100.0;
    for (size_t t = 0; t < N_EVENTS; ++t) {
        current_mid += price_shock(rng);
        current_mid = std::max(10.0, current_mid);
        books[t].timestamp_ns = t * 1000000ULL; // 1ms intervals
        const double best_bid = std::round((current_mid - 0.005) * 100.0) / 100.0;
        const double best_ask = best_bid + 0.01;

        for (size_t lvl = 0; lvl < MAX_LOB_LEVELS; ++lvl) {
            books[t].bids[lvl].price = best_bid - (lvl * 0.01);
            books[t].bids[lvl].quantity = vol_dist(rng);
            books[t].asks[lvl].price = best_ask + (lvl * 0.01);
            books[t].asks[lvl].quantity = vol_dist(rng);
        }
    }

    std::cout << "[+] Compiling C++ Tree Ensemble (50 trees x 3 classes, depth 6)...\n";
    FastGBDTEnsemble model = create_benchmark_tree_ensemble(50, 6);
    FeatureEngine feature_engine(1000.0, 50);
    MicrosecondSimulator simulator(1000000.0);

    std::cout << "[+] Running Warmup Iterations (5,000 events)...\n";
    for (size_t i = 0; i < 5000; ++i) {
        auto feat = feature_engine.compute(books[i]);
        float feat_arr[16] = {
            static_cast<float>(feat.ofi_l1),
            static_cast<float>(feat.imbalance_l1),
            static_cast<float>(feat.micro_deviation),
            static_cast<float>(feat.relative_spread),
            static_cast<float>(feat.vpin),
            static_cast<float>(feat.imbalance_l3),
            static_cast<float>(feat.imbalance_l10),
            static_cast<float>(feat.mlofi[1]),
            static_cast<float>(feat.mlofi[2]),
            static_cast<float>(feat.mlofi[3]),
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        int32_t action = model.predict_class(feat_arr);
        if (action == 1) simulator.execute_aggressive(1, 100.0, books[i]);
    }

    std::cout << "[+] Executing Ultra-Low Latency Profiling across " << N_EVENTS << " Events...\n";
    feature_engine.reset();
    simulator.reset();

    std::vector<double> latencies_features_ns(N_EVENTS);
    std::vector<double> latencies_inference_ns(N_EVENTS);
    std::vector<double> latencies_e2e_ns(N_EVENTS);

    auto t_start_total = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < N_EVENTS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. SIMD Feature Extraction
        auto feat = feature_engine.compute(books[t]);
        auto t1 = std::chrono::high_resolution_clock::now();

        // 2. Tree Inference
        float feat_arr[16] = {
            static_cast<float>(feat.ofi_l1),
            static_cast<float>(feat.imbalance_l1),
            static_cast<float>(feat.micro_deviation),
            static_cast<float>(feat.relative_spread),
            static_cast<float>(feat.vpin),
            static_cast<float>(feat.imbalance_l3),
            static_cast<float>(feat.imbalance_l10),
            static_cast<float>(feat.mlofi[1]),
            static_cast<float>(feat.mlofi[2]),
            static_cast<float>(feat.mlofi[3]),
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        int32_t action = model.predict_class(feat_arr);
        auto t2 = std::chrono::high_resolution_clock::now();

        // 3. Execution Simulation
        if (action == 1) {
            simulator.execute_aggressive(1, 100.0, books[t]);
        } else if (action == 2) {
            simulator.execute_aggressive(-1, 100.0, books[t]);
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        latencies_features_ns[t] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        latencies_inference_ns[t] = std::chrono::duration<double, std::nano>(t2 - t1).count();
        latencies_e2e_ns[t] = std::chrono::duration<double, std::nano>(t3 - t0).count();
    }

    auto t_end_total = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end_total - t_start_total).count();
    double throughput_mps = static_cast<double>(N_EVENTS) / total_sec;

    // Sort to compute percentiles
    std::sort(latencies_e2e_ns.begin(), latencies_e2e_ns.end());
    std::sort(latencies_features_ns.begin(), latencies_features_ns.end());
    std::sort(latencies_inference_ns.begin(), latencies_inference_ns.end());

    auto get_perc = [](const std::vector<double>& v, double p) -> double {
        size_t idx = static_cast<size_t>(p * (v.size() - 1));
        return v[idx] / 1000.0; // Return in microseconds
    };

    double e2e_p50 = get_perc(latencies_e2e_ns, 0.50);
    double e2e_p90 = get_perc(latencies_e2e_ns, 0.90);
    double e2e_p99 = get_perc(latencies_e2e_ns, 0.99);
    double e2e_p999 = get_perc(latencies_e2e_ns, 0.999);
    double feat_p50 = get_perc(latencies_features_ns, 0.50);
    double feat_p99 = get_perc(latencies_features_ns, 0.99);
    double inf_p50 = get_perc(latencies_inference_ns, 0.50);
    double inf_p99 = get_perc(latencies_inference_ns, 0.99);

    std::cout << "\n================ LATENCY BENCHMARK RESULTS ================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Throughput:             " << throughput_mps << " messages/sec\n";
    std::cout << "Total Events Processed: " << N_EVENTS << "\n";
    std::cout << "Total Wall Time:        " << total_sec << " seconds\n\n";

    std::cout << "--- Stage Breakdown ---\n";
    std::cout << "SIMD Feature Extraction: P50 = " << feat_p50 << " us | P99 = " << feat_p99 << " us\n";
    std::cout << "Fast Tree Inference:     P50 = " << inf_p50 << " us | P99 = " << inf_p99 << " us\n\n";

    std::cout << "--- End-to-End Pipeline Latency (Wire-to-Order) ---\n";
    std::cout << "P50  (Median): " << e2e_p50 << " microseconds\n";
    std::cout << "P90:           " << e2e_p90 << " microseconds\n";
    std::cout << "P99:           " << e2e_p99 << " microseconds\n";
    std::cout << "P99.9 (Tail):  " << e2e_p999 << " microseconds\n";
    std::cout << "===========================================================\n";

    // Write output to CSV for research analysis
    std::ofstream out_file("artifacts/cpp_latency_benchmarks.csv");
    if (out_file.is_open()) {
        out_file << "metric,throughput_mps,feat_p50_us,feat_p99_us,inf_p50_us,inf_p99_us,e2e_p50_us,e2e_p90_us,e2e_p99_us,e2e_p999_us\n";
        out_file << "cpp_pipeline," << throughput_mps << "," << feat_p50 << "," << feat_p99 << ","
                 << inf_p50 << "," << inf_p99 << "," << e2e_p50 << "," << e2e_p90 << ","
                 << e2e_p99 << "," << e2e_p999 << "\n";
        out_file.close();
        std::cout << "\n[+] Benchmark metrics saved to artifacts/cpp_latency_benchmarks.csv\n";
    }

    return 0;
}
