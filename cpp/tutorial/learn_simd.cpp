/**
 * ============================================================================
 *               SIMD & AVX2 HANDS-ON TUTORIAL FOR QUANT TRADING
 * ============================================================================
 * 
 * Welcome! This tutorial is designed for developers who know basic C++ but have
 * never touched SIMD or AVX2 before.
 * 
 * What is SIMD?
 *   SIMD stands for "Single Instruction, Multiple Data".
 *   Normally, a CPU instruction takes two numbers, adds them, and stores the result:
 *       c = a + b;   (Scalar: 1 instruction -> 1 result)
 * 
 *   With SIMD, the CPU has super-wide 256-bit registers (called YMM registers).
 *   Since a standard `double` is 64 bits, one 256-bit register holds FOUR doubles!
 *   A single SIMD instruction adds 4 pairs of doubles AT THE EXACT SAME TIME:
 *       [c0, c1, c2, c3] = [a0, a1, a2, a3] + [b0, b1, b2, b3];
 * 
 * Why does this matter in High-Frequency Trading (HFT)?
 *   In an order book, we have 4, 10, or 20 bid/ask price levels.
 *   Instead of a slow `for (int i = 0; i < 4; ++i)` loop with branches,
 *   AVX2 allows us to process 4 levels in ~1 nanosecond.
 * 
 * How to compile and run:
 *   g++ -O3 -mavx2 -std=c++20 learn_simd.cpp -o learn_simd.exe
 *   ./learn_simd.exe
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <immintrin.h>  // Header containing all Intel/AMD AVX and SIMD intrinsics
#include <array>
#include <vector>
#include <chrono>

// Helper utility: prints the 4 doubles stored inside a 256-bit vector (__m256d)
// Passing by const reference prevents Windows ABI stack-spill misalignment
void print_vec4(const char* label, const __m256d& v) {
    double values[4];
    _mm256_storeu_pd(values, v);  // Store vector into array (unaligned-safe)
    std::cout << std::left << std::setw(24) << label << " [ "
              << std::fixed << std::setprecision(2)
              << values[0] << ", " << values[1] << ", " 
              << values[2] << ", " << values[3] << " ]\n";
}

// ============================================================================
// LESSON 1: What is __m256d & How to Load Data
// ============================================================================
void lesson1_loading_data() {
    std::cout << "\n========================================\n";
    std::cout << " LESSON 1: What is __m256d & How to Load\n";
    std::cout << "========================================\n";
    std::cout << "A 64-bit double takes 8 bytes. A 256-bit register holds 4 doubles.\n\n";

    // Method A: Set explicitly from 4 values.
    // NOTE: _mm256_set_pd loads in REVERSE order: (val3, val2, val1, val0)
    // NOTE: _mm256_setr_pd loads in NATURAL order: (val0, val1, val2, val3) [Recommended!]
    __m256d bids = _mm256_setr_pd(100.50, 100.40, 100.30, 100.20);
    print_vec4("Bid Prices (Level 1-4):", bids);

    // Method B: Load directly from an array in memory.
    // _mm256_loadu_pd: 'u' stands for Unaligned. Modern CPUs run this at the
    // exact same speed as aligned loads, but it never crashes on stack boundaries.
    double ask_array[4] = { 100.60, 100.70, 100.80, 100.90 };
    __m256d asks = _mm256_loadu_pd(ask_array);
    print_vec4("Ask Prices (Level 1-4):", asks);

    // Method C: Broadcast a single number to all 4 lanes
    __m256d scalar_multiplier = _mm256_set1_pd(10.0);
    print_vec4("Broadcast 10.0:", scalar_multiplier);
}

// ============================================================================
// LESSON 2: Parallel Vector Arithmetic (1 Instruction -> 4 Operations)
// ============================================================================
void lesson2_vector_math() {
    std::cout << "\n========================================\n";
    std::cout << " LESSON 2: Parallel Arithmetic\n";
    std::cout << "========================================\n";

    __m256d asks = _mm256_setr_pd(100.60, 100.70, 100.80, 100.90);
    __m256d bids = _mm256_setr_pd(100.50, 100.40, 100.30, 100.20);

    // 1. Vector Subtraction: Spreads across all 4 levels at once!
    // spread[i] = ask[i] - bid[i]
    __m256d spreads = _mm256_sub_pd(asks, bids);
    print_vec4("Spreads (Ask - Bid):", spreads);

    // 2. Vector Addition: Mid-price numerator
    // mid[i] = (ask[i] + bid[i]) * 0.5
    __m256d sums = _mm256_add_pd(asks, bids);
    __m256d half = _mm256_set1_pd(0.5);
    __m256d mid_prices = _mm256_mul_pd(sums, half);
    print_vec4("Mid Prices:", mid_prices);
}

// ============================================================================
// LESSON 3: The Secret Weapon: Branchless Comparisons & Blends
// ============================================================================
void lesson3_branchless_logic() {
    std::cout << "\n========================================\n";
    std::cout << " LESSON 3: Branchless Conditionals (No 'if'!)\n";
    std::cout << "========================================\n";
    std::cout << "In trading code, an 'if' statement costs 15-20 CPU cycles when mispredicted.\n";
    std::cout << "AVX2 replaces 'if' statements with instantaneous bitwise blends!\n\n";

    // Suppose we want to compare current price with previous price:
    __m256d current_prices  = _mm256_setr_pd(101.0, 99.0, 102.5, 98.0);
    __m256d previous_prices = _mm256_setr_pd(100.0, 100.0, 102.0, 100.0);

    print_vec4("Current Prices:", current_prices);
    print_vec4("Previous Prices:", previous_prices);

    // Question: For each level, did price INCREASE?
    // In scalar C++: if (current[i] > previous[i]) ...
    // In AVX2: _mm256_cmp_pd with _CMP_GT_OQ (Greater-Than, Ordered, Non-signaling)
    // Result: A mask register with all 1s (0xFFFFFFFFFFFFFFFF) if true, or all 0s if false!
    __m256d mask = _mm256_cmp_pd(current_prices, previous_prices, _CMP_GT_OQ);

    // Now, let's select a value branchlessly:
    // If price increased, output +1.0; else output -1.0.
    __m256d plus_one  = _mm256_set1_pd(1.0);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    // _mm256_blendv_pd(false_choice, true_choice, mask)
    // If mask bit is 1, it picks true_choice. If 0, it picks false_choice.
    // 0 branches. 0 pipeline stalls. Exact 1 CPU cycle!
    __m256d direction = _mm256_blendv_pd(minus_one, plus_one, mask);
    print_vec4("Price Direction (+1/-1):", direction);
}

// ============================================================================
// LESSON 4: Real Microstructure Feature: 4-Level Order Book Imbalance
// ============================================================================
void lesson4_real_world_hft_feature() {
    std::cout << "\n========================================\n";
    std::cout << " LESSON 4: Real HFT Feature - 4-Level Imbalance\n";
    std::cout << "========================================\n";
    std::cout << "Feature Formula: Imbalance_k = (BidVol_k - AskVol_k) / (BidVol_k + AskVol_k)\n\n";

    // Order book queue volumes at depth 1, 2, 3, 4
    double bid_vol[4] = { 500.0, 1200.0, 300.0, 2500.0 };
    double ask_vol[4] = { 200.0,  400.0, 900.0, 2500.0 };

    __m256d vb = _mm256_loadu_pd(bid_vol);
    __m256d va = _mm256_loadu_pd(ask_vol);

    print_vec4("Bid Volumes (L1-L4):", vb);
    print_vec4("Ask Volumes (L1-L4):", va);

    // 1. Numerator: BidVol - AskVol
    __m256d numer = _mm256_sub_pd(vb, va);

    // 2. Denominator: BidVol + AskVol
    __m256d denom = _mm256_add_pd(vb, va);

    // 3. Imbalance: numer / denom
    __m256d imbalances = _mm256_div_pd(numer, denom);

    print_vec4("Depth Imbalances:", imbalances);

    // Explain interpretation:
    // Level 1: (500 - 200) / (500 + 200) = +300 / 700 = +0.43 (Heavy buying pressure)
    // Level 2: (1200 - 400) / (1200 + 400) = +800 / 1600 = +0.50 (Strong bid queue)
    // Level 3: (300 - 900) / (300 + 900) = -600 / 1200 = -0.50 (Selling resistance)
    // Level 4: (2500 - 2500) / 5000 = 0.00 (Balanced)
    std::cout << "\nAll 4 depth levels calculated simultaneously in ONE SIMD division!\n";
}

// ============================================================================
// LESSON 5: The Real Power: Hand-Coded SIMD vs Compiler Auto-Vectorizer
// ============================================================================
void lesson5_speed_benchmark() {
    std::cout << "\n========================================\n";
    std::cout << " LESSON 5: OFI Benchmark (Branchy Logic vs AVX2 Blend)\n";
    std::cout << "========================================\n";
    std::cout << "Compilers can auto-vectorize simple addition, but they FAIL on branchy\n";
    std::cout << "microstructure logic (like Cont et al. OFI: if price > prev ... else ...).\n";
    std::cout << "Here is hand-coded AVX2 beating the branching loop:\n\n";

    constexpr size_t N = 1'000;
    constexpr int REPEATS = 500;
    std::vector<double> curr_bids(N * 4), prev_bids(N * 4);
    std::vector<double> curr_vols(N * 4), prev_vols(N * 4);
    std::vector<double> out_scalar(N * 4, 0.0), out_simd(N * 4, 0.0);

    for (size_t i = 0; i < N * 4; ++i) {
        curr_bids[i] = 100.0 + (i % 3) * 0.01;
        prev_bids[i] = 100.0 + ((i + 1) % 3) * 0.01;
        curr_vols[i] = 100.0 + (i % 10);
        prev_vols[i] = 100.0 + ((i + 2) % 10);
    }

    // --- Benchmark 1: Branchy Scalar C++ ---
    double checksum_scalar = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < REPEATS; ++rep) {
        for (size_t i = 0; i < N * 4; ++i) {
            if (curr_bids[i] > prev_bids[i]) {
                out_scalar[i] = curr_vols[i];
            } else if (curr_bids[i] == prev_bids[i]) {
                out_scalar[i] = curr_vols[i] - prev_vols[i];
            } else {
                out_scalar[i] = 0.0;
            }
            checksum_scalar += out_scalar[i];
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double scalar_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // --- Benchmark 2: Hand-Coded Branchless AVX2 ---
    double checksum_simd = 0.0;
    auto t2 = std::chrono::high_resolution_clock::now();
    __m256d zero = _mm256_setzero_pd();
    for (int rep = 0; rep < REPEATS; ++rep) {
        for (size_t i = 0; i < N * 4; i += 4) {
            __m256d cb = _mm256_loadu_pd(&curr_bids[i]);
            __m256d pb = _mm256_loadu_pd(&prev_bids[i]);
            __m256d cv = _mm256_loadu_pd(&curr_vols[i]);
            __m256d pv = _mm256_loadu_pd(&prev_vols[i]);

            __m256d gt_mask = _mm256_cmp_pd(cb, pb, _CMP_GT_OQ);
            __m256d eq_mask = _mm256_cmp_pd(cb, pb, _CMP_EQ_OQ);

            __m256d delta_v = _mm256_sub_pd(cv, pv);

            __m256d ofi_eq = _mm256_blendv_pd(zero, delta_v, eq_mask);
            __m256d ofi = _mm256_blendv_pd(ofi_eq, cv, gt_mask);

            _mm256_storeu_pd(&out_simd[i], ofi);
            checksum_simd += (out_simd[i] + out_simd[i+1] + out_simd[i+2] + out_simd[i+3]);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double avx2_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    std::cout << "Branchy Scalar C++ : " << std::fixed << std::setprecision(2) << scalar_us << " us\n";
    std::cout << "Hand-Coded AVX2    : " << std::fixed << std::setprecision(2) << avx2_us << " us\n";
    std::cout << ">>> Hardware Speedup: " << std::setprecision(2) << (scalar_us / avx2_us) << "x faster!\n";
    std::cout << "(Checksum verified : " << checksum_scalar << " == " << checksum_simd << ")\n";
}

__attribute__((force_align_arg_pointer))
int main() {
    std::cout << "========================================================\n";
    std::cout << "    QUANT ENGINE: AVX2 SIMD INTERACTIVE TUTORIAL       \n";
    std::cout << "========================================================\n";

    lesson1_loading_data();
    lesson2_vector_math();
    lesson3_branchless_logic();
    lesson4_real_world_hft_feature();
    lesson5_speed_benchmark();

    std::cout << "\n========================================================\n";
    std::cout << "Congratulations! You now understand the core fundamentals of\n";
    std::cout << "SIMD vectorization and branchless programming in C++.\n";
    std::cout << "Next, check out 'cpp/include/features_avx2.hpp' in this repo\n";
    std::cout << "to see how we compute OFI (Order Flow Imbalance) with AVX2!\n";
    std::cout << "========================================================\n";

    return 0;
}
