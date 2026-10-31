# SIMD & AVX2 Hands-On Tutorial for Quant Engineers

Welcome to the interactive SIMD / AVX2 tutorial! This module is designed to teach you modern vectorization and branchless programming in C++ from scratch, using high-frequency trading (HFT) and limit order book (LOB) microstructure examples.

---

## 1. Quick Start

Compile and run the tutorial from the repo root (Linux; on Windows with MinGW the same `g++` command works):

```bash
mkdir -p build
g++ -std=c++20 -O3 -march=native cpp/tutorial/learn_simd.cpp -o build/learn_simd
./build/learn_simd
```

This tutorial is part of [LEARNING_PATH.md](../LEARNING_PATH.md) (stage 3). The matching hands-on experiment is [lab 06](../labs/docs/06_simd_ofi.md).

---

## 2. What Is Covered

The tutorial in [`learn_simd.cpp`](learn_simd.cpp) walks through 5 self-contained lessons:

1. **Lesson 1: What is `__m256d` & How to Load Data**
   - Understand 256-bit registers (which hold **4 x 64-bit doubles** simultaneously).
   - Learn how to load data with `_mm256_setr_pd`, broadcast with `_mm256_set1_pd`, and load from memory with `_mm256_loadu_pd`.

2. **Lesson 2: Parallel Vector Arithmetic**
   - Execute 4 subtractions and 4 multiplications in a single CPU cycle (`_mm256_sub_pd`, `_mm256_mul_pd`).
   - Calculate order book spreads and mid-prices across 4 depth levels at once.

3. **Lesson 3: Branchless Conditionals (The HFT Secret Weapon)**
   - Why `if / else` statements stall the CPU pipeline (15–20 wasted cycles on branch mispredictions).
   - Creating bitmasks with `_mm256_cmp_pd`.
   - Instantaneous hardware selection with `_mm256_blendv_pd` (0 branch instructions).

4. **Lesson 4: Real Microstructure Feature — 4-Level Order Book Imbalance**
   - Implement $I_k = \frac{\text{BidVol}_k - \text{AskVol}_k}{\text{BidVol}_k + \text{AskVol}_k}$ for $k \in \{1, 2, 3, 4\}$.
   - All 4 levels computed in parallel with a single SIMD division instruction.

5. **Lesson 5: Microbenchmark — Branchy Scalar C++ vs Hand-Coded AVX2**
   - Compilers fail to auto-vectorize branching microstructure formulas (such as Cont et al. Order Flow Imbalance).
   - Empirically see hand-coded AVX2 achieve a **speedup (about 1.65x on a Ryzen 5 5625U)** with 100% bitwise arithmetic equivalence.

---

## 3. How to Learn Hands-On

1. Open [`cpp/tutorial/learn_simd.cpp`](learn_simd.cpp).
2. Modify the values in `lesson1_loading_data()` or `lesson4_real_world_hft_feature()` to experiment with different bid/ask levels.
3. Try adding a new feature, such as calculating the **weighted mid-price**:
   $$P_{\text{wmid}} = \frac{P_{\text{bid}} \cdot V_{\text{ask}} + P_{\text{ask}} \cdot V_{\text{bid}}}{V_{\text{bid}} + V_{\text{ask}}}$$
   using `_mm256_mul_pd`, `_mm256_add_pd`, and `_mm256_div_pd`.
4. Recompile and run to verify your SIMD calculations against regular C++!

---

## 4. Production Reference in this Repository

Once you're comfortable with the basics:
- Look at [`cpp/include/lob_soa.hpp`](../include/lob_soa.hpp) to see how order book data is organized in a **Struct-of-Arrays (SoA)** layout for cache efficiency.
- Look at [`cpp/include/features_avx2.hpp`](../include/features_avx2.hpp) to see production AVX2 computing Multi-Level OFI across all 10 book levels.
- Look at [`cpp/include/tree_engine_branchless.hpp`](../include/tree_engine_branchless.hpp) to see how decision tree branches are evaluated branchlessly with bitwise operations.
- The head-to-head engine benchmark, [optimization_comparison.cpp](../src/optimization_comparison.cpp), is still Windows-only. Porting it to Linux is exercise 2.4 in the learning path, and the audit section there asks whether its 2.41x figure is a fair comparison.
