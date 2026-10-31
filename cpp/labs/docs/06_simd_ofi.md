# Lab 06: SIMD on a real feature

> **Learning-path stage:** [3](../../LEARNING_PATH.md#stage-3-simd) · **Time:** 2–3 hours · **Needs:** labs 01–05 and the [SIMD tutorial](../../tutorial/README.md)  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** compare scalar and AVX2 versions of the project's multi-level order flow imbalance (OFI), check they agree, and find out when SIMD matters. It uses the repo's [features_avx2.hpp](../../include/features_avx2.hpp).

## Background

**SIMD** (single instruction, multiple data): one instruction operates on several values held side by side in a wide register. With AVX2, a 256-bit `__m256d` holds 4 doubles:

```
  bid_prices (now)   [ p0 | p1 | p2 | p3 ]
  bid_prices (prev)  [ q0 | q1 | q2 | q3 ]
  _mm256_cmp_pd(>)   [ ✓  | ✗  | ✗  | ✓  ]   a mask: all-ones or all-zeros per lane
  _mm256_blendv_pd   picks per lane between two vectors using that mask
```

- **No branches.** Instead of `if` per element, compute every possible answer for all lanes, then select per lane with a mask. That turns OFI's three-way `if` into compares and blends.
- **Loads.** `_mm256_load_pd` needs a 32-byte-aligned address; `_mm256_loadu_pd` doesn't. That's why `lob_soa.hpp` uses `alignas(32)` and pads to 12 levels (a multiple of 4).
- **Auto-vectorization.** The compiler vectorizes simple loops by itself at `-O3 -march=native` if it can prove that's safe: no data-dependent branches it can't convert, no aliasing between inputs and outputs, and results identical to the scalar code.
- **Floating-point rules.** Comparisons involving NaN are always false, and floating-point addition isn't associative. These rules can block vectorization. `-ffast-math` tells the compiler to ignore them: it may assume no NaN or infinity, reorder sums and fuse multiply-adds. Faster, but results can change.

Your CPU has AVX2 and FMA but **not AVX-512**.

## Code tour

1. **`ofi_branchy`** in [lab06_simd_ofi.cpp](../lab06_simd_ofi.cpp): the textbook definition, one `if`/`else if`/`else` per side per level.
2. **`ofi_select`**: the same maths written with ternaries, the form compilers like to vectorize.
3. **[features_avx2.hpp](../../include/features_avx2.hpp)**, `AVX2FeatureEngine::compute`: for each block of 4 levels, it loads current and previous prices and sizes, makes `>` and `==` masks, computes the three candidate answers, and blends twice. Map every intrinsic to a line of `ofi_branchy`.
4. **The correctness check**: 100,000 random book pairs through all three versions. `same()` treats two NaNs as equal.
5. **`random_book` and `--tick`**: how far prices may move between books, which controls how predictable the scalar branches are (lab 03 again).

## Predict
1. How much faster will the AVX2 version be than the plain scalar `if`/`else` version: 2x, 4x, more?
2. `scalar_select` is the same maths written with ternaries. Will the compiler vectorize it?
3. A book has a NaN price (a data glitch). Will all three versions produce the same OFI?

## Run
```bash
./build/lab06_simd_ofi
./build/lab06_simd_ofi_fastmath
```

## Break it
- `--tick=0` (prices never change between books) versus `--tick=2` (prices jump at random). Which versions care, and why? Which setting looks more like real 1-second BTC books?
- See which loops the compiler vectorized:
  ```bash
  g++ -std=c++20 -O3 -march=native -fopt-info-vec-optimized -c lab06_simd_ofi.cpp -o /dev/null
  g++ -std=c++20 -O3 -march=native -ffast-math -fopt-info-vec-optimized -c lab06_simd_ofi.cpp -o /dev/null
  ```
- `make CXX=clang++ -B` and rerun. Does Clang vectorize `scalar_select` without fast-math?
- Introduce a bug on purpose in `features_avx2.hpp`: swap the two arguments of one `_mm256_blendv_pd`. Rebuild. Does the correctness check catch it?

## Explain
- Why does `-ffast-math` make the scalar code so much faster? What exactly is the compiler allowed to assume, and why does that let it vectorize?
- With fast-math, the branchy version returns a different value for the NaN book. Which answer is "right", and why does it matter that the versions disagree?
- Given these results, would you ship the hand-written AVX2 or `scalar_select` with fast-math? What would make you decide?

## What you should see

<details><summary>What you measured on this laptop</summary>

Without fast-math, at `--tick=2`: **scalar_branchy ~119 ns, scalar_select ~124 ns, AVX2 ~6 ns** per book, about 20x. GCC does not vectorize `scalar_select`, and both scalar versions branch on random price moves, so they mispredict a lot. At `--tick=0` the scalar versions drop to **~21–23 ns**: the branches become predictable. The AVX2 version doesn't care (~6 ns), because it has no branches.

With fast-math: **all three ~7 ns**. The compiler may now assume no NaNs, which makes the comparisons simple enough to vectorize. The price is the NaN book: **branchy returns -0 while the others return -30.9**. Fast-math made the program's behaviour on bad data depend on how the code happens to be written.

All 1.2 million random values agree in both builds, so the repo's AVX2 code is correct on normal data.
</details>

## Going further

- **llvm-mca** (installed) estimates the cycles per iteration of a loop from its assembly. Compile the AVX2 loop to assembly (`-S`), extract the loop, and run `llvm-mca -mcpu=znver3` on it. Does the estimate match your measurement?
- **Four books per register.** Write a version that computes level-1 OFI for 4 consecutive books at once (lanes = books, not levels). Which layout do you need? (Learning-path exercise 3.4.)
- **AVX-512 elsewhere.** Colab CPUs are often Intel Xeons; check `lscpu` there for `avx512f`. If it's present, write a 512-bit version and compare.
- **Find the vectorized loops.** Use `-fopt-info-vec-missed` to see *why* GCC refused to vectorize `scalar_select` without fast-math.

---

← [Lab 05: Noise and jitter](05_noise_and_jitter.md) · [All labs](../README.md) · [Lab 07: The trained model in C++](07_trained_model.md) →
