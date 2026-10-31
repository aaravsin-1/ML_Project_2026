# Lab 07: The trained model in C++

> **Learning-path stage:** [2 and 4](../../LEARNING_PATH.md#stage-4-fast-model-inference) · **Time:** 2–4 hours · **Needs:** labs 01–06; Python with `lightgbm` (`pip install --user lightgbm`, or the project's `.venv`)  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** rebuild the BTC LightGBM model as C++, confirm it matches LightGBM exactly, then measure it properly. This closes the project's open "rerun the C++ timing on an idle machine" item. Then work out why it takes ~10 µs.

## Background

A gradient-boosted tree model (LightGBM here) predicts by adding up the outputs of many small decision trees. For 3 classes, there are 3 sets of trees. Each tree is walked from the root to one leaf by comparing one feature with one threshold per node. The class scores then go through a softmax.

```
   tree k:      x[3] <= 0.12 ?
               /              \
        x[7] <= -1.4 ?      x[0] <= 2.0 ?
          /      \             /      \
       leaf     leaf        leaf     leaf      → add the leaf value to its class score
```

There are three common ways to run this in C++:

1. **Walk a node array** (`tree_engine.hpp`): nodes in memory, a loop following child indices. Simple, but it chases pointers and branches on every node.
2. **Generate code** (what this lab does, like Treelite or lleaves): every tree becomes nested `if`/`else`. The thresholds become constants in the instructions, so nothing is loaded from a node array, but the program becomes huge.
3. **Branch-free layouts** (QuickScorer, oblivious trees): reorganize the trees so prediction becomes bit operations, with no unpredictable jumps.

Two hardware limits matter for option 2:
- **Instruction cache.** Code must be fetched just like data. The L1 instruction cache is 32 KB. Code much bigger than that keeps missing it.
- **Branch prediction.** Each node is a branch whose direction depends on the input features, so it's hard to predict (lab 03).

**Exactness.** LightGBM stores thresholds as doubles and compares in double. The generated code casts each `float` feature to `double` before comparing, so its results match LightGBM exactly.

## Code tour

1. **[scripts/lgbm_to_cpp.py](../../../scripts/lgbm_to_cpp.py)**: `tree_to_cpp()` recursively turns LightGBM's JSON dump of one tree into nested `if`/`else`; `generate_cpp()` writes one function per tree plus `predict_raw()`, which sums the trees of each class.
2. **[gen_trained_model.py](../gen_trained_model.py)**: loads the saved BTC model file and calls `generate_cpp()`. Open the generated `cpp/generated/btc_lightgbm_model.hpp` and read one tree function.
3. **[trained_model_benchmark.cpp](../../src/trained_model_benchmark.cpp)**:
   - `read_bin` loads the 20,000 feature rows and LightGBM's own scores, saved from Python.
   - The correctness loop compares every score and the predicted class.
   - The warm-up loop runs 2,000 predictions untimed.
   - The per-row loop times each prediction with `steady_clock` (lab 04: what does that add?).
   - The batch loop times all rows together to amortize the clock cost.

## Setup (once)
```bash
pip install --user lightgbm            # or use a virtual environment
python gen_trained_model.py           # writes cpp/generated/btc_lightgbm_model.hpp
make trained                           # about 10 s: the generated header is ~82,000 lines
```

## Predict
1. The model has 327 trees and 20,601 leaves over 27 features. Each prediction walks one path through every tree. Guess the single-row latency.
2. Will the p99 be close to the p50?

## Run
```bash
./build/trained_model_benchmark ../../artifacts/cpp_trained_model/features.bin \
    ../../artifacts/cpp_trained_model/lightgbm_raw_scores.bin 20000
```

## Break it
- **The proper measurement.** Close everything, plug in the charger, and run it 10 times pinned (`taskset -c 3 ./build/trained_model_benchmark ...`). Report the median and spread of the 10 p50s and p99s. Put the result in the main README's C++ section, replacing the "measured under load" number.
- Count instruction cache misses and branch misses:
  ```bash
  perf stat -e instructions,cycles,branch-misses,L1-icache-load-misses ./build/trained_model_benchmark ...
  ```
  How big is the compiled binary (`size build/trained_model_benchmark`)? Compare with your 32 KB L1 instruction cache.
- Rebuild at `make OPT=-O1 -B trained` and `make OPT=-O2 -B trained`. Faster or slower than `-O3`? Why might more optimization hurt huge generated code?
- Add `-ffast-math` to the trained build. Does `max |diff|` stay 0?

## Explain
- Where do the ~10 µs go: arithmetic, branch mispredictions, or instruction fetching? Back it up with your `perf` numbers.
- Which design from stage 4 (flat arrays, QuickScorer, batching) attacks that cost directly?

## What you should see

<details><summary>What you measured on this laptop</summary>

Exactness: **max |diff| 0 and 100% class agreement** on all 20,000 rows. Latency in my unpinned, not-idle run: **p50 ~9.8 µs, p99 ~19 µs, p99.9 ~26 µs**. That is about 30 ns per tree, far more than ~8 comparisons should cost. The generated code is huge and full of unpredictable branches. Your `perf` numbers will show which of the two dominates. The proper idle, pinned measurement is yours to make.
</details>

## Going further

- **Where does the time go?** `perf record ./build/trained_model_benchmark ...` then `perf report`. Are a few trees hot, or is it spread evenly?
- **Optimize for size.** Build with `make OPT=-Os -B trained`. Smaller code fits the instruction cache better. Faster or slower?
- **Profile-guided optimization (PGO).** Build with `-fprofile-generate`, run once, then rebuild with `-fprofile-use`. The compiler learns which branches go which way. How much does p50 improve?
- **Flat arrays.** Learning-path exercise 4.2: store all trees in arrays and walk them in a loop. Compare latency, and compare binary size with `size`.
- **Clang.** `make CXX=clang++ -B trained`.

---

← [Lab 06: SIMD on a real feature](06_simd_ofi.md) · [All labs](../README.md) · [Lab 08: Parallel scaling](08_parallel_scaling.md) →
