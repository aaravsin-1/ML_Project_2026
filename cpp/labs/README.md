# Hands-on labs

Small programs you build, run, change and break, to see low-latency effects on your own machine instead of reading about them. They are the practical half of [LEARNING_PATH.md](../LEARNING_PATH.md). Labs 01–07 come with code. From stage 5 on you write the code yourself, following the step-by-step milestones in the learning path.

## How to work through a lab

Every lab follows the same five steps. Keep a lab notebook (a plain `LAB_NOTES.md` is enough), and write in it at steps 1, 3 and 5.

1. **Predict.** Before running anything, answer the lab's *Predict* questions in your notebook. Write down numbers, even rough guesses. A wrong prediction you can explain afterwards teaches more than a right one you can't.
2. **Run.** Use the exact commands given.
3. **Observe.** Write down the numbers you got next to your predictions.
4. **Break it.** Do the variations. Each changes *one* thing. Predict each result before you run it.
5. **Explain.** Answer the *Explain* questions in a few sentences each, without looking anything up. Then open *What you should see* and compare.

The numbers under *What you should see* were measured on your laptop (Ryzen 5 5625U, Fedora, GCC 15) in October 2026, unpinned and not idle. Yours will differ somewhat. The *shape* of the results is what matters.

## Setup

```bash
cd cpp/labs
make            # builds labs 01-06 into build/  (make clean to start over)
```

`make CXX=clang++` builds everything with Clang instead. Comparing the two compilers is a good extra experiment for any lab.

---

## Lab 01: The memory hierarchy

**Goal:** measure how long one memory access takes depending on where the data lives, and find your CPU's cache sizes from the numbers alone.

**How it works:** the program links the buffer into one random cycle and follows it. Each load needs the previous one's result to know where to go next, so the CPU can't overlap them. You see the raw latency.

**Predict**
1. Roughly how many ns per load for a 4 KB buffer, and for a 256 MB buffer?
2. At which buffer sizes do you expect the time to jump? (Look up the L1, L2 and L3 sizes of a Ryzen 5 5625U, or run `lscpu --caches`.)
3. Will a *sequential* walk over 256 MB be close to the random one, or much faster? Why?

**Run**
```bash
./build/lab01_memory_hierarchy
```

**Break it**
- `--order=seq`: the same buffer sizes, walked in order. What happened at the large sizes, and which piece of hardware is responsible?
- `--order=seq --stride=4096`: in order, but jumping 4 KB each time. Does the hardware still help? Why not?
- `--core=0` versus `--core=5`: does pinning change anything here?
- Run it while a browser tab plays a video. Which buffer sizes are affected most?

**Explain**
- Why does the random walk measure *latency* while the sequential walk measures something else?
- Your BTC pipeline keeps 1 order book of 384 bytes per update. Which level of the hierarchy does it live in, and so how much does memory matter to it?
- A strategy tracks 5,000 instruments, each with a 384-byte book. Which level does that live in, and what does that imply?

<details><summary>What you should see (open after you predict)</summary>

| Buffer | ns per load | Where it lives |
|:---|:---:|:---|
| 4–32 KB | ~1.0 | L1 data cache (32 KB per core) |
| 64–256 KB | ~3 | L2 (512 KB per core) |
| 512 KB–4 MB | ~7–15 | L3 |
| 8 MB | ~37 | L3 partly missing (L3 is 16 MB, shared, and the TLB starts to struggle) |
| 16 MB and up | ~100–115 | Main memory (RAM) |

A factor of about 100 between L1 and RAM, and the jumps line up with the cache sizes. The sequential walk stays fast even for huge buffers because the **hardware prefetcher** recognises the pattern and fetches lines before you ask. A 4 KB stride defeats it: the prefetcher doesn't cross page boundaries, and each access lands on a new page, which also costs TLB lookups.

For 5,000 instruments: 5,000 × 384 bytes ≈ 1.9 MB. That fits in L3 but not L2, so a random instrument's book costs around 10 ns just to reach. That is why real systems care which instruments are "hot".
</details>

---

## Lab 02: Data layout (array of structs vs struct of arrays)

**Goal:** see that the "optimized" layout is only optimal for some tasks. It uses the repo's own [lob.hpp](../include/lob.hpp) (array of structs) and [lob_soa.hpp](../include/lob_soa.hpp) (struct of arrays).

**Predict**
1. Run with `--layout_report=1` first, before predicting anything else. How many bytes does each layout use per book, and why is the struct-of-arrays version *bigger*?
2. For "sum all 10 bid quantities", which layout wins and by roughly how much?
3. For "level-1 mid price", which wins?

**Run**
```bash
./build/lab02_data_layout --layout_report=1
```

**Break it**
- `--books=20000` (everything fits in cache) versus `--books=2000000` (far bigger than cache). Does the gap grow or shrink, and why?
- Edit `lob_soa.hpp` to use 10 levels instead of 12 (`SIMD_LOB_LEVELS = 10`), rebuild with `make`, and rerun. What changed in size and speed? (Lab 06 will break. That is the point. Change it back afterwards.)
- Add a field to `LimitOrderBook` (say, a `double vwap`) and watch the offsets move with `--layout_report=1`.

**Explain**
- Why does struct-of-arrays win one task and lose the other? Think in cache lines touched per book.
- `best_of` timing hides noise. When is reporting the best run honest, and when is it misleading?

<details><summary>What you should see</summary>

Sizes: 384 bytes (6 cache lines) for array of structs, 448 bytes (7 lines) for struct of arrays. The SoA version pads each array to 12 levels so AVX2 can load 4 at a time, and pays 64 bytes per book for it.

With 500,000 books (~192 MB, much bigger than cache): summing 10 bid quantities took about **9.7 ms AoS vs 4.6 ms SoA**. SoA reads 80 contiguous bytes (2 cache lines). AoS strides through prices and quantities interleaved and touches 3 lines. Level-1 mid price took about **5.1 ms AoS vs 5.8 ms SoA**. AoS has both level-1 prices near each other in one line, while SoA keeps bid and ask prices 192 bytes apart on different lines, and the book is bigger overall.

The lesson: there is no universally best layout. You choose it for the access pattern of the hot path.
</details>

---

## Lab 03: Branch prediction

**Goal:** measure what a wrong guess by the branch predictor costs, and watch the compiler take the problem away.

**How it works:** the loop sums the numbers above a threshold. The data follows an easy pattern (runs of 64 above, then 64 below), and `random_pct`% of it is overwritten with coin flips. The work never changes, only how predictable it is.

**Predict**
1. Sketch the time-vs-randomness curve before running. Is it a straight line? Where does it flatten?
2. Will `--mode=branchless` show the same curve?
3. The `_O3` build is a normal optimized build. Will it show the curve?

**Run**
```bash
./build/lab03_branch_prediction
./build/lab03_branch_prediction --mode=branchless
./build/lab03_branch_prediction_O3
```

**Break it**
- Count the mispredictions directly:
  ```bash
  perf stat -e branches,branch-misses ./build/lab03_branch_prediction --random_pct=0
  perf stat -e branches,branch-misses ./build/lab03_branch_prediction --random_pct=50
  ```
  Divide the extra time by the extra misses. That is your cost per misprediction in ns. Convert to cycles (your CPU runs at roughly 2.3–4.3 GHz).
- Look at the hot loop in each build: `objdump -d -C --no-show-raw-insn build/lab03_branch_prediction | grep -A30 "sum_branchy"`, then the same for `_O3`. Find the `j...` jump in one, and `cmov` or vector instructions in the other.
- Change the run length from 64 to 2 (`(i / 2) % 2` in the source). Is the 0% case still fast? What patterns can the predictor learn?

**Explain**
- Why does the curve flatten around 50% rather than at 100%?
- The compiler removed your branch at `-O3`. Why not always write branchless code then? When is a branch *better* than a conditional move?

<details><summary>What you should see</summary>

The default build (a real branch): **2.6 ms at 0% random**, rising steeply to **~15.5 ms from 70% random on**. About 6x for identical work. It flattens because a single coin flip is mispredicted only half the time, and once most elements are random the misprediction rate saturates. The `-O3` build runs at **~1.25 ms regardless of randomness**: GCC turned the branch into branch-free vector code. That is why the default build of this lab switches those optimizations off (`-fno-if-conversion -fno-tree-vectorize`). Expect a misprediction to cost around 5 ns, roughly 15–20 cycles.

A branch beats a conditional move when it is very predictable and one side is expensive. The branch lets the CPU skip the work entirely, while a conditional move always computes both sides.
</details>

---

## Lab 04: Timers and the disappearing benchmark

**Goal:** find out what reading a clock costs, and see the compiler delete a benchmark. Read [common.hpp](common.hpp) first: every helper the labs use is in there, and you should be able to explain each line.

**Predict**
1. How many ns does it take to read `steady_clock` twice in a row with nothing in between? And `rdtsc`?
2. Part B computes a level-1 imbalance 2 million times in three ways. B1 throws the result away. What will B1 report?

**Run**
```bash
./build/lab04_timers_and_dce
```

**Break it**
- Find Part B's three loops in the assembly (`objdump -d -C --no-show-raw-insn build/lab04_timers_and_dce | less`, search for `divsd` or `vdivsd`). How many division instructions does B1's loop contain?
- Rebuild at `make OPT=-O0` and rerun. What happens to B1, and why is `-O0` never the right way to "fix" this?
- Remove the `"memory"` clobber from `do_not_optimize` in `common.hpp`, rebuild, and rerun B2. Anything change? (Try both compilers.)

**Explain**
- `imbalance()` takes about 1.5 ns. If you wrap a single call in `steady_clock` reads, what will you measure? How *should* you measure a 1.5 ns function?
- Now reread [main_benchmark.cpp](../src/main_benchmark.cpp) lines 118–153. What can its "feature extraction p50" number really tell you? (This is audit question 3 in the learning path.)

<details><summary>What you should see</summary>

Empty region: **steady_clock ~20 ns** (p50), **rdtsc ~10 ns**, **rdtscp ~20 ns**, all with rare maximums of several µs (an interrupt landed in the window). TSC at about 2.3 ticks per ns.

Part B: **B1 0.00 ns**. The compiler deleted the whole loop because nothing used the result. **B2 and B3 ~1.4–1.6 ns per item.** So one `imbalance()` costs about a tenth of a single clock read: timing it one call at a time measures the clock, not the function. Time many calls in a loop and divide (like B2 and B3), or use Google Benchmark, which does this for you.
</details>

---

## Lab 05: Noise and jitter

**Goal:** see that the median hides almost everything that goes wrong, and measure how often your OS takes the CPU away from you.

**Predict**
1. The same 200-step computation runs 100,000 times. How far apart will p50 and p99.9 be?
2. Spinning for 5 seconds, how many times will the program be interrupted for more than 1 µs?

**Run**
```bash
./build/lab05_noise_and_jitter
./build/lab05_noise_and_jitter --mode=gaps --seconds=5
```

**Break it** (one change at a time, rerun both commands each time, and record p50, p99, p99.9 and max)
- Pin to a core: `--core=3`.
- Add load. In a second terminal: `for i in 1 2 3 4; do yes > /dev/null & done`. Run with and without `--core=3`. Then `killall yes`.
- Unplug the laptop charger (power-saving mode), then plug it back in.
- Run while Firefox scrolls a heavy page.
- Advanced (stage 10): boot with `isolcpus=3 nohz_full=3`, rerun with `--core=3`, and compare the gap counts.

**Explain**
- Which percentile changed the most across conditions, and which barely moved? Why is that a problem for anyone who only reports averages?
- The gap mode lost about 0.4% of its time. That sounds small. Why does it matter to a trading system that has to react to one specific packet?

<details><summary>What you should see</summary>

Unpinned, unloaded: fixed work **p50 ~1,040 ns, p99 ~1,080 ns, p99.9 ~4,200 ns, max ~8,700 ns**. The median and p99 are within 4%, while p99.9 is 4x higher. The gap mode saw **about 1,100 interruptions longer than 1 µs per second**: median ~3.6 µs, worst ~280 µs, about 0.44% of the time lost.

Load and power saving mostly move p99.9 and max, not p50. A trading system is judged on whether it reacted to *this* packet in time, and an unlucky 280 µs pause at the wrong moment is a missed trade, however good the average looks.
</details>

---

## Lab 06: SIMD on a real feature

**Goal:** compare scalar and AVX2 versions of the project's multi-level order flow imbalance (OFI), check they agree, and find out when SIMD matters. It uses the repo's [features_avx2.hpp](../include/features_avx2.hpp).

**Predict**
1. How much faster will the AVX2 version be than the plain scalar `if`/`else` version: 2x, 4x, more?
2. `scalar_select` is the same maths written with ternaries. Will the compiler vectorize it?
3. A book has a NaN price (a data glitch). Will all three versions produce the same OFI?

**Run**
```bash
./build/lab06_simd_ofi
./build/lab06_simd_ofi_fastmath
```

**Break it**
- `--tick=0` (prices never change between books) versus `--tick=2` (prices jump at random). Which versions care, and why? Which setting looks more like real 1-second BTC books?
- See which loops the compiler vectorized:
  ```bash
  g++ -std=c++20 -O3 -march=native -fopt-info-vec-optimized -c lab06_simd_ofi.cpp -o /dev/null
  g++ -std=c++20 -O3 -march=native -ffast-math -fopt-info-vec-optimized -c lab06_simd_ofi.cpp -o /dev/null
  ```
- `make CXX=clang++ -B` and rerun. Does Clang vectorize `scalar_select` without fast-math?
- Introduce a bug on purpose in `features_avx2.hpp`: swap the two arguments of one `_mm256_blendv_pd`. Rebuild. Does the correctness check catch it?

**Explain**
- Why does `-ffast-math` make the scalar code so much faster? What exactly is the compiler allowed to assume, and why does that let it vectorize?
- With fast-math, the branchy version returns a different value for the NaN book. Which answer is "right", and why does it matter that the versions disagree?
- Given these results, would you ship the hand-written AVX2 or `scalar_select` with fast-math? What would make you decide?

<details><summary>What you should see</summary>

Without fast-math, at `--tick=2`: **scalar_branchy ~119 ns, scalar_select ~124 ns, AVX2 ~6 ns** per book, about 20x. GCC does not vectorize `scalar_select`, and both scalar versions branch on random price moves, so they mispredict a lot. At `--tick=0` the scalar versions drop to **~21–23 ns**: the branches become predictable. The AVX2 version doesn't care (~6 ns), because it has no branches.

With fast-math: **all three ~7 ns**. The compiler may now assume no NaNs, which makes the comparisons simple enough to vectorize. The price is the NaN book: **branchy returns -0 while the others return -30.9**. Fast-math made the program's behaviour on bad data depend on how the code happens to be written.

All 1.2 million random values agree in both builds, so the repo's AVX2 code is correct on normal data.
</details>

---

## Lab 07: The trained model in C++

**Goal:** rebuild the BTC LightGBM model as C++, confirm it matches LightGBM exactly, then measure it properly. This closes the project's open "rerun the C++ timing on an idle machine" item. Then work out why it takes ~10 µs.

**Setup** (once)
```bash
pip install --user lightgbm            # or use a virtual environment
python gen_trained_model.py           # writes cpp/generated/btc_lightgbm_model.hpp
make trained                           # about 10 s: the generated header is ~82,000 lines
```

**Predict**
1. The model has 327 trees and 20,601 leaves over 27 features. Each prediction walks one path through every tree. Guess the single-row latency.
2. Will the p99 be close to the p50?

**Run**
```bash
./build/trained_model_benchmark ../../artifacts/cpp_trained_model/features.bin \
    ../../artifacts/cpp_trained_model/lightgbm_raw_scores.bin 20000
```

**Break it**
- **The proper measurement.** Close everything, plug in the charger, and run it 10 times pinned (`taskset -c 3 ./build/trained_model_benchmark ...`). Report the median and spread of the 10 p50s and p99s. Put the result in the main README's C++ section, replacing the "measured under load" number.
- Count instruction cache misses and branch misses:
  ```bash
  perf stat -e instructions,cycles,branch-misses,L1-icache-load-misses ./build/trained_model_benchmark ...
  ```
  How big is the compiled binary (`size build/trained_model_benchmark`)? Compare with your 32 KB L1 instruction cache.
- Rebuild at `make OPT=-O1 -B trained` and `make OPT=-O2 -B trained`. Faster or slower than `-O3`? Why might more optimization hurt huge generated code?
- Add `-ffast-math` to the trained build. Does `max |diff|` stay 0?

**Explain**
- Where do the ~10 µs go: arithmetic, branch mispredictions, or instruction fetching? Back it up with your `perf` numbers.
- Which design from stage 4 (flat arrays, QuickScorer, batching) attacks that cost directly?

<details><summary>What you should see</summary>

Exactness: **max |diff| 0 and 100% class agreement** on all 20,000 rows. Latency in my unpinned, not-idle run: **p50 ~9.8 µs, p99 ~19 µs, p99.9 ~26 µs**. That is about 30 ns per tree, far more than ~8 comparisons should cost. The generated code is huge and full of unpredictable branches. Your `perf` numbers will show which of the two dominates. The proper idle, pinned measurement is yours to make.
</details>

---

## After lab 07

From here you write the code. The learning path has step-by-step milestones for each project:

| Stage | You build | Starting point |
|:---|:---|:---|
| 5 | Allocation counter, object pool, CRTP strategy | [Stage 5](../LEARNING_PATH.md#stage-5-low-latency-c-style) |
| 6 | L3 order book, then a fast version, then matching | [Stage 6](../LEARNING_PATH.md#stage-6-the-order-book-engine) |
| 7 | Nasdaq ITCH parser feeding your book | [Stage 7](../LEARNING_PATH.md#stage-7-market-data-binary-feeds) |
| 8 | SPSC queue, thread round-trip benchmark | [Stage 8](../LEARNING_PATH.md#stage-8-concurrency-without-locks) |
| 9 | Multicast replay, io_uring and AF_XDP receivers | [Stage 9](../LEARNING_PATH.md#stage-9-networking-and-kernel-bypass) |
| 10 | Isolated-core jitter measurements | [Stage 10](../LEARNING_PATH.md#stage-10-tuning-the-operating-system) |
| 11 | The full tick-to-trade pipeline | [Stage 11](../LEARNING_PATH.md#stage-11-capstone-tick-to-trade) |

Put each project in its own folder under `cpp/` (for example `cpp/orderbook/`), with its own tests and a benchmark, and keep using the five-step method: predict, run, observe, break, explain.
