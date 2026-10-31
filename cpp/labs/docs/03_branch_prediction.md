# Lab 03: Branch prediction

> **Learning-path stage:** [1](../../LEARNING_PATH.md#stage-1-how-the-cpu-actually-runs-your-code) · **Time:** 1–2 hours · **Needs:** labs 01–02; `perf` (installed)  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** measure what a wrong guess by the branch predictor costs, and watch the compiler take the problem away.

**How it works:** the loop sums the numbers above a threshold. The data follows an easy pattern (runs of 64 above, then 64 below), and `random_pct`% of it is overwritten with coin flips. The work never changes, only how predictable it is.

## Background

A modern core is a pipeline: it fetches, decodes and starts many instructions before earlier ones finish. When it reaches a conditional branch (an `if`), the outcome isn't known yet, so it **guesses** and keeps going down the predicted path (*speculative execution*).

```
   guess right:  ──fetch──decode──execute──retire──▶  no cost
   guess wrong:  ──fetch──decode──execute──✗ flush everything after the branch,
                                              restart on the correct path
```

- The **branch predictor** learns from history: per-branch patterns and the recent global sequence of taken/not-taken outcomes. Regular patterns (always taken, alternating, runs of 64) are learned almost perfectly. Coin flips can't be learned at all.
- A **misprediction** throws away the speculative work and refills the pipeline. On Zen 3 that is roughly 15–20 cycles.
- **Removing the branch.** `x >= t ? x : 0` can compile to a *conditional move* (`cmov`): compute both values, select one. No guess, no flush, but both sides are always computed. Compilers do this themselves (*if-conversion*), and at `-O3` they may also vectorize the loop, which removes the branch entirely.
- So the cost of a branch depends on the **data**, not just the code. That is why the same function can be fast in a benchmark and slow in production.

## Code tour

1. **`sum_branchy` and `sum_branchless`** in [lab03_branch_prediction.cpp](../lab03_branch_prediction.cpp): identical maths. `__attribute__((noinline))` keeps each as a separate function, so you can find it in the disassembly and it can't be merged into `main`.
2. **The data**: `v[i] = ((i / 64) % 2) ? 200 : 50` creates runs of 64 values above the threshold and 64 below, which is very predictable. Then `pct`% of positions are overwritten with random values.
3. **`result`**: printed so you can check every mode computes the same sum (and so the work can't be deleted).
4. **The Makefile**: the default build uses `-O2 -fno-if-conversion -fno-if-conversion2 -fno-tree-vectorize`, which stops GCC from removing the branch. `_O3` is a normal build. Read the comment in [Makefile](../Makefile).

## Predict
1. Sketch the time-vs-randomness curve before running. Is it a straight line? Where does it flatten?
2. Will `--mode=branchless` show the same curve?
3. The `_O3` build is a normal optimized build. Will it show the curve?

## Run
```bash
./build/lab03_branch_prediction
./build/lab03_branch_prediction --mode=branchless
./build/lab03_branch_prediction_O3
```

## Break it
- Count the mispredictions directly:
  ```bash
  perf stat -e branches,branch-misses ./build/lab03_branch_prediction --random_pct=0
  perf stat -e branches,branch-misses ./build/lab03_branch_prediction --random_pct=50
  ```
  Divide the extra time by the extra misses. That is your cost per misprediction in ns. Convert to cycles (your CPU runs at roughly 2.3–4.3 GHz).
- Look at the hot loop in each build: `objdump -d -C --no-show-raw-insn build/lab03_branch_prediction | grep -A30 "sum_branchy"`, then the same for `_O3`. Find the `j...` jump in one, and `cmov` or vector instructions in the other.
- Change the run length from 64 to 2 (`(i / 2) % 2` in the source). Is the 0% case still fast? What patterns can the predictor learn?

## Explain
- Why does the curve flatten around 50% rather than at 100%?
- The compiler removed your branch at `-O3`. Why not always write branchless code then? When is a branch *better* than a conditional move?

## What you should see

<details><summary>What you measured on this laptop</summary>

The default build (a real branch): **2.6 ms at 0% random**, rising steeply to **~15.5 ms from 70% random on**. About 6x for identical work. It flattens because a single coin flip is mispredicted only half the time, and once most elements are random the misprediction rate saturates. The `-O3` build runs at **~1.25 ms regardless of randomness**: GCC turned the branch into branch-free vector code. That is why the default build of this lab switches those optimizations off (`-fno-if-conversion -fno-tree-vectorize`). Expect a misprediction to cost around 5 ns, roughly 15–20 cycles.

A branch beats a conditional move when it is very predictable and one side is expensive. The branch lets the CPU skip the work entirely, while a conditional move always computes both sides.
</details>

## Going further

- **Find the exact instruction.** `perf record -e branch-misses ./build/lab03_branch_prediction --random_pct=50`, then `perf annotate sum_branchy`. Which instruction carries the misses?
- **Measure the predictor's memory.** Change the run length (`i / 64`) to 1, 2, 4, ..., 64 with `random_pct=0`. Which patterns does it learn perfectly?
- **Sort first.** Add `--sorted=1` that sorts `v` before summing. Is sorting plus a predictable branch faster than an unpredictable branch? Include the sort time.
- **Clang.** `make CXX=clang++ -B` and look at what Clang does with the default build's flags (some are GCC-specific).

---

← [Lab 02: Data layout (array of structs vs struct of arrays)](02_data_layout.md) · [All labs](../README.md) · [Lab 04: Timers and the disappearing benchmark](04_timers_and_dce.md) →
