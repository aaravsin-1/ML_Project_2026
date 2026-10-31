# Lab 04: Timers and the disappearing benchmark

> **Learning-path stage:** [2](../../LEARNING_PATH.md#stage-2-measuring-honestly) · **Time:** 1–2 hours · **Needs:** labs 01–03  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** find out what reading a clock costs, and see the compiler delete a benchmark. Read [common.hpp](../common.hpp) first: every helper the labs use is in there, and you should be able to explain each line.

## Background

Every benchmark has two hidden enemies: the cost of measuring, and the compiler.

**Clocks.**
- `std::chrono::steady_clock::now()` usually becomes a `clock_gettime` call through the vDSO (a kernel page mapped into your process, so no real system call). Cheap, but not free.
- `rdtsc` reads the CPU's *time-stamp counter*. On modern CPUs it ticks at a constant rate whatever the core frequency (look for `constant_tsc` and `nonstop_tsc` in `/proc/cpuinfo`), so it measures wall time, not core cycles.
- `rdtsc` can execute before earlier instructions finish (out-of-order execution). `rdtscp` waits for them, which makes it more precise and slightly slower.
- Whichever clock you use, reading it costs time. If that cost is close to what you measure, the measurement is mostly the clock.

**The compiler.** The optimizer may legally delete or move any computation whose result it can prove you don't use (*dead-code elimination*). It can also precompute constants (*constant folding*) and pull repeated work out of loops (*loop-invariant code motion*). A benchmark that ignores its result can end up timing an empty loop.

**The fix.** Make the result *observable*. `lab::do_not_optimize(x)` in [common.hpp](../common.hpp) is an empty inline-assembly statement that claims to read `x`, so the compiler must compute it. The `"memory"` clobber tells the compiler that memory may have changed, so it can't cache values across it. Google Benchmark's `benchmark::DoNotOptimize` works the same way.

## Code tour

1. **[common.hpp](../common.hpp)**: read every helper. `rdtsc`, `rdtscp`, `now_ns`, `pin_to_core` (`sched_setaffinity`), `tsc_per_ns` (calibrates the TSC against `steady_clock` over 200 ms), and `print_percentiles`.
2. **Part A** in [lab04_timers_and_dce.cpp](../lab04_timers_and_dce.cpp): three loops that read a clock twice with nothing in between, two million times each.
3. **Part B**: the same imbalance computed three ways. B1 discards the result, B2 passes it to `do_not_optimize`, and B3 calls a `noinline` function and sums the results (a real data dependency).
4. Compare the three B loops in `objdump -d -C --no-show-raw-insn build/lab04_timers_and_dce`. Search for `vdivsd`.

## Predict
1. How many ns does it take to read `steady_clock` twice in a row with nothing in between? And `rdtsc`?
2. Part B computes a level-1 imbalance 2 million times in three ways. B1 throws the result away. What will B1 report?

## Run
```bash
./build/lab04_timers_and_dce
```

## Break it
- Find Part B's three loops in the assembly (`objdump -d -C --no-show-raw-insn build/lab04_timers_and_dce | less`, search for `divsd` or `vdivsd`). How many division instructions does B1's loop contain?
- Rebuild at `make OPT=-O0` and rerun. What happens to B1, and why is `-O0` never the right way to "fix" this?
- Remove the `"memory"` clobber from `do_not_optimize` in `common.hpp`, rebuild, and rerun B2. Anything change? (Try both compilers.)

## Explain
- `imbalance()` takes about 1.5 ns. If you wrap a single call in `steady_clock` reads, what will you measure? How *should* you measure a 1.5 ns function?
- Now reread [main_benchmark.cpp](../../src/main_benchmark.cpp) lines 118–153. What can its "feature extraction p50" number really tell you? (This is audit question 3 in the learning path.)

## What you should see

<details><summary>What you measured on this laptop</summary>

Empty region: **steady_clock ~20 ns** (p50), **rdtsc ~10 ns**, **rdtscp ~20 ns**, all with rare maximums of several µs (an interrupt landed in the window). TSC at about 2.3 ticks per ns.

Part B: **B1 0.00 ns**. The compiler deleted the whole loop because nothing used the result. **B2 and B3 ~1.4–1.6 ns per item.** So one `imbalance()` costs about a tenth of a single clock read: timing it one call at a time measures the clock, not the function. Time many calls in a loop and divide (like B2 and B3), or use Google Benchmark, which does this for you.
</details>

## Going further

- **Google Benchmark.** `sudo dnf install google-benchmark-devel`, then rewrite Part B as a Google Benchmark (link with `-lbenchmark -lpthread`). Compare its per-iteration time with B2 and B3.
- **Clock cost in cycles.** Use `perf stat -e cycles,instructions` on Part A alone. How many instructions does one `steady_clock::now()` take?
- **Compiler Explorer.** Paste B1's loop and compare `-O0`, `-O1`, `-O2` and `-O3`. At which level does it disappear?
- **`lfence`.** Add a version that wraps the timed region with `_mm_lfence(); rdtsc` on both sides. Does it change the empty-region distribution?

---

← [Lab 03: Branch prediction](03_branch_prediction.md) · [All labs](../README.md) · [Lab 05: Noise and jitter](05_noise_and_jitter.md) →
