# Tools guide

The tools the labs use, how to run them on this laptop (Fedora, Ryzen 5 5625U), and how to read their output. Run commands from `cpp/labs`.

| Tool | Installed? | Used for |
|:---|:---:|:---|
| `perf` | yes | Hardware counters (cycles, cache and branch misses), profiling |
| `objdump` | yes | Disassembling your binaries |
| [Compiler Explorer](https://godbolt.org) | web | Seeing the assembly for a snippet, instantly |
| `taskset`, `chrt` | yes | Pinning to a core, real-time priority |
| `cpupower` | yes | CPU frequency governor |
| `llvm-mca` | yes | Estimating cycles per loop iteration from assembly |
| `gdb` | yes | Debugging (the labs build with `-g`) |
| Google Benchmark | no | `sudo dnf install google-benchmark-devel` |
| `valgrind` (cachegrind) | no | `sudo dnf install valgrind` |
| `hyperfine` | no | `sudo dnf install hyperfine` (timing whole commands) |
| Intel TBB | no | `sudo dnf install tbb-devel` (stage 9) |

---

## perf stat: count hardware events

```bash
perf stat -e cycles,instructions,branches,branch-misses ./build/lab03_branch_prediction --random_pct=50
```

What the output looked like on this laptop:

```
      45,16,87,281      cycles:u                                   (82.96%)
      76,38,66,424      instructions:u                             (82.96%)
      13,31,24,574      branches:u                                 (82.96%)
       1,08,03,297      branch-misses:u                            (83.57%)
```

How to read it:
- **The digit grouping is Indian** (`45,16,87,281` = 451,687,281) because of your system locale. For standard grouping run `LC_ALL=C perf stat ...`.
- **`:u`** means user space only. Your kernel setting `perf_event_paranoid = 2` lets normal users count only their own code, which is all the labs need. Kernel time isn't included.
- **The percentage on the right** shows multiplexing. The CPU has only a few hardware counters, so when you ask for more events than it has, perf rotates them and scales up the counts. 83% means each event was counted for 83% of the run. Ask for fewer events at once for exact counts.
- **Useful ratios:** instructions per cycle (IPC = instructions / cycles; above 2 is good, below 1 means the core is mostly waiting), and branch-miss rate (branch-misses / branches).
- `perf list` shows every event; `perf list cache` shows the cache events your CPU exposes.

## perf record / report / annotate: find the hot code

```bash
perf record -g ./build/lab03_branch_prediction --random_pct=50   # any lab binary; writes perf.data
perf report                              # interactive: which functions took the time
perf annotate sum_branchy                # per-instruction view of one function
perf record -e branch-misses ./build/lab03_branch_prediction --random_pct=50   # sample on an event instead of time
```

The labs build with `-g`, so perf can show source lines next to the assembly. Delete `perf.data` files when you're done (they can get large).

## objdump: read your binary's assembly

```bash
objdump -d -C --no-show-raw-insn build/lab03_branch_prediction | less
# then search (/) for the function name, e.g. sum_branchy
objdump -d -C --no-show-raw-insn build/lab03_branch_prediction | grep -A40 "<sum_branchy"
```

`-C` turns C++ symbol names back into readable form. Things to look for:
- **Jumps:** `jmp` (always) and `j<cond>` such as `jle`, `jne` (conditional, so the predictor has to guess).
- **Conditional moves:** `cmov<cond>`, a select with no branch.
- **Vector instructions:** names starting with `v` and using `ymm` registers (256-bit AVX2), such as `vaddpd ymm0, ...`. `xmm` registers are 128-bit; scalar double maths also uses them (`vaddsd`, where `sd` = scalar double and `pd` = packed double).

## Compiler Explorer (godbolt.org)

The fastest way to see what the compiler does with a few lines of code.
- Choose **x86-64 gcc** (the newest version) and set the flags to `-O3 -march=znver3`, which targets your CPU's architecture (Zen 3). Try `-O2` and `-O0` to compare.
- Coloured highlighting links each source line to its assembly.
- Add a second compiler pane (Clang) to compare compilers side by side.
- For SIMD code, add `#include <immintrin.h>`.

## taskset and chrt: control where and how your program runs

```bash
taskset -c 3 ./build/lab05_noise_and_jitter          # run only on CPU 3
sudo chrt -f 50 taskset -c 3 ./build/lab05_...       # real-time FIFO priority 50, on CPU 3
```

CPUs 0–11 are hardware threads: two per physical core. `lscpu -e` shows which CPU numbers share a core. Real-time priority can freeze the machine if your program spins forever, so only use it with programs that finish.

## cpupower: CPU frequency

Your CPU uses the `amd-pstate` driver with the `powersave` governor and a "balance_performance" preference, so the clock speed changes all the time. For benchmarking:

```bash
sudo cpupower frequency-set -g performance   # before measuring
sudo cpupower frequency-set -g powersave     # afterwards, to restore the default
```

Always benchmark plugged in. On battery the CPU slows down further.

## llvm-mca: estimate a loop's speed from its assembly

```bash
g++ -std=c++20 -O3 -march=native -S -o lab06.s lab06_simd_ofi.cpp
# copy the instructions of one hot loop into loop.s, then:
llvm-mca -mcpu=znver3 loop.s
```

It reports the estimated cycles per iteration and which execution ports are the bottleneck. It's a model, not a measurement, so compare it with what you measure.

## Google Benchmark (after installing)

```cpp
#include <benchmark/benchmark.h>
static void BM_imbalance(benchmark::State& s) {
    double b = 120, a = 95;
    for (auto _ : s) benchmark::DoNotOptimize((b - a) / (b + a));
}
BENCHMARK(BM_imbalance);
BENCHMARK_MAIN();
```

Build with `g++ -O3 -march=native bench.cpp -lbenchmark -lpthread`. It picks the iteration count, handles warm-up and reports time per iteration. Use `--benchmark_repetitions=10` to see the spread.

---

[All labs](../README.md) · [Troubleshooting](troubleshooting.md) · [Glossary](glossary.md)
