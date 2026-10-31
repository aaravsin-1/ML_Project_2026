# Glossary

Short definitions of the terms the labs and the learning path use. The lab in brackets is where it first matters.

**Aliasing** (06): two pointers that may refer to the same memory. It stops the compiler from reordering or vectorizing, because a write through one might change what the other reads. `__restrict` promises it won't happen.

**Alignment** (02): an address being a multiple of some size. `alignas(64)` puts a struct at the start of a cache line; aligned SIMD loads need 32-byte alignment for AVX2.

**Amdahl's law** (08): the speedup on *n* workers is at most 1 / ((1 − p) + p / n), where *p* is the fraction of the work that parallelizes. The serial part caps the speedup.

**Arithmetic intensity** (08): operations per byte moved from memory. Low intensity means memory-bound, high means compute-bound.

**AVX2** (06): the 256-bit SIMD instruction set your CPU supports. Registers are `ymm0`–`ymm15`.

**Bandwidth** (01): bytes per second the memory system can deliver when many requests are in flight. Compare *latency*.

**Branch predictor** (03): the hardware that guesses the direction of each conditional branch before it's known, from past behaviour.

**Branch misprediction** (03): a wrong guess. The CPU throws away the speculative work and restarts, costing roughly 15–20 cycles on Zen 3.

**Cache line** (01): the 64-byte unit in which memory moves between DRAM and the caches.

**Cache coherence** (08): the protocol that keeps each core's cached copies consistent. A core must take ownership of a whole line before writing to it.

**Clobber** (04): telling the compiler, in inline assembly, that something (here, all of memory) may have changed. Used by `lab::clobber_memory()`.

**Conditional move (`cmov`)** (03): an instruction that copies a value only if a condition holds. A select without a branch.

**Constant folding** (04): the compiler computing an expression at compile time when all inputs are known.

**CRTP** (learning path, stage 5): the curiously recurring template pattern, a way to get polymorphism at compile time instead of through virtual functions.

**Dead-code elimination** (04): the compiler removing computations whose results are never used.

**DRAM** (01): main memory. Large and slow compared with the caches.

**False sharing** (08): threads writing different variables that happen to share a cache line, so the line bounces between cores on every write.

**Fast-math** (06): `-ffast-math`, which lets the compiler assume no NaN or infinity and reorder floating-point operations. Faster; results can differ.

**FMA** (06): fused multiply-add, computing a × b + c in one instruction with one rounding.

**Huge pages** (01): 2 MB memory pages instead of 4 KB, so the TLB covers much more memory.

**If-conversion** (03): the compiler replacing a branch with branch-free code such as a conditional move.

**Instruction cache (L1i)** (07): the cache for program code, 32 KB per core on your CPU. Huge code doesn't fit and keeps missing.

**IPC** (tools): instructions per cycle. Low IPC means the core spends most cycles waiting (on memory, mispredictions and so on).

**Jitter** (05): variation in how long the same work takes, caused by interruptions and changing conditions.

**Latency** (01): the time for one operation from start to finish. Compare *throughput* and *bandwidth*.

**Markout** (main project): how a fill's price compares with the mid price some time later. Negative means the fill was adversely selected.

**Memory-bound / compute-bound** (08): whether a program's speed is limited by moving data or by doing arithmetic.

**Multiplexing** (tools): perf rotating more events than there are hardware counters, then scaling the counts.

**Oblivious tree** (main repo, `tree_engine_branchless.hpp`): a decision tree where every node at the same depth tests the same feature, so the leaf can be found with bit operations.

**OpenMP** (08): compiler pragmas (`#pragma omp ...`) for parallel loops and reductions. Enabled with `-fopenmp`.

**Percentile (p50, p99, p99.9)** (05): the value below which that share of samples falls. p50 is the median; p99.9 is the worst 1 in 1,000.

**PGO** (07): profile-guided optimization. Compile, run on representative input, then recompile using the recorded profile.

**Pinning** (05): restricting a thread to one CPU core with `taskset`, `sched_setaffinity` or OpenMP's `OMP_PLACES`.

**Pointer chasing** (01): a chain of loads where each address comes from the previous load, so they can't overlap.

**Prefetcher** (01): hardware that detects access patterns and fetches cache lines before the program asks for them.

**Reduction** (08): combining per-thread partial results (sums, maxima) into one result.

**Roofline model** (08): a picture of the best achievable speed as limited either by compute or by memory bandwidth times arithmetic intensity.

**Scheduling, static vs dynamic** (08): how OpenMP hands out loop iterations: fixed shares decided up front, or chunks taken as threads become free.

**SIMD** (06): single instruction, multiple data. One instruction operates on several values held in a wide register.

**SMT** (08): simultaneous multithreading (Intel calls it hyper-threading). Two hardware threads share one physical core.

**SoA / AoS** (02): struct of arrays (each field stored contiguously) vs array of structs (each record stored contiguously).

**Speculative execution** (03): the CPU running instructions past an unresolved branch on its predicted path.

**Tail latency** (05): the slow end of the latency distribution (p99 and beyond).

**Throughput** (08): work completed per unit of time. Compare *latency*.

**TLB** (01): translation lookaside buffer, the cache of virtual-to-physical address translations.

**TSC** (04): the time-stamp counter, read with `rdtsc` or `rdtscp`. Ticks at a constant rate on modern CPUs.

**Vectorization** (06): turning a scalar loop into SIMD instructions, by hand (intrinsics) or by the compiler (auto-vectorization).

**vDSO** (04): a small piece of kernel code mapped into every process, so calls like `clock_gettime` avoid a real system call.

---

[All labs](../README.md) · [Tools guide](tools.md) · [Troubleshooting](troubleshooting.md)
