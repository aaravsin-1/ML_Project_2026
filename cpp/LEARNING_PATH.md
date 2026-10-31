# Low-Latency C++ and HFT Systems: A Learning Path

A self-study route through the systems side of this project, from the C++ already in [cpp/](.) to the parts a real trading system has that this repo does not have yet (order book engines, market-data feeds, lock-free queues, kernel bypass).

**Two documents work together.** This file is the map: what to learn, in what order, and why. The hands-on part is [labs/](labs/README.md): runnable experiments you predict, run, break and explain, each tied to a stage below. Start there as soon as you finish setup.

Each stage builds on the previous one and has the same five parts:

- **Why it matters:** what problem the stage solves in a trading system.
- **Concepts:** what you should be able to explain afterwards.
- **In this repo:** the code to read, in order.
- **Do:** exercises. Most produce code or numbers you keep.
- **Checkpoint:** questions to answer *without notes* before moving on. If you can't, redo the exercises, not the reading.

Time estimates assume part-time study (around 8–10 hours a week). They are rough; going slower and doing every exercise beats going faster.

---

## Contents

0. [Setup](#stage-0-setup)
1. [How the CPU actually runs your code](#stage-1-how-the-cpu-actually-runs-your-code)
2. [Measuring honestly](#stage-2-measuring-honestly)
3. [SIMD](#stage-3-simd)
4. [Fast model inference](#stage-4-fast-model-inference)
5. [Low-latency C++ style](#stage-5-low-latency-c-style)
6. [The order book engine](#stage-6-the-order-book-engine)
7. [Market data: binary feeds](#stage-7-market-data-binary-feeds)
8. [Concurrency without locks](#stage-8-concurrency-without-locks)
9. [Parallelism and throughput](#stage-9-parallelism-and-throughput)
10. [Networking and kernel bypass](#stage-10-networking-and-kernel-bypass)
11. [Tuning the operating system](#stage-11-tuning-the-operating-system)
12. [Capstone: tick-to-trade](#stage-12-capstone-tick-to-trade)
- [Audit: what is wrong with the existing C++](#audit-what-is-wrong-with-the-existing-c)
- [Resources](#resources)

---

## Stage 0: Setup

**Your machine:** AMD Ryzen 5 5625U (Zen 3), Fedora. It has **AVX2 and FMA, but no AVX-512**, so everything in this repo runs, and AVX-512 material is read-only for you. `g++`, `clang++`, `perf` and `cmake` are installed.

**Build commands:**

```bash
# The labs (start here)
cd cpp/labs && make && ./build/lab01_memory_hierarchy

# From the repo root: the SIMD tutorial
mkdir -p build
g++ -std=c++20 -O3 -march=native cpp/tutorial/learn_simd.cpp -o build/learn_simd && ./build/learn_simd

# The pipeline benchmark (features + random tree ensemble + simulated order)
g++ -std=c++20 -O3 -march=native -I cpp/include cpp/src/main_benchmark.cpp -o build/main_benchmark && ./build/main_benchmark
```

All of these were checked on your laptop. `cpp/src/optimization_comparison.cpp` does **not** build on Linux yet: it includes `<windows.h>`. Porting it is exercise 2.4.

**Tools you will use throughout:**

| Tool | What for |
|:---|:---|
| [Compiler Explorer](https://godbolt.org) | See the assembly any snippet compiles to. Use it constantly |
| `objdump -d -C --no-show-raw-insn` | Same, for your real binaries |
| `perf stat`, `perf record`, `perf report` | Count cycles, cache misses and branch misses; find hot spots |
| `taskset -c N` | Pin a program to one CPU core |
| `valgrind --tool=cachegrind` | Simulate the cache behaviour of a program |

**Checkpoint:** you can build and run the labs and both programs, and open `main_benchmark` in `perf report` (`perf record -g ./build/main_benchmark`, then `perf report`).

---

## Stage 1: How the CPU actually runs your code

*About 2 weeks.*

**Why it matters.** Every later optimization (struct-of-arrays, branchless code, SIMD, pinning) is a response to one of four hardware facts: memory is slow, wrong guesses are expensive, the CPU does several things per cycle, and the OS interrupts you. You cannot reason about speed without this model.

**Concepts**
- The memory hierarchy: registers, L1/L2/L3 caches, RAM. Rough latencies (about 1 ns for L1, tens of ns for L3, around 100 ns for RAM) and why the gap dominates everything.
- Cache lines (64 bytes), spatial and temporal locality, hardware prefetching.
- Pipelines and out-of-order execution: why a modern core can retire several instructions per cycle, and what stops it (dependency chains).
- Branch prediction: what a misprediction costs (around 15–20 cycles on Zen 3), and why random data is the worst case.
- The TLB and why huge pages exist (used again in stages 10–11).

**In this repo**
1. [lob.hpp](include/lob.hpp): the order book as an *array of structs* (each level is a `{price, quantity}` pair).
2. [lob_soa.hpp](include/lob_soa.hpp): the same data as a *struct of arrays* (all bid prices together, all bid quantities together), padded to 12 levels.
3. [tree_engine.hpp](include/tree_engine.hpp): a normal decision tree. Each node decides left or right with an `if`, then jumps to a child somewhere else in memory ("pointer chasing").
4. [tree_engine_branchless.hpp](include/tree_engine_branchless.hpp): an *oblivious* tree, where every node at the same depth tests the same feature, so the leaf index can be computed with bit operations and no jumps.

**Labs:** [01 memory hierarchy](labs/README.md#lab-01-the-memory-hierarchy), [02 data layout](labs/README.md#lab-02-data-layout-array-of-structs-vs-struct-of-arrays), [03 branch prediction](labs/README.md#lab-03-branch-prediction). Do these first; the exercises below go further.

**Do**
- **1.1** Draw the memory layout of `LimitOrderBook` and `LimitOrderBookSoA` byte by byte: offsets, padding, and which cache line each field lands in. Check your drawing with `static_assert(offsetof(...) == ...)` and `sizeof`.
- **1.2** Write a microbenchmark that sums an array of 64 MB by walking it (a) in order, (b) with a stride of 64 bytes, (c) in random order. Plot time per element. Explain the three numbers with the hierarchy above.
- **1.3** Classic branch-prediction experiment: sum the elements of a random array that are above a threshold, once unsorted and once sorted. Then make the loop branchless (`sum += (x > t) * x`). Run all three under `perf stat -e branches,branch-misses`.
- **1.4** Put `DecisionTree::predict` and `ObliviousTreeDepth4::predict` into Compiler Explorer. Find the jumps in the first and confirm the second has none.

**Checkpoint**
- Why is the struct-of-arrays layout better for computing something across all 10 bid levels, and when would the array-of-structs layout be *better*?
- Your sorted-vs-unsorted loop: why does sorting the data change the speed when it does the same work?
- What does an oblivious tree give up in exchange for having no branches? (Hint: is it the same model as LightGBM's trees?)

---

## Stage 2: Measuring honestly

*About 2 weeks. The most important stage: every number after this depends on it.*

**Why it matters.** Latency numbers are easy to produce and easy to get wrong. A benchmark can be dominated by the clock it uses, optimized away by the compiler, distorted by CPU frequency changes, or compare two programs that do different work. The C++ timing in this project's README was measured while other jobs used the CPU, and has never been rerun properly. Rerunning it is your job in this stage.

**Concepts**
- Clocks: `std::chrono::steady_clock`, `rdtsc`/`rdtscp`, the invariant TSC, and what each costs to read (often 20+ ns, comparable to what you are measuring).
- Dead-code elimination: why the compiler can delete the work you are timing, and how to stop it (`benchmark::DoNotOptimize`, `volatile` sinks, using the result).
- Warm-up, steady state, and why the first iterations are slow (cold caches, page faults, branch predictor training, CPU frequency ramp).
- Percentiles versus means. Why trading cares about p99 and p99.9 ("tail latency"), and why you report a distribution, not one number.
- Noise: frequency scaling and boost, other processes, interrupts, SMT siblings. Pinning with `taskset`.
- Fair comparisons: same inputs, same work, same compiler flags.

**In this repo**
1. [main_benchmark.cpp](src/main_benchmark.cpp): times each stage per event with `high_resolution_clock`.
2. [trained_model_benchmark.cpp](src/trained_model_benchmark.cpp): times the *trained* LightGBM model compiled to C++, checks it matches LightGBM exactly, and reports percentiles plus a batch number that amortizes the clock cost.
3. [optimization_comparison.cpp](src/optimization_comparison.cpp): the "baseline vs hand-optimized" comparison behind the claimed 2.41x speedup.

**Labs:** [04 timers and dead-code elimination](labs/README.md#lab-04-timers-and-the-disappearing-benchmark), [05 noise and jitter](labs/README.md#lab-05-noise-and-jitter), [07 the trained model](labs/README.md#lab-07-the-trained-model-in-c).

**Do**
- **2.1** Measure the cost of reading the clock: time an empty region 10 million times with `steady_clock` and with `__rdtsc()`. Compare with the "feature extraction" numbers `main_benchmark` reports. What can you conclude about those numbers?
- **2.2** Learn [Google Benchmark](https://github.com/google/benchmark) (`sudo dnf install google-benchmark-devel`). Rewrite the stage-1 experiments with it.
- **2.3** **Rerun the trained-model benchmark properly** ([lab 07](labs/README.md#lab-07-the-trained-model-in-c)). [gen_trained_model.py](labs/gen_trained_model.py) rebuilds the C++ model from the saved model file (no BTC data needed). Run it idle, pinned and plugged in, 10 times, report the spread, and update the README table. This closes an open item in the project.
- **2.4** Port `optimization_comparison.cpp` to Linux: replace `SetThreadPriority` with `sched_setaffinity` (pinning) and optionally `SCHED_FIFO`. Then read the [audit](#audit-what-is-wrong-with-the-existing-c) questions about it before trusting its output.

**Checkpoint**
- Your clock costs 20 ns to read and the function costs 5 ns. How do you measure the function?
- Why can the p99 latency be 5x the median on a laptop, and which of those numbers matters to a trading system?
- Name three ways `optimization_comparison.cpp`'s two sides differ *other than* the optimization being tested.

---

## Stage 3: SIMD

*About 2 weeks.*

**Why it matters.** One AVX2 instruction operates on 256 bits at once: 4 doubles, 8 floats or 32 bytes. Order-book features across 10 levels, parsing bytes, and evaluating many rows of a model are natural fits.

**Concepts**
- Vector registers (`__m256d`, `__m256`, `__m256i`), lanes, and the intrinsics naming scheme (`_mm256_<op>_<type>`).
- Loads and stores, aligned versus unaligned, and why the struct-of-arrays layout matters here.
- Masks and blends: replacing `if` with "compute both answers, then select per lane".
- Horizontal operations (summing lanes) and why they are comparatively slow.
- Auto-vectorization: when the compiler does it for you (`-O3 -march=native`, `-fopt-info-vec`), and what stops it (branches, aliasing, unknown trip counts).
- Floating-point details: `-ffast-math` allows reordering and assumes no NaN or infinity, which can change results.

**In this repo**
1. [tutorial/learn_simd.cpp](tutorial/learn_simd.cpp) and its [README](tutorial/README.md): five lessons, from loading registers to a branchy-vs-AVX2 benchmark (about 1.65x on your laptop).
2. [features_avx2.hpp](include/features_avx2.hpp): multi-level order flow imbalance for 12 levels in 3 blocks of 4, with `_mm256_cmp_pd` and `_mm256_blendv_pd` replacing the three-way `if`.
3. [features.hpp](include/features.hpp): the scalar version, to compare against.

**Lab:** [06 SIMD on a real feature](labs/README.md#lab-06-simd-on-a-real-feature).

**Do**
- **3.1** Work through the tutorial, then do its suggested exercise (weighted mid-price with intrinsics).
- **3.2** Write a scalar reference for the AVX2 OFI and a test that feeds 1 million random books through both and checks they agree. Include equal prices, padded zero levels, and a NaN. What does `-ffast-math` do to the NaN case?
- **3.3** Write the scalar OFI loop so the compiler auto-vectorizes it (hint: no `if`, use the ternary operator on arrays). Check with `-fopt-info-vec-optimized` and Compiler Explorer, then race it against the hand-written AVX2.
- **3.4** Vectorize *across books instead of across levels*: compute level-1 imbalance for 4 consecutive snapshots at once. Which layout do you need, and when is this the better choice?

**Checkpoint**
- Why does `features_avx2.hpp` pad the book to 12 levels?
- Explain line by line how the two `_mm256_blendv_pd` calls implement the three cases of OFI.
- When is hand-written SIMD worth it over letting the compiler vectorize?

---

## Stage 4: Fast model inference

*About 1–2 weeks.*

**Why it matters.** The model is the part of the pipeline most people try to speed up first. This repo already compiles the trained LightGBM model into C++ that matches LightGBM exactly, which is a good base to experiment on.

**Concepts**
- How a gradient-boosted tree ensemble produces a prediction (sum of tree outputs per class, then softmax).
- Three ways to run trees: walking a node array (pointer chasing), generated `if`/`else` code (what this repo does), and branch-free layouts.
- QuickScorer: evaluating all trees with bitvectors instead of walking them.
- Single-row latency versus batch throughput, and why they lead to different designs.
- Exactness: float versus double thresholds, and why the generated code casts to `double`.

**In this repo**
1. [scripts/lgbm_to_cpp.py](../scripts/lgbm_to_cpp.py): `tree_to_cpp()` turns each tree into nested `if`/`else`.
2. [trained_model_benchmark.cpp](src/trained_model_benchmark.cpp): checks exactness, then times it.
3. [tree_engine.hpp](include/tree_engine.hpp) and [tree_engine_branchless.hpp](include/tree_engine_branchless.hpp) again, now as alternative engines.

**Do**
- **4.1** Read one generated tree function. Count how many comparisons a prediction makes on average (trees × depth). Does that explain the latency you measured in 2.3?
- **4.2** Write a second engine for the same model that stores all trees in flat arrays and walks them in a loop. Check exactness against `lightgbm_raw_scores.bin`, then compare latency with the generated code.
- **4.3** Read the QuickScorer paper (see [resources](#resources)) and explain its idea in your own words in a few paragraphs. Optional: implement it for the trained model.
- **4.4** Batch: evaluate 8 rows at once with AVX2 (one row per float lane). What happens to per-row cost? When would a trading system have 8 rows at once?

**Checkpoint**
- Why does the generated code match LightGBM exactly but would likely stop matching with `-ffast-math`?
- For this model (327 trees, 27 features), which costs more: the features or the model? How do you know?

---

## Stage 5: Low-latency C++ style

*About 2 weeks, overlapping with stages 6–7.*

**Why it matters.** Stages 1–4 make a single function fast. A system also needs code that stays fast: no surprise allocations, no hidden copies, nothing that occasionally takes a slow path at the worst moment.

**Concepts**
- Keeping the hot path predictable: no heap allocation (`new`, growing `std::vector`, `std::string`), no exceptions thrown, no virtual calls in tight loops, no locks, no system calls, no logging that formats strings.
- Allocation strategies: preallocated arrays, object pools, arena allocators, `std::pmr`.
- Compile-time work: templates, `constexpr`, CRTP (static polymorphism instead of virtual functions).
- Move semantics, references versus copies, `[[likely]]`/`[[unlikely]]`, `noexcept`.
- Fast logging: write raw binary records to a ring buffer and format them on another thread.

**In this repo**
- Look for allocations and copies on the hot path of `main_benchmark.cpp` and `features_avx2.hpp`. (For example: what does `prev_book_ = book` copy, and how often?)

**Do**
- **5.1** Write a custom `operator new` that counts calls, link it into `main_benchmark`, and prove the timed loop allocates nothing. If it does, fix it.
- **5.2** Replace a virtual-function interface (write a small `Strategy` base class with two implementations) with CRTP. Compare the generated assembly.
- **5.3** Write a fixed-capacity object pool (preallocated array plus a free list) and use it in stage 6.

**Checkpoint**
- List five things that must never happen on the hot path, and for each, where it hides in normal-looking C++.

---

## Stage 6: The order book engine

*About 3 weeks. Your first big build.*

**Why it matters.** Everything in this repo works on *snapshots*: 10 price levels, refreshed every second. Real exchanges send individual events (add order, cancel, execute, replace), and each trading system maintains its own full order book from them. This is the core data structure of the field.

**Concepts**
- Level 2 (aggregated price levels) versus level 3 (every individual order).
- Price-time priority, and what "queue position" means.
- Data structure choices: `std::map` of levels (simple, slow), a flat array indexed by price tick around the current price (fast, the usual choice), intrusive linked lists of orders per level, a hash map from order ID to order.
- Matching: what happens when an incoming order crosses the spread.

**Do**
- **6.1** Build an L3 order book: `add(id, side, price, qty)`, `cancel(id)`, `execute(id, qty)`, `replace(id, new_price, new_qty)`, `best_bid()`, `best_ask()`, `depth(n)`. Start with `std::map` + `std::unordered_map` and test it thoroughly.
- **6.2** Rewrite it with a price-indexed array, intrusive lists and the object pool from 5.3. Benchmark both with Google Benchmark on a realistic mix of operations (mostly adds and cancels near the top of the book).
- **6.3** Add a matching engine: incoming limit and market orders that cross the spread trade against the book in price-time order.
- **6.4** Port the Python simulator fix to [simulator.hpp](include/simulator.hpp): it still fills orders larger than the visible book at the last price plus $0.05. See `_walk_book` in [src/execution/simulator.py](../src/execution/simulator.py).

**Checkpoint**
- Why is a flat array indexed by price usually faster than a tree map, and what happens when the price drifts far from where the array was centred?
- What is the time complexity of each operation in your fast version?

---

## Stage 7: Market data: binary feeds

*About 2 weeks.*

**Why it matters.** Exchanges publish market data in compact binary formats sent over UDP multicast. A feed handler parses those messages and updates the order book. It is the first stage of every trading system, and a large share of the latency.

**Concepts**
- Nasdaq TotalView-ITCH 5.0: fixed-layout binary messages (add order, cancel, execute, ...), big-endian integers, prices as fixed-point integers.
- Parsing without copies: reading fields straight out of the buffer with `memcpy` and byte swaps; why `reinterpret_cast` to a packed struct is risky.
- Fixed-point prices versus `double` (exactness, comparisons, hashing).
- Sequence numbers, gaps and recovery (how a feed handler notices lost packets).
- A brief look at order-entry protocols: FIX (text) and exchange binary protocols such as Nasdaq OUCH.

**Do**
- **7.1** Download a Nasdaq ITCH 5.0 sample file (see [resources](#resources); several GB, use one day) and write a parser that counts messages by type. Measure parse speed in messages per second.
- **7.2** Feed the parsed messages into your stage-6 order book and rebuild the full book for one stock (for example AAPL). Print the top 5 levels at a few timestamps.
- **7.3** Compute this project's features (OFI, imbalance, micro-price) from your own book, on every event instead of once a second. That is the event-level data the BTC experiments never had.
- **7.4** Profile the whole replay with `perf`. Where does the time go: parsing, book updates or features?

**Checkpoint**
- Why do exchanges use binary formats and UDP rather than JSON and TCP?
- How does your feed handler know it missed a message?

---

## Stage 8: Concurrency without locks

*About 2 weeks.*

**Why it matters.** A trading system usually runs one thread per job (feed handler, strategy, order gateway, logger), each pinned to its own core and spinning instead of sleeping. Passing data between them without locks is a core skill.

**Concepts**
- Why locks hurt latency: contention, and the OS putting threads to sleep and waking them.
- `std::atomic`, the C++ memory model, and memory orders (`relaxed`, `acquire`, `release`, `seq_cst`).
- The single-producer single-consumer (SPSC) ring buffer, the standard building block.
- False sharing: two threads writing variables on the same cache line, and fixing it with `alignas(64)` padding.
- Busy-polling versus blocking, and the CPU cost of spinning.

**Do**
- **8.1** Write an SPSC ring buffer with `std::atomic` head and tail indices. Test it with one producer and one consumer pushing 100 million integers, checking order and completeness.
- **8.2** Measure round-trip latency between two threads pinned to different cores with your queue, then with a `std::mutex` + `std::condition_variable` queue. Compare p50 and p99.
- **8.3** Remove the `alignas(64)` padding between head and tail and measure again (false sharing).
- **8.4** Compare with Erik Rigtorp's SPSCQueue (see [resources](#resources)) and read his code to see what he does differently.
- **8.5** Split your stage-7 replay into a parser thread and a book/feature thread connected by your queue.

**Checkpoint**
- Why does an SPSC queue need `acquire`/`release` and not just `relaxed`? What could go wrong otherwise?
- What is false sharing, and how would you spot it with `perf`?

---

## Stage 9: Parallelism and throughput

*About 2 weeks.*

**Why it matters.** Stages 1–8 make one message fast on one core: that is *latency*. Research and backtesting have the opposite problem: process a whole day (or a year) of data as fast as possible on every core you have, which is *throughput*. The same feature code is written differently for the two, and knowing which one you are optimizing is half the skill. The research spec also proposes GPUs (TensorRT) for DeepLOB inference, which is parallelism on a different kind of hardware.

**Concepts**
- **OpenMP:** `#pragma omp parallel for`, `schedule(static)` vs `schedule(dynamic, chunk)`, `reduction(+:x)`, `#pragma omp simd`, `num_threads`, `OMP_PROC_BIND` / `OMP_PLACES` for pinning.
- **Standard C++ parallelism:** `std::thread` and `std::jthread`, writing a thread pool with a work queue, `std::execution::par` / `par_unseq` algorithms. On GCC these need Intel TBB as the backend; without it they quietly run on one thread.
- **Intel TBB (oneTBB):** `parallel_for`, `parallel_reduce`, task-based work stealing.
- **Measuring scaling:** speedup, parallel efficiency, **Amdahl's law** (the serial part caps the speedup), and why compute-bound work scales while **memory-bandwidth-bound** work stops at a few cores.
- **SMT (hyper-threading):** your 6 cores show up as 12 threads; why the 12th thread can make things slower.
- **False sharing in parallel loops:** per-thread counters or results that share a cache line (stage 8 again, now at full scale).
- **Latency vs throughput:** per-message processing on one pinned core versus batched processing across all cores, and why a trading system usually wants the first while its research pipeline wants the second.
- **GPUs:** the CUDA model (kernels, threads, blocks, warps, memory transfers), why GPUs win on big batches and lose on single small requests, and inference engines (TensorRT) that compile a trained network for low-latency GPU inference. **Your laptop has no NVIDIA GPU**, so CUDA work happens on Colab.

**In this repo**
- [labs/lab08_parallel_scaling.cpp](labs/lab08_parallel_scaling.cpp): OpenMP scaling sweep, false sharing and latency-vs-batch, ready to run.
- The Python side does this already: LightGBM's `n_jobs=-1` and PyTorch's batched GPU training. Experiment H's run took 76 s of wall time but 5 m 45 s of CPU time, a speedup of about 4.5x across 12 threads.

**Lab:** [08 parallel scaling](labs/README.md#lab-08-parallel-scaling).

**Do**
- **9.1** Do lab 08. Fit Amdahl's law to your compute-bound curve: what serial fraction explains it?
- **9.2** Install TBB (`sudo dnf install tbb-devel`), rewrite lab 08's loop with `std::for_each(std::execution::par, ...)` linked with `-ltbb`, then with `tbb::parallel_for`. Compare all three with OpenMP. Then build the `std::execution::par` version *without* `-ltbb` and confirm it runs on one thread.
- **9.3** Write your own thread pool (a fixed set of `std::jthread` workers taking tasks from a mutex-protected queue). Use it to compute features for the five LOBSTER stocks in parallel, one stock per task. Where does it stop scaling, and why?
- **9.4** Port the AVX2 OFI from stage 3 to process a whole day of books with OpenMP *and* SIMD (`#pragma omp parallel for simd`, or your intrinsics inside an OpenMP loop). How close do you get to the machine's memory bandwidth (measure it with lab 08's light mode)?
- **9.5** On Colab: write a CUDA kernel (`%%writefile kernel.cu`, then `nvcc`) that computes level-1 imbalance for millions of books. Time it including the host-to-device copy, then excluding it. At what batch size does the GPU beat your 12-thread CPU version?
- **9.6** On Colab: export the trained DeepLOB to ONNX, build a TensorRT engine, and measure single-sample and batch-of-1024 latency against plain PyTorch. Compare with the project's measured 5.7 ms PyTorch DeepLOB latency and the ~10 µs C++ LightGBM.

**Checkpoint**
- Why did the light (memory-bound) loop stop scaling at about 1.4x while the heavy one reached about 6.7x?
- 95% of a program parallelizes perfectly. What is the best possible speedup on 12 threads, and on infinitely many?
- A model takes 2 µs per message on one core. Would a GPU make a live trading system faster? Would it make a backtest faster? Why the difference?

---

## Stage 10: Networking and kernel bypass

*About 3 weeks.*

**Why it matters.** Receiving a packet through the normal Linux network stack costs several microseconds: interrupts, copying from kernel to user space, system calls. Kernel bypass lets your program read packets straight from the network card's memory, cutting that to around a microsecond or less. It is the step that separates "fast software" from HFT infrastructure.

**Concepts**
- The normal path: NIC → interrupt → kernel driver → socket buffer → `recv()` copy → your program.
- UDP multicast (how exchange data is distributed).
- `epoll` and busy-polling sockets (`SO_BUSY_POLL`).
- `io_uring`: fewer system calls, shared rings between kernel and program.
- AF_XDP: the kernel hands raw packets to your program through shared memory. Available in standard Linux, so it works on your laptop.
- DPDK: the program drives the NIC itself with poll-mode drivers. The industry-standard open-source option.
- Vendor stacks: Solarflare (AMD) OpenOnload and ef_vi, Mellanox/NVIDIA VMA. They need special network cards.
- Hardware timestamping (NIC timestamps, PTP clock sync) for measuring latency on the wire.
- Beyond software: FPGAs in trading (conceptual only).

**What your laptop can and cannot do.** You can learn sockets, multicast, `epoll`, `io_uring` and AF_XDP (on the loopback or a virtual interface, or your Wi-Fi card in generic mode). DPDK runs with its AF_XDP or virtual devices for learning. Real microsecond-level numbers need a server with a supported NIC (cloud instances with ENA or Mellanox cards, or a used Solarflare card in a desktop), so treat timings on the laptop as relative.

**Do**
- **10.1** Write a UDP multicast sender that replays your ITCH messages and a receiver that feeds them into the stage-7 handler. Measure one-way latency on the same machine (send timestamp in the packet).
- **10.2** Rewrite the receiver with `io_uring`, then with busy-polling. Compare latency distributions.
- **10.3** Write an AF_XDP receiver (start from the `xdp-tutorial` repo in [resources](#resources)). Compare with 9.1.
- **10.4** Optional: set up DPDK with the `net_af_xdp` or `net_tap` driver and receive the same packets.
- **10.5** Write a one-page summary of how ef_vi or OpenOnload differs from AF_XDP and DPDK, from their documentation.

**Checkpoint**
- Walk through every copy and context switch between a packet hitting the NIC and your `recv()` returning. Which ones does each bypass technique remove?
- Why do trading firms busy-poll and burn a whole core instead of waiting for interrupts?

---

## Stage 11: Tuning the operating system

*About 1 week.*

**Why it matters.** Even perfect code is interrupted by the OS: the scheduler moves threads, interrupts arrive, the CPU changes frequency or sleeps. Production systems remove these sources of jitter.

**Concepts**
- Core isolation (`isolcpus`, `nohz_full`, `rcu_nocbs`), and pinning threads to the isolated cores.
- IRQ affinity: moving network interrupts away from (or onto) specific cores.
- CPU frequency and sleep states: the performance governor, disabling deep C-states, turbo and its unpredictability.
- Huge pages to avoid TLB misses; `mlockall` to avoid page faults.
- SMT (hyper-threading): why HFT systems often disable it.

**Do**
- **11.1** Measure the jitter of an empty busy loop (record the largest gaps between consecutive `rdtsc` reads) on a normal core. Then boot with `isolcpus=3 nohz_full=3` (Fedora: `grubby --update-kernel=ALL --args=...`, revert afterwards) and measure on core 3. Compare the gap histograms.
- **11.2** Rerun the 2.3 benchmark on the isolated core with the performance governor. How much did p99.9 improve?
- **11.3** Read Erik Rigtorp's low-latency tuning guide (see [resources](#resources)) and list which items apply to a laptop and which need a server.

**Checkpoint**
- What does `nohz_full` remove, and why does it only help a core running a single thread?

---

## Stage 12: Capstone: tick-to-trade

*About 4 weeks.*

Put the pieces together into one system and measure it end to end, from a packet arriving to an order leaving:

```
UDP multicast replay ─▶ feed handler ─▶ L3 order book ─▶ features ─▶ model ─▶ risk checks ─▶ order out
   (stage 10)            (stage 7)       (stage 6)        (stage 3)  (stage 4)  (new)        (UDP/TCP)
```

- Each stage on its own pinned core, connected by your SPSC queues (stage 8).
- A risk-check layer before any order: maximum position, maximum order size, price sanity bands, and a kill switch.
- Timestamps at every boundary, written to a lock-free log and analysed afterwards (stage 2 methods).
- Replay a full day of ITCH data and report the tick-to-trade latency distribution, with a breakdown by stage.

A finished capstone is a strong portfolio project: a feed handler, order book and full pipeline you built and measured yourself, on real exchange data.

---

## Audit: what is wrong with the existing C++

The C++ in this repo is a starting point, not production code, and some of it overclaims. Investigating it is one of the best exercises here. Try each question before opening the hint.

1. **Does `optimization_comparison.cpp` measure what it says?** It reports a 2.41x speedup for "struct of arrays + AVX2 + branchless trees" over the baseline.
   <details><summary>Hint</summary>Compare the two loops line by line: how many features does each compute, how deep and how big are the two tree ensembles, and are they even the same kind of model? Also look at the thresholds and leaf values: with every threshold 0 and every leaf the same, how hard is the baseline's branch prediction really?</details>

2. **Are the comments in `features_avx2.hpp` and `tree_engine_branchless.hpp` true?** For example "4 levels per cycle", "100% pipeline saturation", "branchless class decision via ternary cmov".
   <details><summary>Hint</summary>Look at <code>FastBranchlessEnsemble::predict_class</code>: it ends in <code>if</code> statements. Check in Compiler Explorer whether they become conditional moves or jumps. For "per cycle" claims, count the instructions in one loop iteration and look up their throughput in Agner Fog's instruction tables.</details>

3. **What does `main_benchmark.cpp` actually time?** The README says feature extraction is "below timer resolution" and the output calls the total "wire-to-order".
   <details><summary>Hint</summary>Measure what one <code>high_resolution_clock::now()</code> call costs (exercise 2.1) and compare. And is there any wire in this benchmark?</details>

4. **Is `-ffast-math` safe here?** `CMakeLists.txt` and the README build command use it.
   <details><summary>Hint</summary>Fast-math lets the compiler assume no NaNs and reorder floating-point sums. Which guards in <code>features.hpp</code> rely on comparisons that could change? Would the trained-model benchmark still match LightGBM exactly with it?</details>

5. **Is the C++ simulator correct?** Compare `MicrosecondSimulator::execute_aggressive` in [simulator.hpp](include/simulator.hpp) with the Python simulator.
   <details><summary>Hint</summary>Look at what happens to an order larger than the visible book, and how slippage is signed. Exercise 6.4.</details>

6. **Does the build system build everything?**
   <details><summary>Hint</summary>Read <code>CMakeLists.txt</code>: which of the three programs does it build? Fixing it so <code>cmake --build</code> builds all three, plus your tests, is a good first CMake exercise.</details>

---

## Resources

**CPU and memory (stages 1–2)**
- Randal Bryant and David O'Hallaron, *Computer Systems: A Programmer's Perspective*, chapters 5–6 (program optimization, memory hierarchy). The best single foundation.
- Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007, free PDF). Old but still the reference on caches.
- Denis Bakhvalov, *Performance Analysis and Tuning on Modern CPUs* (free online) and his blog [easyperf.net](https://easyperf.net). Practical `perf` and microarchitecture.
- Agner Fog, [optimization manuals](https://www.agner.org/optimize/): *Optimizing software in C++*, the microarchitecture guide, and the instruction tables.
- Brendan Gregg, *Systems Performance* (2nd ed.) and [brendangregg.com/perf.html](https://www.brendangregg.com/perf.html).
- Fedor Pikus, *The Art of Writing Efficient Programs* (2021). C++-focused, covers branches, memory and concurrency.

**SIMD (stage 3)**
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html): look up every intrinsic you use.
- Daniel Lemire's blog ([lemire.me/blog](https://lemire.me/blog/)): many short, measured SIMD articles.

**Model inference (stage 4)**
- Lucchese et al., "QuickScorer: a Fast Algorithm to Rank Documents with Additive Ensembles of Regression Trees", SIGIR 2015.
- [Treelite / TL2cgen](https://github.com/dmlc/treelite): production tree compilers, the same idea as this repo's code generator.

**HFT systems in C++ (stages 5–8, 11)**
- Carl Cook, "When a Microsecond Is an Eternity: High Performance Trading Systems in C++", CppCon 2017 (YouTube). Start here.
- David Gross, "Trading at light speed: designing low latency systems in C++", Meeting C++ 2022 (YouTube).
- Sourav Ghosh, *Building Low Latency Applications with C++* (Packt, 2023). Builds a full trading system: exchange, order book, feed handler, strategies.
- Anthony Williams, *C++ Concurrency in Action* (2nd ed.). The reference for atomics and the memory model.
- Jeff Preshing's blog ([preshing.com](https://preshing.com)), especially the posts on acquire and release semantics.
- Erik Rigtorp: [SPSCQueue](https://github.com/rigtorp/SPSCQueue) and his [low-latency tuning guide](https://rigtorp.se/low-latency-guide/).
- WK Selph, "How to Build a Fast Limit Order Book" (2011 blog post, widely mirrored). A short classic on order book data structures.

**Market data (stage 7)**
- Nasdaq TotalView-ITCH 5.0 specification (PDF, on nasdaqtrader.com).
- Sample ITCH files: [emi.nasdaq.com/ITCH](https://emi.nasdaq.com/ITCH/). Real full-day feeds, several GB each.

**Parallelism and GPUs (stage 9)**
- Ruud van der Pas, Eric Stotzer and Christian Terboven, *Using OpenMP: The Next Step* (MIT Press, 2017).
- Michael Voss, Rafael Asenjo and James Reinders, *Pro TBB: C++ Parallel Programming with Threading Building Blocks* (Apress, 2019, free to download).
- Wen-mei Hwu, David Kirk and Izzat El Hajj, *Programming Massively Parallel Processors* (4th ed., 2022). The standard CUDA textbook.
- NVIDIA's CUDA C++ Programming Guide and TensorRT documentation (docs.nvidia.com).

**Networking and kernel bypass (stages 10–11)**
- Jens Axboe, "Efficient IO with io_uring" (kernel.dk), and the *Lord of the io_uring* guide.
- [xdp-project/xdp-tutorial](https://github.com/xdp-project/xdp-tutorial): hands-on AF_XDP and XDP lessons.
- [DPDK documentation](https://doc.dpdk.org), starting with the getting-started guide and the sample applications.
- AMD (Solarflare) OpenOnload and ef_vi user guides, for how vendor kernel bypass works.
- The Linux kernel documentation on `isolcpus`, `nohz_full` and IRQ affinity.
