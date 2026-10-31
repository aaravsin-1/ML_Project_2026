# Lab 08: Parallel scaling

> **Learning-path stage:** [9](../../LEARNING_PATH.md#stage-9-parallelism-and-throughput) · **Time:** 2–3 hours · **Needs:** labs 01–05; OpenMP (comes with GCC, already works)  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** see when more cores help and when they stop helping, see false sharing at full scale, and compare latency with throughput for the same computation. Part of [stage 9](../../LEARNING_PATH.md#stage-9-parallelism-and-throughput); do it after stage 8.

## Background

**Threads, cores and SMT.** Your CPU has 6 physical cores, and each runs 2 hardware threads (SMT, AMD's version of hyper-threading), so the OS sees 12 CPUs. Two threads on one core share its execution units and caches, so a second thread adds much less than a second core.

**OpenMP** adds parallelism to loops with pragmas:

```cpp
#pragma omp parallel for schedule(static)   // split the iterations evenly into one chunk per thread
for (long i = 0; i < n; ++i) out[i] = f(books[i]);
```

- `schedule(static)`: each thread gets a fixed, equal share. Cheapest, best when every iteration costs the same.
- `schedule(dynamic, chunk)`: threads grab chunks from a shared counter as they finish. Better when iteration costs vary, at the price of some coordination.
- `reduction(+:sum)`: each thread keeps a private partial sum, and OpenMP combines them at the end.

**Why speedup stops.**
- **Amdahl's law.** If a fraction *p* of the work parallelizes and the rest is serial, the speedup on *n* threads is at most 1 / ((1 − p) + p / n). The serial part sets a ceiling however many threads you add.
- **Memory bandwidth.** All cores share one memory system. If each iteration does little arithmetic per byte loaded (low *arithmetic intensity*), a few cores saturate the bandwidth and the rest wait. This is the idea behind the *roofline model*.
- **False sharing.** Caches keep copies consistent by passing ownership of whole 64-byte lines between cores. If two threads write *different* variables on the *same* line, the line bounces between their cores on every write, even though they never share data.

**Latency vs throughput.** Processing one message as fast as possible (latency) and processing a whole dataset as fast as possible (throughput) lead to different designs. Batching across cores maximizes throughput, but each item then waits for its batch.

## Code tour

1. **`light_feature` and `heavy_feature`** in [lab08_parallel_scaling.cpp](../lab08_parallel_scaling.cpp): the same input; the heavy version adds 200 rounds of `sin` and `sqrt`, which makes the work compute-bound.
2. **`run()`**: the parallel loop, in a `static` and a `dynamic` version, each repeated 3 times keeping the best. `num_threads(threads)` sets the thread count for the sweep.
3. **`Packed` vs `Padded`**: 8-byte counters next to each other, against counters each on their own 64-byte line. `count_events` has each thread increment only its own counter. `lab::clobber_memory()` forces a real store on every iteration, as a live counter would.
4. **`latency_vs_batch`**: times each book on one core with `rdtscp`, then the whole batch across all threads.
5. **The Makefile** adds `-fopenmp` for this lab only.

## Predict
1. Your CPU has 6 cores and 12 hardware threads. For compute-heavy work, what speedup do you expect at 6 threads, and at 12?
2. For the light work (a few arithmetic operations per 448-byte book, 1.8 GB of books), will 12 threads be 12x faster?
3. Eight threads each increment their own counter. In one version the counters sit next to each other in memory; in the other each has its own cache line. How big is the difference?
4. One book takes about 2.3 µs on one core. What does it cost per book when a whole batch runs on all 12 threads?

## Run
```bash
./build/lab08_parallel_scaling --work=heavy
./build/lab08_parallel_scaling --work=light
./build/lab08_parallel_scaling --mode=false_sharing
./build/lab08_parallel_scaling --mode=latency_vs_batch
```

## Break it
- `--schedule=dynamic` with heavy work. Faster or slower than static? Make the work uneven (for example, heavy only for every tenth book) and try again: when does dynamic scheduling win?
- Pin the threads: `OMP_PROC_BIND=close OMP_PLACES=cores ./build/lab08_parallel_scaling --work=heavy`, then `OMP_PLACES=threads`. What changes at 7–12 threads, and why?
- Plot speedup against threads for both kinds of work and fit Amdahl's law to the heavy curve.
- In `--mode=false_sharing`, change `alignas(64)` to `alignas(32)` and `alignas(128)`. Which still shows the problem?

## Explain
- Why does the memory-bound loop stop scaling? (Lab 01 told you what one core sees; what is shared between cores?)
- Why is 12 threads slower than 11 for the heavy work?
- When is the latency number the right one to optimize, and when is the throughput number?

## What you should see

<details><summary>What you measured on this laptop</summary>

Heavy (compute-bound): **6.7x at 11 threads (61% efficiency), 6.4x at 12**. Speedup grows roughly linearly up to the 6 physical cores, then more slowly as the two threads on each core share its execution units, and the 12th thread competes with the OS. Light (memory-bound): **1.4x at best, already at 3 threads**. The cores wait on the shared memory bus, so more threads only add contention. False sharing: **2,038 ms packed vs 25 ms padded with 8 threads, about 80x**. Every increment forces the cache line to move between cores. Latency vs batch: **~2.3 µs per book one at a time (p50) vs ~350 ns per book in a 12-thread batch**. Batching wins throughput, but each book in a batch waits for the whole batch, which is useless when you must react to *this* message.
</details>

## Going further

- **Measure your bandwidth.** In light mode, divide the bytes touched (books × `sizeof(LimitOrderBookSoA)`) by the time at the best thread count. That's roughly your machine's usable memory bandwidth in GB/s. Compare with lab 01's sequential walk.
- **TBB and parallel STL.** Learning-path exercise 9.2: rewrite the loop with `std::execution::par` and with `tbb::parallel_for`, after `sudo dnf install tbb-devel`.
- **Uneven work.** Make every tenth book heavy and the rest light, then compare `static` and `dynamic` scheduling.
- **Thread placement.** Compare `OMP_PLACES=cores` with `OMP_PLACES=threads` under `OMP_PROC_BIND=close` and `spread`.
- **GPU version.** Learning-path exercise 9.5: the same feature as a CUDA kernel on Colab.

---

← [Lab 07: The trained model in C++](07_trained_model.md) · [All labs](../README.md)
