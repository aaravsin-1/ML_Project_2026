# Hands-on labs

Small programs you build, run, change and break, to see low-latency effects on your own machine instead of reading about them. They are the practical half of [LEARNING_PATH.md](../LEARNING_PATH.md): the learning path is the map, the labs are where you get your hands dirty. Labs 01–08 come with code. From stage 5 on (apart from lab 08) you write the code yourself, following the step-by-step milestones in the learning path.

## The labs

| Lab | What you find out | Stage | Time |
|:---|:---|:---:|:---:|
| [01 Memory hierarchy](docs/01_memory_hierarchy.md) | How long one memory access takes from L1 to RAM, and your cache sizes from the numbers alone | 1 | 1–2 h |
| [02 Data layout](docs/02_data_layout.md) | When struct-of-arrays beats array-of-structs, and when it loses | 1 | 1–2 h |
| [03 Branch prediction](docs/03_branch_prediction.md) | What a wrong guess costs, and how the compiler hides it | 1 | 1–2 h |
| [04 Timers and dead code](docs/04_timers_and_dce.md) | What reading a clock costs, and how a benchmark can measure nothing | 2 | 1–2 h |
| [05 Noise and jitter](docs/05_noise_and_jitter.md) | Why the median hides what matters, and how often the OS interrupts you | 2 | 2–3 h |
| [06 SIMD on a real feature](docs/06_simd_ofi.md) | AVX2 vs scalar for the project's OFI feature, and what `-ffast-math` really does | 3 | 2–3 h |
| [07 The trained model in C++](docs/07_trained_model.md) | The BTC LightGBM model compiled to C++: exactness, real latency, and why it's slow | 2, 4 | 2–4 h |
| [08 Parallel scaling](docs/08_parallel_scaling.md) | When more cores help, memory-bandwidth limits, false sharing, latency vs throughput | 9 | 2–3 h |

Do them in order; each assumes the ones before it. Labs 01–06 fit in the first few weeks of the learning path; lab 07 closes an open item in the project (the C++ timing rerun); lab 08 belongs to stage 9.

**Reference pages:**
- [Tools guide](docs/tools.md): perf, objdump, Compiler Explorer, taskset, cpupower, llvm-mca, Google Benchmark, and how to read their output on this laptop.
- [Troubleshooting](docs/troubleshooting.md): build errors, perf permissions, noisy numbers, and what to do if something freezes.
- [Glossary](docs/glossary.md): every term the labs use, in one place.
- [Lab notebook template](docs/lab_notebook_template.md): copy it and fill one block per lab.

## How to work through a lab

Each lab page has the same parts:
- **Background**: the concept, explained before you touch the code.
- **Code tour**: what to read in the source, and in what order.
- **Predict → Run → Break it → Explain**: the experiment itself.
- **What you should see**: what I measured on this laptop, hidden until you open it.
- **Going further**: optional extensions.

The five steps of the experiment:

1. **Predict.** Before running anything, answer the lab's *Predict* questions in your notebook. Write down numbers, even rough guesses. A wrong prediction you can explain afterwards teaches more than a right one you can't.
2. **Run.** Use the exact commands given.
3. **Observe.** Write down the numbers you got next to your predictions.
4. **Break it.** Do the variations. Each changes *one* thing. Predict each result before you run it.
5. **Explain.** Answer the *Explain* questions in a few sentences each, without looking anything up. Then open *What you should see* and compare.

The numbers under *What you should see* were measured on this laptop (Ryzen 5 5625U, Fedora, GCC 15) in October 2026, unpinned and not idle. Yours will differ somewhat. The *shape* of the results is what matters.

## Setup

```bash
cd cpp/labs
make            # builds labs 01-06 and 08 into build/  (make clean to start over)
make trained    # lab 07 only, after: python gen_trained_model.py
```

`make CXX=clang++` builds everything with Clang instead, and `make OPT=-O1` changes the optimization level. Comparing compilers and levels is a good extra experiment for any lab. Build problems are covered in [troubleshooting](docs/troubleshooting.md).

## Files

| File | What it is |
|:---|:---|
| `common.hpp` | Shared helpers: clocks, `do_not_optimize`, pinning, percentiles, argument parsing. Lab 04 walks through it |
| `labNN_*.cpp` | One program per lab, with the usage at the top of each file |
| `gen_trained_model.py` | Regenerates the trained model's C++ for lab 07 |
| `Makefile` | Build rules, including the special flags for labs 03, 06 and 08 |
| `docs/` | The lab pages and reference pages |

---

## After the labs

From here you write the code. The learning path has step-by-step milestones for each project:

| Stage | You build | Starting point |
|:---|:---|:---|
| 5 | Allocation counter, object pool, CRTP strategy | [Stage 5](../LEARNING_PATH.md#stage-5-low-latency-c-style) |
| 6 | L3 order book, then a fast version, then matching | [Stage 6](../LEARNING_PATH.md#stage-6-the-order-book-engine) |
| 7 | Nasdaq ITCH parser feeding your book | [Stage 7](../LEARNING_PATH.md#stage-7-market-data-binary-feeds) |
| 8 | SPSC queue, thread round-trip benchmark | [Stage 8](../LEARNING_PATH.md#stage-8-concurrency-without-locks) |
| 9 | Thread pool, TBB and parallel STL versions, CUDA kernel and TensorRT on Colab | [Stage 9](../LEARNING_PATH.md#stage-9-parallelism-and-throughput) (start with lab 08) |
| 10 | Multicast replay, io_uring and AF_XDP receivers | [Stage 10](../LEARNING_PATH.md#stage-10-networking-and-kernel-bypass) |
| 11 | Isolated-core jitter measurements | [Stage 11](../LEARNING_PATH.md#stage-11-tuning-the-operating-system) |
| 12 | The full tick-to-trade pipeline | [Stage 12](../LEARNING_PATH.md#stage-12-capstone-tick-to-trade) |

Put each project in its own folder under `cpp/` (for example `cpp/orderbook/`), with its own tests and a benchmark, and keep using the five-step method: predict, run, observe, break, explain.
