# Lab 01: The memory hierarchy

> **Learning-path stage:** [1](../../LEARNING_PATH.md#stage-1-how-the-cpu-actually-runs-your-code) · **Time:** 1–2 hours · **Needs:** nothing beyond `make`  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** measure how long one memory access takes depending on where the data lives, and find your CPU's cache sizes from the numbers alone.

**How it works:** the program links the buffer into one random cycle and follows it. Each load needs the previous one's result to know where to go next, so the CPU can't overlap them. You see the raw latency.

## Background

A core can do a simple arithmetic operation in a fraction of a nanosecond, but fetching a value from main memory (DRAM) takes far longer. Caches hide that gap by keeping recently used data in small, fast memories close to the core:

```
   core ── L1d ── L2 ──┬── L3 (shared by all 6 cores) ── memory controller ── DRAM
   core ── L1d ── L2 ──┤
   ...                 │
   (each core has its own L1 and L2)
```

On your Ryzen 5 5625U (`lscpu --caches`): L1 data 32 KB per core, L2 512 KB per core, L3 16 MB shared.

Four ideas explain almost everything this lab shows:

- **Cache lines.** Memory moves in 64-byte blocks. Touching one byte brings in its whole line, so data used together should live together (that is lab 02).
- **Latency vs bandwidth.** *Latency* is how long one access takes from request to answer. *Bandwidth* is how many bytes per second can flow when many accesses are in flight. A core can have many misses outstanding at once, but only if it knows the addresses in advance.
- **Dependent loads.** In `p = p->next`, the next address is inside the data being loaded, so the CPU cannot start the next load until the current one finishes. Every hop pays the full latency. This is called *pointer chasing*, and it is exactly how this lab measures latency.
- **Prefetching.** The hardware watches access patterns. If you walk memory in order, it fetches lines before you ask for them, turning a latency problem into a bandwidth problem. It works within 4 KB pages and on simple strides.

A fifth, for the large buffers: the **TLB** caches the translation from virtual to physical addresses, one entry per 4 KB page. A buffer bigger than the TLB covers causes page walks on top of cache misses. *Huge pages* (2 MB) reduce this, which matters again in stages 10–11.

## Code tour

Read [lab01_memory_hierarchy.cpp](../lab01_memory_hierarchy.cpp) in this order:

1. **`struct alignas(64) Node`**: one pointer plus padding to exactly 64 bytes, so every node is its own cache line. Ask yourself what would change if nodes were 8 bytes.
2. **Building the cycle** (inside the size loop): `idx` lists the nodes the walk visits, `std::shuffle` randomizes all but the first, and each node's `next` points at the following index. The result is one big loop through every node in random order, so the prefetcher has no pattern to learn.
3. **The warm-up pass**: one trip around the cycle before timing, so the first measured hop doesn't include page faults from touching new memory.
4. **The timed loop**: `for (...) p = p->next;`. Each iteration depends on the last. `lab::do_not_optimize(p)` afterwards stops the compiler from deleting the loop (lab 04 explains why that is needed).
5. **Options**: `--order=seq` skips the shuffle, `--stride` changes how many nodes apart consecutive hops are, `--core` pins the thread.

## Predict
1. Roughly how many ns per load for a 4 KB buffer, and for a 256 MB buffer?
2. At which buffer sizes do you expect the time to jump? (Look up the L1, L2 and L3 sizes of a Ryzen 5 5625U, or run `lscpu --caches`.)
3. Will a *sequential* walk over 256 MB be close to the random one, or much faster? Why?

## Run
```bash
./build/lab01_memory_hierarchy
```

## Break it
- `--order=seq`: the same buffer sizes, walked in order. What happened at the large sizes, and which piece of hardware is responsible?
- `--order=seq --stride=4096`: in order, but jumping 4 KB each time. Does the hardware still help? Why not?
- `--core=0` versus `--core=5`: does pinning change anything here?
- Run it while a browser tab plays a video. Which buffer sizes are affected most?

## Explain
- Why does the random walk measure *latency* while the sequential walk measures something else?
- Your BTC pipeline keeps 1 order book of 384 bytes per update. Which level of the hierarchy does it live in, and so how much does memory matter to it?
- A strategy tracks 5,000 instruments, each with a 384-byte book. Which level does that live in, and what does that imply?

## What you should see

<details><summary>What you measured on this laptop (open after you predict)</summary>

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

## Going further

- **Count the misses.** `perf list cache` shows which cache events your CPU exposes. Try `perf stat -e L1-dcache-loads,L1-dcache-load-misses ./build/lab01_memory_hierarchy --max_kb=64` and then with `--max_kb=65536`.
- **Huge pages.** Allocate the buffer with `std::aligned_alloc(2 << 20, bytes)` and call `madvise(ptr, bytes, MADV_HUGEPAGE)` before building the cycle. Rerun the 8–64 MB sizes. Which part of the cost did huge pages remove, and which part is left?
- **Find the adjacent-line prefetcher.** Run `--order=seq` with strides of 64, 128, 256 and 512 bytes. Where does the sequential walk start to cost as much as the random one?
- **Bandwidth instead of latency.** Change the timed loop to sum a large `std::vector<double>` in order and compute GB/s. Lab 08 uses this number.

---

[All labs](../README.md) · [Lab 02: Data layout (array of structs vs struct of arrays)](02_data_layout.md) →
