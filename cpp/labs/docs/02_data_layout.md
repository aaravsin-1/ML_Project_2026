# Lab 02: Data layout (array of structs vs struct of arrays)

> **Learning-path stage:** [1](../../LEARNING_PATH.md#stage-1-how-the-cpu-actually-runs-your-code) · **Time:** 1–2 hours · **Needs:** lab 01  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** see that the "optimized" layout is only optimal for some tasks. It uses the repo's own [lob.hpp](../../include/lob.hpp) (array of structs) and [lob_soa.hpp](../../include/lob_soa.hpp) (struct of arrays).

## Background

There are two common ways to store many records with several fields:

```
Array of structs (AoS): lob.hpp            Struct of arrays (SoA): lob_soa.hpp
  level 0: [price][qty]                      bid_prices:     [p0][p1][p2]...[p11]
  level 1: [price][qty]                      bid_quantities: [q0][q1][q2]...[q11]
  level 2: [price][qty]                      ask_prices:     ...
  ...                                        ask_quantities: ...
```

- **AoS** keeps everything about one item together. Good when you use most fields of one item at a time.
- **SoA** keeps each field together. Good when you scan one field across many items, and it is what SIMD wants (lab 06): four consecutive prices fill one AVX2 register with a single load.

What decides performance is **how many cache lines the hot loop touches**, and how many of the bytes in those lines it actually uses. Two more details matter:

- **Alignment and padding.** `alignas(64)` starts a struct on a cache-line boundary. The compiler also inserts padding so each field is aligned, and rounds the struct size up to its alignment. Padding is wasted cache space.
- **Hot/cold splitting.** Fields used on every update (best bid and ask) can be separated from fields used rarely (deep levels), so the hot data packs densely.

`sizeof` and `offsetof` tell you the real layout. Never guess it.

## Code tour

1. [lob.hpp](../../include/lob.hpp) and [lob_soa.hpp](../../include/lob_soa.hpp): work out the size of each struct and the offset of each field **by hand** before running `--layout_report=1`. Check `alignas` on both the structs and the fields.
2. **`time_ms(f, reps)`** in [lab02_data_layout.cpp](../lab02_data_layout.cpp): runs a task several times and keeps the best. Note the comment questioning when that is honest.
3. **The four timed lambdas**: "depth" sums `bids[l].quantity` (AoS) or `bid_quantities[l]` (SoA) for all 10 levels; "top" calls `mid_price()`, which reads only level 1. Same arithmetic, different memory access.
4. **`sink`** collects every result and goes through `do_not_optimize`, so no task can be optimized away.

## Predict
1. Run with `--layout_report=1` first, before predicting anything else. How many bytes does each layout use per book, and why is the struct-of-arrays version *bigger*?
2. For "sum all 10 bid quantities", which layout wins and by roughly how much?
3. For "level-1 mid price", which wins?

## Run
```bash
./build/lab02_data_layout --layout_report=1
```

## Break it
- `--books=20000` (everything fits in cache) versus `--books=2000000` (far bigger than cache). Does the gap grow or shrink, and why?
- Edit `lob_soa.hpp` to use 10 levels instead of 12 (`SIMD_LOB_LEVELS = 10`), rebuild with `make`, and rerun. What changed in size and speed? (Lab 06 will break. That is the point. Change it back afterwards.)
- Add a field to `LimitOrderBook` (say, a `double vwap`) and watch the offsets move with `--layout_report=1`.

## Explain
- Why does struct-of-arrays win one task and lose the other? Think in cache lines touched per book.
- `best_of` timing hides noise. When is reporting the best run honest, and when is it misleading?

## What you should see

<details><summary>What you measured on this laptop</summary>

Sizes: 384 bytes (6 cache lines) for array of structs, 448 bytes (7 lines) for struct of arrays. The SoA version pads each array to 12 levels so AVX2 can load 4 at a time, and pays 64 bytes per book for it.

With 500,000 books (~192 MB, much bigger than cache): summing 10 bid quantities took about **9.7 ms AoS vs 4.6 ms SoA**. SoA reads 80 contiguous bytes (2 cache lines). AoS strides through prices and quantities interleaved and touches 3 lines. Level-1 mid price took about **5.1 ms AoS vs 5.8 ms SoA**. AoS has both level-1 prices near each other in one line, while SoA keeps bid and ask prices 192 bytes apart on different lines, and the book is bigger overall.

The lesson: there is no universally best layout. You choose it for the access pattern of the hot path.
</details>

## Going further

- **Hot/cold split.** Design a third layout: a 32-byte struct with only the level-1 bid and ask, plus a separate array for the deeper levels. Add it to both tasks. Where does it land?
- **Measure lines, not time.** Run each task under `perf stat -e L1-dcache-load-misses` (comment out the other tasks or add a `--task` option). Do the miss counts explain the times?
- **Look at the loops.** Paste the "depth" loops into Compiler Explorer (`-O3 -march=znver3`). Which one is vectorized?
- **Prefetching by hand.** Add `__builtin_prefetch(&books[i + 8])` to the AoS "top" loop. Does it help on 2 million books? Why might it not?

---

← [Lab 01: The memory hierarchy](01_memory_hierarchy.md) · [All labs](../README.md) · [Lab 03: Branch prediction](03_branch_prediction.md) →
