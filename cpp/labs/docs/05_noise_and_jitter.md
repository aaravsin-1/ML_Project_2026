# Lab 05: Noise and jitter

> **Learning-path stage:** [2](../../LEARNING_PATH.md#stage-2-measuring-honestly) · **Time:** 2–3 hours (several conditions to try) · **Needs:** lab 04  
> Run every command from `cpp/labs` after `make`. Keep notes in your lab notebook ([template](lab_notebook_template.md)).

**Goal:** see that the median hides almost everything that goes wrong, and measure how often your OS takes the CPU away from you.

## Background

Your program doesn't own the CPU. Things that take it away, from most to least frequent:

- **The scheduler** moves your thread or lets another run on the same core.
- **Interrupts**: the timer tick (hundreds of times a second), network, disk, USB and so on. Each one runs kernel code on some core.
- **The SMT sibling**: the other hardware thread on the same core shares its execution units and caches.
- **Frequency changes**: boost up, power-saving down, thermal throttling when the laptop gets hot. Your CPU uses AMD's `amd-pstate` driver; its governor is `powersave` with a "balance_performance" preference, which keeps changing the clock speed.
- **Page faults and cache pollution** from other programs evicting your data.

Each source shows up in the **tail** of the latency distribution, not the median. That's why trading systems report p99, p99.9 and the maximum: an occasional 200 µs pause is invisible in the average but means missing the one packet that mattered.

One more trap, for later: **coordinated omission**. If a load generator waits for each reply before sending the next request, a slow reply also delays the measurements that would have caught it, and the tail looks better than it is. The fixed-work loop here doesn't have that problem, but the network labs will.

## Code tour

1. **`fixed_work`** in [lab05_noise_and_jitter.cpp](../lab05_noise_and_jitter.cpp): 200 dependent floating-point steps, `noinline`, so every call does exactly the same work.
2. **Work mode**: warm up 10,000 calls, then time each call with `rdtscp` and convert to ns using the calibrated TSC rate.
3. **Gaps mode**: a tight loop reading `rdtsc`. Normally consecutive reads are nanoseconds apart. Any gap longer than 1 µs means something took the CPU away, so the program records it.
4. Both modes print percentiles with `lab::print_percentiles`.

## Predict
1. The same 200-step computation runs 100,000 times. How far apart will p50 and p99.9 be?
2. Spinning for 5 seconds, how many times will the program be interrupted for more than 1 µs?

## Run
```bash
./build/lab05_noise_and_jitter
./build/lab05_noise_and_jitter --mode=gaps --seconds=5
```

## Break it (one change at a time, rerun both commands each time, and record p50, p99, p99.9 and max)
- Pin to a core: `--core=3`.
- Add load. In a second terminal: `for i in 1 2 3 4; do yes > /dev/null & done`. Run with and without `--core=3`. Then `killall yes`.
- Unplug the laptop charger (power-saving mode), then plug it back in.
- Run while Firefox scrolls a heavy page.
- Advanced (stage 11): boot with `isolcpus=3 nohz_full=3`, rerun with `--core=3`, and compare the gap counts.

## Explain
- Which percentile changed the most across conditions, and which barely moved? Why is that a problem for anyone who only reports averages?
- The gap mode lost about 0.4% of its time. That sounds small. Why does it matter to a trading system that has to react to one specific packet?

## What you should see

<details><summary>What you measured on this laptop</summary>

Unpinned, unloaded: fixed work **p50 ~1,040 ns, p99 ~1,080 ns, p99.9 ~4,200 ns, max ~8,700 ns**. The median and p99 are within 4%, while p99.9 is 4x higher. The gap mode saw **about 1,100 interruptions longer than 1 µs per second**: median ~3.6 µs, worst ~280 µs, about 0.44% of the time lost.

Load and power saving mostly move p99.9 and max, not p50. A trading system is judged on whether it reacted to *this* packet in time, and an unlucky 280 µs pause at the wrong moment is a missed trade, however good the average looks.
</details>

## Going further

- **Plot the distribution.** Add a `--dump=file.csv` option that writes every sample, then plot a histogram on a log scale (Python is fine). Where does the tail start?
- **Real-time priority.** `sudo chrt -f 50 ./build/lab05_noise_and_jitter --core=3` runs under `SCHED_FIFO`. What changes, and what risks does real-time priority carry?
- **Frequency.** Compare the default with `sudo cpupower frequency-set -g performance` (set it back with `-g powersave` afterwards). Watch `grep MHz /proc/cpuinfo` while it runs.
- **Interrupt counts.** `cat /proc/interrupts` before and after a run, and see which interrupts landed on your pinned core.
- **Isolated core.** Stage 11 has you boot with `isolcpus=3 nohz_full=3` and repeat the gaps mode.

---

← [Lab 04: Timers and the disappearing benchmark](04_timers_and_dce.md) · [All labs](../README.md) · [Lab 06: SIMD on a real feature](06_simd_ofi.md) →
