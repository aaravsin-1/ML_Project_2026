# Troubleshooting

Problems you're likely to hit in the labs, and what to do.

## Building

**`make: *** No rule to make target`**
Run `make` from `cpp/labs`, not the repo root. `make trained` also needs `cpp/generated/btc_lightgbm_model.hpp`, so run `python gen_trained_model.py` first.

**`illegal instruction` when running**
The program was built with `-march=native` on a machine with instructions this one lacks (for example AVX-512 built on Colab, run on the laptop). Rebuild here with `make -B`.

**`error: inlining failed ... target specific option mismatch`** with intrinsics
The AVX2 intrinsics need `-mavx2` or `-march=native`. The Makefile sets `-march=native`; if you compile by hand, add it.

**`ModuleNotFoundError: No module named 'lightgbm'`** (lab 07)
Use the project's environment: `../../.venv/bin/python gen_trained_model.py`, or `pip install --user lightgbm`.

**`fatal error: tbb/...: No such file`** (stage 9)
`sudo dnf install tbb-devel`, and link with `-ltbb`.

**`std::execution::par` compiles but uses one core**
GCC's parallel algorithms need TBB at link time (`-ltbb`). Without it they quietly run sequentially. Check with `top` or by timing it.

## perf

**`Access to performance monitoring and observability operations is limited`**
Your setting is `perf_event_paranoid = 2`, which allows counting your own programs in user space (events show a `:u` suffix). If you need kernel events, `sudo sysctl kernel.perf_event_paranoid=1` changes it until reboot. The labs don't need that.

**Numbers like `45,16,87,281`**
That's Indian digit grouping from your locale. Use `LC_ALL=C perf stat ...`.

**`<not supported>` next to an event**
Your CPU doesn't expose that event under that name. Run `perf list` to see what's available. AMD and Intel name some cache events differently.

**Percentages like `(82.96%)` on the right**
Multiplexing: more events requested than hardware counters. Request fewer events for exact counts.

## Noisy or surprising numbers

**Results change a lot between runs**
Close the browser and other heavy programs, plug in the charger, pin to one core (`taskset -c 3` or the lab's `--core=3`), and repeat each run 5–10 times. Report the spread, not one run. Lab 05 shows how big the noise is.

**The first run is slower than later ones**
Warm-up effects: page faults on fresh memory, cold caches, the branch predictor learning, the CPU ramping up its clock. Every lab warms up before timing; when you write your own benchmarks, do the same.

**Times get slower during a long run**
Thermal throttling: the laptop gets hot and lowers its clock. Watch `grep MHz /proc/cpuinfo` in another terminal. Let it cool down between long runs.

**A "faster" version measures as 0 ns, or far too fast**
The compiler probably deleted the work because nothing used the result. See lab 04, and use `lab::do_not_optimize`.

**More threads made it slower**
Normal past the number of physical cores (6), and for memory-bound work much earlier. See lab 08.

## Something froze

**The machine stopped responding during `chrt`**
A real-time thread that never yields can starve everything else. Wait for it to finish, or switch to a text console (Ctrl+Alt+F3) and kill it. Only use real-time priority with programs that end.

**Booting with `isolcpus` went wrong** (stage 11)
At the GRUB menu, press `e` on the boot entry and delete the added parameters for that boot. Then remove them permanently with `sudo grubby --update-kernel=ALL --remove-args="isolcpus=3 nohz_full=3"`.

---

[All labs](../README.md) · [Tools guide](tools.md) · [Glossary](glossary.md)
