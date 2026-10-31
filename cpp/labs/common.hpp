// Shared helpers for the labs. Lab 04 asks you to read and question every line of this file.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sched.h>
#include <x86intrin.h>

namespace lab {

// Stops the compiler from deleting a computation whose result is never used.
// The empty asm statement claims to read `value` and touch memory, so the value must exist.
template <typename T>
inline void do_not_optimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

// Stops the compiler from moving memory reads and writes across this point.
inline void clobber_memory() { asm volatile("" : : : "memory"); }

// Time-stamp counter: counts at a fixed rate on modern CPUs (check for constant_tsc in /proc/cpuinfo).
inline uint64_t rdtsc() { return __rdtsc(); }

// rdtscp waits for earlier instructions to finish before reading the counter.
inline uint64_t rdtscp() {
    unsigned aux;
    return __rdtscp(&aux);
}

inline uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Pins the calling thread to one CPU core. Returns false if the core does not exist.
inline bool pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// Estimates TSC ticks per nanosecond by comparing against steady_clock over ~200 ms.
inline double tsc_per_ns() {
    const uint64_t t0 = now_ns(), c0 = rdtsc();
    while (now_ns() - t0 < 200'000'000) {}
    const uint64_t t1 = now_ns(), c1 = rdtsc();
    return static_cast<double>(c1 - c0) / static_cast<double>(t1 - t0);
}

// Sorts `v` and prints min / p50 / p90 / p99 / p99.9 / max.
inline void print_percentiles(const char* label, std::vector<double>& v, const char* unit = "ns") {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto p = [&](double q) { return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))]; };
    std::printf("%-28s min %8.1f | p50 %8.1f | p90 %8.1f | p99 %8.1f | p99.9 %8.1f | max %10.1f %s\n",
                label, v.front(), p(0.50), p(0.90), p(0.99), p(0.999), v.back(), unit);
}

// Tiny command-line parsing: --name=value, with a default.
inline long arg_long(int argc, char** argv, const char* name, long def) {
    const std::string key = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], key.c_str(), key.size()) == 0) return std::strtol(argv[i] + key.size(), nullptr, 10);
    return def;
}

inline std::string arg_str(int argc, char** argv, const char* name, const char* def) {
    const std::string key = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], key.c_str(), key.size()) == 0) return std::string(argv[i] + key.size());
    return def;
}

}  // namespace lab
