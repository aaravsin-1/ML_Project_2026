// Lab 01: the memory hierarchy, measured.
//
// Builds a random cycle through a buffer and follows it: each load's address depends on the
// previous load, so the CPU cannot overlap them and you see the raw latency of wherever the
// buffer lives (L1, L2, L3 or RAM). Then repeats with a sequential walk for comparison.
//
//   ./build/lab01_memory_hierarchy                       # sweep 4 KB .. 256 MB, random order
//   ./build/lab01_memory_hierarchy --order=seq           # same sweep, sequential order
//   ./build/lab01_memory_hierarchy --stride=4096 --order=seq
//   ./build/lab01_memory_hierarchy --max_kb=1024 --core=2

#include <numeric>
#include <random>

#include "common.hpp"

// One element per cache line, so every hop touches a new line (unless you change the stride).
struct alignas(64) Node {
    Node* next;
    char pad[64 - sizeof(Node*)];
};

int main(int argc, char** argv) {
    const long max_kb = lab::arg_long(argc, argv, "max_kb", 256 * 1024);
    const long stride_bytes = lab::arg_long(argc, argv, "stride", 64);
    const std::string order = lab::arg_str(argc, argv, "order", "random");
    const long core = lab::arg_long(argc, argv, "core", -1);
    if (core >= 0 && !lab::pin_to_core(static_cast<int>(core))) std::printf("could not pin to core %ld\n", core);

    const long stride_nodes = std::max<long>(1, stride_bytes / static_cast<long>(sizeof(Node)));
    std::printf("order=%s stride=%ld bytes, %zu-byte nodes\n", order.c_str(), stride_nodes * sizeof(Node), sizeof(Node));
    std::printf("%12s %14s\n", "buffer", "ns per load");

    std::mt19937_64 rng(42);
    for (long kb = 4; kb <= max_kb; kb *= 2) {
        const size_t n_nodes = static_cast<size_t>(kb) * 1024 / sizeof(Node);
        std::vector<Node> buf(n_nodes);

        // The nodes this walk visits: every stride_nodes-th one
        std::vector<size_t> idx;
        for (size_t i = 0; i < n_nodes; i += static_cast<size_t>(stride_nodes)) idx.push_back(i);
        if (idx.size() < 2) continue;
        if (order == "random") std::shuffle(idx.begin() + 1, idx.end(), rng);
        for (size_t i = 0; i < idx.size(); ++i) buf[idx[i]].next = &buf[idx[(i + 1) % idx.size()]];

        // Enough hops to take ~0.1-0.5 s regardless of size
        const size_t hops = std::max<size_t>(idx.size() * 4, 20'000'000);
        Node* p = &buf[idx[0]];
        for (size_t i = 0; i < idx.size(); ++i) p = p->next;  // warm-up pass

        const uint64_t t0 = lab::now_ns();
        for (size_t i = 0; i < hops; ++i) p = p->next;
        const uint64_t t1 = lab::now_ns();
        lab::do_not_optimize(p);

        const double ns = static_cast<double>(t1 - t0) / static_cast<double>(hops);
        if (kb < 1024) std::printf("%9ld KB %14.2f\n", kb, ns);
        else std::printf("%9ld MB %14.2f\n", kb / 1024, ns);
    }
    return 0;
}
