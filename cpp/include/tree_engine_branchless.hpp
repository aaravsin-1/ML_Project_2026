#pragma once

#include <vector>
#include <cstdint>
#include <array>

namespace hft {

// Hand-Coded Branchless Oblivious Tree (Zero Branch Mispredictions)
// Evaluates entire depth D via bitmask shifts instead of pointer chasing
struct alignas(32) ObliviousTreeDepth4 {
    int32_t feature_indices[4]{0, 1, 2, 3};
    float thresholds[4]{0.0f, 0.0f, 0.0f, 0.0f};
    alignas(32) float leaves[16]{0.0f}; // 2^4 = 16 leaves

    inline float predict(const float* __restrict features) const noexcept {
        // Bitmask evaluation: 0 branches, 0 pointer chasing, 100% pipeline saturation
        const uint32_t b0 = static_cast<uint32_t>(features[feature_indices[0]] > thresholds[0]);
        const uint32_t b1 = static_cast<uint32_t>(features[feature_indices[1]] > thresholds[1]);
        const uint32_t b2 = static_cast<uint32_t>(features[feature_indices[2]] > thresholds[2]);
        const uint32_t b3 = static_cast<uint32_t>(features[feature_indices[3]] > thresholds[3]);

        const uint32_t leaf_idx = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
        return leaves[leaf_idx];
    }
};

class FastBranchlessEnsemble {
public:
    std::vector<ObliviousTreeDepth4> trees_flat;
    std::vector<ObliviousTreeDepth4> trees_up;
    std::vector<ObliviousTreeDepth4> trees_down;
    float base_score{0.0f};

    inline int32_t predict_class(const float* __restrict features) const noexcept {
        float s0 = base_score;
        float s1 = base_score;
        float s2 = base_score;

        const size_t n = trees_flat.size();
        for (size_t i = 0; i < n; ++i) {
            s0 += trees_flat[i].predict(features);
            s1 += trees_up[i].predict(features);
            s2 += trees_down[i].predict(features);
        }

        // Branchless class decision via ternary cmov
        const bool up_beats_flat = (s1 > s0);
        const bool up_beats_down = (s1 > s2);
        const bool down_beats_flat = (s2 > s0);
        const bool down_beats_up = (s2 > s1);

        if (up_beats_flat && up_beats_down) return 1;
        if (down_beats_flat && down_beats_up) return 2;
        return 0;
    }
};

} // namespace hft
