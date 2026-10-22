#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace hft {

struct alignas(16) TreeNode {
    int32_t feature_idx{-1};     // -1 if leaf
    float threshold{0.0f};
    int32_t left_child{-1};
    int32_t right_child{-1};
    float leaf_value{0.0f};

    inline bool is_leaf() const noexcept {
        return feature_idx < 0;
    }
};

class DecisionTree {
public:
    std::vector<TreeNode> nodes;

    inline float predict(const float* features) const noexcept {
        if (nodes.empty()) return 0.0f;
        int32_t curr = 0;
        while (!nodes[curr].is_leaf()) {
            const int32_t f_idx = nodes[curr].feature_idx;
            if (features[f_idx] <= nodes[curr].threshold) {
                curr = nodes[curr].left_child;
            } else {
                curr = nodes[curr].right_child;
            }
        }
        return nodes[curr].leaf_value;
    }
};

class FastGBDTEnsemble {
public:
    // 3 classes: FLAT (0), UP (1), DOWN (2)
    std::vector<DecisionTree> trees_flat;
    std::vector<DecisionTree> trees_up;
    std::vector<DecisionTree> trees_down;
    float base_score{0.0f};

    inline void predict_logits(const float* features, float* out_logits) const noexcept {
        float s0 = base_score;
        float s1 = base_score;
        float s2 = base_score;

        for (const auto& t : trees_flat) {
            s0 += t.predict(features);
        }
        for (const auto& t : trees_up) {
            s1 += t.predict(features);
        }
        for (const auto& t : trees_down) {
            s2 += t.predict(features);
        }

        out_logits[0] = s0;
        out_logits[1] = s1;
        out_logits[2] = s2;
    }

    inline int32_t predict_class(const float* features) const noexcept {
        float logits[3];
        predict_logits(features, logits);
        if (logits[1] > logits[0] && logits[1] > logits[2]) return 1; // UP
        if (logits[2] > logits[0] && logits[2] > logits[1]) return 2; // DOWN
        return 0; // FLAT
    }
};

} // namespace hft
