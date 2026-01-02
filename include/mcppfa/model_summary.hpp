#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcppfa::model_summary {

inline int64_t count_total_params(torch::nn::Module& module) {
    int64_t total = 0;
    for (const auto& p : module.parameters(/*recurse=*/true)) {
        total += p.numel();
    }
    return total;
}

inline int64_t count_trainable_params(torch::nn::Module& module) {
    int64_t total = 0;
    for (const auto& p : module.parameters(/*recurse=*/true)) {
        if (p.requires_grad()) {
            total += p.numel();
        }
    }
    return total;
}

inline std::string shape_to_string(const torch::Tensor& t) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < t.dim(); ++i) {
        if (i) oss << ", ";
        oss << t.size(i);
    }
    oss << "]";
    return oss.str();
}

struct SummaryOptions {
    bool print_each_param = false;
    int max_groups = 60;
    int group_name_width = 65;
};

inline void print_model_summary(torch::nn::Module& module, SummaryOptions opts = {}) {
    // Group totals by prefix (everything before the last '.') and print sorted.
    int64_t total = 0;
    int64_t trainable = 0;
    std::unordered_map<std::string, int64_t> group_totals;
    std::unordered_map<std::string, int64_t> top_totals;

    std::vector<std::pair<std::string, torch::Tensor>> params;
    for (const auto& np : module.named_parameters(/*recurse=*/true)) {
        params.emplace_back(np.key(), np.value());
    }

    if (opts.print_each_param) {
        std::cout << "\n=== Model Parameters ===\n";
    }

    for (const auto& kv : params) {
        const auto& name = kv.first;
        const auto& p = kv.second;
        const int64_t n = p.numel();
        total += n;
        if (p.requires_grad()) trainable += n;

        std::string prefix = name;
        auto pos = prefix.find_last_of('.');
        if (pos != std::string::npos) prefix = prefix.substr(0, pos);
        group_totals[prefix] += n;

        std::string top = name;
        auto dot = top.find('.');
        if (dot != std::string::npos) top = top.substr(0, dot);
        top_totals[top] += n;

        if (opts.print_each_param) {
            std::cout << std::left << std::setw(opts.group_name_width) << name
                      << " shape=" << std::setw(18) << shape_to_string(p)
                      << " params=" << n
                      << (p.requires_grad() ? "" : " (frozen)")
                      << "\n";
        }
    }

    std::vector<std::pair<std::string, int64_t>> groups;
    groups.reserve(group_totals.size());
    for (const auto& g : group_totals) groups.emplace_back(g.first, g.second);
    std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<std::pair<std::string, int64_t>> tops;
    tops.reserve(top_totals.size());
    for (const auto& g : top_totals) tops.emplace_back(g.first, g.second);
    std::sort(tops.begin(), tops.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\n=== Top-Level Param Totals ===\n";
    for (const auto& g : tops) {
        std::cout << std::left << std::setw(20) << g.first << " params=" << g.second << "\n";
    }

    std::cout << "\n=== Grouped Parameter Totals (top) ===\n";
    const int limit = std::min<int>(static_cast<int>(groups.size()), opts.max_groups);
    for (int i = 0; i < limit; ++i) {
        const auto& g = groups[i];
        std::cout << std::left << std::setw(opts.group_name_width) << g.first << " params=" << g.second << "\n";
    }
    if (static_cast<int>(groups.size()) > opts.max_groups) {
        std::cout << "... (" << (groups.size() - opts.max_groups) << " more groups not shown)\n";
    }

    std::cout << "\nTRAINABLE PARAMS: " << trainable << "\n";
    std::cout << "TOTAL PARAMS:     " << total << "\n";
}

} // namespace mcppfa::model_summary
