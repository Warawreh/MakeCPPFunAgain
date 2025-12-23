#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_set>
#include <vector>

// Depends on mcppfa::type_name(...) and DataFrame get_columns_info<...>()
#include "psql_dataframe.hpp"

namespace mcppfa {

// Return unique (name,size,type) for common DB-loaded frames.
template <typename DF>
inline std::vector<std::tuple<std::string, std::size_t, std::type_index>> columns_info_basic_unique(const DF &df) {
    const auto infos = df.template get_columns_info<long long, double, std::string>();
    std::vector<std::tuple<std::string, std::size_t, std::type_index>> out;
    out.reserve(infos.size());

    std::unordered_set<std::string> seen;
    seen.reserve(infos.size());

    for (const auto &[name_any, size, type_idx] : infos) {
        const std::string name{name_any.c_str()};
        if (seen.insert(name).second) {
            out.emplace_back(name, static_cast<std::size_t>(size), type_idx);
        }
    }

    return out;
}

// One-liner for notebooks.
template <typename DF>
inline void print_columns(const DF &df, std::ostream &os = std::cout) {
    for (const auto &[name, size, type_idx] : columns_info_basic_unique(df)) {
        os << "Column: " << name << ", Size: " << size
           << ", Type: " << type_name(type_idx) << '\n';
    }
}

} // namespace mcppfa
