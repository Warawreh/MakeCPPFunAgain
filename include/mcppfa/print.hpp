#pragma once

#include <cstddef>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Note: This overload is intentionally in the global namespace so
// `std::cout << std::vector<T>{...}` works without requiring ADL into mcppfa.
// It prints elements separated by a single space.

template <class T, class Alloc>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& v) {
    os << "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i != 0) os << ", ";
    if constexpr (std::is_same_v<T, std::string>) {
      os << '"' << v[i] << '"';
    } else {
      os << v[i];
    }
  }
    os << "]";
  return os;
}

namespace mcppfa {
namespace detail {

template <typename T, typename = void>
struct has_c_str : std::false_type {};

template <typename T>
struct has_c_str<T, std::void_t<decltype(std::declval<const T &>().c_str())>> : std::true_type {};

inline std::string truncate_cell(std::string s, std::size_t max_width) {
  if (max_width == 0) return s;
  if (s.size() <= max_width) return s;
  if (max_width <= 3) return s.substr(0, max_width);
  return s.substr(0, max_width - 3) + "...";
}

template <typename T>
inline std::string cell_to_string(const T &v) {
  if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    return v;
  } else if constexpr (std::is_same_v<std::decay_t<T>, const char *>) {
    return v ? std::string(v) : std::string();
  } else if constexpr (has_c_str<std::decay_t<T>>::value) {
    // DataFrame's FixedSizeString has a broken operator<< overload for ostringstream.
    // Using c_str() avoids that and is generally what we want for string-like values.
    return std::string(v.c_str());
  } else {
    std::ostringstream oss;
    oss << v;
    return oss.str();
  }
}

template <typename DF>
inline std::vector<std::string> get_index_strings(const DF &df, std::size_t row_count) {
  std::vector<std::string> out;
  out.reserve(row_count);
  const auto &idx = df.get_index();
  for (std::size_t i = 0; i < row_count; ++i) {
    out.emplace_back(cell_to_string(idx[i]));
  }
  return out;
}

} // namespace detail

// Print a DataFrame in a compact, Python-like table (headers + first N rows).
//
// Notes:
// - Works best when column types are among the common set below.
// - If a column type is not in the supported set, it will be skipped.
// - `max_width` truncates long cells (0 means no truncation).
template <typename DF>
inline void print_df(const DF &df,
           std::size_t n_rows = 10,
           std::size_t max_width = 24,
           std::ostream &os = std::cout)
{
  const std::size_t total_rows = df.get_index().size();
  if (total_rows == 0) {
    os << "<empty dataframe>\n";
    return;
  }

  const std::size_t row_count = std::min(n_rows, total_rows);

  struct Col {
    std::string name;
    std::vector<std::string> values;
  };

  // Fast path: SQL-loaded DataFrames in this repo are string columns.
  // Keeping this to std::string avoids expensive template instantiations in Cling.
  const auto infos = df.template get_columns_info<std::string>();

  std::vector<Col> cols;
  cols.reserve(infos.size());
  for (const auto &[name_any, _size_any, _type_idx] : infos) {
    std::string name = detail::cell_to_string(name_any);
    const auto &col = df.template get_column<std::string>(name.c_str());
    std::vector<std::string> values;
    values.reserve(row_count);
    for (std::size_t i = 0; i < row_count; ++i) {
      values.emplace_back(col[i]);
    }
    cols.push_back(Col{std::move(name), std::move(values)});
  }

  const auto idx_values = detail::get_index_strings(df, row_count);

  // Compute widths (index + each column)
  std::vector<std::size_t> widths;
  widths.reserve(1 + cols.size());
  std::size_t idx_w = std::string("idx").size();
  for (const auto &s : idx_values) idx_w = std::max(idx_w, detail::truncate_cell(s, max_width).size());
  widths.push_back(idx_w);

  for (const auto &c : cols) {
    std::size_t w = c.name.size();
    for (const auto &v : c.values) w = std::max(w, detail::truncate_cell(v, max_width).size());
    widths.push_back(w);
  }

  auto print_sep = [&]() {
    os << std::string(widths[0], '-') << "+";
    for (std::size_t i = 0; i < cols.size(); ++i) {
      os << std::string(widths[i + 1], '-');
      os << (i + 1 == cols.size() ? "\n" : "+");
    }
    if (cols.empty()) os << "\n";
  };

  // Header
  os << std::left << std::setw(static_cast<int>(widths[0])) << "idx";
  if (!cols.empty()) os << "|";
  for (std::size_t i = 0; i < cols.size(); ++i) {
    os << std::left << std::setw(static_cast<int>(widths[i + 1])) << cols[i].name;
    os << (i + 1 == cols.size() ? "\n" : "|");
  }
  if (cols.empty()) os << "\n";
  print_sep();

  // Rows
  for (std::size_t r = 0; r < row_count; ++r) {
    os << std::left << std::setw(static_cast<int>(widths[0]))
       << detail::truncate_cell(idx_values[r], max_width);
    if (!cols.empty()) os << "|";
    for (std::size_t c = 0; c < cols.size(); ++c) {
      os << std::left << std::setw(static_cast<int>(widths[c + 1]))
         << detail::truncate_cell(cols[c].values[r], max_width);
      os << (c + 1 == cols.size() ? "\n" : "|");
    }
    if (cols.empty()) os << "\n";
  }

  if (row_count < total_rows) {
    os << "... (" << total_rows << " rows total)\n";
  }
}

// Streamable view so you can do:
//   std::cout << mcppfa::table(df) << '\n';
// This avoids overriding DataFrame's own operator<< (which may exist).
template <typename DF>
struct table_view {
  const DF *df;
  std::size_t n_rows;
  std::size_t max_width;
};

template <typename DF>
inline table_view<DF> table(const DF &df,
              std::size_t n_rows = 10,
              std::size_t max_width = 24) {
  return table_view<DF>{&df, n_rows, max_width};
}

template <typename DF>
inline std::ostream &operator<<(std::ostream &os, const table_view<DF> &v) {
  mcppfa::print_df(*v.df, v.n_rows, v.max_width, os);
  return os;
}

} // namespace mcppfa
