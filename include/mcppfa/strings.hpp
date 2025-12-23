#pragma once

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mcppfa {

namespace detail {

inline bool is_space(char c) {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

inline std::string_view trim_view(std::string_view s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && is_space(s[b])) ++b;
  while (e > b && is_space(s[e - 1])) --e;
  return s.substr(b, e - b);
}

}  // namespace detail

// Python-like str.split() with sep=None (whitespace splitting).
// - Consecutive whitespace is treated as a single separator.
// - Leading/trailing whitespace is ignored.
// - If maxsplit == 0: returns a single element containing the trimmed string (or empty vector if only whitespace).
inline std::vector<std::string> split(std::string_view s, std::ptrdiff_t maxsplit = -1) {
  s = detail::trim_view(s);
  if (s.empty()) return {};

  std::vector<std::string> out;
  std::size_t i = 0;

  while (i < s.size()) {
    while (i < s.size() && detail::is_space(s[i])) ++i;
    if (i >= s.size()) break;

    if (maxsplit >= 0 && static_cast<std::ptrdiff_t>(out.size()) >= maxsplit) {
      out.emplace_back(s.substr(i));
      break;
    }

    std::size_t j = i;
    while (j < s.size() && !detail::is_space(s[j])) ++j;
    out.emplace_back(s.substr(i, j - i));
    i = j;
  }

  return out;
}

// Python-like str.split(sep[, maxsplit]) with an explicit separator.
// - Empty separator is invalid (Python raises ValueError); here we throw std::invalid_argument.
// - Empty fields are preserved (e.g. "a,,b" split on "," => {"a", "", "b"}).
// - If maxsplit == 0: returns a single element containing the original string.
inline std::vector<std::string> split(std::string_view s, std::string_view sep, std::ptrdiff_t maxsplit = -1) {
  if (sep.empty()) {
    throw std::invalid_argument("mcppfa::split: empty separator");
  }

  std::vector<std::string> out;
  if (maxsplit == 0) {
    out.emplace_back(s);
    return out;
  }

  std::size_t pos = 0;
  std::ptrdiff_t splits_done = 0;

  while (pos <= s.size()) {
    const bool can_split_more = (maxsplit < 0) || (splits_done < maxsplit);
    if (!can_split_more) break;

    const std::size_t found = s.find(sep, pos);
    if (found == std::string_view::npos) break;

    out.emplace_back(s.substr(pos, found - pos));
    pos = found + sep.size();
    ++splits_done;
  }

  out.emplace_back(s.substr(pos));
  return out;
}

}  // namespace mcppfa
