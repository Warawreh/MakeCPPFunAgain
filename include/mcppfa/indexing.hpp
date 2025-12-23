#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace mcppfa {

// Convert a possibly-negative index into a valid [0, len) index.
// If allow_end=true, also allows index == len (useful for slicing APIs).
inline std::size_t normalize_index(std::size_t len, std::ptrdiff_t index, bool allow_end = false) {
  std::ptrdiff_t i = index;
  const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(len);

  if (i < 0) i += n;

  if (allow_end) {
    if (i < 0 || i > n) throw std::out_of_range("mcppfa::normalize_index: index out of range");
  } else {
    if (i < 0 || i >= n) throw std::out_of_range("mcppfa::normalize_index: index out of range");
  }

  return static_cast<std::size_t>(i);
}

// Python-like negative indexing access for containers supporting size() and operator[].
// Example: mcppfa::at(v, -1) == v[v.size()-1]
template <class Container>
inline decltype(auto) at(Container& c, std::ptrdiff_t index) {
  const std::size_t i = normalize_index(c.size(), index, false);
  return c[i];
}

template <class Container>
inline decltype(auto) at(const Container& c, std::ptrdiff_t index) {
  const std::size_t i = normalize_index(c.size(), index, false);
  return c[i];
}

// Python-like substr start index. If pos is negative, counts from the end.
// Example: substr("hello", -1) => "o"
inline std::string_view substr(std::string_view s, std::ptrdiff_t pos,
                              std::size_t count = std::string_view::npos) {
  const std::size_t p = normalize_index(s.size(), pos, true);
  return s.substr(p, count);
}

inline std::string substr(const std::string& s, std::ptrdiff_t pos,
                          std::size_t count = std::string::npos) {
  const std::size_t p = normalize_index(s.size(), pos, true);
  return s.substr(p, count);
}

}  // namespace mcppfa
