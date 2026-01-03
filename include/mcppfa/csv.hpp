#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mcppfa::csv {

// Minimal RFC4180-ish CSV splitter.
// - Handles quoted fields.
// - Handles escaped quotes inside quoted field ("" -> ").
// - Does not do multiline fields.
inline std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string cur;
  bool in_quotes = false;

  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        cur.push_back(c);
      }
    } else {
      if (c == ',') {
        fields.push_back(cur);
        cur.clear();
      } else if (c == '"') {
        in_quotes = true;
      } else {
        cur.push_back(c);
      }
    }
  }

  fields.push_back(cur);
  return fields;
}

}  // namespace mcppfa::csv
