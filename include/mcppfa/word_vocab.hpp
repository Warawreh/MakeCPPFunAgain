#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcppfa::text {

inline bool is_word_char(unsigned char c) {
  return std::isalnum(c) != 0 || c == '_';
}

// Simple lowercase ASCII-ish word tokenizer.
// - Splits on non [A-Za-z0-9_]
// - Lowercases using std::tolower on unsigned char
inline std::vector<std::string> word_tokenize_lower_ascii(std::string_view s) {
  std::vector<std::string> out;
  std::string cur;

  for (unsigned char uc : s) {
    const unsigned char lower = static_cast<unsigned char>(std::tolower(uc));
    if (is_word_char(lower)) {
      cur.push_back(static_cast<char>(lower));
    } else {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    }
  }

  if (!cur.empty()) out.push_back(cur);
  return out;
}

struct WordVocab {
  static constexpr int64_t pad_id = 0;
  static constexpr int64_t unk_id = 1;

  int64_t max_size{10000};
  std::unordered_map<std::string, int64_t> word_to_id;
  std::vector<std::string> id_to_word;

  explicit WordVocab(int64_t max_size = 10000) : max_size(max_size) {
    id_to_word.reserve(static_cast<std::size_t>(std::max<int64_t>(2, max_size)));
    id_to_word.push_back("<pad>");
    id_to_word.push_back("<unk>");
    word_to_id.emplace("<pad>", pad_id);
    word_to_id.emplace("<unk>", unk_id);
  }

  int64_t size() const { return static_cast<int64_t>(id_to_word.size()); }

  void build_from_texts(const std::vector<std::string_view>& texts) {
    std::unordered_map<std::string, int64_t> freq;
    freq.reserve(texts.size() * 4);

    for (auto t : texts) {
      std::string cur;
      cur.reserve(32);

      for (unsigned char uc : t) {
        const unsigned char lower = static_cast<unsigned char>(std::tolower(uc));
        if (is_word_char(lower)) {
          cur.push_back(static_cast<char>(lower));
        } else {
          if (!cur.empty()) {
            ++freq[cur];
            cur.clear();
          }
        }
      }
      if (!cur.empty()) ++freq[cur];
    }

    std::vector<std::pair<std::string, int64_t>> items;
    items.reserve(freq.size());
    for (auto& kv : freq) items.emplace_back(std::move(kv.first), kv.second);

    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });

    const int64_t target = std::max<int64_t>(2, max_size);
    for (const auto& it : items) {
      if (static_cast<int64_t>(id_to_word.size()) >= target) break;
      const auto& w = it.first;
      if (word_to_id.find(w) != word_to_id.end()) continue;
      const int64_t id = static_cast<int64_t>(id_to_word.size());
      word_to_id.emplace(w, id);
      id_to_word.push_back(w);
    }
  }

  // Fast, non-allocating encoder that writes directly into an output buffer.
  void encode_padded_to(std::string_view text, int64_t* out, int64_t max_len) const {
    int64_t j = 0;
    std::string cur;
    cur.reserve(32);

    auto flush = [&]() {
      if (cur.empty() || j >= max_len) return;
      auto it = word_to_id.find(cur);
      out[j++] = (it == word_to_id.end()) ? unk_id : it->second;
      cur.clear();
    };

    for (unsigned char uc : text) {
      const unsigned char lower = static_cast<unsigned char>(std::tolower(uc));
      if (is_word_char(lower)) {
        if (j < max_len) cur.push_back(static_cast<char>(lower));
      } else {
        flush();
        if (j >= max_len) break;
      }
    }

    flush();
    while (j < max_len) out[j++] = pad_id;
  }

  std::vector<int64_t> encode_padded(std::string_view text, int64_t max_len) const {
    std::vector<int64_t> ids(static_cast<std::size_t>(max_len), pad_id);
    encode_padded_to(text, ids.data(), max_len);
    return ids;
  }
};

}  // namespace mcppfa::text
