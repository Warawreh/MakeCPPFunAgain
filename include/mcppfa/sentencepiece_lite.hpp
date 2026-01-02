#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcppfa::spm_lite {

inline void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

inline std::string ltrim_spaces(std::string s) {
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    s.erase(0, i);
    return s;
}

inline std::string rtrim_spaces(std::string s) {
    while (!s.empty()) {
        const char c = s.back();
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') s.pop_back();
        else break;
    }
    return s;
}

inline std::string trim_spaces(std::string s) {
    return rtrim_spaces(ltrim_spaces(std::move(s)));
}

// Minimal protobuf reader (enough for SentencePiece ModelProto -> pieces -> piece string).
// We intentionally avoid bringing a full protobuf dependency into xcpp17.
class ProtoReader {
public:
    explicit ProtoReader(std::vector<std::uint8_t> data) : data_(std::move(data)) {}

    std::size_t pos() const { return pos_; }
    std::size_t size() const { return data_.size(); }
    bool eof() const { return pos_ >= data_.size(); }

    std::uint8_t read_u8() {
        if (pos_ >= data_.size()) throw std::runtime_error("ProtoReader: unexpected EOF");
        return data_[pos_++];
    }

    std::uint64_t read_varint() {
        std::uint64_t result = 0;
        int shift = 0;
        for (int i = 0; i < 10; ++i) {
            const std::uint8_t b = read_u8();
            result |= static_cast<std::uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        throw std::runtime_error("ProtoReader: varint too long");
    }

    std::vector<std::uint8_t> read_bytes(const std::size_t n) {
        if (pos_ + n > data_.size()) throw std::runtime_error("ProtoReader: unexpected EOF (bytes)");
        std::vector<std::uint8_t> out(data_.begin() + static_cast<std::ptrdiff_t>(pos_),
                                      data_.begin() + static_cast<std::ptrdiff_t>(pos_ + n));
        pos_ += n;
        return out;
    }

    std::string read_string(const std::size_t n) {
        if (pos_ + n > data_.size()) throw std::runtime_error("ProtoReader: unexpected EOF (string)");
        std::string out(reinterpret_cast<const char*>(data_.data() + pos_), n);
        pos_ += n;
        return out;
    }

    void skip_bytes(const std::size_t n) {
        if (pos_ + n > data_.size()) throw std::runtime_error("ProtoReader: unexpected EOF (skip)");
        pos_ += n;
    }

private:
    std::vector<std::uint8_t> data_;
    std::size_t pos_ = 0;
};

inline std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);

    if (n < 0) throw std::runtime_error("Failed to stat file: " + path);

    std::vector<std::uint8_t> buf(static_cast<std::size_t>(n));
    if (!buf.empty()) {
        in.read(reinterpret_cast<char*>(buf.data()), n);
        if (!in) throw std::runtime_error("Failed to read file: " + path);
    }
    return buf;
}

inline bool is_special_piece(const std::string& p) {
    // SentencePiece/T5-style specials
    if (p.size() >= 3 && p.front() == '<' && p.back() == '>') return true;
    // BERT-style
    if (p.size() >= 3 && p.front() == '[' && p.back() == ']') return true;
    return false;
}

class SentencePieceLite {
public:
    SentencePieceLite() = default;

    void load_from_file(const std::string& model_path) {
        const auto bytes = read_file_bytes(model_path);
        ProtoReader r(bytes);

        id_to_piece_.clear();
        piece_to_id_.clear();
        unk_id_ = 0;

        // ModelProto:
        //   repeated SentencePiece pieces = 1; (length-delimited embedded messages)
        // We'll scan the top-level message, and for each field #1, parse the embedded SentencePiece
        // to extract `piece` (field #1 string).
        while (!r.eof()) {
            const std::uint64_t key = r.read_varint();
            const std::uint32_t field = static_cast<std::uint32_t>(key >> 3);
            const std::uint32_t wire = static_cast<std::uint32_t>(key & 0x7);

            if (field == 1 && wire == 2) {
                const std::size_t len = static_cast<std::size_t>(r.read_varint());
                const auto msg_bytes = r.read_bytes(len);
                parse_sentence_piece_message(msg_bytes);
            } else {
                skip_field(r, wire);
            }
        }

        if (id_to_piece_.empty()) {
            throw std::runtime_error("SentencePieceLite: parsed 0 pieces (unexpected model format)");
        }

        // Determine unk id if present.
        auto it = piece_to_id_.find("<unk>");
        if (it != piece_to_id_.end()) unk_id_ = it->second;
        it = piece_to_id_.find("[UNK]");
        if (it != piece_to_id_.end()) unk_id_ = it->second;
    }

    std::size_t vocab_size() const { return id_to_piece_.size(); }

    // Returns the raw SentencePiece piece string for an id (no whitespace marker replacement).
    // Useful for debugging vocab contents.
    std::string piece_for_id(const int64_t id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_piece_.size()) {
            return "[UNK]";
        }
        return id_to_piece_[static_cast<std::size_t>(id)];
    }

    // Adds/overrides a piece at an explicit id (used for HF "added_tokens_decoder").
    // If the vocab is smaller than id+1, it is resized and filled with "[UNK]" placeholders.
    void add_piece_with_id(const int64_t id, const std::string& piece) {
        if (id < 0) return;
        const auto idx = static_cast<std::size_t>(id);
        if (id_to_piece_.size() <= idx) {
            id_to_piece_.resize(idx + 1, "[UNK]");
        }

        // Avoid clobbering an existing non-placeholder piece with a different value.
        if (id_to_piece_[idx] != "[UNK]" && id_to_piece_[idx] != piece) {
            return;
        }

        id_to_piece_[idx] = piece;
        piece_to_id_[piece] = id;
    }

    int64_t unk_id() const { return unk_id_; }

    std::string decode(const std::vector<int64_t>& ids, const bool skip_special_tokens = false) const {
        std::string out;
        out.reserve(ids.size() * 4);
        for (const auto id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_piece_.size()) {
                out += "[UNK]";
                continue;
            }
            const std::string& p = id_to_piece_[static_cast<std::size_t>(id)];
            if (skip_special_tokens && is_special_piece(p)) continue;
            out += p;
        }
        // SentencePiece uses U+2581 ("▁") for whitespace.
        replace_all(out, u8"▁", " ");
        // Trim leading/trailing spaces for nicer roundtrip comparisons.
        out = trim_spaces(std::move(out));
        return out;
    }

    // Minimal encoding intended for simple/custom vocabs:
    // - Split on ASCII whitespace.
    // - Prefer "▁" + word pieces.
    // - Fallback to exact word pieces.
    // - Fallback to greedy longest-match subpieces (best-effort).
    // - Otherwise emit unk.
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> ids;
        std::string cur;
        auto flush_word = [&](const std::string& w) {
            if (w.empty()) return;
            const std::string w_space = std::string(u8"▁") + w;
            if (auto it = piece_to_id_.find(w_space); it != piece_to_id_.end()) {
                ids.push_back(it->second);
                return;
            }
            if (auto it = piece_to_id_.find(w); it != piece_to_id_.end()) {
                ids.push_back(it->second);
                return;
            }
            // Greedy fallback: try to segment using existing pieces.
            encode_greedy_word(w, /*prepend_space_marker=*/true, ids);
        };

        for (unsigned char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                flush_word(cur);
                cur.clear();
            } else {
                cur.push_back(static_cast<char>(c));
            }
        }
        flush_word(cur);
        return ids;
    }

    bool has_piece(const std::string& piece) const {
        return piece_to_id_.find(piece) != piece_to_id_.end();
    }

    int64_t id_for_piece(const std::string& piece) const {
        auto it = piece_to_id_.find(piece);
        return (it == piece_to_id_.end()) ? -1 : it->second;
    }

private:
    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, int64_t> piece_to_id_;
    int64_t unk_id_ = 0;

    static void skip_field(ProtoReader& r, const std::uint32_t wire) {
        switch (wire) {
            case 0: (void)r.read_varint(); return; // varint
            case 1: r.skip_bytes(8); return;      // 64-bit
            case 2: {
                const std::size_t len = static_cast<std::size_t>(r.read_varint());
                r.skip_bytes(len);
                return;
            }
            case 5: r.skip_bytes(4); return;      // 32-bit
            default: throw std::runtime_error("ProtoReader: unsupported wire type");
        }
    }

    void parse_sentence_piece_message(const std::vector<std::uint8_t>& msg_bytes) {
        ProtoReader r(msg_bytes);
        std::string piece;

        while (!r.eof()) {
            const std::uint64_t key = r.read_varint();
            const std::uint32_t field = static_cast<std::uint32_t>(key >> 3);
            const std::uint32_t wire = static_cast<std::uint32_t>(key & 0x7);

            if (field == 1 && wire == 2) {
                const std::size_t len = static_cast<std::size_t>(r.read_varint());
                piece = r.read_string(len);
            } else {
                skip_field(r, wire);
            }
        }

        // Keep insertion order as ID assignment.
        const int64_t id = static_cast<int64_t>(id_to_piece_.size());
        id_to_piece_.push_back(piece);
        piece_to_id_.emplace(piece, id);
    }

    void encode_greedy_word(const std::string& w, const bool prepend_space_marker, std::vector<int64_t>& out_ids) const {
        // Try longest-match using existing vocab. This is not full unigram/Viterbi.
        std::size_t i = 0;
        bool first = true;
        while (i < w.size()) {
            bool matched = false;
            const std::size_t remaining = w.size() - i;
            for (std::size_t len = remaining; len >= 1; --len) {
                std::string cand = w.substr(i, len);
                if (first && prepend_space_marker) {
                    cand = std::string(u8"▁") + cand;
                }
                auto it = piece_to_id_.find(cand);
                if (it != piece_to_id_.end()) {
                    out_ids.push_back(it->second);
                    i += len;
                    matched = true;
                    first = false;
                    break;
                }
                if (len == 1) break;
            }

            if (!matched) {
                // Try single char without space marker.
                std::string cand(1, w[i]);
                auto it = piece_to_id_.find(cand);
                if (it != piece_to_id_.end()) {
                    out_ids.push_back(it->second);
                } else {
                    out_ids.push_back(unk_id_);
                }
                ++i;
                first = false;
            }
        }
    }
};

} // namespace mcppfa::spm_lite
