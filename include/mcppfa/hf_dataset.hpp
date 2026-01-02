#pragma once

#include "mcppfa/huggingface.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcppfa::hf_dataset {

struct Table {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
};

namespace detail {

inline std::string url_encode_component(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3);
    for (const unsigned char c : s) {
        const bool ok = (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9')
            || c == '-' || c == '_' || c == '.' || c == '~';
        if (ok) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hex[(c >> 4) & 0xF]);
            out.push_back(hex[c & 0xF]);
        }
    }
    return out;
}

inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string trim(std::string s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

inline std::string http_get_json(const std::string& url, const std::string& token) {
    if (!mcppfa::hf::detail::has_command("curl")) {
        throw std::runtime_error("curl not found; required for HuggingFace dataset listing");
    }

    std::ostringstream cmd;
    cmd << "curl -L -s -f";
    if (!token.empty()) {
        cmd << " -H " << mcppfa::hf::detail::shell_quote(std::string("Authorization: Bearer ") + token);
    }
    cmd << " " << mcppfa::hf::detail::shell_quote(url);

    const auto r = mcppfa::hf::run_verbose(cmd.str(), token);
    if (r.exit_status != 0) {
        std::ostringstream oss;
        oss << "HTTP GET failed (exit=" << r.exit_status << ") for: " << url;
        throw std::runtime_error(oss.str());
    }
    return r.output;
}

inline std::string parse_json_string(const std::string& s, std::size_t& i) {
    // expects s[i] == '"'
    if (i >= s.size() || s[i] != '"') throw std::runtime_error("Expected JSON string");
    ++i;
    std::string out;
    while (i < s.size()) {
        const char c = s[i++];
        if (c == '"') break;
        if (c == '\\') {
            if (i >= s.size()) break;
            const char e = s[i++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                // Note: we ignore \uXXXX for simplicity.
                default: out.push_back(e); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return out;
}

inline void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

inline std::vector<std::string> extract_json_string_field(const std::string& json, const std::string& field) {
    // Very small JSON extractor: finds occurrences of "field":"..." and returns the strings.
    std::vector<std::string> out;
    const std::string needle = std::string("\"") + field + "\"";

    std::size_t i = 0;
    while (true) {
        const std::size_t pos = json.find(needle, i);
        if (pos == std::string::npos) break;
        std::size_t j = pos + needle.size();
        j = json.find(':', j);
        if (j == std::string::npos) break;
        ++j;
        skip_ws(json, j);
        if (j < json.size() && json[j] == '"') {
            try {
                out.push_back(parse_json_string(json, j));
            } catch (...) {
                // ignore
            }
        }
        i = pos + needle.size();
    }

    // De-dup while preserving order.
    std::vector<std::string> uniq;
    uniq.reserve(out.size());
    std::unordered_map<std::string, bool> seen;
    for (const auto& s : out) {
        if (seen.emplace(s, true).second) uniq.push_back(s);
    }
    return uniq;
}

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

inline std::unordered_map<std::string, std::string> parse_json_object_flat(const std::string& line) {
    // Parses a single JSON object line into key->value string.
    // Values are returned as:
    // - string contents (unescaped) for JSON strings
    // - raw substring for numbers/bools/null/arrays/objects (best-effort)
    std::unordered_map<std::string, std::string> out;
    std::size_t i = 0;
    skip_ws(line, i);
    if (i >= line.size() || line[i] != '{') return out;
    ++i;

    while (i < line.size()) {
        skip_ws(line, i);
        if (i < line.size() && line[i] == '}') break;
        if (i >= line.size() || line[i] != '"') break;
        const std::string key = parse_json_string(line, i);
        skip_ws(line, i);
        if (i >= line.size() || line[i] != ':') break;
        ++i;
        skip_ws(line, i);

        std::string val;
        if (i < line.size() && line[i] == '"') {
            val = parse_json_string(line, i);
        } else {
            // capture until comma at depth 0
            std::size_t start = i;
            int depth = 0;
            bool in_str = false;
            while (i < line.size()) {
                const char c = line[i];
                if (in_str) {
                    if (c == '\\') {
                        i += 2;
                        continue;
                    }
                    if (c == '"') in_str = false;
                    ++i;
                    continue;
                }
                if (c == '"') { in_str = true; ++i; continue; }
                if (c == '{' || c == '[') { ++depth; ++i; continue; }
                if (c == '}' || c == ']') { if (depth > 0) --depth; ++i; continue; }
                if (depth == 0 && c == ',') break;
                if (depth == 0 && c == '}') break;
                ++i;
            }
            val = trim(line.substr(start, i - start));
        }

        out.emplace(key, val);

        skip_ws(line, i);
        if (i < line.size() && line[i] == ',') { ++i; continue; }
        if (i < line.size() && line[i] == '}') break;
    }

    return out;
}

inline std::vector<std::pair<std::string, std::string>> parse_json_object_flat_ordered(const std::string& text) {
    // Like parse_json_object_flat, but returns fields in appearance order.
    std::vector<std::pair<std::string, std::string>> out;
    std::size_t i = 0;
    skip_ws(text, i);
    if (i >= text.size() || text[i] != '{') return out;
    ++i;

    while (i < text.size()) {
        skip_ws(text, i);
        if (i < text.size() && text[i] == '}') break;
        if (i >= text.size() || text[i] != '"') break;
        const std::string key = parse_json_string(text, i);
        skip_ws(text, i);
        if (i >= text.size() || text[i] != ':') break;
        ++i;
        skip_ws(text, i);

        std::string val;
        if (i < text.size() && text[i] == '"') {
            val = parse_json_string(text, i);
        } else {
            std::size_t start = i;
            int depth = 0;
            bool in_str = false;
            while (i < text.size()) {
                const char c = text[i];
                if (in_str) {
                    if (c == '\\') {
                        i += 2;
                        continue;
                    }
                    if (c == '"') in_str = false;
                    ++i;
                    continue;
                }
                if (c == '"') { in_str = true; ++i; continue; }
                if (c == '{' || c == '[') { ++depth; ++i; continue; }
                if (c == '}' || c == ']') { if (depth > 0) --depth; ++i; continue; }
                if (depth == 0 && c == ',') break;
                if (depth == 0 && c == '}') break;
                ++i;
            }
            val = trim(text.substr(start, i - start));
        }

        out.emplace_back(key, val);

        skip_ws(text, i);
        if (i < text.size() && text[i] == ',') { ++i; continue; }
        if (i < text.size() && text[i] == '}') break;
    }

    return out;
}

inline std::string extract_balanced_object_from(const std::string& s, std::size_t brace_pos) {
    if (brace_pos >= s.size() || s[brace_pos] != '{') {
        throw std::runtime_error("Expected '{' at object start");
    }
    std::size_t i = brace_pos;
    int depth = 0;
    bool in_str = false;
    while (i < s.size()) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') {
                i += 2;
                continue;
            }
            if (c == '"') in_str = false;
            ++i;
            continue;
        }
        if (c == '"') { in_str = true; ++i; continue; }
        if (c == '{') { ++depth; ++i; continue; }
        if (c == '}') {
            --depth;
            ++i;
            if (depth == 0) {
                return s.substr(brace_pos, i - brace_pos);
            }
            continue;
        }
        ++i;
    }
    throw std::runtime_error("Unterminated JSON object");
}

inline std::vector<std::string> extract_first_list(const std::vector<std::string>& v) {
    if (v.empty()) return {};
    return {v.front()};
}

inline std::pair<std::string, std::string> pick_config_split_from_splits_json(const std::string& json) {
    // datasets-server /splits response contains fields: "config" and "split".
    const auto configs = extract_json_string_field(json, "config");
    const auto splits = extract_json_string_field(json, "split");

    std::string config = !configs.empty() ? configs.front() : std::string("default");
    std::string split;
    for (const auto& s : splits) {
        if (to_lower(s) == "train") { split = s; break; }
    }
    if (split.empty()) split = !splits.empty() ? splits.front() : std::string("train");
    return {config, split};
}

inline Table parse_rows_response_to_table(const std::string& json, const std::size_t n_rows_limit) {
    Table t;

    // Pull each "row": { ... } object from the response.
    std::size_t pos = 0;
    std::vector<std::vector<std::pair<std::string, std::string>>> ordered_rows;

    while (ordered_rows.size() < n_rows_limit) {
        const std::size_t k = json.find("\"row\"", pos);
        if (k == std::string::npos) break;
        std::size_t j = json.find(':', k + 5);
        if (j == std::string::npos) break;
        ++j;
        skip_ws(json, j);
        // Expect object
        const std::size_t brace = json.find('{', j);
        if (brace == std::string::npos) break;
        std::string obj = extract_balanced_object_from(json, brace);
        auto fields = parse_json_object_flat_ordered(obj);
        if (!fields.empty()) ordered_rows.push_back(std::move(fields));
        pos = brace + 1;
    }

    if (ordered_rows.empty()) {
        throw std::runtime_error("datasets-server rows response did not contain any rows");
    }

    // Columns: first row order, then append any new keys discovered later.
    std::unordered_map<std::string, bool> seen;
    for (const auto& kv : ordered_rows[0]) {
        if (seen.emplace(kv.first, true).second) t.columns.push_back(kv.first);
    }
    for (std::size_t r = 1; r < ordered_rows.size(); ++r) {
        for (const auto& kv : ordered_rows[r]) {
            if (seen.emplace(kv.first, true).second) t.columns.push_back(kv.first);
        }
    }

    // Rows: map into column vector
    for (const auto& row_fields : ordered_rows) {
        std::unordered_map<std::string, std::string> m;
        m.reserve(row_fields.size());
        for (const auto& kv : row_fields) m.emplace(kv.first, kv.second);

        std::vector<std::string> row;
        row.reserve(t.columns.size());
        for (const auto& c : t.columns) {
            auto it = m.find(c);
            row.push_back(it == m.end() ? std::string() : it->second);
        }
        t.rows.push_back(std::move(row));
    }

    return t;
}

inline void print_table(const Table& t, std::size_t max_rows, std::size_t max_width, std::ostream& os) {
    const std::size_t rows = std::min(max_rows, t.rows.size());
    if (t.columns.empty()) {
        os << "<no columns>\n";
        return;
    }

    // compute widths
    std::vector<std::size_t> w;
    w.resize(t.columns.size());
    for (std::size_t c = 0; c < t.columns.size(); ++c) {
        w[c] = std::min(max_width, t.columns[c].size());
    }
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < t.columns.size(); ++c) {
            if (c >= t.rows[r].size()) continue;
            w[c] = std::max(w[c], std::min(max_width, t.rows[r][c].size()));
        }
    }

    auto trunc = [&](const std::string& s) {
        if (max_width == 0 || s.size() <= max_width) return s;
        if (max_width <= 3) return s.substr(0, max_width);
        return s.substr(0, max_width - 3) + "...";
    };

    // header
    for (std::size_t c = 0; c < t.columns.size(); ++c) {
        os << std::left << std::setw(static_cast<int>(w[c])) << trunc(t.columns[c]);
        os << (c + 1 == t.columns.size() ? "\n" : "|");
    }
    for (std::size_t c = 0; c < t.columns.size(); ++c) {
        os << std::string(w[c], '-');
        os << (c + 1 == t.columns.size() ? "\n" : "+");
    }

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < t.columns.size(); ++c) {
            const std::string v = (c < t.rows[r].size()) ? t.rows[r][c] : std::string();
            os << std::left << std::setw(static_cast<int>(w[c])) << trunc(v);
            os << (c + 1 == t.columns.size() ? "\n" : "|");
        }
    }

    if (rows < t.rows.size()) {
        os << "... (" << t.rows.size() << " rows total)\n";
    }
}

} // namespace detail

inline std::vector<std::string> list_files(
    const std::string& dataset_repo,
    const std::string& revision = "main",
    const std::string& token = std::string()) {

    // HuggingFace API: list dataset files.
    // Response is JSON array with objects containing "path".
    std::ostringstream url;
    url << "https://huggingface.co/api/datasets/" << dataset_repo
        << "/tree/" << revision << "?recursive=1";

    const std::string json = detail::http_get_json(url.str(), token);
    return detail::extract_json_string_field(json, "path");
}

inline std::string pick_data_file(const std::vector<std::string>& paths) {
    // Prefer common dataset formats (and likely split names).
    const std::vector<std::string> exts = {
        ".csv", ".jsonl", ".json"
    };
    const std::vector<std::string> prefer_contains = {
        "train", "data", "dataset"
    };

    auto score = [&](const std::string& p) -> int {
        const std::string pl = detail::to_lower(p);
        int s = 0;
        for (const auto& e : exts) {
            if (detail::ends_with(pl, e)) {
                s += 100;
                if (e == ".csv") s += 10;
                if (e == ".jsonl") s += 5;
                break;
            }
        }
        for (const auto& k : prefer_contains) {
            if (pl.find(k) != std::string::npos) s += 3;
        }
        if (pl.find("test") != std::string::npos) s -= 1;
        if (pl.find("readme") != std::string::npos) s -= 50;
        return s;
    };

    int best_s = -100000;
    std::string best;
    for (const auto& p : paths) {
        const int s = score(p);
        if (s > best_s) { best_s = s; best = p; }
    }
    if (best.empty() || best_s < 50) {
        throw std::runtime_error("Could not find a CSV/JSONL/JSON file in dataset repo");
    }
    return best;
}

inline Table load_head_via_datasets_server(
    const std::string& dataset_repo,
    const std::string& token = std::string(),
    const std::string& config = std::string(),
    const std::string& split = std::string(),
    const std::size_t n_rows = 5,
    const std::size_t offset = 0) {

    // This works for Parquet-backed datasets without downloading/parsing parquet in C++.
    // It relies on the hosted datasets-server API.
    const std::string ds_q = detail::url_encode_component(dataset_repo);

    std::string chosen_config = config;
    std::string chosen_split = split;

    if (chosen_config.empty() || chosen_split.empty()) {
        std::ostringstream splits_url;
        splits_url << "https://datasets-server.huggingface.co/splits?dataset=" << ds_q;
        const std::string splits_json = detail::http_get_json(splits_url.str(), token);
        const auto cs = detail::pick_config_split_from_splits_json(splits_json);
        if (chosen_config.empty()) chosen_config = cs.first;
        if (chosen_split.empty()) chosen_split = cs.second;
    }

    std::ostringstream rows_url;
    rows_url << "https://datasets-server.huggingface.co/rows?dataset=" << ds_q
             << "&config=" << detail::url_encode_component(chosen_config)
             << "&split=" << detail::url_encode_component(chosen_split)
             << "&offset=" << offset
             << "&length=" << n_rows;

    const std::string rows_json = detail::http_get_json(rows_url.str(), token);
    return detail::parse_rows_response_to_table(rows_json, n_rows);
}

inline std::vector<std::string> list_splits(
    const std::string& dataset_repo,
    const std::string& token = std::string()) {

    const std::string ds_q = detail::url_encode_component(dataset_repo);
    std::ostringstream url;
    url << "https://datasets-server.huggingface.co/splits?dataset=" << ds_q;
    const std::string json = detail::http_get_json(url.str(), token);
    // Response contains repeated "split": "train" / "validation" / "test".
    return detail::extract_json_string_field(json, "split");
}

inline Table load_rows_split(
    const std::string& dataset_repo,
    const std::string& split,
    const std::size_t offset,
    const std::size_t length,
    const std::string& token = std::string(),
    const std::string& config = std::string()) {

    // Always uses datasets-server (works for parquet-only datasets).
    return load_head_via_datasets_server(dataset_repo, token, config, split, length, offset);
}

inline Table load_head(
    const std::string& dataset_repo,
    const std::string& local_dir = ".hf/_datasets",
    const std::string& token = std::string(),
    const std::string& revision = "main",
    const std::size_t n_rows = 5) {

    const auto paths = list_files(dataset_repo, revision, token);

    // First try "download + parse" for simple CSV/JSONL datasets.
    // If the repo is Parquet-only (common), fall back to datasets-server.
    std::string data_path_in_repo;
    try {
        data_path_in_repo = pick_data_file(paths);
    } catch (const std::exception&) {
        return load_head_via_datasets_server(dataset_repo, token, /*config=*/std::string(), /*split=*/"train", n_rows, /*offset=*/0);
    }

    const std::filesystem::path out_dir = std::filesystem::path(local_dir) / dataset_repo;
    std::filesystem::create_directories(out_dir);

    const std::filesystem::path local_path = out_dir / std::filesystem::path(data_path_in_repo).filename();

    (void)mcppfa::hf::download_file_http(
        dataset_repo,
        data_path_in_repo,
        local_path.string(),
        mcppfa::hf::RepoType::dataset,
        revision,
        token);

    // Parse file
    const std::string pl = detail::to_lower(local_path.string());

    Table t;

    std::ifstream in(local_path);
    if (!in) {
        throw std::runtime_error("Failed to open downloaded dataset file: " + local_path.string());
    }

    if (detail::ends_with(pl, ".csv")) {
        std::string line;
        if (!std::getline(in, line)) throw std::runtime_error("CSV appears empty");
        t.columns = detail::split_csv_line(line);

        std::size_t count = 0;
        while (count < n_rows && std::getline(in, line)) {
            auto row = detail::split_csv_line(line);
            // pad
            if (row.size() < t.columns.size()) row.resize(t.columns.size());
            t.rows.push_back(std::move(row));
            ++count;
        }
        return t;
    }

    if (detail::ends_with(pl, ".jsonl")) {
        std::string line;
        std::vector<std::unordered_map<std::string, std::string>> objects;
        objects.reserve(n_rows);

        std::size_t count = 0;
        while (count < n_rows && std::getline(in, line)) {
            const auto obj = detail::parse_json_object_flat(line);
            if (!obj.empty()) {
                objects.push_back(obj);
                ++count;
            }
        }

        // Determine columns from union of keys (stable-ish order: first object keys, then others).
        std::unordered_map<std::string, bool> seen;
        if (!objects.empty()) {
            for (const auto& kv : objects[0]) {
                if (seen.emplace(kv.first, true).second) t.columns.push_back(kv.first);
            }
            for (std::size_t i = 1; i < objects.size(); ++i) {
                for (const auto& kv : objects[i]) {
                    if (seen.emplace(kv.first, true).second) t.columns.push_back(kv.first);
                }
            }
        }

        for (const auto& obj : objects) {
            std::vector<std::string> row;
            row.reserve(t.columns.size());
            for (const auto& c : t.columns) {
                auto it = obj.find(c);
                row.push_back(it == obj.end() ? std::string() : it->second);
            }
            t.rows.push_back(std::move(row));
        }

        return t;
    }

    if (detail::ends_with(pl, ".json")) {
        throw std::runtime_error("JSON file detected; only JSONL/CSV are supported for printing rows");
    }

    throw std::runtime_error("Unsupported dataset format (expected .csv/.jsonl/.json)");
}

inline void print_columns(const Table& t, std::ostream& os = std::cout) {
    os << "Columns (" << t.columns.size() << "): ";
    for (std::size_t i = 0; i < t.columns.size(); ++i) {
        os << t.columns[i];
        os << (i + 1 == t.columns.size() ? "\n" : ", ");
    }
}

inline void print_head(const Table& t, const std::size_t n = 5, std::ostream& os = std::cout) {
    detail::print_table(t, n, /*max_width=*/28, os);
}

} // namespace mcppfa::hf_dataset
