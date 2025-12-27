#pragma once

#include <cstdlib>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mcppfa::hf {

enum class RepoType {
    model,
    dataset,
    space
};

inline const char* to_string(const RepoType t) {
    switch (t) {
        case RepoType::model: return "model";
        case RepoType::dataset: return "dataset";
        case RepoType::space: return "space";
    }
    return "model";
}

struct CommandResult {
    int exit_code{};
    std::string command;
};

struct CapturedCommandResult {
    int system_rc{};
    int exit_status{};
    double seconds{};
    std::string command;
    std::string output;
};

struct GitUploadLog {
    int system_rc{};
    int exit_status{};
    double seconds_total{};
    std::string report;
};

namespace detail {

inline CommandResult run_system(const std::string& command);

inline bool has_command(const std::string& name);

// Forward declarations (used by run_capture before definitions).
inline std::string shell_quote_posix(const std::string& s);
inline std::string shell_quote_windows_cmd(const std::string& s);
inline std::string shell_quote(const std::string& s);

inline int system_rc_to_exit_status(const int system_rc) {
    // `std::system` returns implementation-defined status.
    // In typical POSIX shells, the exit code is in the high byte.
    // For our notebook usage, this heuristic is the most portable without extra headers.
    if (system_rc < 0) return system_rc;
    if (system_rc > 255) return (system_rc >> 8) & 0xFF;
    return system_rc;
}

inline std::string read_all_file(const std::filesystem::path& p) {
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return std::string();
    std::string out;
    char buf[4096];
    while (true) {
        const std::size_t n = std::fread(buf, 1, sizeof(buf), f);
        if (n > 0) out.append(buf, n);
        if (n < sizeof(buf)) break;
    }
    std::fclose(f);
    return out;
}

inline std::string redact_token(std::string text, const std::string& token) {
    if (token.empty()) return text;
    std::size_t pos = 0;
    while ((pos = text.find(token, pos)) != std::string::npos) {
        text.replace(pos, token.size(), "***REDACTED***");
        pos += 12;
    }
    return text;
}

inline CapturedCommandResult run_capture(const std::string& command, const std::string& redact = std::string()) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // Capture stdout+stderr into a temporary file.
    // Use system_clock for timestamp, steady_clock doesn't have time_since_epoch
    const auto stamp = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    const std::filesystem::path tmp = std::filesystem::path(".hf") / std::filesystem::path("_tmp");
    std::filesystem::create_directories(tmp);
    const std::filesystem::path out_path = tmp / ("cmd_" + std::to_string(stamp) + ".log");

    std::ostringstream wrapped;
#ifdef _WIN32
    // Best effort; output capture is less reliable on cmd.exe.
    wrapped << command << " > " << shell_quote_windows_cmd(out_path.string()) << " 2>&1";
#else
    wrapped << command << " > " << shell_quote_posix(out_path.string()) << " 2>&1";
#endif

    const int rc = std::system(wrapped.str().c_str());
    const auto t1 = clock::now();

    CapturedCommandResult res;
    res.system_rc = rc;
    res.exit_status = system_rc_to_exit_status(rc);
    res.seconds = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    res.command = redact_token(command, redact);
    res.output = redact_token(read_all_file(out_path), redact);
    (void)std::remove(out_path.string().c_str());
    return res;
}

inline CapturedCommandResult run_capture_streaming(
    const std::string& command,
    const std::string& redact = std::string(),
    const bool stream_to_stdout = false,
    const std::size_t max_stream_bytes = 200000) {

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // Best-effort streaming capture.
    // On POSIX we use popen() and forward output as it arrives.
    // On Windows we fall back to file-based capture.
#ifdef _WIN32
    (void)stream_to_stdout;
    (void)max_stream_bytes;
    return run_capture(command, redact);
#else
    std::string cmd = command;
    cmd += " 2>&1";

    std::FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return run_capture(command, redact);
    }

    std::string out;
    out.reserve(4096);

    char buf[4096];
    std::size_t streamed = 0;
    while (true) {
        const std::size_t n = std::fread(buf, 1, sizeof(buf), pipe);
        if (n > 0) {
            out.append(buf, n);

            if (stream_to_stdout && streamed < max_stream_bytes) {
                std::string chunk(buf, n);
                chunk = redact_token(std::move(chunk), redact);

                const std::size_t remaining = max_stream_bytes - streamed;
                const std::size_t to_write = (chunk.size() <= remaining) ? chunk.size() : remaining;
                if (to_write > 0) {
                    (void)std::fwrite(chunk.data(), 1, to_write, stdout);
                    (void)std::fflush(stdout);
                    streamed += to_write;
                }
                if (streamed >= max_stream_bytes) {
                    const char* msg = "\n... (stream output truncated)\n";
                    (void)std::fwrite(msg, 1, std::strlen(msg), stdout);
                    (void)std::fflush(stdout);
                }
            }
        }

        if (n < sizeof(buf)) {
            if (std::feof(pipe)) break;
            if (std::ferror(pipe)) break;
        }
    }

    const int rc = pclose(pipe);
    const auto t1 = clock::now();

    CapturedCommandResult res;
    res.system_rc = rc;
    res.exit_status = system_rc_to_exit_status(rc);
    res.seconds = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    res.command = redact_token(command, redact);
    res.output = redact_token(std::move(out), redact);
    return res;
#endif
}

inline bool contains_repo_not_found(const std::string& s) {
    // Heuristics for git/hf errors.
    return (s.find("Repository not found") != std::string::npos)
        || (s.find("repository '") != std::string::npos && s.find("not found") != std::string::npos)
        || (s.find("HTTP_STATUS:404") != std::string::npos);
}

inline std::pair<std::string, std::string> split_repo_id(const std::string& repo_id) {
    const std::size_t slash = repo_id.find('/');
    if (slash == std::string::npos) {
        return {std::string(), repo_id};
    }
    return {repo_id.substr(0, slash), repo_id.substr(slash + 1)};
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

inline CapturedCommandResult repo_create_http_attempt(
    const std::string& name,
    const std::string& organization,
    const char* type,
    const bool is_private,
    const std::string& token) {

    if (!has_command("curl")) {
        throw std::runtime_error("curl not found. Install curl (apt-get install curl) to create repos via HTTP.");
    }
    if (token.empty()) {
        throw std::runtime_error("Missing token: cannot create a repo without HF token.");
    }

    std::ostringstream payload;
    payload << "{";
    payload << "\"name\":\"" << json_escape(name) << "\"";
    if (!organization.empty()) {
        payload << ",\"organization\":\"" << json_escape(organization) << "\"";
    }
    payload << ",\"type\":\"" << type << "\"";
    payload << ",\"private\":" << (is_private ? "true" : "false");
    payload << "}";

    // Print HTTP status into output so callers can interpret 404/409, etc.
    std::ostringstream cmd;
    cmd << "curl -sS -L -X POST";
    cmd << " -H " << shell_quote(std::string("Authorization: Bearer ") + token);
    cmd << " -H " << shell_quote("Content-Type: application/json");
    cmd << " -d " << shell_quote(payload.str());
    cmd << " -w " << shell_quote("\nHTTP_STATUS:%{http_code}\n");
    cmd << " " << shell_quote("https://huggingface.co/api/repos/create");
    return run_capture(cmd.str(), token);
}

inline bool http_status_is_ok_or_exists(const std::string& output) {
    // HF returns 201 for created; 409 for already exists.
    return (output.find("HTTP_STATUS:200") != std::string::npos)
        || (output.find("HTTP_STATUS:201") != std::string::npos)
        || (output.find("HTTP_STATUS:409") != std::string::npos);
}

inline std::string shell_quote_posix(const std::string& s) {
    // Quote for /bin/sh. Safe for typical Linux notebook environments (WSL/Ubuntu).
    // Produces single-quoted string, escaping embedded single quotes.
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (const char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

inline std::string shell_quote_windows_cmd(const std::string& s) {
    // Best-effort quoting for cmd.exe. This is not perfect for all edge cases.
    // For Windows users, prefer logging in once and keeping paths simple.
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

inline std::string shell_quote(const std::string& s) {
#ifdef _WIN32
    return shell_quote_windows_cmd(s);
#else
    return shell_quote_posix(s);
#endif
}

inline bool has_command(const std::string& name) {
#ifdef _WIN32
    // `where` returns 0 when found.
    const auto cmd = std::string("where ") + shell_quote_windows_cmd(name) + " >NUL 2>NUL";
    return run_system(cmd).exit_code == 0;
#else
    // `command -v` returns 0 when found.
    const auto cmd = std::string("command -v ") + shell_quote_posix(name) + " >/dev/null 2>&1";
    return run_system(cmd).exit_code == 0;
#endif
}

inline bool has_git_lfs() {
    const auto r = run_capture("git lfs version");
    return r.exit_status == 0;
}

inline std::string detect_default_remote_branch(const std::filesystem::path& repo_dir) {
    // Try to resolve origin/HEAD -> origin/<branch>
    {
        std::ostringstream cmd;
        cmd << "git -C " << shell_quote(repo_dir.string())
            << " rev-parse --abbrev-ref origin/HEAD";
        const auto r = run_capture(cmd.str());
        if (r.exit_status == 0) {
            const std::string out = r.output;
            const std::string prefix = "origin/";
            const std::size_t p = out.find(prefix);
            if (p != std::string::npos) {
                std::size_t start = p + prefix.size();
                std::size_t end = start;
                while (end < out.size() && out[end] != '\n' && out[end] != '\r' && out[end] != ' ' && out[end] != '\t') {
                    ++end;
                }
                const std::string b = out.substr(start, end - start);
                if (!b.empty()) return b;
            }
        }
    }

    // Fallback: main.
    return "main";
}

inline CapturedCommandResult reset_worktree_to_origin(const std::filesystem::path& repo_dir, const std::string& branch) {
    // Make repeated notebook runs deterministic:
    // - discard old local commits (including accidental big blobs that cause 408)
    // - remove untracked files
    std::ostringstream cmd;
    cmd << "git -C " << shell_quote(repo_dir.string())
        << " fetch origin --prune"
        << " && git -C " << shell_quote(repo_dir.string())
        << " checkout -B " << shell_quote(branch) << " origin/" << shell_quote(branch)
        << " && git -C " << shell_quote(repo_dir.string())
        << " reset --hard origin/" << shell_quote(branch)
        << " && git -C " << shell_quote(repo_dir.string())
        << " clean -fd";
    return run_capture(cmd.str());
}

inline bool looks_like_lfs_pointer_file(const std::filesystem::path& p) {
    // Check the on-disk file content quickly. If it's a pointer, it starts with:
    // version https://git-lfs.github.com/spec/v1
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return false;
    char buf[128] = {0};
    const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    if (n == 0) return false;
    const std::string s(buf, buf + n);
    return s.find("version https://git-lfs.github.com/spec/v1") != std::string::npos;
}

inline bool should_force_lfs_for_path(const std::string& path_in_repo) {
    // Heuristic: typical weight extensions.
    const auto dot = path_in_repo.find_last_of('.');
    if (dot == std::string::npos) return false;
    const std::string ext = path_in_repo.substr(dot);
    return (ext == ".bin") || (ext == ".pt") || (ext == ".safetensors") || (ext == ".onnx");
}

inline void print_progress_bar(
    const int current_step,
    const int total_steps,
    const char* label,
    const int bar_width = 28) {

    if (total_steps <= 0) return;
    int cur = current_step;
    if (cur < 0) cur = 0;
    if (cur > total_steps) cur = total_steps;

    const int filled = (cur * bar_width) / total_steps;
    const int percent = (cur * 100) / total_steps;

    std::ostringstream oss;
    oss << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        oss << (i < filled ? '#' : '.');
    }
    oss << "] " << percent << "%";
    if (label && *label) oss << " - " << label;

    const std::string line = oss.str();
    (void)std::fwrite(line.data(), 1, line.size(), stdout);
    (void)std::fflush(stdout);

    if (cur == total_steps) {
        const char nl = '\n';
        (void)std::fwrite(&nl, 1, 1, stdout);
        (void)std::fflush(stdout);
    }
}

inline std::string getenv_or_empty(const char* name) {
    if (name == nullptr) return std::string();
    if (const char* v = std::getenv(name)) return std::string(v);
    return std::string();
}

inline CommandResult run_system(const std::string& command) {
    const int rc = std::system(command.c_str());
    return CommandResult{rc, command};
}

} // namespace detail

struct GitUploadOptions {
    bool create_if_missing{true};
    bool create_private{true};
    bool use_lfs{false};
    bool stream_progress{false};
    std::size_t max_stream_bytes{200000};
    int push_retries{2};
    int push_retry_delay_seconds{2};
};

// --- Public API ---

inline CommandResult run(const std::string& command) {
    return detail::run_system(command);
}

inline CapturedCommandResult run_verbose(const std::string& command, const std::string& redact_token = std::string()) {
    return detail::run_capture(command, redact_token);
}

inline std::string read_token_file(const std::string& path) {
    // Minimal reader for xcpp17 environments where <fstream> can be problematic.
    // Reads the first line and trims common whitespace/newline characters.
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Failed to open token file: " + path);
    }

    char buf[8192];
    if (!std::fgets(buf, static_cast<int>(sizeof(buf)), f)) {
        std::fclose(f);
        throw std::runtime_error("Failed to read token from file: " + path);
    }
    std::fclose(f);

    std::size_t n = 0;
    while (n < sizeof(buf) && buf[n] != '\0') ++n;
    while (n > 0) {
        const char c = buf[n - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            --n;
        } else {
            break;
        }
    }
    std::size_t start = 0;
    while (start < n) {
        const char c = buf[start];
        if (c == ' ' || c == '\t') {
            ++start;
        } else {
            break;
        }
    }

    if (start >= n) {
        throw std::runtime_error("Token file is empty/whitespace: " + path);
    }

    return std::string(buf + start, n - start);
}

inline void require_huggingface_cli() {
    // `huggingface-cli --help` should return 0 when installed.
    // Note: On some systems it may return nonzero; if so, callers can bypass this.
    if (!detail::has_command("huggingface-cli")) {
        throw std::runtime_error(
            "huggingface-cli not found (or not runnable). Install huggingface_hub (pip/conda) and ensure it's on PATH.");
    }
}

// --- No-Python path: direct HTTP download via `curl` binary ---

inline std::string resolve_url(
    const std::string& repo_id,
    const std::string& path_in_repo,
    const RepoType repo_type = RepoType::model,
    const std::string& revision = "main") {

    // Hugging Face "resolve" URLs are stable and don't require python tooling.
    // model:   https://huggingface.co/<repo_id>/resolve/<rev>/<path>
    // dataset: https://huggingface.co/datasets/<repo_id>/resolve/<rev>/<path>
    // space:   https://huggingface.co/spaces/<repo_id>/resolve/<rev>/<path>
    std::ostringstream url;
    url << "https://huggingface.co/";
    if (repo_type == RepoType::dataset) url << "datasets/";
    if (repo_type == RepoType::space) url << "spaces/";
    url << repo_id << "/resolve/" << revision << "/" << path_in_repo;
    return url.str();
}

inline CommandResult download_file_http(
    const std::string& repo_id,
    const std::string& path_in_repo,
    const std::string& out_path,
    const RepoType repo_type = RepoType::model,
    const std::string& revision = "main",
    const std::string& token = std::string()) {

    if (!detail::has_command("curl")) {
        throw std::runtime_error("curl not found. Install curl (apt-get install curl) or use the huggingface-cli path.");
    }

    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());

    const auto url = resolve_url(repo_id, path_in_repo, repo_type, revision);

    std::ostringstream cmd;
    cmd << "curl -L -f";
    if (!token.empty()) {
        cmd << " -H " << detail::shell_quote(std::string("Authorization: Bearer ") + token);
    }
    cmd << " -o " << detail::shell_quote(out_path);
    cmd << " " << detail::shell_quote(url);
    return run(cmd.str());
}

inline CommandResult download_file_http_from_env(
    const std::string& repo_id,
    const std::string& path_in_repo,
    const std::string& out_path,
    const RepoType repo_type = RepoType::model,
    const std::string& revision = "main",
    const char* token_env = "HF_TOKEN") {

    const std::string tok = detail::getenv_or_empty(token_env);
    return download_file_http(repo_id, path_in_repo, out_path, repo_type, revision, tok);
}

// --- No-Python path: upload via git (works without HF python libs) ---

inline std::string repo_git_url(
    const std::string& repo_id,
    const RepoType repo_type = RepoType::model) {

    std::ostringstream url;
    url << "https://huggingface.co/";
    if (repo_type == RepoType::dataset) url << "datasets/";
    if (repo_type == RepoType::space) url << "spaces/";
    url << repo_id;
    return url.str();
}

inline std::string repo_git_url_with_token(
    const std::string& repo_id,
    const RepoType repo_type,
    const std::string& token,
    const std::string& username = "hf") {

    // Use token as password in HTTPS URL.
    // NOTE: This may leak the token into shell history / process lists.
    // Prefer running inside a trusted environment.
    std::ostringstream url;
    url << "https://" << username << ":" << token << "@huggingface.co/";
    if (repo_type == RepoType::dataset) url << "datasets/";
    if (repo_type == RepoType::space) url << "spaces/";
    url << repo_id;
    return url.str();
}

inline CommandResult upload_file_git(
    const std::string& repo_id,
    const std::string& local_path,
    const std::string& path_in_repo,
    const RepoType repo_type = RepoType::model,
    const std::string& commit_message = "upload from C++ (git)",
    const std::string& token = std::string(),
    const std::string& username = std::string(),
    const std::string& workdir = ".hf/_git_upload") {

    if (!detail::has_command("git")) {
        throw std::runtime_error("git not found. Install git (apt-get install git) or use the huggingface-cli path.");
    }
    if (!std::filesystem::exists(std::filesystem::path(local_path))) {
        throw std::runtime_error("upload_file_git: local file not found: " + local_path);
    }

    const std::filesystem::path base = std::filesystem::path(workdir) / (repo_id);
    std::filesystem::create_directories(base.parent_path());

    const bool repo_exists = std::filesystem::exists(base / ".git");
    const std::string remote = !token.empty()
        ? repo_git_url_with_token(repo_id, repo_type, token, username.empty() ? std::string("hf") : username)
        : repo_git_url(repo_id, repo_type);

    // Clone (first time) or pull.
    CommandResult res{};
    if (!repo_exists) {
        std::ostringstream cmd;
        cmd << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
        res = run(cmd.str());
        if (res.exit_code != 0) return res;
    } else {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " pull --rebase";
        res = run(cmd.str());
        // Don't early return: repo may be up-to-date but still nonzero in some configs.
    }

    // Copy file into repo working tree.
    const std::filesystem::path dst = base / std::filesystem::path(path_in_repo);
    std::filesystem::create_directories(dst.parent_path());
    std::filesystem::copy_file(std::filesystem::path(local_path), dst,
                               std::filesystem::copy_options::overwrite_existing);

    // Add + commit + push.
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " add " << detail::shell_quote(path_in_repo);
        res = run(cmd.str());
        if (res.exit_code != 0) return res;
    }
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " commit -m " << detail::shell_quote(commit_message);
        // commit may fail if there are no changes; allow that.
        (void)run(cmd.str());
    }
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " push";
        res = run(cmd.str());
        return res;
    }
}

inline GitUploadLog upload_file_git_verbose(
    const std::string& repo_id,
    const std::string& local_path,
    const std::string& path_in_repo,
    const RepoType repo_type = RepoType::model,
    const std::string& commit_message = "upload from C++ (git)",
    const std::string& token = std::string(),
    const std::string& username = std::string(),
    const std::string& workdir = ".hf/_git_upload",
    const bool create_if_missing = true,
    const bool create_private = true) {

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    if (!detail::has_command("git")) {
        throw std::runtime_error("git not found. Install git (apt-get install git) or use the huggingface-cli path.");
    }
    if (!std::filesystem::exists(std::filesystem::path(local_path))) {
        throw std::runtime_error("upload_file_git_verbose: local file not found: " + local_path);
    }

    GitUploadLog log;
    const std::size_t max_step_out = 2000;
    const std::size_t max_total_out = 12000;

    const auto append_step = [&](const char* name, const CapturedCommandResult& r) {
        std::ostringstream head;
        head << "\n== step: " << name
             << " (exit_status=" << r.exit_status
             << ", system_rc=" << r.system_rc
             << ", s=" << r.seconds << ") ==\n";
        head << "$ " << r.command << "\n";
        log.report += head.str();

        if (!r.output.empty()) {
            if (r.output.size() > max_step_out) {
                log.report.append(r.output.data(), max_step_out);
                log.report += "\n... (truncated)\n";
            } else {
                log.report += r.output;
                if (log.report.empty() || log.report.back() != '\n') log.report += "\n";
            }
        }

        if (log.report.size() > max_total_out) {
            log.report.resize(max_total_out);
            log.report += "\n... (total output truncated)\n";
        }
    };
    const std::filesystem::path base = std::filesystem::path(workdir) / (repo_id);
    std::filesystem::create_directories(base.parent_path());

    const bool repo_exists = std::filesystem::exists(base / ".git");
    const std::string remote = !token.empty()
        ? repo_git_url_with_token(repo_id, repo_type, token, username.empty() ? std::string("hf") : username)
        : repo_git_url(repo_id, repo_type);

    // Clone (first time) or pull.
    if (!repo_exists) {
        std::ostringstream cmd;
        cmd << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("clone", r);
        if (r.exit_status != 0) {
            // If repo is missing, optionally create it via HTTP then retry clone.
            if (create_if_missing && detail::contains_repo_not_found(r.output)) {
                log.report += "\n(note) Repo not found; attempting to create it via https://huggingface.co/api/repos/create\n";

                const auto parts = detail::split_repo_id(repo_id);
                const std::string owner = parts.first;
                const std::string name = parts.second;
                const char* api_type = to_string(repo_type);

                bool created = false;
                if (!owner.empty()) {
                    const auto c1 = detail::repo_create_http_attempt(name, owner, api_type, create_private, token);
                    append_step("create_repo(org)", c1);
                    created = detail::http_status_is_ok_or_exists(c1.output);
                }
                if (!created) {
                    const auto c2 = detail::repo_create_http_attempt(name, std::string(), api_type, create_private, token);
                    append_step("create_repo(user)", c2);
                    created = detail::http_status_is_ok_or_exists(c2.output);
                }

                if (created) {
                    std::ostringstream cmd2;
                    cmd2 << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
                    const auto r2 = detail::run_capture(cmd2.str(), token);
                    append_step("clone_retry", r2);
                    if (r2.exit_status == 0) {
                        // Continue with upload.
                    } else {
                        const auto t1 = clock::now();
                        log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                        log.system_rc = r2.system_rc;
                        log.exit_status = r2.exit_status;
                        return log;
                    }
                } else {
                    const auto t1 = clock::now();
                    log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                    // Keep the original clone failure rc.
                    log.system_rc = r.system_rc;
                    log.exit_status = r.exit_status;
                    return log;
                }
            } else {
                const auto t1 = clock::now();
                log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                log.system_rc = r.system_rc;
                log.exit_status = r.exit_status;
                return log;
            }
        }
    } else {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " pull --rebase";
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("pull", r);
        // pull is not necessarily fatal; continue.
    }

    // Copy file into repo working tree.
    const std::filesystem::path dst = base / std::filesystem::path(path_in_repo);
    std::filesystem::create_directories(dst.parent_path());
    std::filesystem::copy_file(std::filesystem::path(local_path), dst,
                               std::filesystem::copy_options::overwrite_existing);

    // Add.
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " add " << detail::shell_quote(path_in_repo);
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("add", r);
        if (r.exit_status != 0) {
            const auto t1 = clock::now();
            log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
            log.system_rc = r.system_rc;
            log.exit_status = r.exit_status;
            return log;
        }
    }

    // Commit (may fail if no changes).
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " commit -m " << detail::shell_quote(commit_message);
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("commit", r);
    }

    // Push.
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " push";
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("push", r);
        const auto t1 = clock::now();
        log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
        log.system_rc = r.system_rc;
        log.exit_status = r.exit_status;
        return log;
    }
}

inline GitUploadLog upload_files_git_verbose(
    const std::string& repo_id,
    const std::vector<std::pair<std::string, std::string>>& local_and_repo_paths,
    const RepoType repo_type = RepoType::model,
    const std::string& commit_message = "upload from C++ (git)",
    const std::string& token = std::string(),
    const std::string& username = std::string(),
    const std::string& workdir = ".hf/_git_upload",
    const bool create_if_missing = true,
    const bool create_private = true) {

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    if (!detail::has_command("git")) {
        throw std::runtime_error("git not found. Install git (apt-get install git) or use the huggingface-cli path.");
    }
    if (local_and_repo_paths.empty()) {
        throw std::runtime_error("upload_files_git_verbose: no files provided");
    }
    for (const auto& p : local_and_repo_paths) {
        if (p.first.empty()) {
            throw std::runtime_error("upload_files_git_verbose: empty local_path");
        }
        if (p.second.empty()) {
            throw std::runtime_error("upload_files_git_verbose: empty path_in_repo");
        }
        if (!std::filesystem::exists(std::filesystem::path(p.first))) {
            throw std::runtime_error("upload_files_git_verbose: local file not found: " + p.first);
        }
    }

    GitUploadLog log;
    const std::size_t max_step_out = 2000;
    const std::size_t max_total_out = 12000;

    const auto append_step = [&](const char* name, const CapturedCommandResult& r) {
        std::ostringstream head;
        head << "\n== step: " << name
             << " (exit_status=" << r.exit_status
             << ", system_rc=" << r.system_rc
             << ", s=" << r.seconds << ") ==\n";
        head << "$ " << r.command << "\n";
        log.report += head.str();

        if (!r.output.empty()) {
            if (r.output.size() > max_step_out) {
                log.report.append(r.output.data(), max_step_out);
                log.report += "\n... (truncated)\n";
            } else {
                log.report += r.output;
                if (log.report.empty() || log.report.back() != '\n') log.report += "\n";
            }
        }

        if (log.report.size() > max_total_out) {
            log.report.resize(max_total_out);
            log.report += "\n... (total output truncated)\n";
        }
    };

    const std::filesystem::path base = std::filesystem::path(workdir) / (repo_id);
    std::filesystem::create_directories(base.parent_path());

    const bool repo_exists = std::filesystem::exists(base / ".git");
    const std::string remote = !token.empty()
        ? repo_git_url_with_token(repo_id, repo_type, token, username.empty() ? std::string("hf") : username)
        : repo_git_url(repo_id, repo_type);

    // Clone (first time) or pull.
    if (!repo_exists) {
        std::ostringstream cmd;
        cmd << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("clone", r);
        if (r.exit_status != 0) {
            // If repo is missing, optionally create it via HTTP then retry clone.
            if (create_if_missing && detail::contains_repo_not_found(r.output)) {
                log.report += "\n(note) Repo not found; attempting to create it via https://huggingface.co/api/repos/create\n";

                const auto parts = detail::split_repo_id(repo_id);
                const std::string owner = parts.first;
                const std::string name = parts.second;
                const char* api_type = to_string(repo_type);

                bool created = false;
                if (!owner.empty()) {
                    const auto c1 = detail::repo_create_http_attempt(name, owner, api_type, create_private, token);
                    append_step("create_repo(org)", c1);
                    created = detail::http_status_is_ok_or_exists(c1.output);
                }
                if (!created) {
                    const auto c2 = detail::repo_create_http_attempt(name, std::string(), api_type, create_private, token);
                    append_step("create_repo(user)", c2);
                    created = detail::http_status_is_ok_or_exists(c2.output);
                }

                if (created) {
                    std::ostringstream cmd2;
                    cmd2 << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
                    const auto r2 = detail::run_capture(cmd2.str(), token);
                    append_step("clone_retry", r2);
                    if (r2.exit_status != 0) {
                        const auto t1 = clock::now();
                        log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                        log.system_rc = r2.system_rc;
                        log.exit_status = r2.exit_status;
                        return log;
                    }
                } else {
                    const auto t1 = clock::now();
                    log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                    log.system_rc = r.system_rc;
                    log.exit_status = r.exit_status;
                    return log;
                }
            } else {
                const auto t1 = clock::now();
                log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                log.system_rc = r.system_rc;
                log.exit_status = r.exit_status;
                return log;
            }
        }
    } else {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " pull --rebase";
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("pull", r);
        // pull is not necessarily fatal; continue.
    }

    // Copy files into repo working tree.
    log.report += "\n== step: copy_files ==\n";
    for (const auto& p : local_and_repo_paths) {
        const std::filesystem::path dst = base / std::filesystem::path(p.second);
        std::filesystem::create_directories(dst.parent_path());
        std::filesystem::copy_file(std::filesystem::path(p.first), dst, std::filesystem::copy_options::overwrite_existing);
        log.report += "- " + p.first + " -> " + p.second + "\n";
        if (log.report.size() > max_total_out) {
            log.report.resize(max_total_out);
            log.report += "\n... (total output truncated)\n";
            break;
        }
    }

    // Add all paths.
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " add";
        for (const auto& p : local_and_repo_paths) {
            cmd << " " << detail::shell_quote(p.second);
        }
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("add", r);
        if (r.exit_status != 0) {
            const auto t1 = clock::now();
            log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
            log.system_rc = r.system_rc;
            log.exit_status = r.exit_status;
            return log;
        }
    }

    // Commit (may fail if no changes).
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " commit -m " << detail::shell_quote(commit_message);
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("commit", r);
    }

    // Push.
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " push --progress";
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("push", r);
        const auto t1 = clock::now();
        log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
        log.system_rc = r.system_rc;
        log.exit_status = r.exit_status;
        return log;
    }
}

inline GitUploadLog upload_files_git_verbose(
    const std::string& repo_id,
    const std::vector<std::pair<std::string, std::string>>& local_and_repo_paths,
    const RepoType repo_type,
    const std::string& commit_message,
    const std::string& token,
    const GitUploadOptions& options,
    const std::string& username = std::string(),
    const std::string& workdir = ".hf/_git_upload") {

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    if (!detail::has_command("git")) {
        throw std::runtime_error("git not found. Install git (apt-get install git) or use the huggingface-cli path.");
    }
    if (local_and_repo_paths.empty()) {
        throw std::runtime_error("upload_files_git_verbose: no files provided");
    }
    for (const auto& p : local_and_repo_paths) {
        if (p.first.empty()) throw std::runtime_error("upload_files_git_verbose: empty local_path");
        if (p.second.empty()) throw std::runtime_error("upload_files_git_verbose: empty path_in_repo");
        if (!std::filesystem::exists(std::filesystem::path(p.first))) {
            throw std::runtime_error("upload_files_git_verbose: local file not found: " + p.first);
        }
    }

    GitUploadLog log;
    const std::size_t max_step_out = 2000;
    const std::size_t max_total_out = 12000;

    const auto append_step = [&](const char* name, const CapturedCommandResult& r) {
        std::ostringstream head;
        head << "\n== step: " << name
             << " (exit_status=" << r.exit_status
             << ", system_rc=" << r.system_rc
             << ", s=" << r.seconds << ") ==\n";
        head << "$ " << r.command << "\n";
        log.report += head.str();

        if (!r.output.empty()) {
            if (r.output.size() > max_step_out) {
                log.report.append(r.output.data(), max_step_out);
                log.report += "\n... (truncated)\n";
            } else {
                log.report += r.output;
                if (log.report.empty() || log.report.back() != '\n') log.report += "\n";
            }
        }

        if (log.report.size() > max_total_out) {
            log.report.resize(max_total_out);
            log.report += "\n... (total output truncated)\n";
        }
    };

    const std::filesystem::path base = std::filesystem::path(workdir) / (repo_id);
    std::filesystem::create_directories(base.parent_path());

    const bool repo_exists = std::filesystem::exists(base / ".git");
    const std::string remote = !token.empty()
        ? repo_git_url_with_token(repo_id, repo_type, token, username.empty() ? std::string("hf") : username)
        : repo_git_url(repo_id, repo_type);

    int step = 0;
    const int total_steps = 5 + (options.use_lfs ? 1 : 0);
    const auto tick = [&](const char* label) {
        ++step;
        detail::print_progress_bar(step, total_steps, label);
    };

    tick(repo_exists ? "pull" : "clone");
    if (!repo_exists) {
        std::ostringstream cmd;
        cmd << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("clone", r);
        if (r.exit_status != 0) {
            if (options.create_if_missing && detail::contains_repo_not_found(r.output)) {
                log.report += "\n(note) Repo not found; attempting to create it via https://huggingface.co/api/repos/create\n";

                const auto parts = detail::split_repo_id(repo_id);
                const std::string owner = parts.first;
                const std::string name = parts.second;
                const char* api_type = to_string(repo_type);

                bool created = false;
                if (!owner.empty()) {
                    const auto c1 = detail::repo_create_http_attempt(name, owner, api_type, options.create_private, token);
                    append_step("create_repo(org)", c1);
                    created = detail::http_status_is_ok_or_exists(c1.output);
                }
                if (!created) {
                    const auto c2 = detail::repo_create_http_attempt(name, std::string(), api_type, options.create_private, token);
                    append_step("create_repo(user)", c2);
                    created = detail::http_status_is_ok_or_exists(c2.output);
                }

                if (created) {
                    std::ostringstream cmd2;
                    cmd2 << "git clone " << detail::shell_quote(remote) << " " << detail::shell_quote(base.string());
                    const auto r2 = detail::run_capture(cmd2.str(), token);
                    append_step("clone_retry", r2);
                    if (r2.exit_status != 0) {
                        const auto t1 = clock::now();
                        log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                        log.system_rc = r2.system_rc;
                        log.exit_status = r2.exit_status;
                        return log;
                    }
                } else {
                    const auto t1 = clock::now();
                    log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                    log.system_rc = r.system_rc;
                    log.exit_status = r.exit_status;
                    return log;
                }
            } else {
                const auto t1 = clock::now();
                log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
                log.system_rc = r.system_rc;
                log.exit_status = r.exit_status;
                return log;
            }
        }
    } else {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " pull --rebase";
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("pull", r);
    }

    // Always normalize the worktree to remote HEAD to avoid accumulating local commits.
    // This is critical when a previous attempt accidentally committed big blobs (causing HTTP 408).
    {
        const std::string branch = detail::detect_default_remote_branch(base);
        const auto r = detail::reset_worktree_to_origin(base, branch);
        append_step("reset_to_origin", r);
        // Non-fatal: some repos might be empty or not have origin/HEAD set.
    }

    if (options.use_lfs) {
        tick("git-lfs");
        if (!detail::has_git_lfs()) {
            log.report += "\n(note) git-lfs does not appear to be installed. On Ubuntu/WSL: sudo apt-get install -y git-lfs && git lfs install\n";
        }

        {
            std::ostringstream cmd;
            cmd << "git -C " << detail::shell_quote(base.string()) << " lfs install --local";
            const auto r = detail::run_capture(cmd.str(), token);
            append_step("lfs_install", r);
        }

        const char* patterns[] = {"*.bin", "*.pt", "*.safetensors", "*.onnx"};
        for (const char* pat : patterns) {
            std::ostringstream cmd;
            cmd << "git -C " << detail::shell_quote(base.string()) << " lfs track " << detail::shell_quote(pat);
            const auto r = detail::run_capture(cmd.str(), token);
            append_step("lfs_track", r);
        }

        {
            std::ostringstream cmd;
            cmd << "git -C " << detail::shell_quote(base.string()) << " add .gitattributes";
            const auto r = detail::run_capture(cmd.str(), token);
            append_step("lfs_add_gitattributes", r);
        }
    }

    tick("copy files");
    log.report += "\n== step: copy_files ==\n";
    for (const auto& p : local_and_repo_paths) {
        const std::filesystem::path dst = base / std::filesystem::path(p.second);
        std::filesystem::create_directories(dst.parent_path());
        std::filesystem::copy_file(std::filesystem::path(p.first), dst, std::filesystem::copy_options::overwrite_existing);
        log.report += "- " + p.first + " -> " + p.second + "\n";
        if (log.report.size() > max_total_out) {
            log.report.resize(max_total_out);
            log.report += "\n... (total output truncated)\n";
            break;
        }
    }

    // If using LFS, ensure large/weight files are added as LFS pointers.
    // If a file gets added as a normal blob, it may create a huge pack and trigger HTTP 408.
    if (options.use_lfs) {
        for (const auto& p : local_and_repo_paths) {
            const std::filesystem::path dst = base / std::filesystem::path(p.second);
            std::error_code ec;
            const auto sz = std::filesystem::file_size(dst, ec);
            const bool big = (!ec && sz >= (10ull * 1024ull * 1024ull));
            if (!big && !detail::should_force_lfs_for_path(p.second)) continue;

            // If the on-disk file is already a pointer, skip.
            if (detail::looks_like_lfs_pointer_file(dst)) continue;

            // Force re-add through clean filter: rm --cached + add.
            {
                std::ostringstream cmd;
                cmd << "git -C " << detail::shell_quote(base.string())
                    << " rm --cached -- " << detail::shell_quote(p.second);
                const auto r = detail::run_capture(cmd.str(), token);
                append_step("lfs_rm_cached", r);
            }
            {
                std::ostringstream cmd;
                cmd << "git -C " << detail::shell_quote(base.string())
                    << " add " << detail::shell_quote(p.second);
                const auto r = detail::run_capture(cmd.str(), token);
                append_step("lfs_readd", r);
            }
        }
    }

    tick("git add");
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " add";
        for (const auto& p : local_and_repo_paths) {
            cmd << " " << detail::shell_quote(p.second);
        }
        if (options.use_lfs) {
            cmd << " .gitattributes";
        }
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("add", r);
        if (r.exit_status != 0) {
            const auto t1 = clock::now();
            log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
            log.system_rc = r.system_rc;
            log.exit_status = r.exit_status;
            return log;
        }
    }

    tick("git commit");
    {
        std::ostringstream cmd;
        cmd << "git -C " << detail::shell_quote(base.string()) << " commit -m " << detail::shell_quote(commit_message);
        const auto r = detail::run_capture(cmd.str(), token);
        append_step("commit", r);
    }

    tick("git push");
    {
        CapturedCommandResult last;
        const int attempts = (options.push_retries < 1) ? 1 : options.push_retries;
        for (int i = 1; i <= attempts; ++i) {
            std::ostringstream cmd;
            // Be more tolerant of slow links; this helps avoid curl 22 / HTTP 408 in long uploads.
            cmd << "git -C " << detail::shell_quote(base.string())
                << " -c http.lowSpeedLimit=0"
                << " -c http.lowSpeedTime=999999"
                << " -c http.postBuffer=524288000"
                << " push --progress";

            last = detail::run_capture_streaming(cmd.str(), token, options.stream_progress, options.max_stream_bytes);
            append_step(i == 1 ? "push" : "push_retry", last);
            if (last.exit_status == 0) break;

            // Retry only for likely transient network timeouts.
            const bool looks_like_timeout =
                (last.output.find("HTTP 408") != std::string::npos)
                || (last.output.find("RPC failed") != std::string::npos)
                || (last.output.find("remote end hung up") != std::string::npos)
                || (last.output.find("unexpected disconnect") != std::string::npos);
            if (!looks_like_timeout) break;
            if (i < attempts && options.push_retry_delay_seconds > 0) {
                std::this_thread::sleep_for(std::chrono::seconds(options.push_retry_delay_seconds));
            }
        }

        const auto t1 = clock::now();
        log.seconds_total = std::chrono::duration<double>(t1 - t0).count();
        log.system_rc = last.system_rc;
        log.exit_status = last.exit_status;
        detail::print_progress_bar(total_steps, total_steps, "done");
        return log;
    }
}

// -----------------------------------------------------------------------------
// High-level Hub-style uploader (notebook-friendly)
// -----------------------------------------------------------------------------

class HubUploader {
public:
    struct Model {
        explicit Model(HubUploader* owner) : owner_(owner) {}
        void upload(const std::string& local_path, const std::string& path_in_repo = "pytorch_model.bin") {
            owner_->upload(local_path, path_in_repo);
        }

    private:
        HubUploader* owner_{};
    };

    struct Tokenizer {
        explicit Tokenizer(HubUploader* owner) : owner_(owner) {}
        void upload(const std::string& local_path, const std::string& path_in_repo = "tokenizer.json") {
            owner_->upload(local_path, path_in_repo);
        }

    private:
        HubUploader* owner_{};
    };

    struct Config {
        explicit Config(HubUploader* owner) : owner_(owner) {}
        void upload(const std::string& local_path, const std::string& path_in_repo = "config.json") {
            owner_->upload(local_path, path_in_repo);
        }

    private:
        HubUploader* owner_{};
    };

    HubUploader(
        std::string repo_id,
        const RepoType repo_type,
        std::string token,
        GitUploadOptions options = GitUploadOptions{})
        : repo_id_(std::move(repo_id)),
          repo_type_(repo_type),
          token_(std::move(token)),
          options_(options),
          model(this),
          tokenizer(this),
          config(this) {

        // Notebook-friendly defaults.
        if (!token_.empty()) {
            // no-op: just here to emphasize that token is optional for public repos.
        }
    }

    // Convenience constructor for model repos.
    explicit HubUploader(std::string repo_id, std::string token = std::string(), GitUploadOptions options = GitUploadOptions{})
        : HubUploader(std::move(repo_id), RepoType::model, std::move(token), options) {}

    // Generic upload (for any file).
    void upload(const std::string& local_path, const std::string& path_in_repo) {
        files_.emplace_back(local_path, path_in_repo);
    }

    // Push everything in one commit.
    GitUploadLog push(const std::string& commit_message = "upload from C++ (git)") {
        return upload_files_git_verbose(
            repo_id_,
            files_,
            repo_type_,
            commit_message,
            token_,
            options_);
    }

    // Exposed sub-APIs to mimic python-ish usage.
    Model model;
    Tokenizer tokenizer;
    Config config;

private:
    std::string repo_id_;
    RepoType repo_type_{RepoType::model};
    std::string token_;
    GitUploadOptions options_{};
    std::vector<std::pair<std::string, std::string>> files_{};
};

// -----------------------------------------------------------------------------
// Simple artifact wrappers (PyTorch-ish: obj.upload(repo_id, token))
// -----------------------------------------------------------------------------

class Model {
public:
    Model() = default;
    explicit Model(std::string local_weights_path) { load(std::move(local_weights_path)); }

    void load(std::string local_weights_path) {
        if (local_weights_path.empty()) {
            throw std::runtime_error("Model::load: local_weights_path is empty");
        }
        const std::filesystem::path p(local_weights_path);
        if (!std::filesystem::exists(p)) {
            throw std::runtime_error("Model::load: file does not exist: " + local_weights_path);
        }
        if (!std::filesystem::is_regular_file(p)) {
            throw std::runtime_error("Model::load: not a regular file: " + local_weights_path);
        }
        local_path_ = std::move(local_weights_path);
    }

    const std::string& local_path() const { return local_path_; }

    GitUploadLog upload(
        const std::string& repo_id,
        const std::string& token,
        GitUploadOptions options = GitUploadOptions{},
        const std::string& path_in_repo = "pytorch_model.bin",
        const std::string& commit_message = "upload model from C++ (git)") const {

        if (local_path_.empty()) {
            throw std::runtime_error("Model::upload: call load(...) first");
        }
        if (repo_id.empty()) {
            throw std::runtime_error("Model::upload: repo_id is empty");
        }
        if (path_in_repo.empty()) {
            throw std::runtime_error("Model::upload: path_in_repo is empty");
        }

        std::vector<std::pair<std::string, std::string>> files;
        files.emplace_back(local_path_, path_in_repo);
        return upload_files_git_verbose(repo_id, files, RepoType::model, commit_message, token, options);
    }

private:
    std::string local_path_{};
};

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(std::string local_tokenizer_path) { load(std::move(local_tokenizer_path)); }

    void load(std::string local_tokenizer_path) {
        if (local_tokenizer_path.empty()) {
            throw std::runtime_error("Tokenizer::load: local_tokenizer_path is empty");
        }
        const std::filesystem::path p(local_tokenizer_path);
        if (!std::filesystem::exists(p)) {
            throw std::runtime_error("Tokenizer::load: file does not exist: " + local_tokenizer_path);
        }
        if (!std::filesystem::is_regular_file(p)) {
            throw std::runtime_error("Tokenizer::load: not a regular file: " + local_tokenizer_path);
        }
        local_path_ = std::move(local_tokenizer_path);
    }

    const std::string& local_path() const { return local_path_; }

    GitUploadLog upload(
        const std::string& repo_id,
        const std::string& token,
        GitUploadOptions options = GitUploadOptions{},
        const std::string& path_in_repo = "tokenizer.json",
        const std::string& commit_message = "upload tokenizer from C++ (git)") const {

        if (local_path_.empty()) {
            throw std::runtime_error("Tokenizer::upload: call load(...) first");
        }
        if (repo_id.empty()) {
            throw std::runtime_error("Tokenizer::upload: repo_id is empty");
        }
        if (path_in_repo.empty()) {
            throw std::runtime_error("Tokenizer::upload: path_in_repo is empty");
        }

        std::vector<std::pair<std::string, std::string>> files;
        files.emplace_back(local_path_, path_in_repo);
        return upload_files_git_verbose(repo_id, files, RepoType::model, commit_message, token, options);
    }

private:
    std::string local_path_{};
};

inline CommandResult login_with_token(const std::string& token, const bool add_to_git_credential = true) {
    std::ostringstream cmd;
    cmd << "huggingface-cli login";
    cmd << " --token " << detail::shell_quote(token);
    if (add_to_git_credential) cmd << " --add-to-git-credential";
    return run(cmd.str());
}

inline CommandResult login_with_token(const char* token, const bool add_to_git_credential = true) {
    if (token == nullptr || *token == '\0') {
        throw std::runtime_error("Token is empty.");
    }
    return login_with_token(std::string(token), add_to_git_credential);
}

// Convenience overload:
// - If `token` is provided, uses it directly.
// - Otherwise falls back to `login_from_env(env_name, ...)`.
//
// Note: Passing token on the command line can leak it via process lists/history.
// Prefer env vars or credential helpers when possible.
inline CommandResult login_from_env(const char* env_name = "HF_TOKEN", const bool add_to_git_credential = true) {
    const std::string token = detail::getenv_or_empty(env_name);
    if (token.empty()) {
        throw std::runtime_error(std::string("Missing environment variable ") + (env_name ? env_name : "(null)") + ".");
    }
    return login_with_token(token, add_to_git_credential);
}

// Token-first convenience overload.
// If `token` is empty/null, falls back to env var `env_name`.
inline CommandResult login(const char* token, const char* env_name = "HF_TOKEN", const bool add_to_git_credential = true) {
    if (token != nullptr && *token != '\0') {
        return login_with_token(token, add_to_git_credential);
    }
    return login_from_env(env_name, add_to_git_credential);
}

inline CommandResult login(const std::string& token, const char* env_name = "HF_TOKEN", const bool add_to_git_credential = true) {
    return login(token.empty() ? nullptr : token.c_str(), env_name, add_to_git_credential);
}

inline CommandResult download(
    const std::string& repo_id,
    const std::string& local_dir,
    const RepoType repo_type = RepoType::model,
    const std::string& revision = std::string(),
    const std::string& include_glob = std::string(),
    const std::string& exclude_glob = std::string(),
    const std::string& token = std::string()) {

    std::ostringstream cmd;
    cmd << "huggingface-cli download " << detail::shell_quote(repo_id);
    cmd << " --repo-type " << to_string(repo_type);
    cmd << " --local-dir " << detail::shell_quote(local_dir);
    cmd << " --local-dir-use-symlinks False";

    if (!revision.empty()) {
        cmd << " --revision " << detail::shell_quote(revision);
    }
    if (!include_glob.empty()) {
        cmd << " --include " << detail::shell_quote(include_glob);
    }
    if (!exclude_glob.empty()) {
        cmd << " --exclude " << detail::shell_quote(exclude_glob);
    }
    if (!token.empty()) {
        cmd << " --token " << detail::shell_quote(token);
    }

    return run(cmd.str());
}

inline CommandResult upload_file(
    const std::string& repo_id,
    const std::string& local_path,
    const std::string& path_in_repo,
    const RepoType repo_type = RepoType::model,
    const std::string& commit_message = std::string("upload from C++"),
    const std::string& token = std::string()) {

    std::ostringstream cmd;
    cmd << "huggingface-cli upload " << detail::shell_quote(repo_id);
    cmd << " " << detail::shell_quote(local_path);
    cmd << " " << detail::shell_quote(path_in_repo);
    cmd << " --repo-type " << to_string(repo_type);

    if (!commit_message.empty()) {
        cmd << " --commit-message " << detail::shell_quote(commit_message);
    }
    if (!token.empty()) {
        cmd << " --token " << detail::shell_quote(token);
    }

    return run(cmd.str());
}

inline CommandResult repo_create(
    const std::string& repo_id,
    const RepoType repo_type = RepoType::model,
    const bool is_private = true,
    const std::string& token = std::string()) {

    std::ostringstream cmd;
    cmd << "huggingface-cli repo create " << detail::shell_quote(repo_id);
    cmd << " --type " << to_string(repo_type);
    cmd << (is_private ? " --private" : " --public");
    if (!token.empty()) {
        cmd << " --token " << detail::shell_quote(token);
    }
    return run(cmd.str());
}

} // namespace mcppfa::hf
