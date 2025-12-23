#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <istream>
#include <optional>
#include <chrono>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef PLOTPP_FORCE_XCPP_DISPLAY
#define PLOTPP_FORCE_XCPP_DISPLAY 0
#endif

#if defined(__CLING__) && (PLOTPP_FORCE_XCPP_DISPLAY || __has_include(<xcpp/xdisplay.hpp>))
#include <xcpp/xdisplay.hpp>
#include <nlohmann/json.hpp>
#define PLOTPP_HAS_XCPP_DISPLAY 1
#else
#define PLOTPP_HAS_XCPP_DISPLAY 0
#endif

namespace plotpp {

struct inline_png {
    std::string path;
};

struct inline_svg {
    std::string path;
};

// In-memory display helpers (no file needed)
struct inline_png_bytes {
    std::string bytes;
};

struct inline_svg_text {
    std::string svg;
};

namespace detail {

inline std::string unique_token() {
    using namespace std::chrono;
    const auto now = steady_clock::now().time_since_epoch();
    const auto n = duration_cast<nanoseconds>(now).count();
    // Mix in rand() to reduce collision risk across rapid calls.
    return std::to_string(static_cast<long long>(n)) + "_" + std::to_string(std::rand());
}

inline std::filesystem::path plotpp_temp_dir() {
    auto dir = std::filesystem::temp_directory_path() / "plotpp";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

inline std::filesystem::path temp_path_with_ext(const char* ext_with_dot) {
    const auto base = plotpp_temp_dir();
    const auto tok = unique_token();
    return base / (std::string("plotpp_") + tok + ext_with_dot);
}

struct temp_paths {
    std::filesystem::path data;
    std::filesystem::path script;
    std::filesystem::path out;
    bool cleanup = true;

    ~temp_paths() {
        if (!cleanup) return;
        std::error_code ec;
        if (!out.empty()) std::filesystem::remove(out, ec);
        if (!script.empty()) std::filesystem::remove(script, ec);
        if (!data.empty()) std::filesystem::remove(data, ec);
    }
};

inline bool is_default_path(const std::string& path, const char* def) {
    return path == def;
}

struct Series {
    std::string label;
    std::vector<double> x;
    std::vector<double> y;
    std::string style;
};

struct FigureState {
    int width = 900;
    int height = 450;

    std::string title;
    std::string xlabel;
    std::string ylabel;

    bool grid = false;
    bool legend = false;

    // Bar plots (gnuplot boxes)
    bool any_boxes = false;
    double boxwidth = 0.8; // relative
    std::string boxes_fill = "solid 0.7";

    std::vector<Series> series;

    void clear_series() {
        series.clear();
        any_boxes = false;
        boxwidth = 0.8;
        boxes_fill = "solid 0.7";
    }
};

inline std::unordered_map<int, FigureState>& figures() {
    static std::unordered_map<int, FigureState> figs;
    if (figs.empty()) {
        figs.emplace(0, FigureState{});
    }
    return figs;
}

inline int& current_figure_id() {
    static int id = 0;
    return id;
}

inline int& next_figure_id() {
    static int id = 1;
    return id;
}

inline FigureState& state() {
    auto& figs = figures();
    const int id = current_figure_id();
    auto it = figs.find(id);
    if (it == figs.end()) {
        it = figs.emplace(id, FigureState{}).first;
    }
    return it->second;
}

inline std::string escape_single_quotes(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\'') {
            out.push_back('\'');
            out.push_back('\'');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

inline void ensure_xy_sizes_match(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("plotpp: x and y must have the same length");
    }
}

inline std::vector<double> make_default_x(std::size_t n) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<double>(i);
    return x;
}

inline void write_data_tsv(const FigureState& st, const std::filesystem::path& data_path) {
    std::ofstream out(data_path);
    if (!out) {
        throw std::runtime_error("plotpp: failed to write data file: " + data_path.string());
    }

    // One block per series, separated by a blank line.
    for (const auto& s : st.series) {
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            out << s.x[i] << '\t' << s.y[i] << '\n';
        }
        out << "\n";
    }
}

inline std::string build_plot_command(const FigureState& st, const std::filesystem::path& data_path) {
    std::ostringstream cmd;
    cmd << "plot ";
    for (std::size_t i = 0; i < st.series.size(); ++i) {
        const auto& s = st.series[i];
        if (i != 0) cmd << ", \\\n     ";

        cmd << "'" << escape_single_quotes(data_path.string()) << "' index " << i << " using 1:2 ";
        if (!s.style.empty()) {
            cmd << s.style;
        } else {
            cmd << "with lines";
        }

        if (st.legend) {
            const std::string label = s.label.empty() ? ("series_" + std::to_string(i)) : s.label;
            cmd << " title '" << escape_single_quotes(label) << "'";
        } else {
            cmd << " notitle";
        }
    }
    cmd << "\n";
    return cmd.str();
}

inline void write_gnuplot_script(const FigureState& st,
                                const std::filesystem::path& script_path,
                                const std::filesystem::path& data_path,
                                const std::optional<std::filesystem::path>& output_path) {
    std::ofstream gp(script_path);
    if (!gp) {
        throw std::runtime_error("plotpp: failed to write gnuplot script: " + script_path.string());
    }

    if (output_path.has_value()) {
        const std::string out = output_path->string();
        auto ends_with = [](const std::string& a, const std::string& b) {
            return a.size() >= b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0;
        };

        if (ends_with(out, ".svg")) {
            gp << "set terminal svg size " << st.width << "," << st.height << " dynamic\n";
        } else {
            gp << "set terminal pngcairo size " << st.width << "," << st.height << " enhanced\n";
        }
        gp << "set output '" << escape_single_quotes(out) << "'\n";
    }

    if (!st.title.empty()) gp << "set title '" << escape_single_quotes(st.title) << "'\n";
    if (!st.xlabel.empty()) gp << "set xlabel '" << escape_single_quotes(st.xlabel) << "'\n";
    if (!st.ylabel.empty()) gp << "set ylabel '" << escape_single_quotes(st.ylabel) << "'\n";

    if (st.grid) gp << "set grid\n"; else gp << "unset grid\n";
    if (st.legend) gp << "set key\n"; else gp << "unset key\n";

    if (st.any_boxes) {
        gp << "set style fill " << st.boxes_fill << "\n";
        gp << "set boxwidth " << st.boxwidth << " relative\n";
    }

    gp << build_plot_command(st, data_path);
}

inline int run_gnuplot_file(const std::filesystem::path& script_path) {
    const std::string cmd = std::string("gnuplot ") + script_path.string();
    return std::system(cmd.c_str());
}

inline std::string read_file_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("plotpp: failed to read file: " + path.string());
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

inline bool wait_for_file_nonempty(const std::filesystem::path& path,
                                  std::chrono::milliseconds timeout = std::chrono::milliseconds(3000),
                                  std::chrono::milliseconds interval = std::chrono::milliseconds(50)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            const auto sz = std::filesystem::file_size(path, ec);
            if (!ec && sz > 0) return true;
        }
        std::this_thread::sleep_for(interval);
    }

    std::error_code ec;
    return std::filesystem::exists(path, ec) && !ec && std::filesystem::file_size(path, ec) > 0;
}

inline std::string base64_encode(const unsigned char* data, std::size_t len) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    std::size_t i = 0;
    for (; i + 2 < len; i += 3) {
        const unsigned int n = (static_cast<unsigned int>(data[i]) << 16) |
                               (static_cast<unsigned int>(data[i + 1]) << 8) |
                               (static_cast<unsigned int>(data[i + 2]));
        out.push_back(k[(n >> 18) & 63]);
        out.push_back(k[(n >> 12) & 63]);
        out.push_back(k[(n >> 6) & 63]);
        out.push_back(k[n & 63]);
    }

    if (i < len) {
        unsigned int n = static_cast<unsigned int>(data[i]) << 16;
        out.push_back(k[(n >> 18) & 63]);
        if (i + 1 < len) {
            n |= static_cast<unsigned int>(data[i + 1]) << 8;
            out.push_back(k[(n >> 12) & 63]);
            out.push_back(k[(n >> 6) & 63]);
            out.push_back('=');
        } else {
            out.push_back(k[(n >> 12) & 63]);
            out.push_back('=');
            out.push_back('=');
        }
    }

    return out;
}

inline void inject_white_background_svg(std::string& s) {
    const std::string tag = "<svg";
    const auto pos = s.find(tag);
    if (pos == std::string::npos) return;

    const auto gt = s.find('>', pos);
    if (gt == std::string::npos) return;

    const std::string marker = "plotpp-bg-white";
    if (s.find(marker, pos) != std::string::npos) return;

    const std::string rect = "\n<rect width=\"100%\" height=\"100%\" fill=\"white\" id=\"plotpp-bg-white\"/>\n";
    s.insert(gt + 1, rect);
}

} // namespace detail

// Matplotlib-like API (small subset)

inline void figure_size(int width, int height) {
    auto& st = detail::state();
    st.width = width;
    st.height = height;
}

inline void title(std::string t) { detail::state().title = std::move(t); }
inline void xlabel(std::string t) { detail::state().xlabel = std::move(t); }
inline void ylabel(std::string t) { detail::state().ylabel = std::move(t); }

inline void grid(bool on = true) { detail::state().grid = on; }
inline void legend(bool on = true) { detail::state().legend = on; }

// Figure management (matplotlib-like)
// - figure() creates and activates a new figure (fresh state)
// - figure(id) activates figure id (creates it if missing)
inline int figure() {
    const int id = detail::next_figure_id()++;
    detail::figures()[id] = detail::FigureState{};
    detail::current_figure_id() = id;
    return id;
}

inline int figure(int id) {
    if (id < 0) {
        throw std::invalid_argument("plotpp: figure id must be >= 0");
    }
    detail::current_figure_id() = id;
    (void)detail::state();
    if (detail::next_figure_id() <= id) {
        detail::next_figure_id() = id + 1;
    }
    return id;
}

inline int gcf() { return detail::current_figure_id(); }

inline void cla() { detail::state().clear_series(); }

inline void plot(const std::vector<double>& y, const std::string& style = "") {
    auto& st = detail::state();
    detail::Series s;
    s.x = detail::make_default_x(y.size());
    s.y = y;
    s.style = style;
    st.series.push_back(std::move(s));
}

inline void plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& style = "") {
    detail::ensure_xy_sizes_match(x, y);
    auto& st = detail::state();
    detail::Series s;
    s.x = x;
    s.y = y;
    s.style = style;
    st.series.push_back(std::move(s));
}

inline void named_plot(const std::string& label,
                       const std::vector<double>& x,
                       const std::vector<double>& y,
                       const std::string& style = "") {
    detail::ensure_xy_sizes_match(x, y);
    auto& st = detail::state();
    detail::Series s;
    s.label = label;
    s.x = x;
    s.y = y;
    s.style = style;
    st.series.push_back(std::move(s));
}

inline void bar(const std::vector<double>& y, double width = 0.8, const std::string& style = "") {
    auto& st = detail::state();
    st.any_boxes = true;
    st.boxwidth = width;

    detail::Series s;
    s.x = detail::make_default_x(y.size());
    s.y = y;
    s.style = style.empty() ? "with boxes" : style;
    st.series.push_back(std::move(s));
}

inline void bar(const std::vector<double>& x,
                const std::vector<double>& y,
                double width = 0.8,
                const std::string& style = "") {
    detail::ensure_xy_sizes_match(x, y);
    auto& st = detail::state();
    st.any_boxes = true;
    st.boxwidth = width;

    detail::Series s;
    s.x = x;
    s.y = y;
    s.style = style.empty() ? "with boxes" : style;
    st.series.push_back(std::move(s));
}

inline void named_bar(const std::string& label,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      double width = 0.8,
                      const std::string& style = "") {
    detail::ensure_xy_sizes_match(x, y);
    auto& st = detail::state();
    st.any_boxes = true;
    st.boxwidth = width;

    detail::Series s;
    s.label = label;
    s.x = x;
    s.y = y;
    s.style = style.empty() ? "with boxes" : style;
    st.series.push_back(std::move(s));
}

inline int save(const std::string& output_path,
                const std::string& script_path = "plotpp_script.gp",
                const std::string& data_path = "plotpp_data.tsv") {
    auto& st = detail::state();
    if (st.series.empty()) {
        throw std::runtime_error("plotpp: nothing to plot (no series added)");
    }

    // If caller uses the defaults, write temp files and clean them up.
    detail::temp_paths tmp;
    if (detail::is_default_path(script_path, "plotpp_script.gp") && detail::is_default_path(data_path, "plotpp_data.tsv")) {
        const auto base = detail::plotpp_temp_dir();
        const auto tok = detail::unique_token();
        tmp.data = base / ("plotpp_" + tok + ".tsv");
        tmp.script = base / ("plotpp_" + tok + ".gp");
        // Don't touch the user-specified output path.
        detail::write_data_tsv(st, tmp.data);
        detail::write_gnuplot_script(st, tmp.script, tmp.data, std::filesystem::path(output_path));
        const int rc = detail::run_gnuplot_file(tmp.script);
        return rc;
    }

    // Otherwise, respect explicit paths and do not delete them.
    tmp.cleanup = false;
    detail::write_data_tsv(st, std::filesystem::path(data_path));
    detail::write_gnuplot_script(st,
                                std::filesystem::path(script_path),
                                std::filesystem::path(data_path),
                                std::filesystem::path(output_path));
    return detail::run_gnuplot_file(std::filesystem::path(script_path));
}

inline int show(const std::string& script_path = "plotpp_show.gp",
                const std::string& data_path = "plotpp_data.tsv") {
    auto& st = detail::state();
    if (st.series.empty()) {
        throw std::runtime_error("plotpp: nothing to plot (no series added)");
    }

#if PLOTPP_HAS_XCPP_DISPLAY
    // In notebooks: render to a temporary PNG and display inline.
    detail::temp_paths tmp;
    const bool use_defaults = detail::is_default_path(script_path, "plotpp_show.gp") && detail::is_default_path(data_path, "plotpp_data.tsv");
    const auto base = detail::plotpp_temp_dir();
    const auto tok = detail::unique_token();
    tmp.data = use_defaults ? (base / ("plotpp_" + tok + ".tsv")) : std::filesystem::path(data_path);
    tmp.script = use_defaults ? (base / ("plotpp_" + tok + ".gp")) : std::filesystem::path(script_path);
    tmp.out = base / ("plotpp_" + tok + ".png");
    tmp.cleanup = true;

    // If caller provided explicit script/data paths, keep them.
    if (!use_defaults) {
        tmp.data.clear();
        tmp.script.clear();
    }

    detail::write_data_tsv(st, use_defaults ? tmp.data : std::filesystem::path(data_path));
    detail::write_gnuplot_script(st,
                                use_defaults ? tmp.script : std::filesystem::path(script_path),
                                use_defaults ? tmp.data : std::filesystem::path(data_path),
                                tmp.out);
    const int rc = detail::run_gnuplot_file(use_defaults ? tmp.script : std::filesystem::path(script_path));
    if (rc != 0) return rc;

    xcpp::display(plotpp::inline_png{tmp.out.string()});
    return rc;
#else
    // Non-notebook fallback: respect provided paths.
    detail::write_data_tsv(st, std::filesystem::path(data_path));
    detail::write_gnuplot_script(st,
                                std::filesystem::path(script_path),
                                std::filesystem::path(data_path),
                                std::nullopt);
    return detail::run_gnuplot_file(std::filesystem::path(script_path));
#endif
}

} // namespace plotpp

#if PLOTPP_HAS_XCPP_DISPLAY
namespace xcpp {
// IMPORTANT: specialize the existing function template (xcpp::display uses a non-dependent
// using-declaration, so overloads declared later would be ignored).
template <>
inline nlohmann::json mime_bundle_repr<plotpp::inline_png>(const plotpp::inline_png& img) {
    (void)plotpp::detail::wait_for_file_nonempty(std::filesystem::path(img.path));
    const std::string bytes = plotpp::detail::read_file_binary(img.path);
    const std::string b64 = plotpp::detail::base64_encode(
        reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());

    auto bundle = nlohmann::json::object();
    bundle["image/png"] = b64;
    bundle["text/html"] = std::string("<img src=\"data:image/png;base64,") + b64 + "\"/>";
    bundle["text/plain"] = std::string("<plotpp image: ") + img.path + ">";
    return bundle;
}

template <>
inline nlohmann::json mime_bundle_repr<plotpp::inline_svg>(const plotpp::inline_svg& img) {
    (void)plotpp::detail::wait_for_file_nonempty(std::filesystem::path(img.path));
    std::string svg = plotpp::detail::read_file_binary(img.path);

    // Make SVG readable on dark notebook themes.
    plotpp::detail::inject_white_background_svg(svg);

    auto bundle = nlohmann::json::object();
    bundle["image/svg+xml"] = svg;
    // Wrap in an explicit white background container (helps some frontends).
    bundle["text/html"] = std::string("<div style=\"background:#fff; display:inline-block; padding:8px;\">\n") + svg + "\n</div>";
    bundle["text/plain"] = std::string("<plotpp svg: ") + img.path + ">";
    return bundle;
}

template <>
inline nlohmann::json mime_bundle_repr<plotpp::inline_svg_text>(const plotpp::inline_svg_text& v) {
    std::string svg = v.svg;
    plotpp::detail::inject_white_background_svg(svg);

    auto bundle = nlohmann::json::object();
    bundle["image/svg+xml"] = svg;
    bundle["text/html"] = std::string("<div style=\"background:#fff; display:inline-block; padding:8px;\">\n") + svg + "\n</div>";
    bundle["text/plain"] = "<plotpp svg (in-memory)>";
    return bundle;
}

template <>
inline nlohmann::json mime_bundle_repr<plotpp::inline_png_bytes>(const plotpp::inline_png_bytes& v) {
    const std::string b64 = plotpp::detail::base64_encode(
        reinterpret_cast<const unsigned char*>(v.bytes.data()), v.bytes.size());

    auto bundle = nlohmann::json::object();
    bundle["image/png"] = b64;
    bundle["text/html"] = std::string("<img style=\"background:#fff;\" src=\"data:image/png;base64,") + b64 + "\"/>";
    bundle["text/plain"] = "<plotpp png (in-memory)>";
    return bundle;
}
} // namespace xcpp
#endif

// Convenience: display SVG without you managing filenames.
// NOTE: Matplot++ only supports saving to a filename, so this creates a temp SVG file internally.
#if PLOTPP_HAS_XCPP_DISPLAY
namespace plotpp {
template <class SaveFn>
inline void display_svg_temp(SaveFn&& save_fn) {
    const auto path = detail::temp_path_with_ext(".svg");
    save_fn(path.string());
    xcpp::display(plotpp::inline_svg{path.string()});
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
} // namespace plotpp
#endif
