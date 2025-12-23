#pragma once

#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <typeindex>
#include <unordered_set>
#include <vector>

#include <cstdlib>
#include <memory>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

#include <pqxx/pqxx>

#include <DataFrame/DataFrame.h>

namespace mcppfa::psql {

namespace detail {
// PostgreSQL type OIDs: https://www.postgresql.org/docs/current/datatype-oid.html
// We keep this intentionally small and practical for notebooks.
enum class col_kind { int64_kind, double_kind, string_kind };

inline col_kind kind_from_oid(pqxx::oid oid) {
    switch (oid) {
        // bool
        case 16: return col_kind::int64_kind; // store as 0/1

        // integers
        case 20: // int8
        case 21: // int2
        case 23: // int4
            return col_kind::int64_kind;

        // floating / numeric
        case 700:  // float4
        case 701:  // float8
        case 1700: // numeric
            return col_kind::double_kind;

        default:
            return col_kind::string_kind;
    }
}
} // namespace detail

// Execute any SQL and return pqxx::result.
// For DDL/INSERT/UPDATE/DELETE you can ignore the returned result.
inline pqxx::result exec(std::string_view conn_str, std::string_view sql) {
    pqxx::connection conn{std::string(conn_str)};
    pqxx::work txn{conn};
    pqxx::result r = txn.exec(std::string(sql));
    txn.commit();
    return r;
}

// Backwards-compatible alias: exec0() is like exec() but ignores the result.
inline void exec0(std::string_view conn_str, std::string_view sql) {
    (void)exec(conn_str, sql);
}

// Run a SELECT and load the result into an hmdf::StdDataFrame.
// For simplicity and flexibility, all columns are loaded as std::string.
//
// - Index is 0..N-1 (row number)
// - If `requested_columns` is empty, all result columns are loaded
// - NULL values are loaded as empty strings
template <typename IndexT = unsigned long>
inline hmdf::StdDataFrame<IndexT> select_to_dataframe_strings(
    std::string_view conn_str,
    std::string_view sql,
    const std::vector<std::string> &requested_columns = {})
{
    pqxx::connection conn{std::string(conn_str)};
    pqxx::read_transaction txn{conn};
    pqxx::result res = txn.exec(std::string(sql));

    std::vector<std::string> col_names;
    if (!requested_columns.empty()) {
        col_names = requested_columns;
    } else {
        col_names.reserve(res.columns());
        for (pqxx::row::size_type c = 0; c < res.columns(); ++c) {
            col_names.emplace_back(res.column_name(c));
        }
    }

    std::vector<IndexT> idx;
    idx.reserve(res.size());
    for (pqxx::result::size_type i = 0; i < res.size(); ++i) {
        idx.push_back(static_cast<IndexT>(i));
    }

    hmdf::StdDataFrame<IndexT> df;
    df.load_index(std::move(idx));

    for (const std::string &name : col_names) {
        std::vector<std::string> values;
        values.reserve(res.size());

        for (pqxx::result::size_type i = 0; i < res.size(); ++i) {
            const pqxx::field f = res[i][name.c_str()];
            if (f.is_null())
                values.emplace_back("");
            else
                values.emplace_back(f.c_str());
        }

        df.template load_column<std::string>(name.c_str(), std::move(values),
                                             hmdf::nan_policy::dont_pad_with_nans);
    }

    return df;
}

// Run a SELECT and load the result into an hmdf::StdDataFrame with basic type autodetect.
//
// Column types chosen per PostgreSQL OID:
// - bool/int2/int4/int8  -> long long
// - float4/float8/numeric -> double
// - everything else -> std::string
//
// NULL handling:
// - long long: 0 (or 0/1 for bool)
// - double: NaN
// - string: ""
template <typename IndexT = unsigned long>
inline hmdf::StdDataFrame<IndexT> select_to_dataframe(
    std::string_view conn_str,
    std::string_view sql,
    const std::vector<std::string> &requested_columns = {})
{
    pqxx::connection conn{std::string(conn_str)};
    pqxx::read_transaction txn{conn};
    pqxx::result res = txn.exec(std::string(sql));

    std::vector<std::string> col_names;
    if (!requested_columns.empty()) {
        col_names = requested_columns;
    } else {
        col_names.reserve(res.columns());
        for (pqxx::row::size_type c = 0; c < res.columns(); ++c) {
            col_names.emplace_back(res.column_name(c));
        }
    }

    std::vector<IndexT> idx;
    idx.reserve(res.size());
    for (pqxx::result::size_type i = 0; i < res.size(); ++i) {
        idx.push_back(static_cast<IndexT>(i));
    }

    hmdf::StdDataFrame<IndexT> df;
    df.load_index(std::move(idx));

    for (const std::string &name : col_names) {
        const pqxx::row::size_type c = res.column_number(name.c_str());
        const pqxx::oid oid = res.column_type(c);
        const auto kind = detail::kind_from_oid(oid);

        if (kind == detail::col_kind::int64_kind) {
            std::vector<long long> values;
            values.reserve(res.size());
            for (pqxx::result::size_type i = 0; i < res.size(); ++i) {
                const pqxx::field f = res[i][c];
                if (f.is_null()) {
                    values.emplace_back(0);
                } else if (oid == 16) {
                    values.emplace_back(f.as<bool>() ? 1LL : 0LL);
                } else {
                    values.emplace_back(f.as<long long>());
                }
            }
            df.template load_column<long long>(name.c_str(), std::move(values),
                                               hmdf::nan_policy::dont_pad_with_nans);
        } else if (kind == detail::col_kind::double_kind) {
            std::vector<double> values;
            values.reserve(res.size());
            const double nan = std::numeric_limits<double>::quiet_NaN();
            for (pqxx::result::size_type i = 0; i < res.size(); ++i) {
                const pqxx::field f = res[i][c];
                if (f.is_null()) values.emplace_back(nan);
                else values.emplace_back(f.as<double>());
            }
            df.template load_column<double>(name.c_str(), std::move(values),
                                            hmdf::nan_policy::dont_pad_with_nans);
        } else {
            std::vector<std::string> values;
            values.reserve(res.size());
            for (pqxx::result::size_type i = 0; i < res.size(); ++i) {
                const pqxx::field f = res[i][c];
                if (f.is_null()) values.emplace_back("");
                else values.emplace_back(f.c_str());
            }
            df.template load_column<std::string>(name.c_str(), std::move(values),
                                                 hmdf::nan_policy::dont_pad_with_nans);
        }
    }

    return df;
}

} // namespace mcppfa::psql

// Convenience wrappers: allow calling as mcppfa::exec0(...) etc.
namespace mcppfa {
inline pqxx::result exec(std::string_view conn_str, std::string_view sql) {
    return mcppfa::psql::exec(conn_str, sql);
}

inline void exec0(std::string_view conn_str, std::string_view sql) {
    mcppfa::psql::exec0(conn_str, sql);
}

template <typename IndexT = unsigned long>
inline hmdf::StdDataFrame<IndexT> select_to_dataframe_strings(
    std::string_view conn_str,
    std::string_view sql,
    const std::vector<std::string> &requested_columns = {})
{
    return mcppfa::psql::select_to_dataframe_strings<IndexT>(conn_str, sql, requested_columns);
}

// Default: autodetect types (long long / double / std::string)
template <typename IndexT = unsigned long>
inline hmdf::StdDataFrame<IndexT> select_to_dataframe(
    std::string_view conn_str,
    std::string_view sql,
    const std::vector<std::string> &requested_columns = {})
{
    return mcppfa::psql::select_to_dataframe<IndexT>(conn_str, sql, requested_columns);
}

// DataFrame column introspection (DataFrame-2.0.0)
// Usage: mcppfa::columns_info<int,double,std::string>(df)
template <typename... Ts, typename DF>
inline auto columns_info(const DF &df) {
    return df.template get_columns_info<Ts...>();
}

template <typename DF>
inline auto columns_info_strings(const DF &df) {
    return df.template get_columns_info<std::string>();
}

// Common set for DB-loaded frames with autodetect.
template <typename DF>
inline auto columns_info_basic(const DF &df) {
    const auto infos = df.template get_columns_info<long long, double, std::string>();
    std::vector<std::tuple<std::string, std::size_t, std::type_index>> out;
    out.reserve(infos.size());
    std::unordered_set<std::string> seen;
    seen.reserve(infos.size());
    for (const auto &[name_any, size, type_idx] : infos) {
        const std::string name{name_any.c_str()};
        if (seen.insert(name).second) out.emplace_back(name, static_cast<std::size_t>(size), type_idx);
    }
    return out;
}

inline std::string full_type_name(std::type_index ti) {
    const char *raw = ti.name();
#if defined(__GNUG__)
    int status = 0;
    std::unique_ptr<char, void(*)(void*)> buf{
        abi::__cxa_demangle(raw, nullptr, nullptr, &status),
        std::free
    };
    if (status == 0 && buf) return std::string{buf.get()};
#endif
    return std::string{raw};
}

inline std::string type_name(std::type_index ti) {
    std::string name = full_type_name(ti);
    // Normalize the most common long spellings.
    if (name == "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >" ||
        name == "std::basic_string<char, std::char_traits<char>, std::allocator<char> >") {
        return "std::string";
    }
    return name;
}

// Backwards compatible name (older notebook cells)
inline std::string pretty_type_name(std::type_index ti) { return type_name(ti); }
} // namespace mcppfa
