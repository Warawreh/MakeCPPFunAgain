#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <typeindex>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace mcppfa {

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
    if (name == "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >" ||
        name == "std::basic_string<char, std::char_traits<char>, std::allocator<char> >") {
        return "std::string";
    }
    return name;
}

inline std::string pretty_type_name(std::type_index ti) { return type_name(ti); }

} // namespace mcppfa
