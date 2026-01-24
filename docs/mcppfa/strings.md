# strings.hpp

Python-like split utilities.

## Include
- mcppfa/core/strings.hpp
- Legacy: mcppfa/strings.hpp

## What it does
- `split(s)` splits on whitespace like Python `str.split()`.
- `split(s, sep)` splits on a specific separator and preserves empty fields.

## Key APIs
- `mcppfa::split(std::string_view s, std::ptrdiff_t maxsplit = -1)`
- `mcppfa::split(std::string_view s, std::string_view sep, std::ptrdiff_t maxsplit = -1)`

## Usage
```cpp
#include "mcppfa/core/strings.hpp"

auto parts1 = mcppfa::split("a  b  c");
auto parts2 = mcppfa::split("a,,b", ",");
```

## Notes
- `maxsplit == 0` returns the original string (trimmed for whitespace splitting).
- Empty separator throws `std::invalid_argument`.
