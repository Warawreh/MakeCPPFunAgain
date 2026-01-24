# indexing.hpp

Python-like negative indexing helpers.

## Include
- mcppfa/core/indexing.hpp
- Legacy: mcppfa/indexing.hpp

## What it does
- Normalizes negative indices.
- Provides `at()` for containers and `substr()` for strings.

## Key APIs
- `mcppfa::normalize_index(len, index, allow_end=false)`
- `mcppfa::at(container, index)`
- `mcppfa::substr(std::string_view, pos, count)`

## Usage
```cpp
#include "mcppfa/core/indexing.hpp"

std::vector<int> v{10,20,30};
int last = mcppfa::at(v, -1); // 30

std::string s = "hello";
auto tail = mcppfa::substr(s, -2); // "lo"
```
