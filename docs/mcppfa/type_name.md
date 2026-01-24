# type_name.hpp

Human-readable type name utilities.

## Include
- mcppfa/core/type_name.hpp
- Legacy: mcppfa/type_name.hpp

## What it does
- Demangles type names on GNU toolchains.
- Normalizes `std::string` types.

## Key APIs
- `mcppfa::full_type_name(std::type_index)`
- `mcppfa::type_name(std::type_index)`
- `mcppfa::pretty_type_name(std::type_index)`

## Usage
```cpp
#include "mcppfa/core/type_name.hpp"

std::type_index ti = typeid(std::string);
std::cout << mcppfa::type_name(ti) << "\n";
```
