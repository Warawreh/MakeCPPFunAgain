# columns.hpp

Utilities for DataFrame column introspection and quick printing.

## Include
- mcppfa/core/columns.hpp
- Legacy: mcppfa/columns.hpp

## What it does
- Collects unique column metadata (name, size, type) from DataFrame-2.0.0 frames.
- Provides a one-liner to print column information.

## Key APIs
- `mcppfa::columns_info_basic_unique(df)`
- `mcppfa::print_columns(df, os)`

## Usage
```cpp
#include "mcppfa/core/columns.hpp"

// df is an hmdf::StdDataFrame or compatible type
mcppfa::print_columns(df);
```

## Notes
- Relies on `DF::get_columns_info<...>()` from DataFrame-2.0.0.
- Uses `mcppfa::type_name()` for human-readable type strings.
