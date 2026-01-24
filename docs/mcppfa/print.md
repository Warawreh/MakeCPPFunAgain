# print.hpp

Printing helpers for vectors and DataFrame-like tables.

## Include
- mcppfa/core/print.hpp
- Legacy: mcppfa/print.hpp

## What it does
- Adds a global `operator<<` for `std::vector<T>` for quick inspection.
- Pretty-prints a DataFrame as a compact table.
- Provides `mcppfa::table(df)` view for streaming.

## Key APIs
- `operator<<(std::ostream&, const std::vector<T>&)`
- `mcppfa::print_df(df, n_rows, max_width, os)`
- `mcppfa::table(df, n_rows, max_width)`

## Usage
```cpp
#include "mcppfa/core/print.hpp"

std::vector<int> v{1,2,3};
std::cout << v << "\n";

// DataFrame printing
mcppfa::print_df(df, 5);
std::cout << mcppfa::table(df, 10) << "\n";
```

## Notes
- DataFrame printing assumes string columns by default for speed in notebooks.
- Long cells can be truncated via `max_width`.
