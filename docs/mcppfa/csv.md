# csv.hpp

Minimal CSV line splitter (RFC4180-ish).

## Include
- mcppfa/core/csv.hpp
- Legacy: mcppfa/csv.hpp

## What it does
- Splits a single CSV line into fields.
- Handles quoted fields and escaped quotes (`""` -> `"`).
- Does not handle multi-line fields.

## Key APIs
- `mcppfa::csv::split_csv_line(line)`

## Usage
```cpp
#include "mcppfa/core/csv.hpp"

auto fields = mcppfa::csv::split_csv_line("a,\"b,c\",d");
```
