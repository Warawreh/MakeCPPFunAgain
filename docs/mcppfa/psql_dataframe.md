# psql_dataframe.hpp

PostgreSQL helpers that load results into DataFrame-2.0.0.

## Include
- mcppfa/db/psql_dataframe.hpp
- Legacy: mcppfa/psql_dataframe.hpp

## What it does
- Runs SQL via libpqxx.
- Loads result sets into `hmdf::StdDataFrame` with string or basic type inference.
- Exposes convenience wrappers in the `mcppfa` namespace.

## Key APIs
- `mcppfa::psql::exec(conn_str, sql)`
- `mcppfa::psql::exec0(conn_str, sql)`
- `mcppfa::psql::select_to_dataframe_strings(conn_str, sql, columns)`
- `mcppfa::psql::select_to_dataframe(conn_str, sql, columns)`
- Convenience wrappers: `mcppfa::exec`, `mcppfa::exec0`, `mcppfa::select_to_dataframe*`

## Usage
```cpp
#include "mcppfa/db/psql_dataframe.hpp"

auto df = mcppfa::select_to_dataframe(
    "host=localhost dbname=test user=me", "select * from table");
```

## Notes
- Requires libpqxx and libpq headers. Define `MCPPFA_ENABLE_PG` to enable.
- Basic type inference maps Postgres types to `long long`, `double`, or `std::string`.
