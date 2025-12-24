# MakeCPPGreatAgain

C++ notebooks playground using **xcpp17** (xeus-cling / cling JIT), with:

- **DataFrame** (C++17-friendly) for DataFrame-style analysis
- **PostgreSQL** access via **libpqxx** with a small helper layer
- Inline plotting via **plotpp**, and optional **Matplot++** (shared library)

## Quick start

- Read the setup guide: [docs/SETUP_XCPP17.md](docs/SETUP_XCPP17.md)
- Open and run the notebooks:
  - [notebooks/cpp_ga.ipynb](notebooks/cpp_ga.ipynb) (DataFrame + PostgreSQL + plotting)
  - [notebooks/pg_dataframe_workflow.ipynb](notebooks/pg_dataframe_workflow.ipynb) (end-to-end DB→DataFrame workflow + validation)
  - [notebooks/AOC_CPP_TEST/problems.ipynb](notebooks/AOC_CPP_TEST/problems.ipynb) (mcppfa helper examples)

## Repo layout

- `notebooks/` — xcpp17 notebooks
- `include/` — header-only helpers
  - `include/mcppfa/` — PostgreSQL helpers + printing + small utilities
  - `include/plotpp/` — inline plot display helpers
- `vendor/` — vendored third-party libraries used by the notebooks
  - `vendor/DataFrame-2.0.0/` — C++17-compatible DataFrame release
  - `vendor/matplotplusplus/` — Matplot++ source (optional)
- `docs/` — documentation
- `tools/` — local tooling/environment artifacts (optional)

## PostgreSQL → DataFrame helpers (mcppfa)

The notebook uses small helpers under `include/mcppfa/`:

- `mcppfa::exec(conn_str, sql)` / `mcppfa::exec0(conn_str, sql)`
  - Run any SQL (DDL/DML).
- `mcppfa::select_to_dataframe<IndexT>(conn_str, sql)`
  - Runs a `SELECT` and loads results into `hmdf::StdDataFrame<IndexT>` with basic type autodetect (via PostgreSQL OIDs):
    - `bool`, `int2`, `int4`, `int8` → `long long` (bool stored as `0/1`)
    - `float4`, `float8`, `numeric` → `double`
    - everything else → `std::string`
  - NULL handling:
    - `long long`: `0`
    - `double`: `NaN`
    - `std::string`: `""`
- `mcppfa::print_columns(df)`
  - One-liner “schema” print: column name + size + readable type (deduped).
- `mcppfa::print_df(df)`
  - Compact Python-like table printing (optimized for cling).

Example:

```cpp
const std::string conn_str = "postgresql://user:pass@localhost:5432/db";

mcppfa::exec(conn_str, "CREATE TABLE IF NOT EXISTS test (id SERIAL PRIMARY KEY, name TEXT, age INT);");
mcppfa::exec(conn_str, "INSERT INTO test (name, age) VALUES ('Alice', 30);");

auto df = mcppfa::select_to_dataframe<unsigned long>(
    conn_str,
    "SELECT id, name, age FROM test ORDER BY id");

mcppfa::print_columns(df);
std::cout << df << '\n';
```

## Notes

- **cling/xcpp17 header caching**: after editing headers under `include/` or `vendor/`, you may need to restart the notebook kernel and re-run from the include cell.
- **gnuplot**: Matplot++ uses gnuplot at runtime.

### Troubleshooting

- **`xeus::xinterpreter::display_data(...) unresolved while linking`** (often shows `nlohmann::json_abi_v...` in the symbol): your kernel’s `xeus` binary and the `nlohmann_json` headers it is compiling against are out of sync.
  - Fix by aligning `nlohmann_json` to the ABI that `libxeus.so` was built with, in the *same conda env as the kernel*.

```bash
# inside the env that provides your xcpp17 kernel
nm -D $CONDA_PREFIX/lib/libxeus.so | c++filt | grep 'xinterpreter::display_data' | head -n 1

# example: if it prints json_abi_v3_11_2, align nlohmann_json accordingly
conda install -c conda-forge -y nlohmann_json=3.11.2
```

  - Then restart the Jupyter server and restart the notebook kernel.

- **Matplot++ build on Ubuntu/WSL**: building the vendored Matplot++ shared library may require FFTW headers.

```bash
sudo apt-get update
sudo apt-get install -y gnuplot libfftw3-dev
```

## Setup

See [docs/SETUP_XCPP17.md](docs/SETUP_XCPP17.md).
