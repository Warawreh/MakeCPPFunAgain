# MakeCPPFunAgain

## Why this repo exists

I’m building this to make **C++ more appealing for new people learning programming** and easier to teach:

- **Fast feedback loop**: use C++ in notebooks (JIT via xeus-cling/cling) so you can experiment like you would in Python.
- **Beginner-friendly workflow**: keep examples small, runnable, and focused on concepts.
- **Practical motivation**: show where C++ shines (e.g., speed, control, and portability) while keeping the ergonomics closer to a REPL/notebook experience.

The goal is to have a place where you can learn and prototype in a “Python-like” way, then keep the benefits of C++ when you scale up.

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

## Examples

### 1) Minimal notebook include

In a C++ notebook cell:

```cpp
#include "include/mcppfa/notebook_setup.hpp"
```

### 2) Opt out of optional integrations (per cell)

By default, the notebook setup attempts to load Postgres + Matplot++ support. If you don’t want that in a particular cell:

```cpp
#define MCPPFA_NOTEBOOK_ENABLE_PG 0
#define MCPPFA_NOTEBOOK_ENABLE_MATPLOT 0
#include "include/mcppfa/notebook_setup.hpp"
```

### 3) PostgreSQL → DataFrame (end-to-end)

Set a connection string (optional) and run the Postgres notebook demo:

```bash
# Example
export PGURI="postgresql://user:pass@localhost:5432/mydb"
```

Then open and run:
- [notebooks/cpp_ga.ipynb](notebooks/cpp_ga.ipynb)
- [notebooks/pg_dataframe_workflow.ipynb](notebooks/pg_dataframe_workflow.ipynb)

### 4) Plotting

Run the PlotPP section in [notebooks/cpp_ga.ipynb](notebooks/cpp_ga.ipynb). If plots fail to render, install `gnuplot` (see Troubleshooting below).

## What we have so far

- xcpp17 notebooks for an interactive C++ workflow
- Vendored **DataFrame-2.0.0** wired into notebooks
- A small `mcppfa` helper layer for **PostgreSQL + libpqxx** and “query → DataFrame” loading
- Inline plotting via **plotpp** and optional **Matplot++**

## Future work

- **Hugging Face integration**: load datasets/models, fine-tune/train, and export artifacts for fast inference (e.g., exporting to a format usable by `llama.cpp`/`llama.c` workflows).

## Repo layout

- `notebooks/` — xcpp17 notebooks
- `include/` — header-only helpers
  - `include/mcppfa/` — PostgreSQL helpers + printing + small utilities
  - `include/plotpp/` — inline plot display helpers
- `vendor/` — vendored third-party libraries used by the notebooks
  - `vendor/DataFrame-2.0.0/` — C++17-compatible DataFrame release
  - `vendor/matplotplusplus/` — Matplot++ source (optional)
- `docs/` — documentation

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

## Vibe code alert

This repo is **vibe-coded**.

Like… *a lot* of it.

- ~99% of the code was generated with GPT-5.2
- It’s here to be useful (and fun), but expect a few rough edges
- I’ll keep iterating, fixing, and improving things as I go


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
