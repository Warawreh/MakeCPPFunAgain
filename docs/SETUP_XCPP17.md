# xcpp17 setup (Windows / macOS / Linux)

This repo is built around the **xcpp17** Jupyter kernel (xeus-cling / cling), which executes C++17 in notebooks.

## Prerequisites

- Jupyter (Lab or Notebook)
- A C++17-capable toolchain (recommended)
- `conda` / `mamba` (recommended install path)

> Note: xeus-cling support can vary by OS and environment. If you hit installation issues on Windows, the most reliable path is usually **WSL2**.

---

## Linux (recommended, including WSL2)

### 1) Install xcpp17 via conda

```bash
# Miniconda/Miniforge/Mambaforge all work
conda create -n cpp-notebooks -c conda-forge -y \
  jupyterlab \
  xeus-cling \
  xtensor \
  xtensor-blas

conda activate cpp-notebooks
```

### 1.1) Verify the xcpp kernels are installed

Installing `xeus-cling` is what makes the **xcpp** kernels appear (e.g. `xcpp11`, `xcpp14`, `xcpp17`).

```bash
jupyter kernelspec list
```

You should see entries similar to:

- `xcpp11` / `xcpp14` / `xcpp17`

If you don’t, double-check that you installed `xeus-cling` into the same environment you’re using.

### 2) PostgreSQL client libs (optional, for the DB notebook)

```bash
sudo apt-get update
sudo apt-get install -y libpq-dev libpqxx-dev
```

#### Note (xeus-cling): you may need to explicitly load shared libraries

Unlike a normal compiled C++ binary, xeus-cling JITs each cell into a module that must be able to *link* against `libpqxx`/`libpq`.
If you see errors like:

- `IncrementalExecutor::executeFunction: symbol ...demangle_type_name... unresolved`
- or cling failing to resolve pqxx symbols during execution

Load the libraries in a notebook cell *before* including Postgres-dependent headers.
Prefer loading by library name (portable) and make sure the libraries are discoverable via your environment (e.g. start Jupyter from an activated conda env):

```cpp
// Prefer conda env libs when CONDA_PREFIX is set (portable across users)
#pragma cling add_library_path("$CONDA_PREFIX/lib")

// Common system locations (optional but helps on Linux/WSL)
#pragma cling add_library_path("/usr/lib/x86_64-linux-gnu")
#pragma cling add_library_path("/usr/local/lib")

// Load by SONAME
#pragma cling load("libpq.so")
#pragma cling load("libpqxx.so")
```

Then (in the same cell or a later cell) include and use the helpers.

#### Optional: condense pragmas into a single include

This repo provides a convenience header you can include in notebooks to avoid repeating many `#pragma cling ...` lines:

```cpp
#define MCPPFA_NOTEBOOK_ENABLE_PG 1      // optional (Postgres)
#define MCPPFA_NOTEBOOK_ENABLE_MATPLOT 1 // optional (Matplot++)
#include "include/mcppfa/notebook_setup.hpp"
```

It sets up common include paths and (optionally) loads `libpq`/`libpqxx` for Postgres examples and the Matplot++ shared library.

Tip: you can set `PGURI` to avoid hard-coding credentials in notebooks:

```bash
export PGURI=postgresql://user:pass@localhost:5432/dbname
```

### 3) Launch Jupyter

```bash
jupyter lab
```

If you prefer the classic Notebook server (useful for remote/WSL workflows):

```bash
jupyter-notebook --no-browser --port=8888
```

Note: the port flag is a single value (e.g. `--port=8888`), not split across arguments.

#### Avoiding “system Jupyter” vs “conda Jupyter” confusion

On some systems (especially WSL/Ubuntu), `apt` can install `/usr/bin/jupyter-notebook` which may *not* match your conda environment.
Make sure the Jupyter executable you launch comes from your env:

```bash
which jupyter
which jupyter-notebook
```

Both should point under `$CONDA_PREFIX/bin/...`.
If `jupyter-notebook` still resolves to `/usr/bin/jupyter-notebook`, install it into the env:

```bash
conda install -n cpp-notebooks -c conda-forge -y notebook
```

Or launch without relying on shell activation:

```bash
conda run -n cpp-notebooks jupyter-notebook --no-browser --port=8888
```

Open [notebooks/cpp_ga.ipynb](../notebooks/cpp_ga.ipynb) and run cells top-to-bottom.

---

## macOS

### 1) Install xcpp17 via conda

```bash
conda create -n cpp-notebooks -c conda-forge -y \
  jupyterlab \
  xeus-cling \
  xtensor \
  xtensor-blas

conda activate cpp-notebooks
jupyter lab
```

### 2) PostgreSQL client libs (optional)

If you need libpq/libpqxx for the PostgreSQL notebook, install via Homebrew and ensure headers/libs are discoverable.

---

## Windows

### Option A (recommended): WSL2 + Linux instructions

1) Install WSL2 and Ubuntu
2) Follow the **Linux (recommended)** instructions above
3) Open the folder in VS Code (Remote - WSL) and run Jupyter

### Option B (native Windows): conda + xeus-cling

Some setups work natively, but if anything fails, switch to WSL2.

```powershell
conda create -n cpp-notebooks -c conda-forge -y jupyterlab xeus-cling
conda activate cpp-notebooks
jupyter lab
```

---

## Matplot++ (optional)

If you want Matplot++ inside xcpp17, build it as a shared library (Linux/macOS recommended):

On Ubuntu/WSL you may need a couple system dependencies for the vendored Matplot++ build (notably FFTW headers):

```bash
sudo apt-get update
sudo apt-get install -y gnuplot libfftw3-dev
```

```bash
cd vendor/matplotplusplus
rm -rf build-xcpp17
mkdir -p build-xcpp17
cd build-xcpp17
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMATPLOTPP_BUILD_EXAMPLES=OFF \
  -DMATPLOTPP_BUILD_TESTS=OFF
cmake --build . --parallel
```

The notebook expects a shared library under:

- `vendor/matplotplusplus/build-xcpp17/source/matplot/`

---

## Troubleshooting

- If a notebook keeps “not seeing” changes in headers under `include/` or `vendor/`, restart the kernel (cling caches aggressively).
- If Matplot++ fails at runtime, ensure `gnuplot` is installed and in your PATH.
- If you use `mamba` and see `critical libmamba Shell not initialized` on `mamba activate`, initialize once per shell:

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate cpp-notebooks
```

To make it permanent for future shells, run `mamba shell init --shell bash` and restart your terminal.

- If you see an error like `xeus::xinterpreter::display_data(...) unresolved while linking` (often showing `nlohmann::json_abi_v...` in the symbol name), your **kernel’s** `xeus` binary and the `nlohmann_json` headers it is compiling against are out of sync.
  - Fix by ensuring `xeus-cling` and `nlohmann_json` come from the same conda environment (and ideally the same channel, `conda-forge`):

```bash
conda install -n cpp-notebooks -c conda-forge -y xeus-cling nlohmann_json
```

  - Then restart the Jupyter server and restart the notebook kernel.
