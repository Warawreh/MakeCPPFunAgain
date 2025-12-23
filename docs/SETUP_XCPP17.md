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

### 2) PostgreSQL client libs (optional, for the DB notebook)

```bash
sudo apt-get update
sudo apt-get install -y libpq-dev libpqxx-dev
```

### 3) Launch Jupyter

```bash
jupyter lab
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
