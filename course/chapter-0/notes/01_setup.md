# 01 — Install xcpp and verify kernel

## Goals
- Install xcpp/xeus-cling.
- Verify the xcpp kernel is available in Jupyter/VS Code.

## Steps (mamba + xcpp)

### 1) Create a conda environment with mamba
Use a dedicated conda env (e.g., `cpp-notebooks`) for Jupyter/xcpp.

**WSL / Ubuntu / macOS**
```bash
# create and activate the environment
mamba create -n cpp-notebooks -c conda-forge python=3.11 xeus-cling notebook jupyterlab
mamba activate cpp-notebooks

# (optional) install extra tools for this repo
mamba install -c conda-forge cmake ninja
```

**Notes**
- If you don’t have mamba installed, install Miniforge/Mambaforge first.
- After activation, `python -V` should show the env’s Python.

### 2) Start a notebook server
Pick the command that matches your OS. All commands assume the `cpp-notebooks` env is activated.

**WSL (recommended binding for Windows browser access)**
```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```
- Then open the URL shown in the terminal from Windows (http://localhost:8888/?token=...).

**Ubuntu (local machine)**
```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

**macOS (local machine)**
```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

### 3) Select the xcpp kernel
When opening a notebook, pick the **xcpp** kernel in the kernel selector.

---

## Install mcppfa system-wide (optional)
If you want `#include <mcppfa/...>` to work like standard headers, install to `/usr/local`.

```bash
cmake -S . -B build
cmake --build build
sudo cmake --install build --prefix /usr/local
```

If xcpp still can’t find the headers, add the include path when launching Jupyter:
```bash
export CPLUS_INCLUDE_PATH=/usr/local/include
jupyter lab
```
Or add a notebook cell:
```cpp
#pragma cling add_include_path("/usr/local/include")
```

## Exercises
- Create the `xcpp` environment using mamba.
- Start Jupyter Lab and open the Chapter 0 notebook.
- Run the first cell and confirm the output.
