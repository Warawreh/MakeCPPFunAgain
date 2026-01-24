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
- Use the **conda/mamba** environment (e.g., `cpp-notebooks`) for Jupyter/xcpp.
- Open and run the notebooks:
  - [notebooks/cpp_ga.ipynb](notebooks/cpp_ga.ipynb) (DataFrame + PostgreSQL + plotting)
  - [notebooks/pg_dataframe_workflow.ipynb](notebooks/pg_dataframe_workflow.ipynb) (end-to-end DB→DataFrame workflow + validation)
  - [notebooks/t5_pretrain.ipynb](notebooks/t5_pretrain.ipynb) (T5-style **one-token** fine-tuning in pure C++ with LibTorch)
  - [notebooks/AOC_CPP_TEST/problems.ipynb](notebooks/AOC_CPP_TEST/problems.ipynb) (mcppfa helper examples)
  - [docs/SETUP_HUGGINGFACE_XCPP17.ipynb](docs/SETUP_HUGGINGFACE_XCPP17.ipynb) (Complete BERT fine-tuning workflow: Load → Inference (Before) → Fine-tune → Inference (After) → Save → Upload)

## Install as a C++ package (CMake)

This repo provides a header-only CMake package target `mcppfa::mcppfa`.

### Install (system default include path)
This installs to `/usr/local/include` so you can `#include <mcppfa/...>` normally.
```bash
cmake -S . -B build
cmake --build build
sudo cmake --install build --prefix /usr/local
```

### Install (user-local)
```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ~/.local
```

### Use in another project (CMake)
```cmake
find_package(mcppfa CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE mcppfa::mcppfa)
```

Then in code:
```cpp
#include <mcppfa/core/strings.hpp>
```

### Notes for xcpp/xeus-cling notebooks
If xcpp does not find headers after install, add the include path when launching Jupyter:
```bash
export CPLUS_INCLUDE_PATH=/usr/local/include
jupyter lab
```
Or add this line in a notebook cell:
```cpp
#pragma cling add_include_path("/usr/local/include")
```

## Hugging Face + LibTorch: Fine-tuning BERT in Pure C++

This repo includes a **complete C++ workflow** for fine-tuning BERT/DistilBERT models:

### Complete Workflow

The [SETUP_HUGGINGFACE_XCPP17.ipynb](docs/SETUP_HUGGINGFACE_XCPP17.ipynb) notebook demonstrates:

1. **Setup**: Install tools (`curl`, `git`, `git-lfs`), authenticate with HuggingFace, configure LibTorch
2. **Load Model**: Download or load a pre-trained BERT/DistilBERT model from HuggingFace (reuses local files if available)
3. **Inference (Before)**: Test model generation before fine-tuning to establish a baseline
4. **Load Dataset**: Download and prepare training data (e.g., Tiny Shakespeare dataset)
5. **Fine-tune Model**: Train the model on your dataset using LibTorch (pure C++)
6. **Inference (After)**: Test model generation after fine-tuning to see improvements
7. **Save Model**: Save the fine-tuned checkpoint to disk
8. **Upload to HuggingFace**: Upload your fine-tuned model back to the Hub

### Key Features

- **Pure C++**: No Python dependencies for model training or inference
- **HuggingFace Integration**: Download models and upload checkpoints using `curl` and `git`
- **Before/After Comparison**: See how fine-tuning affects model behavior
- **Local File Caching**: Automatically reuses downloaded models to avoid re-downloading

### What's Included

- `include/mcppfa/huggingface.hpp`
  - `download_file_http(...)` using stable `resolve/` URLs
  - `upload_files_git_verbose(...)` which clones/pulls, copies files, `git add/commit/push`
  - `GitUploadOptions` with LFS support and progress streaming
- `include/mcppfa/bert_huggingface.hpp`
  - `BERTModelWrapper`: Load BERT/DistilBERT models from HuggingFace
  - `BERTTokenizerWrapper`: Load tokenizers from HuggingFace
  - Automatic model architecture detection (BERT vs DistilBERT)
- `include/mcppfa/tokenizer_decoder.hpp`
  - `TokenizerDecoder`: Encode/decode text using HuggingFace tokenizer.json
  - Supports vocabulary parsing and token ID mapping

## T5 one-token fine-tuning (pure C++ notebook)

The notebook [notebooks/t5_pretrain.ipynb](notebooks/t5_pretrain.ipynb) is a focused demo that fine-tunes a small T5-style model for a **one-token classification target**.

### Key points

- **One-token objective**: training predicts only 1 output token (decoder length is fixed to `1`).
- **Fast training**: tokenizes each split once into tensors, then trains/evaluates in mini-batches (no per-row re-tokenization).
- **Tokenizer/model vocab correctness**:
  - `spiece.model` can be small (e.g., 128 pieces), but Hugging Face tokenizers often add tokens via `tokenizer_config.json`.
  - The notebook treats the repo as a **tokenizer-assets repo only** and merges `added_tokens_decoder` into `SentencePieceLite` so token IDs match.
  - The model is initialized with the *effective* vocab size after merging.
- **Live progress output**: the notebook disables stdout/stderr buffering and the trainer flushes after key prints so you can see where it reached while running.

### Main headers involved

- `include/mcppfa/hf_trainer.hpp`
  - HF-style `TrainingArguments` + `Trainer<ModelT>` used by the notebook.
- `include/mcppfa/sentencepiece_lite.hpp`
  - `piece_for_id(...)` for vocab inspection.
  - `add_piece_with_id(...)` to merge HF added tokens at explicit IDs.

### Notebook knobs (what they mean)

- `g_max_len`: max encoder input length in tokens (truncate/pad the input sequence to this length).
- `g_batch_size`, `g_lr`, `g_epochs`: standard training controls.
- `g_pad_id`: derived from tokenizer assets; used for padding input and decoder inputs.

### Setup Requirements

#### 1. Token Handling

Create a `secrets.txt` file (first line is your HF token) and **do not commit it**:

```bash
echo "hf_..." > secrets.txt
```

The notebook prefers `secrets.txt` and falls back to `HF_TOKEN` environment variable if needed.

#### 2. Git LFS (required for large weights)

If you're uploading model files, install LFS:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
```

#### 3. LibTorch (required for training)

Download LibTorch and set the environment variable **before** starting the notebook kernel:

```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

# Set environment variable (update path as needed)
export LIBTORCH=$HOME/libtorch
```

#### 4. System Tools

Install required tools:

```bash
sudo apt-get update
sudo apt-get install -y curl git git-lfs
```

### Usage

1. Open [docs/SETUP_HUGGINGFACE_XCPP17.ipynb](docs/SETUP_HUGGINGFACE_XCPP17.ipynb)
2. Run cells sequentially following the workflow
3. Compare inference results before and after fine-tuning
4. Upload your fine-tuned model to share with others

### Important Notes

- **BERT Architecture**: BERT/DistilBERT are bidirectional encoders, not designed for autoregressive text generation
- **Generation Quality**: You may see `[UNK]` tokens in generated text - this is expected behavior
- **Better Generation**: For improved text generation, consider using a decoder model (GPT-style)
- **Local Files**: The notebook automatically detects and reuses locally downloaded models

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

- **More Hugging Face artifacts**: expand the C++ helpers to support more Hub layouts (e.g. extra tokenizer files, sharded weights, model cards) while staying notebook-friendly.
- **Easy-mode multithreading**: simple, beginner-friendly parallel helpers (e.g., `parallel_for`, a tiny thread-pool, and “do this on all cores” utilities) so you can get speedups without turning the notebook into a concurrency lecture.

## Repo layout

- `notebooks/` — xcpp17 notebooks
- `include/` — header-only helpers
  - `include/mcppfa/` — PostgreSQL helpers + printing + small utilities
    - `include/mcppfa/huggingface.hpp` — Hub download/upload helpers (curl + git + git-lfs)
    - `include/mcppfa/torch_char_lm.hpp` — minimal LibTorch training demo (pure C++)
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
