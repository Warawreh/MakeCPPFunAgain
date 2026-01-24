# hf_dataset.hpp

Fetch and preview HuggingFace datasets without a Python dependency.

## Include
- mcppfa/hf/hf_dataset.hpp
- Legacy: mcppfa/hf_dataset.hpp

## What it does
- Lists dataset files via the Hub API.
- Downloads small CSV/JSONL files to preview rows.
- Falls back to datasets-server for Parquet-only datasets.

## Key APIs
- `mcppfa::hf_dataset::Table`
- `list_files(dataset_repo, revision, token)`
- `load_head(dataset_repo, local_dir, token, revision, n_rows)`
- `load_head_via_datasets_server(dataset_repo, token, config, split, n_rows, offset)`
- `list_splits(dataset_repo, token)`
- `load_rows_split(dataset_repo, split, offset, length, token, config)`
- `print_columns(table)` / `print_head(table, n)`

## Usage
```cpp
#include "mcppfa/hf/hf_dataset.hpp"

auto t = mcppfa::hf_dataset::load_head("ag_news", ".hf/_datasets");
mcppfa::hf_dataset::print_head(t, 5);
```

## Notes
- Requires `curl` for HTTP calls.
- Only CSV and JSONL are parsed locally; Parquet uses datasets-server.
