# huggingface.hpp

Lightweight HuggingFace Hub helpers for notebooks and C++ demos.

## Include
- mcppfa/hf/huggingface.hpp
- Legacy: mcppfa/huggingface.hpp

## What it does
- Wraps `huggingface-cli` commands and git-based uploads.
- Provides HTTP download helpers for repo files.
- Adds a simple uploader API for model/tokenizer/config artifacts.

## Key APIs
- `mcppfa::hf::run(cmd)` / `run_verbose(cmd)`
- `mcppfa::hf::read_token_file(path)`
- `mcppfa::hf::login(...)` and `login_from_env(...)`
- `mcppfa::hf::download(repo_id, local_dir, ...)`
- `mcppfa::hf::upload_file(...)`
- `mcppfa::hf::repo_create(...)`
- `mcppfa::hf::download_file_http(...)`
- `mcppfa::hf::upload_files_git_verbose(...)`
- `mcppfa::hf::HubUploader` (with `.model`, `.tokenizer`, `.config`)
- `mcppfa::hf::Model` / `mcppfa::hf::Tokenizer` upload wrappers

## Usage
```cpp
#include "mcppfa/hf/huggingface.hpp"

mcppfa::hf::login_from_env();

mcppfa::hf::HubUploader uploader("user/my-model");
uploader.model.upload("model.pt");
const auto log = uploader.push();
```

## Notes
- Uses system commands (git, git-lfs, curl, huggingface-cli).
- Intended for notebook workflows and quick experiments.
