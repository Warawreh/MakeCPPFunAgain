# bert_huggingface.hpp

Load BERT/DistilBERT artifacts from HuggingFace and run generation-like loops.

## Include
- mcppfa/hf/bert_huggingface.hpp
- Legacy: mcppfa/bert_huggingface.hpp

## What it does
- Downloads config and weights (safetensors preferred) from a repo.
- Builds a matching BERT or DistilBERT model.
- Provides a tokenizer wrapper for tokenizer.json.
- Implements a simple iterative `predict()` loop for next-token sampling.

## Key APIs
- `mcppfa::hf::BERTModelWrapper`
  - `load_from_hf(repo_id, token, repo_type, revision)`
  - `reset(tokenizer, text)` / `predict(tokenizer, ...)`
  - `save(path)`
- `mcppfa::hf::BERTTokenizerWrapper`
  - `load_from_hf(repo_id, token, repo_type, revision)`
  - `save(path)`

## Usage
```cpp
#include "mcppfa/hf/bert_huggingface.hpp"

mcppfa::hf::BERTTokenizerWrapper tok;
tok.load_from_hf("distilbert-base-uncased");

mcppfa::hf::BERTModelWrapper model;
model.load_from_hf("distilbert-base-uncased");
```

## Notes
- PyTorch `.bin` files may not load due to pickle format; safetensors is preferred.
- Generation uses the tokenizer to filter special tokens and sample logits.
