# torch_bert_train.hpp

Masked language modeling training utilities for BERT-style models.

## Include
- mcppfa/torch/torch_bert_train.hpp
- Legacy: mcppfa/torch_bert_train.hpp

## What it does
- Creates MLM batches with [MASK] tokens.
- Provides a simple training loop with warmup and optional eval.

## Key APIs
- `mcppfa::torchlm::BERTTrainConfig`
- `mcppfa::torchlm::BERTDataLoader`
- `mcppfa::torchlm::train_bert(...)`
- `save_bert_checkpoint(...)` / `load_bert_checkpoint(...)`

## Usage
```cpp
#include "mcppfa/torch/torch_bert_train.hpp"

mcppfa::torchlm::BERTDataLoader loader(ids, vocab_size);

auto loss = mcppfa::torchlm::train_bert(model, loader, cfg, device);
```

## Notes
- Assumes BERT-style token IDs with CLS/SEP/MASK.
- `BERTModel` is expected to expose `forward(input_ids, attention_mask)`.
