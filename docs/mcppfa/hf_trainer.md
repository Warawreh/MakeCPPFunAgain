# hf_trainer.hpp

Small training loop for T5-like models in C++.

## Include
- mcppfa/hf/hf_trainer.hpp
- Legacy: mcppfa/hf_trainer.hpp

## What it does
- Tokenizes dataset tables once and caches tensors.
- Trains a model on a one-token target objective.
- Evaluates on validation/test splits and prints examples.

## Key APIs
- `mcppfa::TrainingArguments`
- `mcppfa::Trainer<ModelT>`
- `Trainer::set_splits(train, valid, test)`
- `Trainer::tokenize_splits_once()`
- `Trainer::train()` / `evaluate_valid()` / `evaluate_test()`

## Usage
```cpp
#include "mcppfa/hf/hf_trainer.hpp"

mcppfa::TrainingArguments args;
mcppfa::Trainer model_trainer(model, tokenizer, args);
model_trainer.set_splits(train, valid, test);
model_trainer.tokenize_splits_once();
model_trainer.train();
```

## Notes
- Uses `mcppfa::spm_lite::SentencePieceLite` for tokenization.
- Optimized for notebook-scale experiments.
