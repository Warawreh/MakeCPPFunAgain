# torch_distilbert.hpp

DistilBERT architecture implemented in LibTorch.

## Include
- mcppfa/torch/torch_distilbert.hpp
- Legacy: mcppfa/torch_distilbert.hpp

## What it does
- Implements DistilBERT embeddings, attention, FFN, and transformer blocks.
- Provides `DistilBERTModel` and `DistilBERTForMaskedLM`.

## Key APIs
- `mcppfa::torchlm::DistilBERTModel`
- `mcppfa::torchlm::DistilBERTForMaskedLM`

## Usage
```cpp
#include "mcppfa/torch/torch_distilbert.hpp"

mcppfa::torchlm::DistilBERTForMaskedLM model(vocab_size);

auto logits = model->forward(input_ids, attention_mask);
```

## Notes
- Uses a pre-norm transformer layout matching HuggingFace naming.
- `forward_iv()` is provided for IValue-based invocation.
