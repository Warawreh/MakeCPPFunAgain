# torch_char_lm.hpp

Character-level language model utilities.

## Include
- mcppfa/torch/torch_char_lm.hpp
- Legacy: mcppfa/torch_char_lm.hpp

## What it does
- Builds a character tokenizer from raw text.
- Provides a small LSTM-based char LM model.
- Includes a simple training loop and text generation.

## Key APIs
- `mcppfa::torchlm::CharTokenizer`
- `mcppfa::torchlm::CharLM`
- `mcppfa::torchlm::train_char_lm(...)`
- `mcppfa::torchlm::generate(...)`

## Usage
```cpp
#include "mcppfa/torch/torch_char_lm.hpp"

auto tok = mcppfa::torchlm::CharTokenizer::from_text(text);
auto ids = tok.encode(text);

mcppfa::torchlm::CharLM model(tok.vocab_size(), 128, 256, 2);
```

## Notes
- The training API expects a `TextStream` built from token IDs.
- Use `default_device()` to pick CUDA if available.
