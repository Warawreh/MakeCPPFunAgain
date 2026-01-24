# torch_bert.hpp

Minimal transformer building blocks and a toy T5-like model.

## Include
- mcppfa/torch/torch_bert.hpp
- Legacy: mcppfa/torch_bert.hpp

## What it does
- Implements a minimal Transformer encoder block.
- Provides a small, T5-like model for quick experiments.

## Key APIs
- `mcppfa::torchlm::TransformerEncoderBlock`
- `mcppfa::torchlm::T5Model`

## Usage
```cpp
#include "mcppfa/torch/torch_bert.hpp"

mcppfa::torchlm::T5Model model(
    vocab_size, d_model, num_heads, enc_layers, dec_layers, d_ff, max_len);
```

## Notes
- The decoder is self-attention only (no cross-attention).
- Intended for quick tests rather than production-quality modeling.
