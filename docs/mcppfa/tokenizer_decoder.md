# tokenizer_decoder.hpp

Lightweight tokenizer.json reader and decoder.

## Include
- mcppfa/nlp/tokenizer_decoder.hpp
- Legacy: mcppfa/tokenizer_decoder.hpp

## What it does
- Parses vocab from HuggingFace tokenizer.json (simple JSON parser).
- Decodes token IDs to text and filters special tokens.
- Provides basic sampling utilities for logits.

## Key APIs
- `mcppfa::tokenizer::TokenizerDecoder::load_from_file(path)`
- `decode(ids, skip_special_tokens)`
- `decode_tensor(tensor, max_tokens)`
- `is_special_token(id)`
- `mask_special_tokens(logits)`
- `sample_next_token(logits, temperature, top_k, greedy)`
- `encode(text)` (simple word-based encoder)

## Usage
```cpp
#include "mcppfa/nlp/tokenizer_decoder.hpp"

mcppfa::tokenizer::TokenizerDecoder tok;
tok.load_from_file("tokenizer.json");

auto ids = tok.encode("hello world");
std::string text = tok.decode(ids);
```

## Notes
- The JSON parsing is intentionally minimal; complex tokenizers may not fully parse.
- Special tokens are auto-detected by name patterns (e.g., [CLS], <pad>).
