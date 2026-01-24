# sentencepiece_lite.hpp

Minimal SentencePiece model loader and encoder/decoder.

## Include
- mcppfa/nlp/sentencepiece_lite.hpp
- Legacy: mcppfa/sentencepiece_lite.hpp

## What it does
- Parses SentencePiece `ModelProto` without protobuf deps.
- Decodes piece IDs to text (replaces ▁ with space).
- Provides a lightweight, best-effort encoder.

## Key APIs
- `mcppfa::spm_lite::SentencePieceLite::load_from_file(path)`
- `mcppfa::spm_lite::SentencePieceLite::encode(text)`
- `mcppfa::spm_lite::SentencePieceLite::decode(ids, skip_special)`
- `mcppfa::spm_lite::SentencePieceLite::add_piece_with_id(id, piece)`

## Usage
```cpp
#include "mcppfa/nlp/sentencepiece_lite.hpp"

mcppfa::spm_lite::SentencePieceLite sp;
sp.load_from_file("tokenizer.model");

auto ids = sp.encode("hello world");
std::string text = sp.decode(ids);
```

## Notes
- Encoder is heuristic and not a full unigram/Viterbi implementation.
- Useful for notebooks and lightweight inference.
