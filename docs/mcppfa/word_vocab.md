# word_vocab.hpp

Simple word-level vocabulary and tokenizer.

## Include
- mcppfa/nlp/word_vocab.hpp
- Legacy: mcppfa/word_vocab.hpp

## What it does
- Tokenizes ASCII-ish words (letters, digits, underscore).
- Builds a frequency-based vocabulary.
- Encodes text into padded ID sequences.

## Key APIs
- `mcppfa::text::word_tokenize_lower_ascii(text)`
- `mcppfa::text::WordVocab::build_from_texts(texts)`
- `mcppfa::text::WordVocab::encode_padded(text, max_len)`

## Usage
```cpp
#include "mcppfa/nlp/word_vocab.hpp"

mcppfa::text::WordVocab vocab(10000);
vocab.build_from_texts({"hello world", "hello cpp"});

auto ids = vocab.encode_padded("hello cpp", 8);
```

## Notes
- `pad_id` is 0, `unk_id` is 1.
- Tokenization is ASCII-only and case-folds to lowercase.
