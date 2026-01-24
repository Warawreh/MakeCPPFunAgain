# mcppfa headers

This folder documents the headers under include/mcppfa and their recommended include paths.

## Recommended include paths

### Core utilities
- mcppfa/core/columns.hpp
- mcppfa/core/csv.hpp
- mcppfa/core/indexing.hpp
- mcppfa/core/print.hpp
- mcppfa/core/strings.hpp
- mcppfa/core/type_name.hpp

### NLP/tokenization
- mcppfa/nlp/word_vocab.hpp
- mcppfa/nlp/sentencepiece_lite.hpp
- mcppfa/nlp/tokenizer_decoder.hpp

### Torch helpers/models
- mcppfa/torch/torch_lstm.hpp
- mcppfa/torch/torch_char_lm.hpp
- mcppfa/torch/torch_bert.hpp
- mcppfa/torch/torch_bert_train.hpp
- mcppfa/torch/torch_distilbert.hpp
- mcppfa/torch/model_loader.hpp
- mcppfa/torch/model_summary.hpp
- mcppfa/torch/safetensors.hpp

### HuggingFace helpers
- mcppfa/hf/huggingface.hpp
- mcppfa/hf/hf_dataset.hpp
- mcppfa/hf/hf_trainer.hpp
- mcppfa/hf/bert_huggingface.hpp

### Data/DB
- mcppfa/db/psql_dataframe.hpp

### Notebook setup
- mcppfa/notebook/notebook_setup.hpp

## Legacy include paths
The original flat include paths (for example mcppfa/strings.hpp) remain available for compatibility.

## Per-header docs
Each header has its own documentation file in this folder:
- columns.md
- csv.md
- print.md
- strings.md
- indexing.md
- type_name.md
- psql_dataframe.md
- model_loader.md
- model_summary.md
- notebook_setup.md
- word_vocab.md
- torch_lstm.md
- torch_char_lm.md
- torch_bert.md
- torch_bert_train.md
- torch_distilbert.md
- huggingface.md
- hf_dataset.md
- hf_trainer.md
- bert_huggingface.md
- sentencepiece_lite.md
- tokenizer_decoder.md
- safetensors.md
