# model_loader.hpp

Load PyTorch state dicts into LibTorch modules.

## Include
- mcppfa/torch/model_loader.hpp
- Legacy: mcppfa/model_loader.hpp

## What it does
- Matches parameter and buffer names against a state dict.
- Optionally enforces strict loading.
- Supports name mapping for mismatched conventions.

## Key APIs
- `mcppfa::model_loader::load_state_dict(model, state_dict, strict)`
- `mcppfa::model_loader::load_state_dict_with_mapping(model, state_dict, mapping, strict)`

## Usage
```cpp
#include "mcppfa/torch/model_loader.hpp"

std::map<std::string, torch::Tensor> state;
// fill state...

mcppfa::model_loader::load_state_dict(model, state, false);
```
