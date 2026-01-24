# safetensors.hpp

Read safetensors files into LibTorch tensors.

## Include
- mcppfa/torch/safetensors.hpp
- Legacy: mcppfa/safetensors.hpp

## What it does
- Parses the safetensors header and loads tensors into memory.
- Maps dtype strings to `torch::Dtype`.
- Provides a helper to apply tensors to a model.

## Key APIs
- `mcppfa::safetensors::load_safetensors(path)`
- `mcppfa::safetensors::apply_safetensors_to_model(model, tensors)`

## Usage
```cpp
#include "mcppfa/torch/safetensors.hpp"

auto tensors = mcppfa::safetensors::load_safetensors("model.safetensors");
mcppfa::safetensors::apply_safetensors_to_model(model, tensors);
```

## Notes
- The loader reads the full file into memory.
- Debug output prints header snippets and tensor stats.
