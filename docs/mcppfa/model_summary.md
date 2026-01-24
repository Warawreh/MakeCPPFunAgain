# model_summary.hpp

Quick parameter statistics for LibTorch modules.

## Include
- mcppfa/torch/model_summary.hpp
- Legacy: mcppfa/model_summary.hpp

## What it does
- Counts total and trainable parameters.
- Prints grouped parameter totals by prefix.

## Key APIs
- `mcppfa::model_summary::count_total_params(module)`
- `mcppfa::model_summary::count_trainable_params(module)`
- `mcppfa::model_summary::print_model_summary(module, options)`

## Usage
```cpp
#include "mcppfa/torch/model_summary.hpp"

mcppfa::model_summary::print_model_summary(model);
```
