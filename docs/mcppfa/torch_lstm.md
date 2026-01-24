# torch_lstm.hpp

Simple LSTM wrapper module.

## Include
- mcppfa/torch/torch_lstm.hpp
- Legacy: mcppfa/torch_lstm.hpp

## What it does
- Wraps `torch::nn::LSTM` with a config struct.
- Provides output size helpers and forward methods.

## Key APIs
- `mcppfa::LSTMConfig`
- `mcppfa::LSTMBlock` (module)

## Usage
```cpp
#include "mcppfa/torch/torch_lstm.hpp"

mcppfa::LSTMConfig cfg{.input_size=128, .hidden_size=256};
mcppfa::LSTMBlock lstm(cfg);

auto y = lstm->forward(x);
```

## Notes
- Input shape is [B, T, input_size] when `batch_first` is true.
