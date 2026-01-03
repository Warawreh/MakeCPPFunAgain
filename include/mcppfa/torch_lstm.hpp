#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace mcppfa {

struct LSTMConfig {
    int64_t input_size{};
    int64_t hidden_size{};

    int64_t num_layers{1};
    bool batch_first{true};
    bool bidirectional{false};

    // Only applies when num_layers > 1 in LibTorch.
    double dropout{0.0};

    void validate() const {
        if (input_size <= 0) throw std::runtime_error("LSTMConfig: input_size must be > 0");
        if (hidden_size <= 0) throw std::runtime_error("LSTMConfig: hidden_size must be > 0");
        if (num_layers <= 0) throw std::runtime_error("LSTMConfig: num_layers must be > 0");
        if (!(dropout >= 0.0 && dropout < 1.0)) throw std::runtime_error("LSTMConfig: dropout must be in [0, 1)");
    }
};

// A thin, reusable wrapper around torch::nn::LSTM that:
// - takes a simple config struct
// - exposes output feature size
// - returns the sequence output tensor by default
//
// Input:  [B, T, input_size]  if batch_first=true
// Output: [B, T, hidden_size * dir]
struct LSTMBlockImpl : torch::nn::Module {
    torch::nn::LSTM lstm{nullptr};
    LSTMConfig cfg;

    explicit LSTMBlockImpl(LSTMConfig config) : cfg(std::move(config)) {
        cfg.validate();

        auto opt = torch::nn::LSTMOptions(cfg.input_size, cfg.hidden_size)
                       .num_layers(cfg.num_layers)
                       .batch_first(cfg.batch_first)
                       .bidirectional(cfg.bidirectional);

        // LibTorch only uses dropout when num_layers > 1, but setting it is harmless.
        opt = opt.dropout(cfg.dropout);

        lstm = register_module("lstm", torch::nn::LSTM(opt));
    }

    int64_t directions() const { return cfg.bidirectional ? 2 : 1; }
    int64_t output_size() const { return cfg.hidden_size * directions(); }

    // Returns the full sequence output.
    torch::Tensor forward(const torch::Tensor& x) {
        return std::get<0>(lstm->forward(x));
    }

    // Returns (sequence_output, (h_n, c_n)) for callers that need state.
    std::pair<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>> forward_with_state(const torch::Tensor& x) {
        auto out = lstm->forward(x);
        auto seq = std::get<0>(out);
        auto state = std::get<1>(out);
        return {seq, {std::get<0>(state), std::get<1>(state)}};
    }
};

TORCH_MODULE(LSTMBlock);

} // namespace mcppfa
