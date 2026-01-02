#pragma once

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace mcppfa::torchlm {

// Minimal Transformer encoder-style block.
// Notes:
// - Uses self-attention only (no cross-attention).
// - Expects inputs shaped [B, T, C]. Internally transposes to [T, B, C]
//   for torch::nn::MultiheadAttention compatibility.
struct TransformerEncoderBlockImpl : torch::nn::Module {
    int64_t d_model{}, num_heads{}, d_ff{};
    double dropout_p{};

    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::MultiheadAttention mha{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    TransformerEncoderBlockImpl(int64_t d_model_, int64_t num_heads_, int64_t d_ff_, double dropout_ = 0.1)
        : d_model(d_model_), num_heads(num_heads_), d_ff(d_ff_), dropout_p(dropout_) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{d_model})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{d_model})));

        mha = register_module(
            "mha",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(d_model, num_heads).dropout(dropout_p)));

        fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
        fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(const torch::Tensor& x_in) {
        // x: [B, T, C]
        auto x = x_in;

        // Pre-norm self-attention
        auto y = ln1->forward(x);
        auto y_t = y.transpose(0, 1); // [T, B, C]
        auto attn_tuple = mha->forward(y_t, y_t, y_t);
        auto attn_out_t = std::get<0>(attn_tuple); // [T, B, C]
        auto attn_out = attn_out_t.transpose(0, 1); // [B, T, C]
        x = x + dropout->forward(attn_out);

        // Pre-norm feed-forward
        auto z = ln2->forward(x);
        z = fc1->forward(z);
        z = torch::relu(z);
        z = dropout->forward(z);
        z = fc2->forward(z);
        x = x + dropout->forward(z);
        return x;
    }
};
TORCH_MODULE(TransformerEncoderBlock);

/**
 * Simple T5-like model (toy implementation suitable for quick tests).
 * - Embeddings + positional embeddings
 * - Encoder: stack of TransformerEncoderBlock
 * - Decoder: stack of TransformerEncoderBlock (no cross-attention for simplicity)
 * - LM head projecting back to vocab
 *
 * Constructor arguments match usage in notebook:
 *   T5Model(vocab_size, d_model, num_heads, enc_layers, dec_layers, d_ff, max_len, dropout)
 */
struct T5ModelImpl : torch::nn::Module {
    int64_t vocab_size{}, d_model{}, num_heads{}, enc_layers{}, dec_layers{}, d_ff{}, max_len{};
    double dropout_p{};

    torch::nn::Embedding token_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
    torch::nn::ModuleList encoder{nullptr};
    torch::nn::ModuleList decoder{nullptr};
    torch::nn::Linear lm_head{nullptr};

    T5ModelImpl(
        int64_t vocab_size_,
        int64_t d_model_,
        int64_t num_heads_,
        int64_t enc_layers_,
        int64_t dec_layers_,
        int64_t d_ff_,
        int64_t max_len_,
        double dropout_ = 0.1)
        : vocab_size(vocab_size_),
          d_model(d_model_),
          num_heads(num_heads_),
          enc_layers(enc_layers_),
          dec_layers(dec_layers_),
          d_ff(d_ff_),
          max_len(max_len_),
          dropout_p(dropout_) {

        token_emb = register_module("token_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, d_model)));
        pos_emb = register_module("pos_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(max_len, d_model)));

        encoder = register_module("encoder", torch::nn::ModuleList());
        for (int64_t i = 0; i < enc_layers; ++i) {
            encoder->push_back(TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_p));
        }

        decoder = register_module("decoder", torch::nn::ModuleList());
        for (int64_t i = 0; i < dec_layers; ++i) {
            decoder->push_back(TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_p));
        }

        lm_head = register_module("lm_head", torch::nn::Linear(d_model, vocab_size));
    }

    // Forward: input_ids [B, Tenc], decoder_input_ids [B, Tdec] -> logits [B, Tdec, V]
    torch::Tensor forward(const torch::Tensor& input_ids, const torch::Tensor& decoder_input_ids) {
        auto device = input_ids.device();
        int64_t B = input_ids.size(0);
        int64_t Tenc = input_ids.size(1);
        int64_t Tdec = decoder_input_ids.size(1);

        // Encoder embeddings + pos
        auto enc_pos = torch::arange(Tenc, torch::TensorOptions().dtype(torch::kInt64).device(device))
                           .unsqueeze(0).expand({B, Tenc});
        auto enc = token_emb->forward(input_ids) + pos_emb->forward(enc_pos);

        for (int64_t i = 0; i < enc_layers; ++i) {
            auto layer = encoder[i]->as<TransformerEncoderBlockImpl>();
            enc = layer->forward(enc);
        }

        // Decoder embeddings + pos (note: this toy decoder doesn't use cross-attention)
        auto dec_pos = torch::arange(Tdec, torch::TensorOptions().dtype(torch::kInt64).device(device))
                           .unsqueeze(0).expand({B, Tdec});
        auto dec = token_emb->forward(decoder_input_ids) + pos_emb->forward(dec_pos);

        for (int64_t i = 0; i < dec_layers; ++i) {
            auto layer = decoder[i]->as<TransformerEncoderBlockImpl>();
            dec = layer->forward(dec);
        }

        auto logits = lm_head->forward(dec); // [B, Tdec, V]
        return logits;
    }
};
TORCH_MODULE(T5Model);

} // namespace mcppfa::torchlm

