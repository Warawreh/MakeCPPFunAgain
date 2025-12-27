#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <torch/torch.h>

namespace mcppfa::torchlm {

/**
 * DistilBERT Architecture (matching HuggingFace transformers)
 * 
 * Key differences from BERT:
 * - 6 layers instead of 12
 * - No segment embeddings
 * - Pre-norm architecture (layer norm before attention/FFN)
 * - Simpler structure overall
 * 
 * Architecture:
 * 1. Embeddings (token + position, no segment)
 * 2. Transformer layers (6x)
 *    - Self-attention with pre-norm
 *    - Feed-forward with pre-norm
 * 3. Optional head (for downstream tasks)
 */

// DistilBERT Embeddings (token + position only, no segment)
struct DistilBERTEmbeddingsImpl : torch::nn::Module {
    torch::nn::Embedding word_embeddings{nullptr};  // Named to match HF
    torch::nn::Embedding position_embeddings{nullptr};
    torch::nn::LayerNorm LayerNorm{nullptr};  // Named to match HF
    double dropout_p{};
    
    int64_t vocab_size{};
    int64_t max_position_embeddings{};
    int64_t hidden_dim{};

    DistilBERTEmbeddingsImpl(
        int64_t vocab_size,
        int64_t hidden_dim,
        int64_t max_position_embeddings = 512,
        double dropout = 0.1)
        : vocab_size(vocab_size),
          max_position_embeddings(max_position_embeddings),
          hidden_dim(hidden_dim),
          dropout_p(dropout) {
        
        word_embeddings = register_module(
            "word_embeddings",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, hidden_dim)));
        position_embeddings = register_module(
            "position_embeddings",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(max_position_embeddings, hidden_dim)));
        LayerNorm = register_module(
            "LayerNorm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    torch::Tensor forward(const torch::Tensor& input_ids) {
        const int64_t B = input_ids.size(0);
        const int64_t T = input_ids.size(1);
        
        // Token embeddings
        auto word_emb = word_embeddings->forward(input_ids);  // [B, T, H]
        
        // Position embeddings
        auto positions = torch::arange(T, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
        positions = positions.unsqueeze(0).expand({B, T});  // [B, T]
        auto pos_emb = position_embeddings->forward(positions);  // [B, T, H]
        
        // Sum embeddings
        auto embeddings = word_emb + pos_emb;  // [B, T, H]
        
        // Layer norm (pre-norm style, but applied after sum)
        embeddings = LayerNorm->forward(embeddings);
        
        // Dropout
        if (dropout_p > 0.0 && is_training()) {
            embeddings = torch::dropout(embeddings, dropout_p, is_training());
        }
        
        return embeddings;
    }
};
TORCH_MODULE(DistilBERTEmbeddings);

// Multi-head self-attention (matching DistilBERT structure)
struct DistilBERTSelfAttentionImpl : torch::nn::Module {
    torch::nn::Linear q_lin{nullptr};  // Named to match HF
    torch::nn::Linear k_lin{nullptr};
    torch::nn::Linear v_lin{nullptr};
    torch::nn::Linear out_lin{nullptr};
    
    int64_t num_heads{};
    int64_t head_dim{};
    int64_t hidden_dim{};
    double dropout_p{};

    DistilBERTSelfAttentionImpl(int64_t hidden_dim, int64_t num_heads, double dropout = 0.1)
        : num_heads(num_heads), hidden_dim(hidden_dim), dropout_p(dropout) {
        
        if (hidden_dim % num_heads != 0) {
            throw std::runtime_error("DistilBERTSelfAttention: hidden_dim must be divisible by num_heads");
        }
        
        head_dim = hidden_dim / num_heads;
        
        q_lin = register_module("q_lin", torch::nn::Linear(hidden_dim, hidden_dim));
        k_lin = register_module("k_lin", torch::nn::Linear(hidden_dim, hidden_dim));
        v_lin = register_module("v_lin", torch::nn::Linear(hidden_dim, hidden_dim));
        out_lin = register_module("out_lin", torch::nn::Linear(hidden_dim, hidden_dim));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& mask = torch::Tensor()) {
        const int64_t B = x.size(0);
        const int64_t T = x.size(1);
        const int64_t H = hidden_dim;
        
        // Project to Q, K, V
        auto q = q_lin->forward(x);  // [B, T, H]
        auto k = k_lin->forward(x);  // [B, T, H]
        auto v = v_lin->forward(x);  // [B, T, H]
        
        // Reshape for multi-head: [B, T, H] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);  // [B, num_heads, T, head_dim]
        k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);  // [B, num_heads, T, head_dim]
        v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);  // [B, num_heads, T, head_dim]
        
        // Scaled dot-product attention
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));  // [B, num_heads, T, T]
        
        // Apply mask if provided (1 for valid, 0 for masked)
        if (mask.defined()) {
            // mask: [B, T] -> [B, 1, 1, T] for broadcasting
            auto mask_expanded = mask.unsqueeze(1).unsqueeze(2).to(torch::kFloat32);  // [B, 1, 1, T]
            scores = scores.masked_fill(mask_expanded == 0, -1e9);
        }
        
        auto attn_weights = torch::softmax(scores, -1);  // [B, num_heads, T, T]
        if (dropout_p > 0.0 && is_training()) {
            attn_weights = torch::dropout(attn_weights, dropout_p, is_training());
        }
        
        auto attn_output = torch::matmul(attn_weights, v);  // [B, num_heads, T, head_dim]
        
        // Reshape back: [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, H});
        
        // Output projection
        auto output = out_lin->forward(attn_output);
        
        if (dropout_p > 0.0 && is_training()) {
            output = torch::dropout(output, dropout_p, is_training());
        }
        
        return output;
    }
};
TORCH_MODULE(DistilBERTSelfAttention);

// Feed-forward network (matching DistilBERT)
struct DistilBERTFFNImpl : torch::nn::Module {
    torch::nn::Linear lin1{nullptr};  // Named to match HF
    torch::nn::Linear lin2{nullptr};
    double dropout_p{};

    DistilBERTFFNImpl(int64_t hidden_dim, int64_t dim_ff, double dropout = 0.1)
        : dropout_p(dropout) {
        lin1 = register_module("lin1", torch::nn::Linear(hidden_dim, dim_ff));
        lin2 = register_module("lin2", torch::nn::Linear(dim_ff, hidden_dim));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = lin1->forward(x);
        out = torch::gelu(out);  // GELU activation
        if (dropout_p > 0.0 && is_training()) {
            out = torch::dropout(out, dropout_p, is_training());
        }
        out = lin2->forward(out);
        if (dropout_p > 0.0 && is_training()) {
            out = torch::dropout(out, dropout_p, is_training());
        }
        return out;
    }
};
TORCH_MODULE(DistilBERTFFN);

// Transformer encoder block (matching DistilBERT pre-norm architecture)
struct DistilBERTTransformerBlockImpl : torch::nn::Module {
    DistilBERTSelfAttention attention{nullptr};
    DistilBERTFFN ffn{nullptr};
    torch::nn::LayerNorm sa_layer_norm{nullptr};  // Named to match HF
    torch::nn::LayerNorm output_layer_norm{nullptr};  // Named to match HF
    double dropout_p{};

    DistilBERTTransformerBlockImpl(
        int64_t hidden_dim,
        int64_t num_heads,
        int64_t dim_ff,
        double dropout = 0.1)
        : dropout_p(dropout) {
        attention = register_module("attention", DistilBERTSelfAttention(hidden_dim, num_heads, dropout));
        ffn = register_module("ffn", DistilBERTFFN(hidden_dim, dim_ff, dropout));
        sa_layer_norm = register_module(
            "sa_layer_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        output_layer_norm = register_module(
            "output_layer_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& mask = torch::Tensor()) {
        // Pre-norm architecture: LayerNorm -> Attention -> Residual
        auto sa_norm = sa_layer_norm->forward(x);
        auto attn_out = attention->forward(sa_norm, mask);
        auto sa_output = x + attn_out;  // Residual connection
        
        // Pre-norm architecture: LayerNorm -> FFN -> Residual
        auto ff_norm = output_layer_norm->forward(sa_output);
        auto ff_out = ffn->forward(ff_norm);
        auto output = sa_output + ff_out;  // Residual connection
        
        return output;
    }
};
TORCH_MODULE(DistilBERTTransformerBlock);

// DistilBERT Model (matching HuggingFace structure exactly)
struct DistilBERTModelImpl : torch::nn::Module {
    DistilBERTEmbeddings embeddings{nullptr};
    torch::nn::ModuleList transformer{nullptr};  // Named to match HF
    
    int64_t vocab_size{};
    int64_t hidden_dim{};
    int64_t num_heads{};
    int64_t n_layers{};  // Named to match HF config
    int64_t dim_ff{};  // Named to match HF config
    int64_t max_position_embeddings{};
    double dropout_p{};

    DistilBERTModelImpl(
        int64_t vocab_size,
        int64_t hidden_dim = 768,
        int64_t num_heads = 12,
        int64_t n_layers = 6,  // DistilBERT has 6 layers
        int64_t dim_ff = 3072,
        int64_t max_position_embeddings = 512,
        double dropout = 0.1)
        : vocab_size(vocab_size),
          hidden_dim(hidden_dim),
          num_heads(num_heads),
          n_layers(n_layers),
          dim_ff(dim_ff),
          max_position_embeddings(max_position_embeddings),
          dropout_p(dropout) {
        
        if (hidden_dim % num_heads != 0) {
            throw std::runtime_error("DistilBERTModel: hidden_dim must be divisible by num_heads");
        }
        
        embeddings = register_module(
            "embeddings",
            DistilBERTEmbeddings(vocab_size, hidden_dim, max_position_embeddings, dropout));
        
        transformer = register_module("transformer", torch::nn::ModuleList());
        for (int64_t i = 0; i < n_layers; ++i) {
            transformer->push_back(
                DistilBERTTransformerBlock(hidden_dim, num_heads, dim_ff, dropout));
        }
    }

    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask = torch::Tensor()) {
        
        // Get embeddings
        auto x = embeddings->forward(input_ids);  // [B, T, H]
        
        // Pass through transformer layers
        for (int64_t i = 0; i < n_layers; ++i) {
            auto layer = (*transformer)[i]->as<DistilBERTTransformerBlock>();
            x = layer->forward(x, attention_mask);
        }
        
        return x;  // [B, T, H]
    }
};
TORCH_MODULE(DistilBERTModel);

// DistilBERT for Masked Language Modeling (with MLM head)
struct DistilBERTForMaskedLMImpl : torch::nn::Module {
    DistilBERTModel distilbert{nullptr};  // Named to match HF
    torch::nn::Linear vocab_projector{nullptr};  // Named to match HF
    torch::nn::LayerNorm vocab_layer_norm{nullptr};  // Named to match HF
    
    int64_t vocab_size{};

    DistilBERTForMaskedLMImpl(
        int64_t vocab_size,
        int64_t hidden_dim = 768,
        int64_t num_heads = 12,
        int64_t n_layers = 6,
        int64_t dim_ff = 3072,
        int64_t max_position_embeddings = 512,
        double dropout = 0.1)
        : vocab_size(vocab_size) {
        
        distilbert = register_module(
            "distilbert",
            DistilBERTModel(vocab_size, hidden_dim, num_heads, n_layers, dim_ff, max_position_embeddings, dropout));
        
        vocab_projector = register_module(
            "vocab_projector",
            torch::nn::Linear(hidden_dim, vocab_size));
        vocab_layer_norm = register_module(
            "vocab_layer_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask = torch::Tensor()) {
        
        // Get hidden states from DistilBERT
        auto hidden_states = distilbert->forward(input_ids, attention_mask);  // [B, T, H]
        
        // Apply layer norm
        hidden_states = vocab_layer_norm->forward(hidden_states);
        
        // Project to vocabulary
        auto logits = vocab_projector->forward(hidden_states);  // [B, T, vocab_size]
        
        return logits;
    }
};
TORCH_MODULE(DistilBERTForMaskedLM);

} // namespace mcppfa::torchlm

