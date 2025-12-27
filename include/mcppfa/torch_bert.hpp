#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace mcppfa::torchlm {

// Multi-head self-attention module
struct MultiHeadAttentionImpl : torch::nn::Module {
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};
    
    int64_t num_heads{};
    int64_t head_dim{};
    int64_t hidden_dim{};
    double dropout_p{};

    MultiHeadAttentionImpl(int64_t hidden_dim, int64_t num_heads, double dropout = 0.1)
        : num_heads(num_heads), hidden_dim(hidden_dim), dropout_p(dropout) {
        
        if (hidden_dim % num_heads != 0) {
            throw std::runtime_error("MultiHeadAttention: hidden_dim must be divisible by num_heads");
        }
        
        head_dim = hidden_dim / num_heads;
        
        q_proj = register_module("q_proj", torch::nn::Linear(hidden_dim, hidden_dim));
        k_proj = register_module("k_proj", torch::nn::Linear(hidden_dim, hidden_dim));
        v_proj = register_module("v_proj", torch::nn::Linear(hidden_dim, hidden_dim));
        out_proj = register_module("out_proj", torch::nn::Linear(hidden_dim, hidden_dim));
    }

    // x: [B, T, H], mask: [B, T] (optional, 1 for valid, 0 for masked)
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& mask = torch::Tensor()) {
        const int64_t B = x.size(0);
        const int64_t T = x.size(1);
        const int64_t H = hidden_dim;
        
        // Project to Q, K, V
        auto q = q_proj->forward(x);  // [B, T, H]
        auto k = k_proj->forward(x);  // [B, T, H]
        auto v = v_proj->forward(x);  // [B, T, H]
        
        // Reshape for multi-head: [B, T, H] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);  // [B, num_heads, T, head_dim]
        k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);  // [B, num_heads, T, head_dim]
        v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);  // [B, num_heads, T, head_dim]
        
        // Scaled dot-product attention
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));  // [B, num_heads, T, T]
        
        // Apply mask if provided
        if (mask.defined()) {
            // mask: [B, T] -> [B, 1, 1, T] for broadcasting
            auto mask_expanded = mask.unsqueeze(1).unsqueeze(2);  // [B, 1, 1, T]
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
        auto output = out_proj->forward(attn_output);
        
        if (dropout_p > 0.0 && is_training()) {
            output = torch::dropout(output, dropout_p, is_training());
        }
        
        return output;
    }
};
TORCH_MODULE(MultiHeadAttention);

// Feed-forward network
struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    double dropout_p{};

    FeedForwardImpl(int64_t hidden_dim, int64_t ff_dim, double dropout = 0.1)
        : dropout_p(dropout) {
        fc1 = register_module("fc1", torch::nn::Linear(hidden_dim, ff_dim));
        fc2 = register_module("fc2", torch::nn::Linear(ff_dim, hidden_dim));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = fc1->forward(x);
        out = torch::gelu(out);
        if (dropout_p > 0.0 && is_training()) {
            out = torch::dropout(out, dropout_p, is_training());
        }
        out = fc2->forward(out);
        if (dropout_p > 0.0 && is_training()) {
            out = torch::dropout(out, dropout_p, is_training());
        }
        return out;
    }
};
TORCH_MODULE(FeedForward);

// Transformer encoder block
struct TransformerEncoderBlockImpl : torch::nn::Module {
    MultiHeadAttention attn{nullptr};
    FeedForward ff{nullptr};
    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    double dropout_p{};

    TransformerEncoderBlockImpl(int64_t hidden_dim, int64_t num_heads, int64_t ff_dim, double dropout = 0.1)
        : dropout_p(dropout) {
        attn = register_module("attn", MultiHeadAttention(hidden_dim, num_heads, dropout));
        ff = register_module("ff", FeedForward(hidden_dim, ff_dim, dropout));
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& mask = torch::Tensor()) {
        // Self-attention with residual connection and layer norm
        auto attn_out = attn->forward(x, mask);
        auto out1 = ln1->forward(x + attn_out);
        
        // Feed-forward with residual connection and layer norm
        auto ff_out = ff->forward(out1);
        auto out2 = ln2->forward(out1 + ff_out);
        
        return out2;
    }
};
TORCH_MODULE(TransformerEncoderBlock);

// BERT embeddings (token + position + segment)
struct BERTEmbeddingsImpl : torch::nn::Module {
    torch::nn::Embedding token_emb{nullptr};
    torch::nn::Embedding position_emb{nullptr};
    torch::nn::Embedding segment_emb{nullptr};
    torch::nn::LayerNorm ln{nullptr};
    double dropout_p{};
    
    int64_t vocab_size{};
    int64_t max_seq_len{};
    int64_t hidden_dim{};

    BERTEmbeddingsImpl(int64_t vocab_size, int64_t hidden_dim, int64_t max_seq_len = 512, double dropout = 0.1)
        : vocab_size(vocab_size), max_seq_len(max_seq_len), hidden_dim(hidden_dim), dropout_p(dropout) {
        
        token_emb = register_module("token_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, hidden_dim)));
        position_emb = register_module("position_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(max_seq_len, hidden_dim)));
        segment_emb = register_module("segment_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(2, hidden_dim)));  // 0 or 1
        ln = register_module("ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    // input_ids: [B, T] int64, segment_ids: [B, T] int64 (optional, defaults to 0)
    torch::Tensor forward(const torch::Tensor& input_ids, const torch::Tensor& segment_ids = torch::Tensor()) {
        const int64_t B = input_ids.size(0);
        const int64_t T = input_ids.size(1);
        
        auto token_embeddings = token_emb->forward(input_ids);  // [B, T, H]
        
        // Position embeddings
        auto positions = torch::arange(T, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
        positions = positions.unsqueeze(0).expand({B, T});  // [B, T]
        auto position_embeddings = position_emb->forward(positions);  // [B, T, H]
        
        // Segment embeddings
        torch::Tensor segment_embeddings;
        if (segment_ids.defined()) {
            segment_embeddings = segment_emb->forward(segment_ids);  // [B, T, H]
        } else {
            auto zeros = torch::zeros({B, T}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
            segment_embeddings = segment_emb->forward(zeros);  // [B, T, H]
        }
        
        // Sum all embeddings
        auto embeddings = token_embeddings + position_embeddings + segment_embeddings;  // [B, T, H]
        
        // Layer norm and dropout
        embeddings = ln->forward(embeddings);
        if (dropout_p > 0.0 && is_training()) {
            embeddings = torch::dropout(embeddings, dropout_p, is_training());
        }
        
        return embeddings;
    }
};
TORCH_MODULE(BERTEmbeddings);

// BERT model for masked language modeling
struct BERTModelImpl : torch::nn::Module {
    BERTEmbeddings embeddings{nullptr};
    torch::nn::ModuleList encoder_layers{nullptr};
    torch::nn::Linear mlm_head{nullptr};
    
    int64_t vocab_size{};
    int64_t hidden_dim{};
    int64_t num_heads{};
    int64_t num_layers{};
    int64_t ff_dim{};
    int64_t max_seq_len{};
    double dropout_p{};

    BERTModelImpl(
        int64_t vocab_size,
        int64_t hidden_dim = 768,
        int64_t num_heads = 12,
        int64_t num_layers = 12,
        int64_t ff_dim = 3072,
        int64_t max_seq_len = 512,
        double dropout = 0.1)
        : vocab_size(vocab_size),
          hidden_dim(hidden_dim),
          num_heads(num_heads),
          num_layers(num_layers),
          ff_dim(ff_dim),
          max_seq_len(max_seq_len),
          dropout_p(dropout) {
        
        if (hidden_dim % num_heads != 0) {
            throw std::runtime_error("BERTModel: hidden_dim must be divisible by num_heads");
        }
        
        embeddings = register_module("embeddings", BERTEmbeddings(vocab_size, hidden_dim, max_seq_len, dropout));
        
        encoder_layers = register_module("encoder_layers", torch::nn::ModuleList());
        for (int64_t i = 0; i < num_layers; ++i) {
            encoder_layers->push_back(TransformerEncoderBlock(hidden_dim, num_heads, ff_dim, dropout));
        }
        
        mlm_head = register_module("mlm_head", torch::nn::Linear(hidden_dim, vocab_size));
    }

    // input_ids: [B, T] int64
    // attention_mask: [B, T] int64 (optional, 1 for valid, 0 for masked)
    // segment_ids: [B, T] int64 (optional)
    // Returns: logits [B, T, vocab_size]
    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask = torch::Tensor(),
        const torch::Tensor& segment_ids = torch::Tensor()) {
        
        // Get embeddings
        auto x = embeddings->forward(input_ids, segment_ids);  // [B, T, H]
        
        // Pass through encoder layers
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = (*encoder_layers)[i]->as<TransformerEncoderBlock>();
            x = layer->forward(x, attention_mask);
        }
        
        // MLM head (predict vocabulary for each position)
        auto logits = mlm_head->forward(x);  // [B, T, vocab_size]
        
        return logits;
    }
    
    // Forward for masked language modeling training
    // Returns: logits [B, T, vocab_size]
    torch::Tensor forward_mlm(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask = torch::Tensor(),
        const torch::Tensor& segment_ids = torch::Tensor()) {
        return forward(input_ids, attention_mask, segment_ids);
    }
};
TORCH_MODULE(BERTModel);

} // namespace mcppfa::torchlm

