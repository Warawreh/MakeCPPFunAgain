#pragma once

#include "torch_bert.hpp"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <vector>

namespace mcppfa::torchlm {

// Training configuration for BERT models
struct BERTTrainConfig {
    int64_t batch_size{32};
    int64_t seq_len{128};
    int64_t steps{1000};
    double lr{3e-4};
    double weight_decay{0.01};
    uint64_t seed{1234};
    int64_t log_every{50};
    int64_t eval_every{500};
    double warmup_steps{0.1};  // Fraction of steps for warmup
    bool use_gradient_clipping{true};
    double max_grad_norm{1.0};
};

// Simple data loader for BERT training
// Takes tokenized sequences and creates masked language modeling batches
class BERTDataLoader {
public:
    BERTDataLoader(
        const std::vector<int64_t>& token_ids,
        int64_t vocab_size,
        int64_t mask_token_id = 103,  // [MASK] token ID for BERT
        int64_t cls_token_id = 101,   // [CLS] token ID
        int64_t sep_token_id = 102,   // [SEP] token ID
        int64_t pad_token_id = 0)     // [PAD] token ID
        : token_ids_(token_ids),
          vocab_size_(vocab_size),
          mask_token_id_(mask_token_id),
          cls_token_id_(cls_token_id),
          sep_token_id_(sep_token_id),
          pad_token_id_(pad_token_id) {}
    
    // Sample a batch for masked language modeling
    struct MLMBatch {
        torch::Tensor input_ids;      // [B, T] with some tokens masked
        torch::Tensor attention_mask; // [B, T] 1 for valid, 0 for padding
        torch::Tensor labels;         // [B, T] -1 for non-masked, token_id for masked
        torch::Tensor masked_positions; // [B, T] 1 where tokens are masked
    };
    
    MLMBatch sample_batch(int64_t batch_size, int64_t seq_len, const torch::Device& device, std::mt19937_64& rng) {
        std::uniform_int_distribution<size_t> dist(0, token_ids_.size() - seq_len - 1);
        
        std::vector<int64_t> batch_input_ids;
        std::vector<int64_t> batch_labels;
        std::vector<int64_t> batch_attention_mask;
        std::vector<int64_t> batch_masked_positions;
        
        // Masking probability (15% of tokens)
        const double mask_prob = 0.15;
        // Of masked tokens: 80% [MASK], 10% random, 10% unchanged
        const double mask_token_prob = 0.8;
        const double random_token_prob = 0.1;
        
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int64_t> vocab_dist(0, vocab_size_ - 1);
        
        for (int64_t b = 0; b < batch_size; ++b) {
            size_t start_idx = dist(rng);
            
            std::vector<int64_t> input_seq;
            std::vector<int64_t> label_seq;
            std::vector<int64_t> mask_seq;
            
            // Add [CLS] token
            input_seq.push_back(cls_token_id_);
            label_seq.push_back(-1);  // Don't predict [CLS]
            mask_seq.push_back(0);
            
            // Add sequence tokens with masking
            for (int64_t i = 0; i < seq_len - 2; ++i) {  // -2 for [CLS] and [SEP]
                if (start_idx + i >= token_ids_.size()) break;
                
                int64_t token_id = token_ids_[start_idx + i];
                double rand_val = prob_dist(rng);
                
                if (rand_val < mask_prob) {
                    // This token should be masked
                    double mask_rand = prob_dist(rng);
                    if (mask_rand < mask_token_prob) {
                        // 80%: replace with [MASK]
                        input_seq.push_back(mask_token_id_);
                    } else if (mask_rand < mask_token_prob + random_token_prob) {
                        // 10%: replace with random token
                        input_seq.push_back(vocab_dist(rng));
                    } else {
                        // 10%: keep original
                        input_seq.push_back(token_id);
                    }
                    label_seq.push_back(token_id);  // Predict original token
                    mask_seq.push_back(1);
                } else {
                    // Not masked
                    input_seq.push_back(token_id);
                    label_seq.push_back(-1);  // Don't predict
                    mask_seq.push_back(0);
                }
            }
            
            // Add [SEP] token
            input_seq.push_back(sep_token_id_);
            label_seq.push_back(-1);
            mask_seq.push_back(0);
            
            // Pad to seq_len
            while (input_seq.size() < static_cast<size_t>(seq_len)) {
                input_seq.push_back(pad_token_id_);
                label_seq.push_back(-1);
                mask_seq.push_back(0);
            }
            
            // Truncate if too long
            if (input_seq.size() > static_cast<size_t>(seq_len)) {
                input_seq.resize(seq_len);
                label_seq.resize(seq_len);
                mask_seq.resize(seq_len);
            }
            
            batch_input_ids.insert(batch_input_ids.end(), input_seq.begin(), input_seq.end());
            batch_labels.insert(batch_labels.end(), label_seq.begin(), label_seq.end());
            batch_masked_positions.insert(batch_masked_positions.end(), mask_seq.begin(), mask_seq.end());
            
            // Attention mask: 1 for non-padding tokens
            std::vector<int64_t> attn_mask(seq_len, 1);
            for (size_t i = 0; i < input_seq.size(); ++i) {
                if (input_seq[i] == pad_token_id_) {
                    attn_mask[i] = 0;
                }
            }
            batch_attention_mask.insert(batch_attention_mask.end(), attn_mask.begin(), attn_mask.end());
        }
        
        MLMBatch batch;
        batch.input_ids = torch::from_blob(
            batch_input_ids.data(),
            {batch_size, seq_len},
            torch::TensorOptions().dtype(torch::kInt64)
        ).clone().to(device);
        
        batch.attention_mask = torch::from_blob(
            batch_attention_mask.data(),
            {batch_size, seq_len},
            torch::TensorOptions().dtype(torch::kInt64)
        ).clone().to(device);
        
        batch.labels = torch::from_blob(
            batch_labels.data(),
            {batch_size, seq_len},
            torch::TensorOptions().dtype(torch::kInt64)
        ).clone().to(device);
        
        batch.masked_positions = torch::from_blob(
            batch_masked_positions.data(),
            {batch_size, seq_len},
            torch::TensorOptions().dtype(torch::kInt64)
        ).clone().to(device);
        
        return batch;
    }

private:
    std::vector<int64_t> token_ids_;
    int64_t vocab_size_;
    int64_t mask_token_id_;
    int64_t cls_token_id_;
    int64_t sep_token_id_;
    int64_t pad_token_id_;
};

// BERT training function - simple, clean training loop inspired by llm.c
inline double train_bert(
    BERTModel& model,
    BERTDataLoader& data_loader,
    const BERTTrainConfig& cfg,
    const torch::Device& device) {
    
    model->to(device);
    model->train();
    
    // Optimizer with learning rate scheduling
    torch::optim::AdamW optim(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
    
    std::mt19937_64 rng(cfg.seed);
    
    double last_loss = 0.0;
    int64_t warmup_steps_count = static_cast<int64_t>(cfg.steps * cfg.warmup_steps);
    
    for (int64_t step = 1; step <= cfg.steps; ++step) {
        // Learning rate warmup
        if (step <= warmup_steps_count) {
            double warmup_lr = cfg.lr * (static_cast<double>(step) / warmup_steps_count);
            for (auto& group : optim.param_groups()) {
                static_cast<torch::optim::AdamWOptions&>(group.options()).lr(warmup_lr);
            }
        }
        
        // Sample batch
        auto batch = data_loader.sample_batch(cfg.batch_size, cfg.seq_len, device, rng);
        
        // Forward pass
        auto logits = model->forward(batch.input_ids, batch.attention_mask);  // [B, T, vocab_size]
        
        // Compute loss only on masked positions
        // Reshape for cross entropy: [B*T, vocab_size] and [B*T]
        auto logits_flat = logits.view({cfg.batch_size * cfg.seq_len, -1});
        auto labels_flat = batch.labels.view({cfg.batch_size * cfg.seq_len});
        auto mask_flat = batch.masked_positions.view({cfg.batch_size * cfg.seq_len});
        
        // Only compute loss on masked positions
        auto loss_per_token = torch::nn::functional::cross_entropy(
            logits_flat,
            labels_flat,
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
        
        // Mask out non-masked positions
        auto masked_loss = loss_per_token * mask_flat.to(torch::kFloat32);
        auto num_masked = mask_flat.sum().item<int64_t>();
        auto loss = (num_masked > 0) ? masked_loss.sum() / num_masked : masked_loss.sum();
        
        // Backward pass
        optim.zero_grad();
        loss.backward();
        
        // Gradient clipping
        if (cfg.use_gradient_clipping) {
            torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.max_grad_norm);
        }
        
        optim.step();
        
        last_loss = loss.item<double>();
        
        // Logging
        if (cfg.log_every > 0 && (step % cfg.log_every) == 0) {
            std::cout << "Step " << step << "/" << cfg.steps 
                      << " | Loss: " << last_loss
                      << " | LR: " << optim.param_groups()[0].options().get_lr()
                      << " | Masked tokens: " << num_masked << std::endl;
        }
        
        // Evaluation (optional)
        if (cfg.eval_every > 0 && (step % cfg.eval_every) == 0) {
            model->eval();
            torch::NoGradGuard no_grad;
            
            // Sample a small eval batch
            auto eval_batch = data_loader.sample_batch(8, cfg.seq_len, device, rng);
            auto eval_logits = model->forward(eval_batch.input_ids, eval_batch.attention_mask);
            
            auto eval_logits_flat = eval_logits.view({8 * cfg.seq_len, -1});
            auto eval_labels_flat = eval_batch.labels.view({8 * cfg.seq_len});
            auto eval_mask_flat = eval_batch.masked_positions.view({8 * cfg.seq_len});
            
            auto eval_loss_per_token = torch::nn::functional::cross_entropy(
                eval_logits_flat,
                eval_labels_flat,
                torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
            
            auto eval_masked_loss = eval_loss_per_token * eval_mask_flat.to(torch::kFloat32);
            auto eval_num_masked = eval_mask_flat.sum().item<int64_t>();
            auto eval_loss = (eval_num_masked > 0) ? eval_masked_loss.sum() / eval_num_masked : eval_masked_loss.sum();
            
            std::cout << "  [Eval] Loss: " << eval_loss.item<double>() << std::endl;
            
            model->train();
        }
    }
    
    return last_loss;
}

// Save BERT model checkpoint
inline void save_bert_checkpoint(
    const BERTModel& model,
    const std::string& path) {
    torch::save(model, path);
}

// Load BERT model checkpoint
inline BERTModel load_bert_checkpoint(
    int64_t vocab_size,
    int64_t hidden_dim,
    int64_t num_heads,
    int64_t num_layers,
    int64_t ff_dim,
    int64_t max_seq_len,
    double dropout,
    const std::string& path,
    const torch::Device& device) {
    BERTModel model(vocab_size, hidden_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout);
    torch::load(model, path);
    model->to(device);
    model->eval();
    return model;
}

} // namespace mcppfa::torchlm

