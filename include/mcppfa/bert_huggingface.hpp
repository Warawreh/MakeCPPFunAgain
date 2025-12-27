#pragma once

#include "huggingface.hpp"
#include "torch_bert.hpp"
#include "torch_distilbert.hpp"
#include "safetensors.hpp"
#include "model_loader.hpp"
#include "tokenizer_decoder.hpp"
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace mcppfa::hf {

// BERT Model wrapper that can load from HuggingFace and save weights
class BERTModelWrapper {
public:
    BERTModelWrapper() = default;

    // Load model from HuggingFace repo
    // Downloads config.json, pytorch_model.bin, and loads into BERTModel
    void load_from_hf(
        const std::string& repo_id,
        const std::string& token = std::string(),
        const RepoType repo_type = RepoType::model,
        const std::string& revision = "main") {

        // Create local directory for downloads
        const std::string model_name = repo_id.substr(repo_id.find_last_of('/') + 1);
        const std::string local_dir = ".hf/" + model_name;
        std::filesystem::create_directories(local_dir);

        // Download config.json
        const std::string config_path = local_dir + "/config.json";
        auto res_config = download_file_http(
            repo_id, "config.json", config_path, repo_type, revision, token);
        if (res_config.exit_code != 0) {
            throw std::runtime_error("Failed to download config.json: exit_code=" + 
                                   std::to_string(res_config.exit_code));
        }

        // Download model weights (try model.safetensors first, then pytorch_model.bin, then tf_model.h5)
        std::string actual_weights_path;
        bool weights_downloaded = false;
        
        // Try 1: model.safetensors (preferred format - safer and faster)
        {
            const std::string safetensors_path = local_dir + "/model.safetensors";
            auto res_safetensors = download_file_http(
                repo_id, "model.safetensors", safetensors_path, repo_type, revision, token);
            if (res_safetensors.exit_code == 0) {
                actual_weights_path = safetensors_path;
                weights_downloaded = true;
            }
        }
        
        // Try 2: pytorch_model.bin (fallback)
        if (!weights_downloaded) {
            const std::string weights_path = local_dir + "/pytorch_model.bin";
            auto res_weights = download_file_http(
                repo_id, "pytorch_model.bin", weights_path, repo_type, revision, token);
            if (res_weights.exit_code == 0) {
                actual_weights_path = weights_path;
                weights_downloaded = true;
            }
        }
        
        // Try 3: tf_model.h5 (TensorFlow format - last resort)
        if (!weights_downloaded) {
            const std::string tf_path = local_dir + "/tf_model.h5";
            auto res_tf = download_file_http(
                repo_id, "tf_model.h5", tf_path, repo_type, revision, token);
            if (res_tf.exit_code == 0) {
                actual_weights_path = tf_path;
                weights_downloaded = true;
            }
        }
        
        if (!weights_downloaded) {
            throw std::runtime_error("Failed to download model weights: tried model.safetensors, pytorch_model.bin, and tf_model.h5");
        }

        // Parse config.json to get model parameters
        // For simplicity, we'll use default DistilBERT parameters if parsing fails
        int64_t vocab_size = 30522;  // Default for BERT-base
        int64_t hidden_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 12;
        int64_t ff_dim = 3072;
        int64_t max_seq_len = 512;
        double dropout = 0.1;

        // Try to read config.json and extract parameters
        // Simple JSON parsing - looks for key: value patterns
        try {
            std::ifstream config_file(config_path);
            if (config_file.is_open()) {
                std::string content((std::istreambuf_iterator<char>(config_file)),
                                   std::istreambuf_iterator<char>());
                config_file.close();

                // Helper lambda to extract integer value after a key
                auto extract_int = [&content](const std::string& key) -> int64_t {
                    size_t pos = content.find(key);
                    if (pos == std::string::npos) return -1;
                    pos = content.find(':', pos);
                    if (pos == std::string::npos) return -1;
                    pos++; // skip ':'
                    // Skip whitespace
                    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
                    if (pos >= content.size()) return -1;
                    // Extract number (may have trailing comma or whitespace)
                    size_t end = pos;
                    while (end < content.size() && 
                           content[end] != ',' && content[end] != '}' && 
                           content[end] != '\n' && content[end] != ' ') end++;
                    try {
                        return std::stoll(content.substr(pos, end - pos));
                    } catch (...) {
                        return -1;
                    }
                };

                int64_t val;
                if ((val = extract_int("\"vocab_size\"")) > 0) vocab_size = val;
                if ((val = extract_int("\"hidden_size\"")) > 0) hidden_dim = val;
                else if ((val = extract_int("\"dim\"")) > 0) hidden_dim = val;
                if ((val = extract_int("\"num_attention_heads\"")) > 0) num_heads = val;
                else if ((val = extract_int("\"n_heads\"")) > 0) num_heads = val;
                if ((val = extract_int("\"num_hidden_layers\"")) > 0) num_layers = val;
                else if ((val = extract_int("\"n_layers\"")) > 0) num_layers = val;
                if ((val = extract_int("\"intermediate_size\"")) > 0) ff_dim = val;
                if ((val = extract_int("\"max_position_embeddings\"")) > 0) max_seq_len = val;
            }
        } catch (...) {
            // Use defaults if parsing fails
        }

        // Detect if this is DistilBERT (6 layers, no segment embeddings)
        bool is_distilbert_model = (num_layers == 6 || model_name.find("distilbert") != std::string::npos);
        
        // Create appropriate model based on architecture
        if (is_distilbert_model) {
            // Use DistilBERT architecture (matches HuggingFace exactly)
            distilbert_model_ = std::make_shared<mcppfa::torchlm::DistilBERTForMaskedLM>(
                vocab_size, hidden_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout);
        } else {
            // Use standard BERT architecture
            bert_model_ = std::make_shared<mcppfa::torchlm::BERTModel>(
                vocab_size, hidden_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout);
        }

        // Load weights based on file format
        // Note: Different formats have different loading capabilities in LibTorch C++
        
        bool loaded = false;
        std::string format_note;
        
        // Determine file format from extension
        std::filesystem::path weights_file(actual_weights_path);
        std::string extension = weights_file.extension().string();
        
        if (extension == ".safetensors") {
            // Safetensors format - use our custom loader
            try {
                auto safetensors_tensors = mcppfa::safetensors::load_safetensors(actual_weights_path);
                
                // For DistilBERTForMaskedLM, the "distilbert." prefix is part of the model structure
                // For BERTModel, we might need to strip prefixes
                std::map<std::string, torch::Tensor> state_dict;
                if (distilbert_model_) {
                    // DistilBERTForMaskedLM structure: distilbert.*, vocab_projector.*, vocab_layer_norm.*
                    // Keep keys as-is (they already have "distilbert." prefix)
                    state_dict = safetensors_tensors;
                } else {
                    // For BERT, might need to strip model prefix if present
                    for (const auto& [key, tensor] : safetensors_tensors) {
                        std::string clean_key = key;
                        // Remove common model prefixes if present
                        if (clean_key.find("bert.") == 0) {
                            clean_key = clean_key.substr(5);
                        } else if (clean_key.find("model.") == 0) {
                            clean_key = clean_key.substr(6);
                        }
                        state_dict[clean_key] = tensor;
                    }
                }
                
                // Use clean load_state_dict (replicates HuggingFace's loading)
                if (distilbert_model_) {
                    mcppfa::model_loader::load_state_dict(*distilbert_model_, state_dict, false);
                } else {
                    mcppfa::model_loader::load_state_dict(*bert_model_, state_dict, false);
                }
                loaded = true;
                format_note = "safetensors";
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to load safetensors file: " << e.what() << std::endl;
                std::cerr << "  The model has been created with the correct architecture but random weights." << std::endl;
                format_note = "safetensors (load failed)";
            }
        } else if (extension == ".h5") {
            // TensorFlow H5 format - LibTorch C++ doesn't support this
            std::cerr << "Warning: TensorFlow H5 format is not directly supported by LibTorch C++." << std::endl;
            std::cerr << "  The model has been created with the correct architecture but random weights." << std::endl;
            std::cerr << "  To load TensorFlow weights, use Python to convert to PyTorch format first." << std::endl;
            format_note = "TensorFlow H5 (not supported in LibTorch C++)";
        } else if (extension == ".bin") {
            // PyTorch .bin format - try to load (may fail due to Python pickle format)
            std::string last_error;
            
            // Strategy 1: Try InputArchive (PyTorch checkpoint format)
            try {
                torch::serialize::InputArchive archive;
                archive.load_from(actual_weights_path);
                
                // Build state dict from archive
                std::map<std::string, torch::Tensor> state_dict;
                auto keys = archive.keys();
                for (const auto& key : keys) {
                    torch::Tensor tensor;
                    if (archive.try_read(key, tensor, false)) {
                        state_dict[key] = tensor;
                    }
                }
                
                // Load using clean load_state_dict
                if (distilbert_model_) {
                    mcppfa::model_loader::load_state_dict(*distilbert_model_, state_dict, false);
                } else {
                    mcppfa::model_loader::load_state_dict(*bert_model_, state_dict, false);
                }
                loaded = true;
                format_note = "PyTorch checkpoint";
            } catch (const std::exception& e) {
                last_error = e.what();
            }
            
            // Strategy 2: Try direct module load (fallback)
            if (!loaded) {
                try {
                    if (distilbert_model_) {
                        torch::load(*distilbert_model_, actual_weights_path);
                    } else {
                        torch::load(*bert_model_, actual_weights_path);
                    }
                    loaded = true;
                    format_note = "PyTorch module";
                } catch (const std::exception& e) {
                    last_error = last_error.empty() ? e.what() : (last_error + " / " + std::string(e.what()));
                }
            }
            
            // If loading failed, warn but continue
            if (!loaded) {
                std::cerr << "Warning: Could not load pre-trained weights from " << actual_weights_path << std::endl;
                std::cerr << "  Reason: HuggingFace pytorch_model.bin files use Python pickle format." << std::endl;
                std::cerr << "  The model has been created with the correct architecture but random weights." << std::endl;
                std::cerr << "  To load pre-trained weights, convert using Python: torch.load() then torch.save()" << std::endl;
                format_note = "PyTorch pickle (not loadable in LibTorch C++)";
            }
        } else {
            std::cerr << "Warning: Unknown weights file format: " << extension << std::endl;
            std::cerr << "  The model has been created with the correct architecture but random weights." << std::endl;
            format_note = "unknown format";
        }
        
        if (loaded) {
            std::cout << "Successfully loaded weights from " << actual_weights_path 
                      << " (format: " << format_note << ")" << std::endl;
        } else {
            std::cout << "Model created with architecture from config.json (weights: random initialization)" << std::endl;
        }

        weights_path_ = actual_weights_path;
        config_path_ = config_path;
        model_name_ = model_name;
    }

    // Type-specific getters
    mcppfa::torchlm::BERTModel& bert_model() { 
        if (!bert_model_) {
            throw std::runtime_error("BERTModelWrapper: model is not a BERTModel");
        }
        return *bert_model_;
    }
    
    mcppfa::torchlm::DistilBERTForMaskedLM& distilbert_model() {
        if (!distilbert_model_) {
            throw std::runtime_error("BERTModelWrapper: model is not a DistilBERTForMaskedLM");
        }
        return *distilbert_model_;
    }
    
    bool is_distilbert() const { return distilbert_model_ != nullptr; }
    
    // Get the underlying module pointer (for generic access)
    torch::nn::Module* get_module() {
        if (distilbert_model_) {
            return distilbert_model_->get();
        }
        return bert_model_->get();
    }
    const torch::nn::Module* get_module() const {
        if (distilbert_model_) {
            return distilbert_model_->get();
        }
        return bert_model_->get();
    }

    // Save model weights to local file
    void save(const std::string& path) {
        if (distilbert_model_) {
            torch::save(*distilbert_model_, path);
        } else if (bert_model_) {
            torch::save(*bert_model_, path);
        } else {
            throw std::runtime_error("BERTModelWrapper::save: model not loaded");
        }
        weights_path_ = path;
    }

    // Get current weights path
    const std::string& weights_path() const { return weights_path_; }
    const std::string& config_path() const { return config_path_; }

    // ===== Text Generation Interface (like Python transformers) =====
    
    /**
     * Reset the generation state with a new prompt.
     * Encodes the text and initializes internal input_ids.
     * Similar to Python: model.generate(tokenizer.encode(text), ...)
     */
    void reset(mcppfa::tokenizer::TokenizerDecoder& tokenizer, const std::string& text) {
        input_ids_ = tokenizer.encode(text);
    }
    
    /**
     * Set the generation state directly from token IDs.
     * Useful for resuming generation or setting custom initial state.
     */
    void set_input_ids(const std::vector<int64_t>& ids) {
        input_ids_ = ids;
    }
    
    /**
     * Get the current input_ids state.
     * Returns the current sequence of token IDs.
     */
    const std::vector<int64_t>& get_input_ids() const {
        return input_ids_;
    }
    
    /**
     * Generate the next token and update internal state.
     * This is the main prediction method, similar to Python transformers' generate().
     * 
     * @param tokenizer Tokenizer decoder for encoding/decoding and special token handling
     * @param temperature Sampling temperature (default: 0.8, lower = more conservative)
     * @param top_k Number of top tokens to sample from (default: 50)
     * @param greedy If true, use greedy decoding (argmax), else use sampling (default: false)
     * @param max_seq_len Maximum sequence length before stopping (default: 512)
     * @return The generated token ID, or -1 if generation should stop
     * 
     * Usage:
     *   bert_model.reset(tokenizer, "The red fox");
     *   for (int i = 0; i < 50; ++i) {
     *       int64_t token = bert_model.predict(tokenizer);
     *       if (token == -1) break;  // Stopped
     *   }
     *   std::string result = tokenizer.decode(bert_model.get_input_ids());
     */
    int64_t predict(
        mcppfa::tokenizer::TokenizerDecoder& tokenizer,
        double temperature = 0.8,
        int64_t top_k = 50,
        bool greedy = false,
        int64_t max_seq_len = 512) {
        
        // Check if model is loaded
        if (!bert_model_ && !distilbert_model_) {
            throw std::runtime_error("BERTModelWrapper::predict: model not loaded");
        }
        
        // Check if state is initialized
        if (input_ids_.empty()) {
            throw std::runtime_error("BERTModelWrapper::predict: input_ids not initialized. Call reset() first.");
        }
        
        // Check sequence length limit
        int64_t seq_len = static_cast<int64_t>(input_ids_.size());
        if (seq_len >= max_seq_len) {
            return -1;  // Signal to stop
        }
        
        // Set model to eval mode via underlying module pointer
        torch::nn::Module* module = get_module();
        if (!module) {
            throw std::runtime_error("BERTModelWrapper::predict: underlying module pointer is null");
        }
        module->eval();

        // Prepare input tensor from current input_ids [B=1, T=seq_len]
        torch::Tensor input_tensor = torch::from_blob(
            input_ids_.data(),
            {1, seq_len},
            torch::TensorOptions().dtype(torch::kInt64)
        ).clone();

        // Create attention mask (1 for all tokens)
        torch::Tensor attention_mask = torch::ones(
            {1, seq_len},
            torch::TensorOptions().dtype(torch::kInt64)
        );

        // Forward pass: try calling concrete Impl wrappers via dynamic_cast
        torch::NoGradGuard no_grad;
        torch::IValue out_iv;
        if (auto d_impl = dynamic_cast<mcppfa::torchlm::DistilBERTForMaskedLMImpl*>(module)) {
            out_iv = d_impl->forward_iv({input_tensor, attention_mask});
        } else if (auto b_impl = dynamic_cast<mcppfa::torchlm::BERTModelImpl*>(module)) {
            out_iv = b_impl->forward_iv({input_tensor, attention_mask});
        } else {
            throw std::runtime_error("BERTModelWrapper::predict: unknown concrete module type; cannot call forward");
        }

        // Convert IValue to logits Tensor (handle Tensor or Tuple outputs)
        torch::Tensor logits;
        if (out_iv.isTensor()) {
            logits = out_iv.toTensor();
        } else if (out_iv.isTuple()) {
            auto elems = out_iv.toTuple()->elements();
            if (!elems.empty() && elems[0].isTensor()) {
                logits = elems[0].toTensor();
            } else {
                throw std::runtime_error("BERTModelWrapper::predict: forward() returned tuple with no tensor at index 0");
            }
        } else {
            throw std::runtime_error("BERTModelWrapper::predict: forward() returned unexpected IValue type");
        }
        // logits shape expected: [1, seq_len, vocab_size]
        
        // Get logits for the last position
        auto last_logits = logits[0][seq_len - 1];  // [vocab_size]
        
        // Use tokenizer's built-in sampling (automatically filters special tokens)
        // This matches how Python's transformers library handles special tokens
        int64_t next_token = tokenizer.sample_next_token(
            last_logits, 
            temperature, 
            top_k, 
            greedy
        );
        
        // Append the predicted token to internal state
        input_ids_.push_back(next_token);
        
        // Check if we should stop (e.g., [SEP] token)
        if (tokenizer.is_special_token(next_token)) {
            std::string special_token_str = tokenizer.decode_token(next_token);
            if (special_token_str == "[SEP]") {
                return -1;  // Signal to stop
            }
        }
        
        return next_token;
    }

private:
    std::shared_ptr<mcppfa::torchlm::BERTModel> bert_model_;
    std::shared_ptr<mcppfa::torchlm::DistilBERTForMaskedLM> distilbert_model_;
    std::string weights_path_;
    std::string config_path_;
    std::string model_name_;
    
    // Generation state (like Python transformers maintains internally)
    std::vector<int64_t> input_ids_;
};

// BERT Tokenizer wrapper that can load from HuggingFace and save
class BERTTokenizerWrapper {
public:
    BERTTokenizerWrapper() = default;

    // Load tokenizer from HuggingFace repo
    void load_from_hf(
        const std::string& repo_id,
        const std::string& token = std::string(),
        const RepoType repo_type = RepoType::model,
        const std::string& revision = "main") {

        // Create local directory for downloads
        const std::string model_name = repo_id.substr(repo_id.find_last_of('/') + 1);
        const std::string local_dir = ".hf/" + model_name;
        std::filesystem::create_directories(local_dir);

        // Download tokenizer.json
        const std::string tokenizer_path = local_dir + "/tokenizer.json";
        auto res = download_file_http(
            repo_id, "tokenizer.json", tokenizer_path, repo_type, revision, token);
        
        if (res.exit_code != 0) {
            throw std::runtime_error("Failed to download tokenizer.json: exit_code=" + 
                                   std::to_string(res.exit_code));
        }

        tokenizer_path_ = tokenizer_path;
        model_name_ = model_name;
    }

    // Save tokenizer to local file (copies the file)
    void save(const std::string& path) {
        if (tokenizer_path_.empty()) {
            throw std::runtime_error("BERTTokenizerWrapper::save: tokenizer not loaded");
        }
        if (!std::filesystem::exists(tokenizer_path_)) {
            throw std::runtime_error("BERTTokenizerWrapper::save: source file does not exist: " + tokenizer_path_);
        }
        
        // Normalize paths to compare them (handle relative vs absolute paths)
        std::filesystem::path source_path, dest_path;
        try {
            source_path = std::filesystem::canonical(tokenizer_path_);
        } catch (...) {
            source_path = std::filesystem::absolute(tokenizer_path_);
        }
        try {
            dest_path = std::filesystem::canonical(path);
        } catch (...) {
            dest_path = std::filesystem::absolute(path);
        }
        
        // If source and destination are the same, no need to copy
        if (source_path == dest_path) {
            return;
        }
        
        // Create parent directory if needed
        std::filesystem::create_directories(dest_path.parent_path());
        
        // Copy file with overwrite option
        std::filesystem::copy_file(tokenizer_path_, path, 
                                  std::filesystem::copy_options::overwrite_existing);
        tokenizer_path_ = path;
    }

    // Get current tokenizer path
    const std::string& tokenizer_path() const { return tokenizer_path_; }

private:
    std::string tokenizer_path_;
    std::string model_name_;
};

} // namespace mcppfa::hf

