#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <set>
#include <torch/torch.h>

namespace mcppfa::tokenizer {

/**
 * Simple tokenizer decoder for HuggingFace tokenizer.json
 * 
 * Parses the vocabulary from tokenizer.json and provides decode functionality.
 * Automatically detects and handles special tokens (like transformers library).
 * This is a simplified parser - for full tokenizer support, use a proper JSON library.
 */
class TokenizerDecoder {
public:
    TokenizerDecoder() = default;
    
    // Load vocabulary from tokenizer.json
    void load_from_file(const std::string& tokenizer_path) {
        std::ifstream file(tokenizer_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open tokenizer file: " + tokenizer_path);
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse vocabulary from tokenizer.json
        // Format: {"model": {"vocab": {"token": id, ...}}, ...}
        parse_vocab(content);
    }
    
    // Decode a single token ID to text
    std::string decode_token(int64_t token_id) const {
        if (id_to_token_.find(token_id) != id_to_token_.end()) {
            return id_to_token_.at(token_id);
        }
        // Return [UNK] for unknown tokens
        return "[UNK]";
    }
    
    // Decode a sequence of token IDs to text
    // Automatically skips special tokens (like transformers does)
    std::string decode(const std::vector<int64_t>& token_ids, bool skip_special_tokens = true) const {
        std::string result;
        bool first = true;
        
        for (int64_t token_id : token_ids) {
            // Skip special tokens if requested (default behavior, like transformers)
            if (skip_special_tokens && is_special_token(token_id)) {
                continue;
            }
            
            std::string token = decode_token(token_id);
            
            // Handle subword tokens (## prefix means continuation, BERT-style)
            if (!first && token.size() > 2 && token.substr(0, 2) == "##") {
                // Remove ## prefix and append without space
                result += token.substr(2);
            } else {
                if (!first && !token.empty() && token[0] != ' ') {
                    result += " ";
                }
                result += token;
            }
            
            first = false;
        }
        
        return result;
    }
    
    // Get vocabulary size
    size_t vocab_size() const {
        return id_to_token_.size();
    }
    
    // Check if a token ID is a special token
    bool is_special_token(int64_t token_id) const {
        return special_token_ids_.find(token_id) != special_token_ids_.end();
    }
    
    // Get special token IDs (for filtering during generation)
    const std::set<int64_t>& get_special_token_ids() const {
        return special_token_ids_;
    }
    
    // Mask special tokens in logits (set to -inf, like transformers does)
    // This prevents special tokens from being sampled during generation
    torch::Tensor mask_special_tokens(const torch::Tensor& logits) const {
        // logits: [vocab_size] or [B, vocab_size]
        auto masked_logits = logits.clone();
        
        // Create a mask tensor (1 for special tokens, 0 for regular tokens)
        auto vocab_size = logits.size(-1);
        auto mask = torch::zeros({vocab_size}, torch::TensorOptions().dtype(torch::kBool));
        
        for (int64_t special_id : special_token_ids_) {
            if (special_id >= 0 && special_id < vocab_size) {
                mask[special_id] = true;
            }
        }
        
        // Set special token logits to -inf (very negative value)
        masked_logits.masked_fill_(mask, -1e9);
        
        return masked_logits;
    }
    
    // Sample next token from logits with special token filtering (like transformers)
    // Returns the sampled token ID, automatically filtering special tokens
    int64_t sample_next_token(
        const torch::Tensor& logits,  // [vocab_size]
        double temperature = 1.0,
        int64_t top_k = 50,
        bool greedy = false) const {
        
        // Apply special token masking (like transformers does)
        auto masked_logits = mask_special_tokens(logits);
        
        // Apply temperature
        if (temperature != 1.0 && temperature > 0.0) {
            masked_logits = masked_logits / temperature;
        }
        
        if (greedy) {
            // Greedy: just take argmax
            return masked_logits.argmax(-1).item<int64_t>();
        }
        
        // Top-k sampling
        auto topk_result = torch::topk(masked_logits, top_k);
        auto topk_values = std::get<0>(topk_result);  // [top_k]
        auto topk_indices = std::get<1>(topk_result);  // [top_k]
        
        // Convert to probabilities
        auto topk_probs = torch::softmax(topk_values, -1);
        
        // Sample from top-k
        auto sampled_idx = topk_probs.multinomial(1);  // [1]
        return topk_indices[sampled_idx.item<int64_t>()].item<int64_t>();
    }
    
    // Encode text to token IDs (simplified approach - matches Python's tokenizer.encode())
    // Python's approach: split text, look up each word in vocab, use UNK if not found
    std::vector<int64_t> encode(const std::string& text) const {
        // Build token_to_id map if not already built (lazy initialization)
        if (token_to_id_.empty() && !id_to_token_.empty()) {
            for (const auto& pair : id_to_token_) {
                token_to_id_[pair.second] = pair.first;
            }
        }
        
        // Special tokens (standard BERT/DistilBERT token IDs)
        const int64_t CLS_TOKEN = 101;   // [CLS]
        const int64_t SEP_TOKEN = 102;   // [SEP]
        const int64_t UNK_TOKEN = 100;   // [UNK]
        
        std::vector<int64_t> token_ids;
        token_ids.push_back(CLS_TOKEN);
        
        // Simple word-based tokenization: split by whitespace
        std::string word;
        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!word.empty()) {
                    // Try exact match first, then lowercase
                    int64_t token_id = UNK_TOKEN;
                    auto it = token_to_id_.find(word);
                    if (it != token_to_id_.end()) {
                        token_id = it->second;
                    } else {
                        // Try lowercase (BERT/DistilBERT are uncased)
                        std::string word_lower = word;
                        std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
                        auto it_lower = token_to_id_.find(word_lower);
                        if (it_lower != token_to_id_.end()) {
                            token_id = it_lower->second;
                        }
                    }
                    token_ids.push_back(token_id);
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        
        // Handle last word
        if (!word.empty()) {
            int64_t token_id = UNK_TOKEN;
            auto it = token_to_id_.find(word);
            if (it != token_to_id_.end()) {
                token_id = it->second;
            } else {
                std::string word_lower = word;
                std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
                auto it_lower = token_to_id_.find(word_lower);
                if (it_lower != token_to_id_.end()) {
                    token_id = it_lower->second;
                }
            }
            token_ids.push_back(token_id);
        }
        
        token_ids.push_back(SEP_TOKEN);
        return token_ids;
    }

private:
    std::map<int64_t, std::string> id_to_token_;
    mutable std::map<std::string, int64_t> token_to_id_;  // Reverse map, built lazily
    std::set<int64_t> special_token_ids_;  // Set of special token IDs (auto-detected)
    
    // Detect if a token string is a special token (like transformers does)
    bool is_special_token_string(const std::string& token) const {
        // Special tokens typically start with [ and end with ]
        if (token.size() >= 3 && token[0] == '[' && token[token.size() - 1] == ']') {
            // Common special tokens: [CLS], [SEP], [PAD], [UNK], [MASK], etc.
            return true;
        }
        return false;
    }
    
    // Detect and register special tokens from vocabulary
    void detect_special_tokens() {
        special_token_ids_.clear();
        for (const auto& [token_id, token_str] : id_to_token_) {
            if (is_special_token_string(token_str)) {
                special_token_ids_.insert(token_id);
            }
        }
        if (!special_token_ids_.empty()) {
            std::cout << "Detected " << special_token_ids_.size() << " special tokens: ";
            bool first = true;
            for (int64_t id : special_token_ids_) {
                if (!first) std::cout << ", ";
                std::cout << id_to_token_.at(id);
                first = false;
            }
            std::cout << std::endl;
        }
    }
    
    void parse_vocab(const std::string& json_content) {
        // Find the vocab section: "vocab": { ... }
        size_t vocab_start = json_content.find("\"vocab\"");
        if (vocab_start == std::string::npos) {
            // Try alternative format: "model": {"vocab": ...}
            size_t model_start = json_content.find("\"model\"");
            if (model_start != std::string::npos) {
                size_t vocab_in_model = json_content.find("\"vocab\"", model_start);
                if (vocab_in_model != std::string::npos) {
                    vocab_start = vocab_in_model;
                }
            }
        }
        
        if (vocab_start == std::string::npos) {
            std::cerr << "Warning: Could not find vocab section in tokenizer.json" << std::endl;
            std::cerr << "Using fallback: will try to parse token-to-id mappings directly" << std::endl;
            // Try to parse any "token": id patterns
            parse_vocab_fallback(json_content);
            return;
        }
        
        // Find the opening brace after "vocab"
        size_t brace_start = json_content.find('{', vocab_start);
        if (brace_start == std::string::npos) {
            throw std::runtime_error("Cannot find vocab object in tokenizer.json");
        }
        
        // Parse token: id pairs
        size_t pos = brace_start + 1;
        int depth = 1;
        
        while (pos < json_content.size() && depth > 0) {
            // Skip whitespace
            while (pos < json_content.size() && 
                   (json_content[pos] == ' ' || json_content[pos] == '\n' || 
                    json_content[pos] == '\t' || json_content[pos] == '\r')) {
                pos++;
            }
            
            if (pos >= json_content.size()) break;
            
            // Check for closing brace
            if (json_content[pos] == '}') {
                depth--;
                if (depth == 0) break;
                pos++;
                continue;
            }
            
            // Check for opening brace
            if (json_content[pos] == '{') {
                depth++;
                pos++;
                continue;
            }
            
            // Parse "token": id
            if (json_content[pos] == '"') {
                // Extract token (quoted string)
                size_t token_start = pos + 1;
                size_t token_end = token_start;
                bool escaped = false;
                while (token_end < json_content.size()) {
                    if (escaped) {
                        escaped = false;
                        token_end++;
                        continue;
                    }
                    if (json_content[token_end] == '\\') {
                        escaped = true;
                        token_end++;
                        continue;
                    }
                    if (json_content[token_end] == '"') {
                        break;
                    }
                    token_end++;
                }
                
                if (token_end >= json_content.size()) break;
                
                std::string token = json_content.substr(token_start, token_end - token_start);
                // Unescape common escape sequences
                unescape_string(token);
                
                pos = token_end + 1;
                
                // Skip whitespace and colon
                while (pos < json_content.size() && 
                       (json_content[pos] == ' ' || json_content[pos] == ':' || 
                        json_content[pos] == '\t')) {
                    pos++;
                }
                
                // Extract ID (number)
                if (pos >= json_content.size()) break;
                
                size_t id_start = pos;
                size_t id_end = id_start;
                while (id_end < json_content.size() && 
                       json_content[id_end] != ',' && 
                       json_content[id_end] != '}' &&
                       json_content[id_end] != ' ' &&
                       json_content[id_end] != '\n') {
                    id_end++;
                }
                
                if (id_end > id_start) {
                    try {
                        int64_t token_id = std::stoll(json_content.substr(id_start, id_end - id_start));
                        id_to_token_[token_id] = token;
                    } catch (...) {
                        // Skip invalid numbers
                    }
                }
                
                pos = id_end;
                // Skip comma if present
                if (pos < json_content.size() && json_content[pos] == ',') {
                    pos++;
                }
            } else {
                pos++;
            }
        }
        
        std::cout << "Loaded " << id_to_token_.size() << " tokens from vocabulary" << std::endl;
        
        // Detect special tokens after loading vocabulary
        detect_special_tokens();
    }
    
    void parse_vocab_fallback(const std::string& json_content) {
        // Fallback: try to find any "token": number patterns
        size_t pos = 0;
        while (pos < json_content.size()) {
            // Look for quoted string followed by colon and number
            size_t quote_start = json_content.find('"', pos);
            if (quote_start == std::string::npos) break;
            
            size_t quote_end = json_content.find('"', quote_start + 1);
            if (quote_end == std::string::npos) break;
            
            std::string token = json_content.substr(quote_start + 1, quote_end - quote_start - 1);
            unescape_string(token);
            
            // Find colon after the quote
            size_t colon_pos = json_content.find(':', quote_end);
            if (colon_pos == std::string::npos) {
                pos = quote_end + 1;
                continue;
            }
            
            // Find number after colon
            size_t num_start = colon_pos + 1;
            while (num_start < json_content.size() && 
                   (json_content[num_start] == ' ' || json_content[num_start] == '\t')) {
                num_start++;
            }
            
            size_t num_end = num_start;
            while (num_end < json_content.size() && 
                   json_content[num_end] >= '0' && json_content[num_end] <= '9') {
                num_end++;
            }
            
            if (num_end > num_start) {
                try {
                    int64_t token_id = std::stoll(json_content.substr(num_start, num_end - num_start));
                    id_to_token_[token_id] = token;
                } catch (...) {
                    // Skip
                }
            }
            
            pos = num_end;
        }
        
        std::cout << "Loaded " << id_to_token_.size() << " tokens from vocabulary (fallback parser)" << std::endl;
        
        // Detect special tokens after loading vocabulary
        detect_special_tokens();
    }
    
    void unescape_string(std::string& str) {
        // Unescape common JSON escape sequences
        size_t pos = 0;
        while ((pos = str.find("\\\"", pos)) != std::string::npos) {
            str.replace(pos, 2, "\"");
            pos++;
        }
        pos = 0;
        while ((pos = str.find("\\\\", pos)) != std::string::npos) {
            str.replace(pos, 2, "\\");
            pos++;
        }
        pos = 0;
        while ((pos = str.find("\\n", pos)) != std::string::npos) {
            str.replace(pos, 2, "\n");
            pos++;
        }
        pos = 0;
        while ((pos = str.find("\\t", pos)) != std::string::npos) {
            str.replace(pos, 2, "\t");
            pos++;
        }
        pos = 0;
        while ((pos = str.find("\\r", pos)) != std::string::npos) {
            str.replace(pos, 2, "\r");
            pos++;
        }
    }
};

} // namespace mcppfa::tokenizer

