#pragma once

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace mcppfa::safetensors {

// Tensor metadata from safetensors header
struct TensorMetadata {
    std::vector<int64_t> shape;
    std::string dtype;
    int64_t data_offsets[2];  // [start, end] byte offsets
};

// Read little-endian uint64 from bytes (matches Python's struct.unpack('<Q', ...))
inline uint64_t read_u64_le(const char* bytes) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (uint64_t)(uint8_t)bytes[i] << (8 * i);
    }
    return v;
}

// Map safetensors dtype string to torch::Dtype
inline torch::Dtype map_dtype(const std::string& dt) {
    // Handle both uppercase and lowercase variants
    if (dt == "F32" || dt == "f32" || dt == "FLOAT32" || dt == "float32") return torch::kFloat32;
    if (dt == "F64" || dt == "f64" || dt == "FLOAT64" || dt == "float64") return torch::kFloat64;
    if (dt == "F16" || dt == "f16" || dt == "FLOAT16" || dt == "float16") return torch::kFloat16;
    if (dt == "BF16" || dt == "bf16" || dt == "BFLOAT16" || dt == "bfloat16") return torch::kBFloat16;
    if (dt == "I64" || dt == "i64" || dt == "INT64" || dt == "int64") return torch::kInt64;
    if (dt == "I32" || dt == "i32" || dt == "INT32" || dt == "int32") return torch::kInt32;
    if (dt == "I16" || dt == "i16" || dt == "INT16" || dt == "int16") return torch::kInt16;
    if (dt == "I8" || dt == "i8" || dt == "INT8" || dt == "int8") return torch::kInt8;
    if (dt == "U8" || dt == "u8" || dt == "UINT8" || dt == "uint8") return torch::kUInt8;
    throw std::runtime_error("Unsupported safetensors dtype: " + dt);
}

// Load safetensors file and return a map of tensor name -> tensor
// This replicates Python HuggingFace's safetensors loading behavior exactly:
// 1. Read 8 bytes (little-endian uint64) = header size
// 2. Read JSON header
// 3. Use data_offsets (relative to data section start) to get tensor bytes
// 4. Use zero-copy from_blob when possible
inline std::map<std::string, torch::Tensor> load_safetensors(const std::string& file_path) {
    // Open file and read entire content into memory (for simplicity, can be optimized with mmap later)
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open safetensors file: " + file_path);
    }
    
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read entire file into memory
    std::vector<char> file_data(static_cast<size_t>(file_size));
    if (!file.read(file_data.data(), file_size)) {
        throw std::runtime_error("Cannot read safetensors file");
    }
    
    const char* bytes = file_data.data();
    
    // Step 1: Read header length (first 8 bytes, little-endian uint64)
    if (file_size < 8) {
        throw std::runtime_error("Safetensors file too small: missing header length");
    }
    uint64_t header_length = read_u64_le(bytes);
    
    // Validate header length (sanity check)
    if (header_length > static_cast<uint64_t>(file_size) - 8 || header_length > 100 * 1024 * 1024) {
        throw std::runtime_error("Invalid header length in safetensors file: " + std::to_string(header_length));
    }
    
    // Step 2: Read JSON header (bytes 8 to 8+header_length-1)
    const char* header_start = bytes + 8;
    std::string header_json(header_start, header_length);
    
    // Step 3: Data section starts immediately after header
    const char* data_section = header_start + header_length;
    const size_t data_section_start_offset = 8 + header_length;
    
    // Debug: print first 500 chars of header
    std::cout << "Safetensors header (first 500 chars): " << header_json.substr(0, 500) << std::endl;
    
    // Parse JSON header - simpler approach: find each tensor entry
    // Format: {"tensor_name": {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [0, 24]}, ...}
    std::map<std::string, TensorMetadata> metadata;
    
    // Find all tensor entries by looking for patterns: "tensor_name": {"dtype"
    size_t pos = 0;
    while (pos < header_json.size()) {
        // Look for pattern: "key": { followed by "dtype"
        size_t key_start = header_json.find("\"", pos);
        if (key_start == std::string::npos) break;
        
        size_t key_end = header_json.find("\"", key_start + 1);
        if (key_end == std::string::npos) break;
        
        std::string tensor_name = header_json.substr(key_start + 1, key_end - key_start - 1);
        
        // Skip __metadata__ entries
        if (tensor_name == "__metadata__") {
            // Find the closing brace for this object
            size_t obj_start = header_json.find('{', key_end);
            if (obj_start == std::string::npos) break;
            int depth = 1;
            pos = obj_start + 1;
            while (pos < header_json.size() && depth > 0) {
                if (header_json[pos] == '{') depth++;
                else if (header_json[pos] == '}') depth--;
                pos++;
            }
            continue;
        }
        
        // Find the object start after the colon
        size_t colon_pos = header_json.find(':', key_end);
        if (colon_pos == std::string::npos) {
            pos = key_end + 1;
            continue;
        }
        
        size_t obj_start = header_json.find('{', colon_pos);
        if (obj_start == std::string::npos) {
            pos = colon_pos + 1;
            continue;
        }
        
        // Find the end of this object (matching closing brace)
        int obj_depth = 1;
        size_t obj_end = obj_start + 1;
        while (obj_end < header_json.size() && obj_depth > 0) {
            if (header_json[obj_end] == '{') obj_depth++;
            else if (header_json[obj_end] == '}') obj_depth--;
            obj_end++;
        }
        if (obj_depth > 0) {
            pos = obj_end;
            continue;
        }
        obj_end--; // Back up to the closing brace
        
        // Extract the object content
        std::string obj_content = header_json.substr(obj_start, obj_end - obj_start + 1);
        
        // Check if this object has "dtype" (indicating it's a tensor, not metadata)
        if (obj_content.find("\"dtype\"") == std::string::npos) {
            pos = obj_end + 1;
            continue;
        }
        
        // Parse this tensor object
        TensorMetadata meta;
        
        // Extract dtype
        size_t dtype_pos = obj_content.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_val_start = obj_content.find('"', dtype_pos + 7);
            if (dtype_val_start != std::string::npos) {
                dtype_val_start++;
                size_t dtype_val_end = obj_content.find('"', dtype_val_start);
                if (dtype_val_end != std::string::npos) {
                    meta.dtype = obj_content.substr(dtype_val_start, dtype_val_end - dtype_val_start);
                }
            }
        }
        
        // Extract shape
        size_t shape_pos = obj_content.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t bracket_start = obj_content.find('[', shape_pos);
            if (bracket_start != std::string::npos) {
                size_t bracket_end = obj_content.find(']', bracket_start);
                if (bracket_end != std::string::npos) {
                    std::string shape_str = obj_content.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                    // Parse shape array - split by comma
                    size_t num_start = 0;
                    while (num_start < shape_str.size()) {
                        // Skip whitespace and commas
                        while (num_start < shape_str.size() && 
                               (shape_str[num_start] == ' ' || shape_str[num_start] == '\t' || shape_str[num_start] == ',')) {
                            num_start++;
                        }
                        if (num_start >= shape_str.size()) break;
                        
                        // Find end of number
                        size_t num_end = num_start;
                        while (num_end < shape_str.size() && 
                               shape_str[num_end] != ',' && 
                               shape_str[num_end] != ']' &&
                               shape_str[num_end] != ' ' &&
                               shape_str[num_end] != '\t') {
                            num_end++;
                        }
                        
                        if (num_end > num_start) {
                            try {
                                std::string num_str = shape_str.substr(num_start, num_end - num_start);
                                meta.shape.push_back(std::stoll(num_str));
                            } catch (...) {
                                // Skip invalid number
                            }
                        }
                        num_start = num_end + 1;
                    }
                }
            }
        }
        
        // Extract data_offsets
        size_t offsets_pos = obj_content.find("\"data_offsets\"");
        if (offsets_pos != std::string::npos) {
            size_t bracket_start = obj_content.find('[', offsets_pos);
            if (bracket_start != std::string::npos) {
                size_t bracket_end = obj_content.find(']', bracket_start);
                if (bracket_end != std::string::npos) {
                    std::string offsets_str = obj_content.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                    // Parse two numbers separated by comma
                    size_t comma_pos = offsets_str.find(',');
                    if (comma_pos != std::string::npos) {
                        try {
                            // First number
                            size_t first_start = 0;
                            while (first_start < comma_pos && (offsets_str[first_start] == ' ' || offsets_str[first_start] == '\t')) first_start++;
                            size_t first_end = comma_pos;
                            while (first_end > first_start && (offsets_str[first_end - 1] == ' ' || offsets_str[first_end - 1] == '\t')) first_end--;
                            if (first_end > first_start) {
                                meta.data_offsets[0] = std::stoll(offsets_str.substr(first_start, first_end - first_start));
                            }
                            
                            // Second number
                            size_t second_start = comma_pos + 1;
                            while (second_start < offsets_str.size() && (offsets_str[second_start] == ' ' || offsets_str[second_start] == '\t')) second_start++;
                            size_t second_end = offsets_str.size();
                            while (second_end > second_start && (offsets_str[second_end - 1] == ' ' || offsets_str[second_end - 1] == '\t')) second_end--;
                            if (second_end > second_start) {
                                meta.data_offsets[1] = std::stoll(offsets_str.substr(second_start, second_end - second_start));
                            }
                        } catch (...) {
                            // Use defaults
                        }
                    }
                }
            }
        }
        
        // Save if we have all required fields
        if (!meta.dtype.empty() && !meta.shape.empty() && meta.data_offsets[1] > meta.data_offsets[0]) {
            metadata[tensor_name] = meta;
        }
        
        // Move to next entry
        pos = obj_end + 1;
    }
    
    std::cout << "Parsed " << metadata.size() << " tensors from safetensors header" << std::endl;
    
    // Step 4: Load tensors using data_offsets (relative to data section start)
    std::map<std::string, torch::Tensor> tensors;
    
    for (const auto& [tensor_name, meta] : metadata) {
        size_t start = static_cast<size_t>(meta.data_offsets[0]);
        size_t end = static_cast<size_t>(meta.data_offsets[1]);
        size_t data_size = end - start;
        
        if (data_size <= 0) {
            std::cerr << "Warning: Invalid data_offsets for tensor " << tensor_name << std::endl;
            continue;
        }
        
        // Map dtype string to torch::Dtype
        torch::Dtype torch_dtype;
        try {
            torch_dtype = map_dtype(meta.dtype);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << " for tensor " << tensor_name << ", skipping" << std::endl;
            continue;
        }
        
        // Calculate element size
        size_t elem_size = torch::elementSize(torch_dtype);
        
        // Calculate expected size from shape
        size_t elem_count = 1;
        for (int64_t dim : meta.shape) {
            elem_count *= static_cast<size_t>(dim);
        }
        size_t expected_size = elem_count * elem_size;
        
        // Validate size (data_offsets should match expected size)
        if (data_size != expected_size) {
            std::cerr << "Warning: Size mismatch for tensor " << tensor_name 
                      << " (expected " << expected_size << " bytes, got " << data_size << " bytes)" << std::endl;
            continue;
        }
        
        // Validate bounds (data_offsets are relative to data section start)
        size_t max_data_size = static_cast<size_t>(file_size) - data_section_start_offset;
        if (start + data_size > max_data_size) {
            std::cerr << "Warning: Data out of bounds for tensor " << tensor_name << std::endl;
            continue;
        }
        
        // Get pointer to tensor raw bytes (data_offsets are relative to data section start)
        const char* tensor_bytes = data_section + start;
        
        // Create tensor shape vector
        std::vector<int64_t> shape_vec(meta.shape.begin(), meta.shape.end());
        
        // Use zero-copy from_blob (matches Python's behavior)
        // Note: from_blob does NOT take ownership, so we need to keep file_data alive
        // For persistent storage, we'll clone the tensor
        auto options = torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU);
        torch::Tensor tensor = torch::from_blob(
            const_cast<void*>(static_cast<const void*>(tensor_bytes)),
            shape_vec,
            options
        ).clone();  // Clone to own the data (since file_data will go out of scope)
        
        tensors[tensor_name] = tensor;
    }
    
    return tensors;
}

// Apply safetensors weights to a model's parameters and buffers
// This is a helper function that matches tensor names and applies them
template<typename ModuleType>
inline void apply_safetensors_to_model(
    ModuleType& model,
    const std::map<std::string, torch::Tensor>& safetensors_tensors) {
    
    auto* module_ptr = model.get();
    auto named_params = module_ptr->named_parameters();
    auto named_buffers = module_ptr->named_buffers();
    
    // Debug: print available model parameters
    std::cout << "\nModel parameters (" << named_params.size() << " total):" << std::endl;
    size_t param_count = 0;
    for (const auto& param : named_params) {
        if (param_count < 10) {  // Print first 10
            std::cout << "  " << param.key() << " " << param.value().sizes() << " " << param.value().dtype() << std::endl;
        }
        param_count++;
    }
    if (param_count > 10) {
        std::cout << "  ... and " << (param_count - 10) << " more" << std::endl;
    }
    
    // Debug: print safetensors tensor names
    std::cout << "\nSafetensors tensors (" << safetensors_tensors.size() << " total):" << std::endl;
    size_t tensor_count = 0;
    for (const auto& [name, tensor] : safetensors_tensors) {
        if (tensor_count < 10) {  // Print first 10
            std::cout << "  " << name << " " << tensor.sizes() << " " << tensor.dtype() << std::endl;
        }
        tensor_count++;
    }
    if (tensor_count > 10) {
        std::cout << "  ... and " << (tensor_count - 10) << " more" << std::endl;
    }
    std::cout << std::endl;
    
    size_t applied_count = 0;
    size_t shape_mismatch_count = 0;
    size_t dtype_mismatch_count = 0;
    size_t not_found_count = 0;
    
    for (const auto& [tensor_name, tensor] : safetensors_tensors) {
        bool applied = false;
        
        // Try exact match in parameters
        if (named_params.contains(tensor_name)) {
            auto param = named_params[tensor_name];
            if (param.sizes() == tensor.sizes() && param.dtype() == tensor.dtype()) {
                // Use set_data to avoid autograd issues with in-place operations
                torch::NoGradGuard no_grad;
                param.set_data(tensor.detach().clone());
                applied = true;
                applied_count++;
            } else {
                if (param.sizes() != tensor.sizes()) {
                    std::cerr << "Shape mismatch for parameter " << tensor_name 
                              << ": model=" << param.sizes() << ", safetensors=" << tensor.sizes() << std::endl;
                    shape_mismatch_count++;
                }
                if (param.dtype() != tensor.dtype()) {
                    std::cerr << "Dtype mismatch for parameter " << tensor_name 
                              << ": model=" << param.dtype() << ", safetensors=" << tensor.dtype() << std::endl;
                    dtype_mismatch_count++;
                }
            }
        }
        // Try exact match in buffers
        else if (named_buffers.contains(tensor_name)) {
            auto buf = named_buffers[tensor_name];
            if (buf.sizes() == tensor.sizes() && buf.dtype() == tensor.dtype()) {
                // Use set_data to avoid autograd issues with in-place operations
                torch::NoGradGuard no_grad;
                buf.set_data(tensor.detach().clone());
                applied = true;
                applied_count++;
            } else {
                if (buf.sizes() != tensor.sizes()) {
                    std::cerr << "Shape mismatch for buffer " << tensor_name 
                              << ": model=" << buf.sizes() << ", safetensors=" << tensor.sizes() << std::endl;
                    shape_mismatch_count++;
                }
                if (buf.dtype() != tensor.dtype()) {
                    std::cerr << "Dtype mismatch for buffer " << tensor_name 
                              << ": model=" << buf.dtype() << ", safetensors=" << tensor.dtype() << std::endl;
                    dtype_mismatch_count++;
                }
            }
        }
        
        if (!applied) {
            // Systematic name mapping from HuggingFace DistilBERT to our model
            // Build mapped name step by step to avoid replacement conflicts
            std::string mapped_name = tensor_name;
            
            // Step 1: Remove distilbert. prefix
            if (mapped_name.find("distilbert.") == 0) {
                mapped_name = mapped_name.substr(11);
            }
            
            // Step 2: Map embeddings components
            // embeddings.LayerNorm -> embeddings.ln
            if (mapped_name.find("embeddings.LayerNorm.") == 0) {
                mapped_name = "embeddings.ln." + mapped_name.substr(21);
            } else if (mapped_name == "embeddings.LayerNorm") {
                mapped_name = "embeddings.ln";
            }
            
            // embeddings.word_embeddings -> embeddings.token_emb
            if (mapped_name.find("embeddings.word_embeddings.") == 0) {
                mapped_name = "embeddings.token_emb." + mapped_name.substr(27);
            } else if (mapped_name == "embeddings.word_embeddings") {
                mapped_name = "embeddings.token_emb";
            }
            
            // embeddings.position_embeddings -> embeddings.position_emb
            if (mapped_name.find("embeddings.position_embeddings.") == 0) {
                mapped_name = "embeddings.position_emb." + mapped_name.substr(32);
            } else if (mapped_name == "embeddings.position_embeddings") {
                mapped_name = "embeddings.position_emb";
            }
            
            // Step 3: Map transformer.layer.X -> encoder_layers.X
            // Find pattern "transformer.layer." and replace with "encoder_layers."
            size_t tf_start = mapped_name.find("transformer.layer.");
            if (tf_start != std::string::npos) {
                // Find the layer number (e.g., "0", "1", etc.)
                size_t layer_num_start = tf_start + 17; // after "transformer.layer."
                size_t layer_num_end = layer_num_start;
                while (layer_num_end < mapped_name.size() && mapped_name[layer_num_end] != '.') {
                    layer_num_end++;
                }
                std::string layer_num = mapped_name.substr(layer_num_start, layer_num_end - layer_num_start);
                std::string rest = (layer_num_end < mapped_name.size()) ? mapped_name.substr(layer_num_end) : "";
                // Build new name: prefix + "encoder_layers." + layer_num + rest
                std::string prefix = mapped_name.substr(0, tf_start);
                mapped_name = prefix + "encoder_layers." + layer_num + rest;
            }
            
            // Step 4: Map attention components
            // .attention. -> .attn.
            size_t attn_pos = mapped_name.find(".attention.");
            while (attn_pos != std::string::npos) {
                mapped_name.replace(attn_pos, 11, ".attn.");
                attn_pos = mapped_name.find(".attention.", attn_pos + 5);
            }
            // .attention at end -> .attn
            if (mapped_name.size() >= 10 && mapped_name.substr(mapped_name.size() - 10) == ".attention") {
                mapped_name.replace(mapped_name.size() - 10, 10, ".attn");
            }
            
            // Step 5: Map attention projection layers
            // .q_lin. -> .q_proj., .k_lin. -> .k_proj., etc.
            // Handle with dot at end first, then with dot in middle
            std::vector<std::pair<std::string, std::string>> proj_mappings = {
                {".q_lin.", ".q_proj."},
                {".k_lin.", ".k_proj."},
                {".v_lin.", ".v_proj."},
                {".out_lin.", ".out_proj."},
                {".q_linear.", ".q_proj."},
                {".k_linear.", ".k_proj."},
                {".v_linear.", ".v_proj."},
                {".out_linear.", ".out_proj."}
            };
            
            for (const auto& mapping : proj_mappings) {
                // Replace all occurrences
                size_t pos = 0;
                while ((pos = mapped_name.find(mapping.first, pos)) != std::string::npos) {
                    mapped_name.replace(pos, mapping.first.size(), mapping.second);
                    pos += mapping.second.size();
                }
            }
            
            // Handle without trailing dot (at end of string)
            if (mapped_name.size() >= 6 && mapped_name.substr(mapped_name.size() - 6) == ".q_lin") {
                mapped_name.replace(mapped_name.size() - 6, 6, ".q_proj");
            }
            if (mapped_name.size() >= 6 && mapped_name.substr(mapped_name.size() - 6) == ".k_lin") {
                mapped_name.replace(mapped_name.size() - 6, 6, ".k_proj");
            }
            if (mapped_name.size() >= 6 && mapped_name.substr(mapped_name.size() - 6) == ".v_lin") {
                mapped_name.replace(mapped_name.size() - 6, 6, ".v_proj");
            }
            if (mapped_name.size() >= 8 && mapped_name.substr(mapped_name.size() - 8) == ".out_lin") {
                mapped_name.replace(mapped_name.size() - 8, 8, ".out_proj");
            }
            
            // Step 6: Map layer norms in encoder blocks
            // .sa_layer_norm. -> .ln1.
            size_t sa_ln_pos = mapped_name.find(".sa_layer_norm.");
            while (sa_ln_pos != std::string::npos) {
                mapped_name.replace(sa_ln_pos, 15, ".ln1.");
                sa_ln_pos = mapped_name.find(".sa_layer_norm.", sa_ln_pos + 5);
            }
            if (mapped_name.size() >= 14 && mapped_name.substr(mapped_name.size() - 14) == ".sa_layer_norm") {
                mapped_name.replace(mapped_name.size() - 14, 14, ".ln1");
            }
            
            // .output_layer_norm. -> .ln2.
            size_t out_ln_pos = mapped_name.find(".output_layer_norm.");
            while (out_ln_pos != std::string::npos) {
                mapped_name.replace(out_ln_pos, 19, ".ln2.");
                out_ln_pos = mapped_name.find(".output_layer_norm.", out_ln_pos + 5);
            }
            if (mapped_name.size() >= 18 && mapped_name.substr(mapped_name.size() - 18) == ".output_layer_norm") {
                mapped_name.replace(mapped_name.size() - 18, 18, ".ln2");
            }
            
            // Step 7: Map feedforward network
            // .ffn. -> .ff.
            size_t ffn_pos = mapped_name.find(".ffn.");
            while (ffn_pos != std::string::npos) {
                mapped_name.replace(ffn_pos, 5, ".ff.");
                ffn_pos = mapped_name.find(".ffn.", ffn_pos + 4);
            }
            if (mapped_name.size() >= 4 && mapped_name.substr(mapped_name.size() - 4) == ".ffn") {
                mapped_name.replace(mapped_name.size() - 4, 4, ".ff");
            }
            
            // .lin1. -> .fc1., .lin2. -> .fc2.
            size_t lin1_pos = mapped_name.find(".lin1.");
            while (lin1_pos != std::string::npos) {
                mapped_name.replace(lin1_pos, 6, ".fc1.");
                lin1_pos = mapped_name.find(".lin1.", lin1_pos + 5);
            }
            if (mapped_name.size() >= 5 && mapped_name.substr(mapped_name.size() - 5) == ".lin1") {
                mapped_name.replace(mapped_name.size() - 5, 5, ".fc1");
            }
            
            size_t lin2_pos = mapped_name.find(".lin2.");
            while (lin2_pos != std::string::npos) {
                mapped_name.replace(lin2_pos, 6, ".fc2.");
                lin2_pos = mapped_name.find(".lin2.", lin2_pos + 5);
            }
            if (mapped_name.size() >= 5 && mapped_name.substr(mapped_name.size() - 5) == ".lin2") {
                mapped_name.replace(mapped_name.size() - 5, 5, ".fc2");
            }
            
            // Try the mapped name
            std::vector<std::string> name_variants = {mapped_name};
            
            // Also try without distilbert prefix if we haven't already
            if (tensor_name.find("distilbert.") == 0) {
                std::string without_prefix = tensor_name.substr(11);
                name_variants.push_back(without_prefix);
            }
            
            // Try each variant
            for (const auto& variant : name_variants) {
                if (named_params.contains(variant)) {
                    auto param = named_params[variant];
                    if (param.sizes() == tensor.sizes() && param.dtype() == tensor.dtype()) {
                        torch::NoGradGuard no_grad;
                        param.set_data(tensor.detach().clone());
                        applied = true;
                        applied_count++;
                        std::cout << "Matched " << tensor_name << " -> " << variant << std::endl;
                        break;
                    }
                } else if (named_buffers.contains(variant)) {
                    auto buf = named_buffers[variant];
                    if (buf.sizes() == tensor.sizes() && buf.dtype() == tensor.dtype()) {
                        torch::NoGradGuard no_grad;
                        buf.set_data(tensor.detach().clone());
                        applied = true;
                        applied_count++;
                        std::cout << "Matched " << tensor_name << " -> " << variant << std::endl;
                        break;
                    }
                }
            }
            
            if (!applied) {
                not_found_count++;
                // Only print first few not found to avoid spam
                if (not_found_count <= 10) {
                    std::cerr << "Tensor not found in model: " << tensor_name << " -> tried: " << mapped_name << std::endl;
                }
            }
        }
    }
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Applied: " << applied_count << " tensors" << std::endl;
    if (shape_mismatch_count > 0) {
        std::cout << "  Shape mismatches: " << shape_mismatch_count << std::endl;
    }
    if (dtype_mismatch_count > 0) {
        std::cout << "  Dtype mismatches: " << dtype_mismatch_count << std::endl;
    }
    if (not_found_count > 0) {
        std::cout << "  Not found: " << not_found_count << " tensors" << std::endl;
    }
}

} // namespace mcppfa::safetensors

