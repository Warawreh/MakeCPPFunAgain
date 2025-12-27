#pragma once

#include <torch/torch.h>
#include <map>
#include <string>
#include <iostream>

namespace mcppfa::model_loader {

/**
 * Load state dict into a model (replicates PyTorch's load_state_dict)
 * 
 * This function matches parameter names from the state dict to the model's
 * named parameters and buffers, and loads them. This is exactly how
 * HuggingFace loads weights in Python.
 * 
 * @param model The model to load weights into
 * @param state_dict Map of parameter name -> tensor
 * @param strict If true, all parameters must be found (default: false)
 * @return Number of parameters successfully loaded
 */
template<typename ModuleType>
inline int load_state_dict(
    ModuleType& model,
    const std::map<std::string, torch::Tensor>& state_dict,
    bool strict = false) {
    
    auto* module_ptr = model.get();
    auto named_params = module_ptr->named_parameters();
    auto named_buffers = module_ptr->named_buffers();
    
    int loaded_count = 0;
    int missing_keys = 0;
    int unexpected_keys = 0;
    
    // Track which state dict keys we've used
    std::map<std::string, bool> used_keys;
    for (const auto& [key, _] : state_dict) {
        used_keys[key] = false;
    }
    
    // Try to load each model parameter from state dict
    for (const auto& param : named_params) {
        const std::string& param_name = param.key();
        bool found = false;
        
        // Try exact match first
        if (state_dict.find(param_name) != state_dict.end()) {
            const auto& tensor = state_dict.at(param_name);
            if (param.value().sizes() == tensor.sizes() && 
                param.value().dtype() == tensor.dtype()) {
                torch::NoGradGuard no_grad;
                param.value().set_data(tensor.detach().clone());
                loaded_count++;
                found = true;
                used_keys[param_name] = true;
            } else {
                std::cerr << "Warning: Shape/dtype mismatch for parameter " << param_name
                          << ": model=" << param.value().sizes() << "/" << param.value().dtype()
                          << ", state_dict=" << tensor.sizes() << "/" << tensor.dtype() << std::endl;
            }
        }
        
        if (!found) {
            missing_keys++;
            if (strict) {
                std::cerr << "Error: Missing parameter in state_dict: " << param_name << std::endl;
            }
        }
    }
    
    // Try to load buffers
    for (const auto& buf : named_buffers) {
        const std::string& buf_name = buf.key();
        bool found = false;
        
        if (state_dict.find(buf_name) != state_dict.end()) {
            const auto& tensor = state_dict.at(buf_name);
            if (buf.value().sizes() == tensor.sizes() && 
                buf.value().dtype() == tensor.dtype()) {
                torch::NoGradGuard no_grad;
                buf.value().set_data(tensor.detach().clone());
                loaded_count++;
                found = true;
                used_keys[buf_name] = true;
            }
        }
        
        if (!found) {
            missing_keys++;
        }
    }
    
    // Count unexpected keys (keys in state_dict not in model)
    for (const auto& [key, used] : used_keys) {
        if (!used) {
            unexpected_keys++;
            if (strict) {
                std::cerr << "Error: Unexpected key in state_dict: " << key << std::endl;
            }
        }
    }
    
    if (strict && (missing_keys > 0 || unexpected_keys > 0)) {
        throw std::runtime_error(
            "load_state_dict failed: " + std::to_string(missing_keys) + " missing keys, " +
            std::to_string(unexpected_keys) + " unexpected keys");
    }
    
    std::cout << "Loaded " << loaded_count << " parameters/buffers from state_dict"
              << " (missing: " << missing_keys << ", unexpected: " << unexpected_keys << ")" << std::endl;
    
    return loaded_count;
}

/**
 * Load state dict with name mapping (for converting between naming conventions)
 * 
 * This is useful when loading HuggingFace weights that use different naming
 * than your model implementation.
 * 
 * @param model The model to load weights into
 * @param state_dict Map of parameter name -> tensor
 * @param name_mapping Map of model_name -> state_dict_name
 * @param strict If true, all parameters must be found
 * @return Number of parameters successfully loaded
 */
template<typename ModuleType>
inline int load_state_dict_with_mapping(
    ModuleType& model,
    const std::map<std::string, torch::Tensor>& state_dict,
    const std::map<std::string, std::string>& name_mapping,
    bool strict = false) {
    
    auto* module_ptr = model.get();
    auto named_params = module_ptr->named_parameters();
    auto named_buffers = module_ptr->named_buffers();
    
    int loaded_count = 0;
    int missing_keys = 0;
    
    // Load parameters
    for (const auto& param : named_params) {
        const std::string& param_name = param.key();
        std::string state_dict_name = param_name;
        
        // Check if there's a mapping
        if (name_mapping.find(param_name) != name_mapping.end()) {
            state_dict_name = name_mapping.at(param_name);
        }
        
        if (state_dict.find(state_dict_name) != state_dict.end()) {
            const auto& tensor = state_dict.at(state_dict_name);
            if (param.value().sizes() == tensor.sizes() && 
                param.value().dtype() == tensor.dtype()) {
                torch::NoGradGuard no_grad;
                param.value().set_data(tensor.detach().clone());
                loaded_count++;
            } else {
                std::cerr << "Warning: Shape/dtype mismatch for " << param_name
                          << " (state_dict key: " << state_dict_name << ")" << std::endl;
                missing_keys++;
            }
        } else {
            missing_keys++;
            if (strict) {
                std::cerr << "Error: Missing parameter: " << param_name 
                          << " (looked for: " << state_dict_name << ")" << std::endl;
            }
        }
    }
    
    // Load buffers
    for (const auto& buf : named_buffers) {
        const std::string& buf_name = buf.key();
        std::string state_dict_name = buf_name;
        
        if (name_mapping.find(buf_name) != name_mapping.end()) {
            state_dict_name = name_mapping.at(buf_name);
        }
        
        if (state_dict.find(state_dict_name) != state_dict.end()) {
            const auto& tensor = state_dict.at(state_dict_name);
            if (buf.value().sizes() == tensor.sizes() && 
                buf.value().dtype() == tensor.dtype()) {
                torch::NoGradGuard no_grad;
                buf.value().set_data(tensor.detach().clone());
                loaded_count++;
            }
        } else {
            missing_keys++;
        }
    }
    
    if (strict && missing_keys > 0) {
        throw std::runtime_error("load_state_dict_with_mapping failed: " + 
                                std::to_string(missing_keys) + " missing keys");
    }
    
    std::cout << "Loaded " << loaded_count << " parameters/buffers (missing: " 
              << missing_keys << ")" << std::endl;
    
    return loaded_count;
}

} // namespace mcppfa::model_loader

