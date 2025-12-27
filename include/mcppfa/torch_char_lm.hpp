#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace mcppfa::torchlm {

struct CharTokenizer {
    std::vector<char> itos;
    std::unordered_map<char, int64_t> stoi;

    int64_t vocab_size() const { return static_cast<int64_t>(itos.size()); }

    static CharTokenizer from_text(std::string_view text) {
        CharTokenizer tok;
        tok.itos.reserve(256);

        std::vector<bool> seen(256, false);
        for (unsigned char uc : text) {
            seen[uc] = true;
        }

        // Stable order for reproducibility.
        for (int i = 0; i < 256; ++i) {
            if (seen[static_cast<size_t>(i)]) {
                const char c = static_cast<char>(i);
                tok.stoi.emplace(c, static_cast<int64_t>(tok.itos.size()));
                tok.itos.push_back(c);
            }
        }

        if (tok.itos.empty()) {
            throw std::runtime_error("CharTokenizer: empty vocabulary (text was empty?)");
        }

        return tok;
    }

    std::vector<int64_t> encode(std::string_view text) const {
        std::vector<int64_t> ids;
        ids.reserve(text.size());
        for (char c : text) {
            const auto it = stoi.find(c);
            if (it == stoi.end()) {
                throw std::runtime_error("CharTokenizer: encountered unseen character");
            }
            ids.push_back(it->second);
        }
        return ids;
    }

    std::string decode(const std::vector<int64_t>& ids) const {
        std::string out;
        out.reserve(ids.size());
        for (const auto id : ids) {
            if (id < 0 || id >= static_cast<int64_t>(itos.size())) {
                throw std::runtime_error("CharTokenizer: id out of range");
            }
            out.push_back(itos[static_cast<size_t>(id)]);
        }
        return out;
    }

    torch::Tensor encode_tensor(std::string_view text, const torch::Device& device) const {
        auto ids = encode(text);
        auto t = torch::from_blob(ids.data(), {static_cast<int64_t>(ids.size())}, torch::kInt64).clone();
        return t.to(device);
    }
};

inline void save_char_tokenizer_json(const CharTokenizer& tok, const std::string& path) {
    // Minimal, self-describing tokenizer artifact.
    // Format: {"tokenizer_type":"char","itos":[<byte0>,<byte1>,...]}
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("Failed to open tokenizer file for writing: " + path);
    }

    std::fputs("{\"tokenizer_type\":\"char\",\"itos\":[", f);
    for (size_t i = 0; i < tok.itos.size(); ++i) {
        const unsigned char uc = static_cast<unsigned char>(tok.itos[i]);
        std::fprintf(f, "%u", static_cast<unsigned>(uc));
        if (i + 1 < tok.itos.size()) std::fputc(',', f);
    }
    std::fputs("]}\n", f);
    std::fclose(f);
}

inline std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string s;
    f.seekg(0, std::ios::end);
    s.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0, std::ios::beg);
    f.read(&s[0], static_cast<std::streamsize>(s.size()));
    return s;
}

struct Batch {
    torch::Tensor x; // [B, T] int64
    torch::Tensor y; // [B, T] int64
};

struct TextStream {
    std::vector<int64_t> ids;

    static TextStream from_ids(std::vector<int64_t> v) {
        if (v.size() < 2) {
            throw std::runtime_error("TextStream: need at least 2 tokens");
        }
        TextStream s;
        s.ids = std::move(v);
        return s;
    }

    Batch sample_batch(
        const int64_t batch_size,
        const int64_t seq_len,
        const torch::Device& device,
        std::mt19937_64& rng) const {

        if (batch_size <= 0 || seq_len <= 0) {
            throw std::runtime_error("TextStream: batch_size/seq_len must be > 0");
        }
        if (ids.size() < static_cast<size_t>(seq_len + 1)) {
            throw std::runtime_error("TextStream: text too small for seq_len");
        }

        std::uniform_int_distribution<size_t> dist(0, ids.size() - static_cast<size_t>(seq_len + 1));

        torch::Tensor x = torch::empty({batch_size, seq_len}, torch::kInt64);
        torch::Tensor y = torch::empty({batch_size, seq_len}, torch::kInt64);

        auto xa = x.accessor<int64_t, 2>();
        auto ya = y.accessor<int64_t, 2>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const size_t start = dist(rng);
            for (int64_t t = 0; t < seq_len; ++t) {
                xa[b][t] = ids[start + static_cast<size_t>(t)];
                ya[b][t] = ids[start + static_cast<size_t>(t) + 1];
            }
        }

        return Batch{ x.to(device), y.to(device) };
    }
};

struct CharLMImpl : torch::nn::Module {
    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear head{nullptr};

    int64_t vocab{};
    int64_t emb_dim{};
    int64_t hidden_dim{};
    int64_t num_layers{};

    CharLMImpl(int64_t vocab_size, int64_t emb, int64_t hidden, int64_t layers)
        : vocab(vocab_size), emb_dim(emb), hidden_dim(hidden), num_layers(layers) {

        if (vocab_size <= 0) throw std::runtime_error("CharLM: vocab_size must be > 0");
        tok_emb = register_module("tok_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, emb)));
        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(emb, hidden).num_layers(layers).batch_first(true)));
        head = register_module("head", torch::nn::Linear(hidden, vocab_size));
    }

    // x: [B, T] int64
    torch::Tensor forward(const torch::Tensor& x) {
        auto e = tok_emb->forward(x);               // [B, T, E]
        auto out = std::get<0>(lstm->forward(e));   // [B, T, H]
        auto logits = head->forward(out);           // [B, T, V]
        return logits;
    }
};
TORCH_MODULE(CharLM);

struct TrainConfig {
    int64_t batch_size{32};
    int64_t seq_len{128};
    int64_t steps{200};
    double lr{3e-4};
    double weight_decay{0.01};
    uint64_t seed{1234};
    int64_t log_every{20};
};

inline torch::Device default_device() {
    if (torch::cuda::is_available()) return torch::Device(torch::kCUDA);
    return torch::Device(torch::kCPU);
}

inline double train_char_lm(CharLM& model, const TextStream& stream, const TrainConfig& cfg, const torch::Device& device) {
    model->to(device);
    model->train();

    torch::optim::AdamW optim(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));

    std::mt19937_64 rng(cfg.seed);

    double last_loss = 0.0;
    for (int64_t step = 1; step <= cfg.steps; ++step) {
        const auto batch = stream.sample_batch(cfg.batch_size, cfg.seq_len, device, rng);
        const auto logits = model->forward(batch.x);  // [B, T, V]

        auto loss = torch::nn::functional::cross_entropy(
            logits.view({cfg.batch_size * cfg.seq_len, -1}),
            batch.y.view({cfg.batch_size * cfg.seq_len}),
            torch::nn::functional::CrossEntropyFuncOptions());

        optim.zero_grad();
        loss.backward();
        optim.step();

        last_loss = loss.item<double>();
        if (cfg.log_every > 0 && (step % cfg.log_every) == 0) {
            // printing is left to the caller in notebooks; return value covers final loss.
        }
    }

    return last_loss;
}

inline std::vector<int64_t> sample_next_token(
    const torch::Tensor& last_logits, // [V]
    std::mt19937_64& rng,
    double temperature,
    bool greedy) {

    // last_logits is on some device; bring to CPU for sampling.
    auto logits = last_logits.to(torch::kCPU);
    if (temperature <= 0.0) temperature = 1.0;
    logits = logits / temperature;

    if (greedy) {
        const auto id = logits.argmax(-1).item<int64_t>();
        return {id};
    }

    auto probs = torch::softmax(logits, -1);
    std::vector<double> p(probs.numel());
    auto pa = probs.accessor<float, 1>();
    for (int64_t i = 0; i < probs.numel(); ++i) {
        p[static_cast<size_t>(i)] = static_cast<double>(pa[i]);
    }

    std::discrete_distribution<int64_t> dist(p.begin(), p.end());
    const int64_t id = dist(rng);
    return {id};
}

inline std::string generate(
    CharLM& model,
    const CharTokenizer& tok,
    const std::string& prompt,
    const int64_t max_new_tokens,
    const torch::Device& device,
    const double temperature = 1.0,
    const bool greedy = true,
    const uint64_t seed = 1234) {

    model->to(device);
    model->eval();

    std::mt19937_64 rng(seed);

    std::vector<int64_t> ids = tok.encode(prompt);
    if (ids.empty()) {
        throw std::runtime_error("generate: prompt produced no tokens");
    }

    torch::NoGradGuard ng;
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        auto x = torch::from_blob(ids.data(), {1, static_cast<int64_t>(ids.size())}, torch::kInt64).clone().to(device);
        auto logits = model->forward(x); // [1, T, V]
        auto last = logits.index({0, -1}); // [V]
        const auto next = sample_next_token(last, rng, temperature, greedy);
        ids.push_back(next[0]);
    }

    return tok.decode(ids);
}

inline void save_checkpoint(const CharLM& model, const std::string& path) {
    torch::save(model, path);
}

inline CharLM load_checkpoint(int64_t vocab_size, int64_t emb, int64_t hidden, int64_t layers, const std::string& path, const torch::Device& device) {
    CharLM model(vocab_size, emb, hidden, layers);
    torch::load(model, path);
    model->to(device);
    model->eval();
    return model;
}

} // namespace mcppfa::torchlm
