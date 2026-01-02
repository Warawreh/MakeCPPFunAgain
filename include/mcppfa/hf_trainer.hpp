#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mcppfa/hf_dataset.hpp"
#include "mcppfa/sentencepiece_lite.hpp"

namespace mcppfa {

struct TrainingArguments {
    int64_t max_len = 32;
    int64_t pad_id = 0;
    int64_t batch_size = 32;
    int epochs = 10;
    double lr = 1e-3;

    // Columns
    std::string input_col = "state";
    std::string label_col = "very_short_result";

    // Enforce one-token target (your requirement)
    bool one_token_target = true;

    // Device
    torch::Device device = torch::kCPU;
};

struct TrainerMetrics {
    double loss = 0.0;
    double acc = 0.0;
};

struct TrainerState {
    int epoch = 0;
};

template <typename ModelT>
class Trainer {
public:
    Trainer(ModelT& model,
            const spm_lite::SentencePieceLite& tokenizer,
            TrainingArguments args)
        : model_(model), tokenizer_(tokenizer), args_(std::move(args)) {
        if (args_.max_len <= 0) throw std::runtime_error("TrainingArguments.max_len must be > 0");
        if (args_.batch_size <= 0) throw std::runtime_error("TrainingArguments.batch_size must be > 0");
        if (args_.epochs <= 0) throw std::runtime_error("TrainingArguments.epochs must be > 0");
        if (args_.lr <= 0.0) throw std::runtime_error("TrainingArguments.lr must be > 0");
    }

    struct TokenizedSplit {
        torch::Tensor X; // [N, max_len] int64
        torch::Tensor y; // [N] int64
        int64_t n = 0;
    };

    void set_splits(const hf_dataset::Table& train,
                    const hf_dataset::Table& valid,
                    const hf_dataset::Table& test) {
        train_tbl_ = &train;
        valid_tbl_ = &valid;
        test_tbl_ = &test;
    }

    void tokenize_splits_once() {
        if (!train_tbl_ || !valid_tbl_ || !test_tbl_) {
            throw std::runtime_error("Trainer: splits not set");
        }
        std::printf("Trainer: tokenizing train split...\n");
        std::fflush(stdout);
        train_tok_ = build_split(*train_tbl_);
        std::printf("Trainer: tokenizing valid split...\n");
        std::fflush(stdout);
        valid_tok_ = build_split(*valid_tbl_);
        std::printf("Trainer: tokenizing test split...\n");
        std::fflush(stdout);
        test_tok_ = build_split(*test_tbl_);
        tokenized_ = true;

        std::printf("Trainer: tokenized sizes train=%lld valid=%lld test=%lld\n",
                    (long long)train_tok_.n, (long long)valid_tok_.n, (long long)test_tok_.n);
        std::fflush(stdout);
    }

    TrainerMetrics train_epoch(torch::optim::Optimizer& opt) {
        ensure_tokenized();
        model_->train();

        // Decoder input length is 1 for your one-token objective.
        const int64_t dec_len = 1;
        auto decoder_full = torch::full({args_.batch_size, dec_len}, args_.pad_id,
                                        torch::TensorOptions().dtype(torch::kInt64).device(args_.device));

        double sum_loss = 0.0;
        int64_t correct = 0;
        const int64_t N = train_tok_.n;

        for (int64_t i = 0; i < N; i += args_.batch_size) {
            const int64_t b = std::min<int64_t>(args_.batch_size, N - i);
            auto xb = train_tok_.X.narrow(0, i, b);
            auto yb = train_tok_.y.narrow(0, i, b);
            auto db = decoder_full.narrow(0, 0, b);

            auto logits = model_->forward(xb, db); // [B,1,V]
            auto logits0 = logits.select(1, 0);    // [B,V]
            auto loss = torch::nn::functional::cross_entropy(logits0, yb);

            opt.zero_grad();
            loss.backward();
            opt.step();

            sum_loss += loss.template item<double>() * static_cast<double>(b);
            auto pred = logits0.argmax(-1);
            correct += pred.eq(yb).sum().template item<int64_t>();
        }

        TrainerMetrics m;
        m.loss = sum_loss / static_cast<double>(N);
        m.acc = static_cast<double>(correct) / static_cast<double>(N);
        return m;
    }

    TrainerMetrics evaluate_valid() {
        ensure_tokenized();
        return evaluate_(valid_tok_);
    }

    TrainerMetrics evaluate_test() {
        ensure_tokenized();
        return evaluate_(test_tok_);
    }

    void print_one_example_valid(const char* tag = "Valid") {
        ensure_tokenized();
        print_one_example_(valid_tok_, tag);
    }

    void print_one_example_test(const char* tag = "Test") {
        ensure_tokenized();
        print_one_example_(test_tok_, tag);
    }

    // HF-like: train loop with eval each epoch.
    void train() {
        ensure_tokenized();
        torch::optim::Adam optimizer(model_->parameters(), torch::optim::AdamOptions(args_.lr));

        std::printf("\nTraining (ONE-TOKEN target) with mini-batches...\n");
        std::fflush(stdout);
        for (int e = 0; e < args_.epochs; ++e) {
            state_.epoch = e + 1;
            const auto train_m = train_epoch(optimizer);
            const auto valid_m = evaluate_valid();
            std::printf("Epoch %d/%d train_loss=%.6f train_acc=%.6f valid_loss=%.6f valid_acc=%.6f\n",
                        (e + 1), args_.epochs, train_m.loss, train_m.acc, valid_m.loss, valid_m.acc);
            std::fflush(stdout);
            print_one_example_valid("Valid");
            std::fflush(stdout);
        }

        const auto test_m = evaluate_test();
        std::printf("Test loss: %.6f test_acc=%.6f\n", test_m.loss, test_m.acc);
        std::fflush(stdout);
        print_one_example_test("Test");
        std::fflush(stdout);
    }

    const TokenizedSplit& train_split() const { return train_tok_; }
    const TokenizedSplit& valid_split() const { return valid_tok_; }
    const TokenizedSplit& test_split() const { return test_tok_; }

private:
    ModelT& model_;
    const spm_lite::SentencePieceLite& tokenizer_;
    TrainingArguments args_;
    TrainerState state_;

    const hf_dataset::Table* train_tbl_ = nullptr;
    const hf_dataset::Table* valid_tbl_ = nullptr;
    const hf_dataset::Table* test_tbl_ = nullptr;

    TokenizedSplit train_tok_;
    TokenizedSplit valid_tok_;
    TokenizedSplit test_tok_;
    bool tokenized_ = false;

    static int col_index_(const hf_dataset::Table& t, const std::string& name) {
        for (size_t i = 0; i < t.columns.size(); ++i) {
            if (t.columns[i] == name) return static_cast<int>(i);
        }
        return -1;
    }

    int64_t label_token_id_(const std::string& label_text) const {
        std::vector<int64_t> ids = tokenizer_.encode(label_text);
        for (auto id : ids) {
            if (id != args_.pad_id) return id;
        }
        return args_.pad_id;
    }

    TokenizedSplit build_split(const hf_dataset::Table& tbl) const {
        const int s_col = col_index_(tbl, args_.input_col);
        const int y_col = col_index_(tbl, args_.label_col);
        if (s_col < 0 || y_col < 0) throw std::runtime_error("Trainer: table missing required columns");

        int64_t n = 0;
        for (const auto& row : tbl.rows) {
            if (static_cast<int>(row.size()) <= std::max(s_col, y_col)) continue;
            const int64_t y_id = label_token_id_(row[y_col]);
            if (y_id != args_.pad_id) ++n;
        }
        if (n <= 0) throw std::runtime_error("Trainer: no usable rows (labels empty/pad?)");

        auto X_cpu = torch::empty({n, args_.max_len}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto y_cpu = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto X = X_cpu.template accessor<int64_t, 2>();
        auto y = y_cpu.template accessor<int64_t, 1>();

        int64_t j = 0;
        for (const auto& row : tbl.rows) {
            if (static_cast<int>(row.size()) <= std::max(s_col, y_col)) continue;
            const std::string& x_txt = row[s_col];
            const std::string& y_txt = row[y_col];
            const int64_t y_id = label_token_id_(y_txt);
            if (y_id == args_.pad_id) continue;

            for (int64_t t = 0; t < args_.max_len; ++t) X[j][t] = args_.pad_id;

            auto ids = tokenizer_.encode(x_txt);
            const int64_t L = std::min<int64_t>(static_cast<int64_t>(ids.size()), args_.max_len);
            for (int64_t t = 0; t < L; ++t) X[j][t] = ids[static_cast<size_t>(t)];

            y[j] = y_id;
            ++j;
        }

        TokenizedSplit out;
        out.n = n;
        if (args_.device.is_cuda()) {
            out.X = X_cpu.to(args_.device, /*non_blocking=*/false);
            out.y = y_cpu.to(args_.device, /*non_blocking=*/false);
        } else {
            out.X = X_cpu;
            out.y = y_cpu;
        }
        return out;
    }

    TrainerMetrics evaluate_(const TokenizedSplit& split) {
        model_->eval();
        torch::NoGradGuard ng;

        const int64_t dec_len = 1;
        auto decoder_full = torch::full({args_.batch_size, dec_len}, args_.pad_id,
                                        torch::TensorOptions().dtype(torch::kInt64).device(args_.device));

        double sum_loss = 0.0;
        int64_t correct = 0;
        const int64_t N = split.n;

        for (int64_t i = 0; i < N; i += args_.batch_size) {
            const int64_t b = std::min<int64_t>(args_.batch_size, N - i);
            auto xb = split.X.narrow(0, i, b);
            auto yb = split.y.narrow(0, i, b);
            auto db = decoder_full.narrow(0, 0, b);

            auto logits = model_->forward(xb, db);
            auto logits0 = logits.select(1, 0);
            auto loss = torch::nn::functional::cross_entropy(logits0, yb);
            sum_loss += loss.template item<double>() * static_cast<double>(b);

            auto pred = logits0.argmax(-1);
            correct += pred.eq(yb).sum().template item<int64_t>();
        }

        TrainerMetrics m;
        m.loss = sum_loss / static_cast<double>(N);
        m.acc = static_cast<double>(correct) / static_cast<double>(N);
        return m;
    }

    void print_one_example_(const TokenizedSplit& split, const char* tag) {
        model_->eval();
        torch::NoGradGuard ng;
        if (split.n <= 0) return;

        const int64_t dec_len = 1;
        auto decoder = torch::full({1, dec_len}, args_.pad_id,
                                  torch::TensorOptions().dtype(torch::kInt64).device(args_.device));

        auto xb = split.X.narrow(0, 0, 1);
        const int64_t y_id = split.y[0].template item<int64_t>();

        auto logits = model_->forward(xb, decoder);
        auto logits0 = logits.select(1, 0);
        const int64_t pred_id = logits0.argmax(-1).template item<int64_t>();

        const auto pred_txt = tokenizer_.decode(std::vector<int64_t>{pred_id});
        const auto true_txt = tokenizer_.decode(std::vector<int64_t>{y_id});

        std::printf("%s one-token check: pred='%s' true='%s' match=%s\n",
                    tag, pred_txt.c_str(), true_txt.c_str(), (pred_id == y_id ? "true" : "false"));
    }

    void ensure_tokenized() const {
        if (!tokenized_) throw std::runtime_error("Trainer: call tokenize_splits_once() first");
    }
};

} // namespace mcppfa
