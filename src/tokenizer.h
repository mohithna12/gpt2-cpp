#pragma once
// GPT-2 byte-level BPE tokenizer. Produces the same ids as Hugging Face's GPT-2 tokenizer
// (checked by tests/test_tokenizer.cpp against ids generated with HF).

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gpt2 {

class Tokenizer {
public:
    static constexpr int kEndOfText = 50256;  // "<|endoftext|>"

    // Loads models/vocab.json and models/merges.txt as written by scripts/export_gpt2.py.
    // Throws std::runtime_error with a readable message on any format problem.
    static Tokenizer load(const std::string& vocab_json, const std::string& merges_txt);

    // Like HF's default: a literal "<|endoftext|>" in the text becomes kEndOfText, and no
    // BOS/EOS is added. Accepts any bytes; invalid UTF-8 is tokenized byte by byte.
    std::vector<int> encode(std::string_view text) const;

    // Exact inverse of encode on bytes (no replacement characters). Throws on a bad id.
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    // Tokens are stored in their byte-level form (e.g. "Ġthe"), as in vocab.json.
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> merge_rank_;  // "left right" -> rank
    std::string byte_to_unicode_[256];                 // UTF-8 of each byte's stand-in
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;

    void encode_piece(std::string_view piece, std::vector<int>& out) const;
};

// Pre-tokenizer: splits text into the pieces GPT-2's regex would produce. Exposed for tests.
std::vector<std::string_view> pretokenize(std::string_view text);

}  // namespace gpt2
