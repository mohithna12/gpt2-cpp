#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "unicode_tables.h"

namespace gpt2 {

namespace {

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error(path + ": " + msg);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(path, "cannot open file");
    return {std::istreambuf_iterator<char>(f), {}};
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

constexpr uint32_t kInvalid = 0xFFFFFFFF;

// Decodes one code point at s[i]. Invalid or truncated sequences (including overlong forms
// and surrogates) yield kInvalid with length 1, so every byte is consumed exactly once.
uint32_t decode_utf8(std::string_view s, size_t i, size_t* len) {
    const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = b(i);
    *len = 1;
    if (c < 0x80) return c;
    size_t n;
    uint32_t cp, min;
    if ((c & 0xE0) == 0xC0) {
        n = 2, cp = c & 0x1F, min = 0x80;
    } else if ((c & 0xF0) == 0xE0) {
        n = 3, cp = c & 0x0F, min = 0x800;
    } else if ((c & 0xF8) == 0xF0) {
        n = 4, cp = c & 0x07, min = 0x10000;
    } else {
        return kInvalid;
    }
    if (i + n > s.size()) return kInvalid;
    for (size_t k = 1; k < n; ++k) {
        if ((b(i + k) & 0xC0) != 0x80) return kInvalid;
        cp = cp << 6 | (b(i + k) & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp < 0xE000)) return kInvalid;
    *len = n;
    return cp;
}

enum class Cls { Letter, Number, Space, Other };

template <size_t N>
bool in_table(const unicode::Range (&table)[N], uint32_t cp) {
    const auto* it = std::upper_bound(std::begin(table), std::end(table), cp,
                                      [](uint32_t v, const unicode::Range& r) { return v < r.lo; });
    return it != std::begin(table) && cp <= (it - 1)->hi;
}

Cls classify(uint32_t cp) {
    if (cp == kInvalid) return Cls::Other;
    if (in_table(unicode::kLetter, cp)) return Cls::Letter;
    if (in_table(unicode::kNumber, cp)) return Cls::Number;
    if (in_table(unicode::kSpace, cp)) return Cls::Space;
    return Cls::Other;
}

// Code point at s[i] with its class and byte length.
struct Char {
    Cls cls;
    size_t len;
};
Char char_at(std::string_view s, size_t i) {
    size_t len;
    const uint32_t cp = decode_utf8(s, i, &len);
    return {classify(cp), len};
}

// Extends a run of `cls` characters starting at byte i; returns the end byte offset.
size_t run_end(std::string_view s, size_t i, Cls cls) {
    while (i < s.size()) {
        const Char c = char_at(s, i);
        if (c.cls != cls) break;
        i += c.len;
    }
    return i;
}

constexpr std::string_view kEndOfTextStr = "<|endoftext|>";

}  // namespace

// Implements 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// with leftmost-first alternation, scanning code points.
std::vector<std::string_view> pretokenize(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t end = 0;
        if (s[i] == '\'') {
            for (std::string_view suf : {"s", "t", "re", "ve", "m", "ll", "d"}) {
                if (s.substr(i + 1, suf.size()) == suf) {
                    end = i + 1 + suf.size();
                    break;
                }
            }
        }
        if (!end) {
            // ` ?X+`: the optional prefix is only U+0020, and only if a non-space follows.
            const size_t j = (s[i] == ' ' && i + 1 < s.size()) ? i + 1 : i;
            const Cls c = char_at(s, j).cls;
            if (c != Cls::Space) {
                end = run_end(s, j, c);
            } else {
                // \s+(?!\S) takes the whole run at end of text, otherwise all but its last
                // character (which then prefixes the next word). A single space falls
                // through to \s+.
                size_t last = i;
                end = i;
                while (end < s.size()) {
                    const Char ch = char_at(s, end);
                    if (ch.cls != Cls::Space) break;
                    last = end;
                    end += ch.len;
                }
                if (end < s.size() && last > i) end = last;
            }
        }
        out.push_back(s.substr(i, end - i));
        i = end;
    }
    return out;
}

Tokenizer Tokenizer::load(const std::string& vocab_json, const std::string& merges_txt) {
    Tokenizer t;

    // Byte -> printable stand-in character, as in GPT-2's bytes_to_unicode(): printable
    // Latin-1 bytes map to themselves, the rest to U+0100 onwards in byte order.
    uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
        const uint32_t cp = printable ? static_cast<uint32_t>(b) : next++;
        append_utf8(t.byte_to_unicode_[b], cp);
        t.unicode_to_byte_[cp] = static_cast<unsigned char>(b);
    }

    // vocab.json: a flat {"token": id, ...} object.
    const std::string js = read_file(vocab_json);
    size_t p = 0;
    const auto ws = [&] {
        while (p < js.size() && (js[p] == ' ' || js[p] == '\n' || js[p] == '\r' || js[p] == '\t'))
            ++p;
    };
    const auto expect = [&](char c) {
        ws();
        if (p >= js.size() || js[p] != c)
            fail(vocab_json, std::string("expected '") + c + "' at byte " + std::to_string(p));
        ++p;
    };
    const auto hex4 = [&]() {
        if (p + 4 > js.size()) fail(vocab_json, "truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = js[p++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= h - '0';
            else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
            else fail(vocab_json, "bad \\u escape");
        }
        return v;
    };
    const auto parse_string = [&]() {
        expect('"');
        std::string out;
        while (true) {
            if (p >= js.size()) fail(vocab_json, "unterminated string");
            const char c = js[p++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (p >= js.size()) fail(vocab_json, "unterminated escape");
            const char e = js[p++];
            switch (e) {
                case '"': case '\\': case '/': out += e; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {  // surrogate pair
                        if (js.compare(p, 2, "\\u") != 0) fail(vocab_json, "lone surrogate");
                        p += 2;
                        const uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo >= 0xE000) fail(vocab_json, "bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        fail(vocab_json, "lone surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail(vocab_json, std::string("bad escape \\") + e);
            }
        }
        return out;
    };

    expect('{');
    ws();
    if (p < js.size() && js[p] == '}') fail(vocab_json, "empty vocabulary");
    while (true) {
        std::string tok = parse_string();
        expect(':');
        ws();
        long id = 0;
        size_t digits = 0;
        while (p < js.size() && js[p] >= '0' && js[p] <= '9' && id <= INT_MAX) {
            id = id * 10 + (js[p++] - '0');
            ++digits;
        }
        if (!digits || id > INT_MAX) fail(vocab_json, "bad id for token '" + tok + "'");
        if (!t.token_to_id_.emplace(std::move(tok), static_cast<int>(id)).second)
            fail(vocab_json, "duplicate token");
        ws();
        if (p < js.size() && js[p] == ',') {
            ++p;
            continue;
        }
        expect('}');
        break;
    }
    ws();
    if (p != js.size()) fail(vocab_json, "trailing data after object");

    t.id_to_token_.resize(t.token_to_id_.size());
    std::vector<bool> seen(t.token_to_id_.size());
    for (const auto& [tok, id] : t.token_to_id_) {
        if (static_cast<size_t>(id) >= seen.size() || seen[id])
            fail(vocab_json, "ids are not a permutation of 0..N-1");
        seen[id] = true;
        t.id_to_token_[id] = tok;
    }
    for (int b = 0; b < 256; ++b)
        if (!t.token_to_id_.count(t.byte_to_unicode_[b]))
            fail(vocab_json, "missing single-byte token for byte " + std::to_string(b));
    const auto eot = t.token_to_id_.find(std::string(kEndOfTextStr));
    if (eot == t.token_to_id_.end() || eot->second != kEndOfText)
        fail(vocab_json, "<|endoftext|> must have id " + std::to_string(kEndOfText));

    // merges.txt: optional "#version" header, then one "left right" pair per line, by rank.
    const std::string ms = read_file(merges_txt);
    size_t line_start = 0, line_no = 0;
    int rank = 0;
    while (line_start < ms.size()) {
        size_t line_end = ms.find('\n', line_start);
        if (line_end == std::string::npos) line_end = ms.size();
        std::string_view line(ms.data() + line_start, line_end - line_start);
        line_start = line_end + 1;
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.empty() || (line_no == 1 && line.substr(0, 8) == "#version")) continue;
        const size_t sp = line.find(' ');
        if (sp == 0 || sp == std::string_view::npos || sp + 1 == line.size() ||
            line.find(' ', sp + 1) != std::string_view::npos)
            fail(merges_txt, "line " + std::to_string(line_no) + " is not 'left right'");
        std::string merged = std::string(line.substr(0, sp)) + std::string(line.substr(sp + 1));
        if (!t.token_to_id_.count(merged))
            fail(merges_txt, "line " + std::to_string(line_no) + ": merge result not in vocab");
        if (!t.merge_rank_.emplace(std::string(line), rank++).second)
            fail(merges_txt, "line " + std::to_string(line_no) + ": duplicate merge");
    }
    if (t.merge_rank_.empty()) fail(merges_txt, "no merges");
    return t;
}

// BPE on one pre-token: start from byte-level symbols and repeatedly merge every occurrence
// of the lowest-ranked adjacent pair, left to right, until no pair has a rank.
void Tokenizer::encode_piece(std::string_view piece, std::vector<int>& out) const {
    std::vector<std::string> syms;
    syms.reserve(piece.size());
    for (unsigned char b : piece) syms.push_back(byte_to_unicode_[b]);

    std::string key;
    while (syms.size() > 1) {
        int best = -1;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            key.assign(syms[i]).append(" ").append(syms[i + 1]);
            const auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && (best < 0 || it->second < best)) {
                best = it->second;
                best_i = i;
            }
        }
        if (best < 0) break;
        const std::string left = syms[best_i], right = syms[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(syms.size());
        for (size_t i = 0; i < syms.size(); ++i) {
            if (i + 1 < syms.size() && syms[i] == left && syms[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(std::move(syms[i]));
            }
        }
        syms = std::move(merged);
    }
    for (const std::string& s : syms) out.push_back(token_to_id_.at(s));
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    std::vector<int> ids;
    while (!text.empty()) {
        const size_t sp = text.find(kEndOfTextStr);
        const std::string_view segment = text.substr(0, sp);
        for (std::string_view piece : pretokenize(segment)) encode_piece(piece, ids);
        if (sp == std::string_view::npos) break;
        ids.push_back(kEndOfText);
        text.remove_prefix(sp + kEndOfTextStr.size());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= vocab_size())
            throw std::runtime_error("decode: token id " + std::to_string(id) + " out of range");
        const std::string& tok = id_to_token_[id];
        for (size_t i = 0; i < tok.size();) {
            size_t len;
            const uint32_t cp = decode_utf8(tok, i, &len);
            const auto it = unicode_to_byte_.find(cp);
            if (it == unicode_to_byte_.end())
                throw std::runtime_error("decode: token " + std::to_string(id) +
                                         " has a character outside the byte alphabet");
            out += static_cast<char>(it->second);
            i += len;
        }
    }
    return out;
}

}  // namespace gpt2
