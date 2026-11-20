#include "io/json.hpp"

#include <charconv>
#include <cmath>
#include <system_error>

#include "core/error.hpp"

namespace titan::io {

namespace {

constexpr int kMaxDepth = 64;
// Largest magnitude below which every integer is exactly representable.
constexpr double kMaxExactInt = 9007199254740992.0;  // 2^53

const char* kind_name(const JsonValue& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return "bool";
    if (v.is_number()) return "number";
    if (v.is_string()) return "string";
    if (v.is_array()) return "array";
    return "object";
}

[[noreturn]] void kind_error(const char* expected, const JsonValue& actual) {
    throw_error(__FILE__, __LINE__,
                std::string("json: expected ") + expected + ", got " +
                    kind_name(actual));
}

// ---------------------------------------------------------------------------
// Parser: recursive descent over the RFC 8259 grammar.
// ---------------------------------------------------------------------------

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    JsonValue parse_document() {
        skip_ws();
        JsonValue v = parse_value(0);
        skip_ws();
        if (pos_ < text_.size()) fail("unexpected trailing content");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        fail_at(pos_, message);
    }

    // Line and column are 1-based; the column counts bytes.
    [[noreturn]] void fail_at(std::size_t pos,
                              const std::string& message) const {
        std::size_t line = 1;
        std::size_t line_start = 0;
        for (std::size_t i = 0; i < pos && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                line_start = i + 1;
            }
        }
        throw_error(__FILE__, __LINE__,
                    "json: " + message + " at line " + std::to_string(line) +
                        ", column " + std::to_string(pos - line_start + 1));
    }

    bool at_end() const { return pos_ >= text_.size(); }
    char peek() const { return text_[pos_]; }

    void skip_ws() {
        while (!at_end()) {
            const char c = peek();
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            ++pos_;
        }
    }

    void expect_literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) {
            fail("invalid literal, expected '" + std::string(word) + "'");
        }
        pos_ += word.size();
    }

    JsonValue parse_value(int depth) {
        if (at_end()) fail("unexpected end of input, expected a value");
        switch (peek()) {
            case 'n':
                expect_literal("null");
                return JsonValue();
            case 't':
                expect_literal("true");
                return JsonValue::boolean(true);
            case 'f':
                expect_literal("false");
                return JsonValue::boolean(false);
            case '"':
                return JsonValue::string(parse_string());
            case '[':
                return parse_array(depth);
            case '{':
                return parse_object(depth);
            default:
                break;
        }
        const char c = peek();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(std::string("unexpected character '") + c + "'");
    }

    JsonValue parse_array(int depth) {
        if (depth + 1 > kMaxDepth) {
            fail("nesting deeper than " + std::to_string(kMaxDepth));
        }
        ++pos_;  // '['
        JsonValue result = JsonValue::array();
        skip_ws();
        if (!at_end() && peek() == ']') {
            ++pos_;
            return result;
        }
        for (;;) {
            skip_ws();
            result.push_back(parse_value(depth + 1));
            skip_ws();
            if (at_end()) fail("unexpected end of input in array");
            if (peek() == ',') {
                ++pos_;
            } else if (peek() == ']') {
                ++pos_;
                return result;
            } else {
                fail("expected ',' or ']' in array");
            }
        }
    }

    JsonValue parse_object(int depth) {
        if (depth + 1 > kMaxDepth) {
            fail("nesting deeper than " + std::to_string(kMaxDepth));
        }
        ++pos_;  // '{'
        JsonValue result = JsonValue::object();
        skip_ws();
        if (!at_end() && peek() == '}') {
            ++pos_;
            return result;
        }
        for (;;) {
            skip_ws();
            if (at_end() || peek() != '"') fail("expected a string key");
            const std::size_t key_pos = pos_;
            std::string key = parse_string();
            if (result.find(key) != nullptr) {
                fail_at(key_pos, "duplicate key \"" + key + "\"");
            }
            skip_ws();
            if (at_end() || peek() != ':') fail("expected ':' after key");
            ++pos_;
            skip_ws();
            result.set(std::move(key), parse_value(depth + 1));
            skip_ws();
            if (at_end()) fail("unexpected end of input in object");
            if (peek() == ',') {
                ++pos_;
            } else if (peek() == '}') {
                ++pos_;
                return result;
            } else {
                fail("expected ',' or '}' in object");
            }
        }
    }

    bool digit_at(std::size_t p) const {
        return p < text_.size() && text_[p] >= '0' && text_[p] <= '9';
    }

    // number = [ "-" ] int [ frac ] [ exp ]; int = "0" / digit1-9 *digit
    JsonValue parse_number() {
        const std::size_t start = pos_;
        std::size_t p = pos_;
        if (text_[p] == '-') ++p;
        if (!digit_at(p)) fail_at(p, "invalid number, expected a digit");
        if (text_[p] == '0') {
            ++p;
            if (digit_at(p)) fail_at(p, "invalid number, leading zero");
        } else {
            while (digit_at(p)) ++p;
        }
        if (p < text_.size() && text_[p] == '.') {
            ++p;
            if (!digit_at(p))
                fail_at(p, "invalid number, digit expected after '.'");
            while (digit_at(p)) ++p;
        }
        if (p < text_.size() && (text_[p] == 'e' || text_[p] == 'E')) {
            ++p;
            if (p < text_.size() && (text_[p] == '+' || text_[p] == '-')) ++p;
            if (!digit_at(p))
                fail_at(p, "invalid number, digit expected in exponent");
            while (digit_at(p)) ++p;
        }
        double value = 0.0;
        const char* first = text_.data() + start;
        const char* last = text_.data() + p;
        const auto res = std::from_chars(first, last, value);
        if (res.ec != std::errc() || res.ptr != last || !std::isfinite(value)) {
            fail_at(start, "number out of range");
        }
        pos_ = p;
        return JsonValue::number(value);
    }

    unsigned parse_hex4() {
        if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_ + static_cast<std::size_t>(i)];
            unsigned d = 0;
            if (c >= '0' && c <= '9') {
                d = static_cast<unsigned>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                d = static_cast<unsigned>(c - 'a') + 10;
            } else if (c >= 'A' && c <= 'F') {
                d = static_cast<unsigned>(c - 'A') + 10;
            } else {
                fail_at(pos_ + static_cast<std::size_t>(i),
                        "invalid hex digit in \\u escape");
            }
            v = v * 16 + d;
        }
        pos_ += 4;
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
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

    std::string parse_string() {
        const std::size_t open = pos_;
        ++pos_;  // opening quote
        std::string out;
        for (;;) {
            if (at_end()) fail_at(open, "unterminated string");
            const unsigned char c = static_cast<unsigned char>(peek());
            if (c == '"') {
                ++pos_;
                return out;
            }
            if (c < 0x20) fail("control character in string");
            if (c != '\\') {
                out += static_cast<char>(c);
                ++pos_;
                continue;
            }
            ++pos_;  // backslash
            if (at_end()) fail_at(open, "unterminated string");
            const char e = peek();
            ++pos_;
            switch (e) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    const std::size_t esc = pos_ - 2;
                    unsigned cp = parse_hex4();
                    if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail_at(esc, "unpaired low surrogate");
                    }
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (text_.substr(pos_, 2) != "\\u") {
                            fail_at(esc, "unpaired high surrogate");
                        }
                        pos_ += 2;
                        const unsigned low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            fail_at(esc, "unpaired high surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    fail_at(pos_ - 1, "invalid escape sequence");
            }
        }
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

void write_string(std::string& out, const std::string& s) {
    static constexpr char kHex[] = "0123456789abcdef";
    out += '"';
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += kHex[c >> 4];
                    out += kHex[c & 0xF];
                } else {
                    out += ch;  // UTF-8 passes through unchanged
                }
        }
    }
    out += '"';
}

void write_number(std::string& out, double v) {
    TITAN_CHECK(std::isfinite(v), "to_json: cannot write a non-finite number");
    char buf[64];
    std::to_chars_result res{};
    if (v == std::trunc(v) && std::fabs(v) < 1e15) {
        res =
            std::to_chars(buf, buf + sizeof buf, static_cast<std::int64_t>(v));
    } else {
        res = std::to_chars(buf, buf + sizeof buf, v);  // shortest round-trip
    }
    out.append(buf, res.ptr);
}

void write_value(std::string& out, const JsonValue& v, int indent, int level) {
    const bool pretty = indent > 0;
    const auto newline = [&](int lvl) {
        if (!pretty) return;
        out += '\n';
        out.append(
            static_cast<std::size_t>(lvl) * static_cast<std::size_t>(indent),
            ' ');
    };
    if (v.is_null()) {
        out += "null";
    } else if (v.is_bool()) {
        out += v.as_bool() ? "true" : "false";
    } else if (v.is_number()) {
        write_number(out, v.as_number());
    } else if (v.is_string()) {
        write_string(out, v.as_string());
    } else if (v.is_array()) {
        const JsonValue::Array& items = v.as_array();
        if (items.empty()) {
            out += "[]";
            return;
        }
        out += '[';
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i > 0) out += ',';
            newline(level + 1);
            write_value(out, items[i], indent, level + 1);
        }
        newline(level);
        out += ']';
    } else {
        const JsonValue::Object& members = v.as_object();
        if (members.empty()) {
            out += "{}";
            return;
        }
        out += '{';
        for (std::size_t i = 0; i < members.size(); ++i) {
            if (i > 0) out += ',';
            newline(level + 1);
            write_string(out, members[i].first);
            out += pretty ? ": " : ":";
            write_value(out, members[i].second, indent, level + 1);
        }
        newline(level);
        out += '}';
    }
}

}  // namespace

JsonValue JsonValue::boolean(bool value) {
    JsonValue v;
    v.data_ = value;
    return v;
}

JsonValue JsonValue::number(double value) {
    JsonValue v;
    v.data_ = value;
    return v;
}

JsonValue JsonValue::string(std::string value) {
    JsonValue v;
    v.data_ = std::move(value);
    return v;
}

JsonValue JsonValue::array() {
    JsonValue v;
    v.data_ = Array{};
    return v;
}

JsonValue JsonValue::object() {
    JsonValue v;
    v.data_ = Object{};
    return v;
}

bool JsonValue::is_null() const {
    return std::holds_alternative<std::monostate>(data_);
}
bool JsonValue::is_bool() const { return std::holds_alternative<bool>(data_); }
bool JsonValue::is_number() const {
    return std::holds_alternative<double>(data_);
}
bool JsonValue::is_string() const {
    return std::holds_alternative<std::string>(data_);
}
bool JsonValue::is_array() const {
    return std::holds_alternative<Array>(data_);
}
bool JsonValue::is_object() const {
    return std::holds_alternative<Object>(data_);
}

bool JsonValue::as_bool() const {
    if (!is_bool()) kind_error("bool", *this);
    return std::get<bool>(data_);
}

double JsonValue::as_number() const {
    if (!is_number()) kind_error("number", *this);
    return std::get<double>(data_);
}

std::int64_t JsonValue::as_int() const {
    const double v = as_number();
    TITAN_CHECK(v == std::trunc(v),
                "json: expected an integer, got " + std::to_string(v));
    TITAN_CHECK(std::fabs(v) <= kMaxExactInt,
                "json: integer " + std::to_string(v) + " is outside +-2^53");
    return static_cast<std::int64_t>(v);
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) kind_error("string", *this);
    return std::get<std::string>(data_);
}

const JsonValue::Array& JsonValue::as_array() const {
    if (!is_array()) kind_error("array", *this);
    return std::get<Array>(data_);
}

const JsonValue::Object& JsonValue::as_object() const {
    if (!is_object()) kind_error("object", *this);
    return std::get<Object>(data_);
}

const JsonValue* JsonValue::find(std::string_view key) const {
    for (const auto& member : as_object()) {
        if (member.first == key) return &member.second;
    }
    return nullptr;
}

const JsonValue& JsonValue::at(std::string_view key) const {
    const JsonValue* v = find(key);
    TITAN_CHECK(v != nullptr, "json: missing key \"" + std::string(key) + "\"");
    return *v;
}

void JsonValue::push_back(JsonValue value) {
    if (!is_array()) kind_error("array", *this);
    std::get<Array>(data_).push_back(std::move(value));
}

void JsonValue::set(std::string key, JsonValue value) {
    if (!is_object()) kind_error("object", *this);
    Object& members = std::get<Object>(data_);
    for (auto& member : members) {
        if (member.first == key) {
            member.second = std::move(value);
            return;
        }
    }
    members.emplace_back(std::move(key), std::move(value));
}

JsonValue parse_json(std::string_view text) {
    return Parser(text).parse_document();
}

std::string to_json(const JsonValue& value, int indent) {
    std::string out;
    write_value(out, value, indent, 0);
    return out;
}

}  // namespace titan::io
