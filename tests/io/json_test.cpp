#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>

#include "core/error.hpp"
#include "harness/test.hpp"
#include "io/file.hpp"
#include "io/json.hpp"

namespace {

using titan::io::JsonValue;
using titan::io::parse_json;
using titan::io::to_json;

std::string error_of(const std::string& text) {
    try {
        parse_json(text);
    } catch (const titan::Error& e) {
        return e.what();
    }
    return "";
}

const std::string kBs = "\\";

// A JSON string literal whose content is a backslash followed by `body`.
std::string quoted(const std::string& body) { return "\"" + kBs + body + "\""; }

// Returns a copy: as_string() on a temporary would dangle in the macros.
std::string parsed_string(const std::string& text) {
    return parse_json(text).as_string();
}

bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

TITAN_TEST(Json, ParsesAllKinds) {
    const JsonValue v = parse_json(R"({"n":null,"t":true,"f":false,"x":-1.5e2,
        "s":"a\"b\\c\/\n\t","a":[1,2,[]],"o":{}})");
    TITAN_EXPECT(v.is_object());
    TITAN_EXPECT(v.at("n").is_null());
    TITAN_EXPECT(v.at("t").as_bool());
    TITAN_EXPECT(!v.at("f").as_bool());
    TITAN_EXPECT_EQ(v.at("x").as_number(), -150.0);
    TITAN_EXPECT_EQ(v.at("s").as_string(), std::string("a\"b\\c/\n\t"));
    TITAN_EXPECT_EQ(v.at("a").as_array().size(), std::size_t{3});
    TITAN_EXPECT_EQ(v.at("a").as_array()[1].as_int(), std::int64_t{2});
    TITAN_EXPECT(v.at("o").as_object().empty());
    TITAN_EXPECT(v.find("missing") == nullptr);
    TITAN_EXPECT_EQ(v.as_object().front().first, std::string("n"));
    TITAN_EXPECT_EQ(v.as_object().back().first, std::string("o"));
    TITAN_EXPECT(parse_json("  7 ").is_number());
}

TITAN_TEST(Json, RoundTrip) {
    JsonValue v = JsonValue::object();
    v.set("name",
          JsonValue::string("tab\t \"q\" \\ \x01 \xC3\xA9 \xF0\x9F\x98\x80"));
    v.set("int", JsonValue::number(42));
    v.set("frac", JsonValue::number(0.1));
    v.set("big", JsonValue::number(1e300));
    v.set("neg", JsonValue::number(-3.25e-8));
    JsonValue arr = JsonValue::array();
    arr.push_back(JsonValue());
    arr.push_back(JsonValue::boolean(true));
    arr.push_back(JsonValue::array());
    arr.push_back(JsonValue::object());
    v.set("list", arr);
    v.set("int", JsonValue::number(43));  // replaces, keeps position

    TITAN_EXPECT_EQ(v.as_object().size(), std::size_t{6});
    TITAN_EXPECT_EQ(v.at("int").as_int(), std::int64_t{43});
    for (const int indent : {2, 0, 4}) {
        const std::string text = to_json(v, indent);
        const JsonValue back = parse_json(text);
        TITAN_EXPECT_EQ(to_json(back, indent), text);
        TITAN_EXPECT_EQ(back.at("name").as_string(), v.at("name").as_string());
        TITAN_EXPECT_EQ(back.at("frac").as_number(), 0.1);
        TITAN_EXPECT_EQ(back.at("big").as_number(), 1e300);
        TITAN_EXPECT_EQ(back.at("neg").as_number(), -3.25e-8);
        TITAN_EXPECT_EQ(back.at("list").as_array().size(), std::size_t{4});
    }
    TITAN_EXPECT_EQ(to_json(JsonValue::number(3), 0), std::string("3"));
    TITAN_EXPECT_EQ(to_json(parse_json("[1,{}]"), 0), std::string("[1,{}]"));
    TITAN_EXPECT_EQ(to_json(parse_json(R"({"a":[1]})")),
                    std::string("{\n  \"a\": [\n    1\n  ]\n}"));
}

TITAN_TEST(Json, UnicodeEscapes) {
    TITAN_EXPECT_EQ(parsed_string(quoted("u00e9")), std::string("\xC3\xA9"));
    TITAN_EXPECT_EQ(parsed_string(quoted("ud83d" + kBs + "ude00")),
                    std::string("\xF0\x9F\x98\x80"));
    TITAN_EXPECT_EQ(parsed_string(quoted("u0041" + kBs + "u20AC")),
                    std::string("A\xE2\x82\xAC"));
    TITAN_EXPECT_EQ(parsed_string(R"("\u0000")"), std::string(1, '\0'));
}

TITAN_TEST(Json, RejectsMalformed) {
    const std::string cases[] = {
        R"({"a":1,})",
        "[1 2]",
        R"("\ud83d")",
        quoted("ud83d" + kBs + "u0041"),
        R"("\ude00")",
        "01",
        R"({"a":1}x)",
        R"({"a":1,"a":2})",
        std::string(65, '[') + "1" + std::string(65, ']'),
        R"("abc)",
        "\"a\nb\"",
        "",
        "-",
        "1.",
        "1e",
        ".5",
        "+1",
        "1e999",
        "tru",
        "[1,]",
        R"({"a" 1})",
        R"({a:1})",
        R"("\x")",
        R"("\u12G4")",
        "\"a\x01\"",
        "[",
    };
    for (const std::string& c : cases) {
        const std::string msg = error_of(c);
        TITAN_EXPECT(has(msg, "json: "));
        TITAN_EXPECT(has(msg, "line "));
        TITAN_EXPECT(has(msg, "column "));
    }
    TITAN_EXPECT(
        has(error_of("{\n  \"a\": 1,\n  \"a\": 2}"), "line 3, column 3"));
    TITAN_EXPECT(has(error_of("[1 2]"), "line 1, column 4"));
    // 64 levels are fine.
    const std::string deep = std::string(64, '[') + std::string(64, ']');
    TITAN_EXPECT(parse_json(deep).is_array());
}

TITAN_TEST(Json, AccessorKindErrors) {
    const JsonValue s = JsonValue::string("x");
    try {
        s.as_number();
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(has(e.what(), "expected number"));
        TITAN_EXPECT(has(e.what(), "got string"));
    }
    try {
        JsonValue().as_object();
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(has(e.what(), "expected object"));
        TITAN_EXPECT(has(e.what(), "got null"));
    }
    TITAN_EXPECT_THROWS(JsonValue::number(1).as_string(), titan::Error);
    TITAN_EXPECT_THROWS(JsonValue::array().find("k"), titan::Error);
    TITAN_EXPECT_THROWS(JsonValue::number(1).push_back(JsonValue()),
                        titan::Error);
    TITAN_EXPECT_THROWS(JsonValue::array().set("k", JsonValue()), titan::Error);
    TITAN_EXPECT_THROWS(JsonValue::number(1.5).as_int(), titan::Error);
    TITAN_EXPECT_THROWS(JsonValue::number(1e16).as_int(), titan::Error);
    TITAN_EXPECT_EQ(JsonValue::number(-9007199254740992.0).as_int(),
                    std::int64_t{-9007199254740992});
    try {
        JsonValue::object().at("nope");
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(has(e.what(), "nope"));
    }
}

TITAN_TEST(File, MissingThrowsWithPath) {
    const std::filesystem::path p = "definitely_missing_dir/no_such_file.txt";
    try {
        titan::io::read_file(p);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(has(e.what(), "no_such_file.txt"));
    }
}

TITAN_TEST(File, RoundTripAndLimit) {
    const auto p =
        std::filesystem::temp_directory_path() / "titan_io_file_test.bin";
    const std::string content("a\0b\r\n\xff", 6);
    titan::io::write_file(p, content);
    TITAN_EXPECT_EQ(titan::io::read_file(p), content);
    try {
        titan::io::read_file(p, 3);
        TITAN_EXPECT(false);
    } catch (const titan::Error& e) {
        TITAN_EXPECT(has(e.what(), "titan_io_file_test.bin"));
    }
    std::filesystem::remove(p);
}

TITAN_TEST(Json, SpecificRejections) {
    const std::string too_deep =
        std::string(65, '[') + "1" + std::string(65, ']');
    TITAN_EXPECT(has(error_of(too_deep), "nesting deeper than 64"));
    TITAN_EXPECT(has(error_of(std::string(65, '[')), "nesting deeper than 64"));
    TITAN_EXPECT(has(error_of("01"), "leading zero"));
    TITAN_EXPECT(has(error_of("[01]"), "leading zero"));
    TITAN_EXPECT(has(error_of("-01"), "leading zero"));
    TITAN_EXPECT(has(error_of("1e999"), "number out of range"));
    TITAN_EXPECT(has(error_of("-1e999"), "number out of range"));
}

TITAN_TEST(Json, InvalidUtf8InStringsIsRejected) {
    const std::string cases[] = {
        "\"\xC3\"",              // truncated sequence
        "\"\xC0\xAF\"",          // overlong
        "\"\xED\xA0\x80\"",      // encoded surrogate
        "\"\x80\"",              // stray continuation
        "\"\xF4\x90\x80\x80\"",  // above U+10FFFF
    };
    for (const std::string& c : cases) {
        const std::string msg = error_of("[\n" + c + "]");
        TITAN_EXPECT(has(msg, "invalid UTF-8"));
        TITAN_EXPECT(has(msg, "line 2, column 2"));
    }
    TITAN_EXPECT_EQ(parsed_string("\"\xC3\xA9\xF0\x9F\x98\x80\""),
                    std::string("\xC3\xA9\xF0\x9F\x98\x80"));
}

TITAN_TEST(Json, UnderflowIsAccepted) {
    TITAN_EXPECT_EQ(parse_json("1e-400").as_number(), 0.0);
    TITAN_EXPECT_EQ(parse_json("-1e-400").as_number(), 0.0);
    TITAN_EXPECT(std::signbit(parse_json("-1e-400").as_number()));
    TITAN_EXPECT_EQ(parse_json("0.000e-9999999999999").as_number(), 0.0);
    const double sub = parse_json("5e-324").as_number();
    TITAN_EXPECT(sub > 0.0 && sub < 1e-300);
}

TITAN_TEST(Json, LargeObjectParsesInLinearTime) {
    std::string text = "{";
    for (int i = 0; i < 20000; ++i) {
        if (i > 0) text += ',';
        text += "\"key" + std::to_string(i) + "\":" + std::to_string(i);
    }
    text += "}";
    const JsonValue v = parse_json(text);
    TITAN_EXPECT_EQ(v.as_object().size(), std::size_t{20000});
    TITAN_EXPECT_EQ(v.at("key19999").as_int(), std::int64_t{19999});
    TITAN_EXPECT(has(error_of(text.substr(0, text.size() - 1) + ",\"key7\":1}"),
                     "duplicate key"));
}
