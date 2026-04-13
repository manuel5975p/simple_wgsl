#include "test_utils.h"

#include <cstring>
#include <string>

using namespace wgsl_test;

TEST(WgslDiagnostics, FreeNullIsNoOp) {
    wgsl_diagnostic_list_free(nullptr);
    SUCCEED();
}

TEST(WgslDiagnostics, SeverityStrings) {
    EXPECT_STREQ("error",   wgsl_diagnostic_severity_string(WGSL_DIAG_ERROR));
    EXPECT_STREQ("warning", wgsl_diagnostic_severity_string(WGSL_DIAG_WARNING));
    EXPECT_STREQ("note",    wgsl_diagnostic_severity_string(WGSL_DIAG_NOTE));
}

TEST(WgslDiagnostics, CodeStrings) {
    EXPECT_STREQ("none",
                 wgsl_diagnostic_code_string(WGSL_DIAG_CODE_NONE));
    EXPECT_STREQ("parse.unexpected_token",
                 wgsl_diagnostic_code_string(WGSL_DIAG_PARSE_UNEXPECTED_TOKEN));
    EXPECT_STREQ("resolve.undefined_symbol",
                 wgsl_diagnostic_code_string(WGSL_DIAG_RESOLVE_UNDEFINED_SYMBOL));
    EXPECT_STREQ("lower.unsupported_builtin",
                 wgsl_diagnostic_code_string(WGSL_DIAG_LOWER_UNSUPPORTED_BUILTIN));
}

/* Spec: Parse syntactically invalid WGSL and assert diagnostics have
 * populated codes and non-empty source spans. */
TEST(WgslDiagnostics, ParseErrorHasCodeAndSpan) {
    const char *src = "fn main( { }";
    WgslParseResult pr = wgsl_parse(src);
    AstGuard guard(pr);

    EXPECT_NE(pr.code, SW_OK);
    ASSERT_NE(pr.diags, nullptr);
    ASSERT_GT(pr.diags->count, 0);

    bool saw_parse_code = false;
    bool saw_nonempty_span = false;
    for (size_t i = 0; i < pr.diags->count; i++) {
        const WgslDiagnostic *d = &pr.diags->items[i];
        if (d->code >= WGSL_DIAG_PARSE_UNEXPECTED_TOKEN &&
            d->code <  WGSL_DIAG_RESOLVE_UNDEFINED_SYMBOL) {
            saw_parse_code = true;
        }
        if (d->code_section_begin && d->code_section_end &&
            d->code_section_end > d->code_section_begin) {
            saw_nonempty_span = true;
        }
    }
    EXPECT_TRUE(saw_parse_code);
    EXPECT_TRUE(saw_nonempty_span);
}

/* Spec: Resolve a program referencing an undefined identifier; assert
 * WGSL_DIAG_RESOLVE_UNDEFINED_SYMBOL and that code_section_begin points
 * at the offending identifier in the source. */
TEST(WgslDiagnostics, ResolveUndefinedSymbolPointsAtIdent) {
    const char *src =
        "fn main() -> i32 {\n"
        "  return nonexistent_variable;\n"
        "}\n";

    WgslParseResult pr = wgsl_parse(src);
    AstGuard ast_guard(pr);
    ASSERT_EQ(pr.code, SW_OK) << "parse should succeed";

    WgslResolveResult rr = wgsl_resolver_build(ast_guard.get());
    ResolverGuard rs_guard(rr);

    EXPECT_NE(rr.code, SW_OK);
    ASSERT_NE(rr.diags, nullptr);
    ASSERT_GT(rr.diags->count, 0);

    bool found = false;
    for (size_t i = 0; i < rr.diags->count; i++) {
        const WgslDiagnostic *d = &rr.diags->items[i];
        if (d->code != WGSL_DIAG_RESOLVE_UNDEFINED_SYMBOL) continue;
        found = true;
        ASSERT_NE(d->code_section_begin, nullptr);
        ASSERT_NE(d->code_section_end, nullptr);
        ASSERT_GE(d->code_section_begin, src);
        size_t len = (size_t)(d->code_section_end - d->code_section_begin);
        std::string got(d->code_section_begin, len);
        EXPECT_EQ(got, "nonexistent_variable")
            << "span should cover the offending identifier, got: " << got;
    }
    EXPECT_TRUE(found) << "expected WGSL_DIAG_RESOLVE_UNDEFINED_SYMBOL";
}

/* Spec: Lower a program using an unsupported builtin; assert
 * WGSL_DIAG_LOWER_UNSUPPORTED_BUILTIN. */
TEST(WgslDiagnostics, LowerUnsupportedBuiltin) {
    const char *src =
        "@fragment fn main(@builtin(not_a_real_builtin) x: f32) {}\n";

    WgslParseResult pr = wgsl_parse(src);
    AstGuard ast_guard(pr);
    ASSERT_EQ(pr.code, SW_OK) << "parse should succeed";

    WgslResolveResult rr = wgsl_resolver_build(ast_guard.get());
    ResolverGuard rs_guard(rr);
    ASSERT_EQ(rr.code, SW_OK) << "resolve should succeed";

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    WgslLowerSpirvResult lsr =
        wgsl_lower_emit_spirv(ast_guard.get(), rs_guard.get(), &opts);

    EXPECT_NE(lsr.code, SW_OK);
    ASSERT_NE(lsr.diags, nullptr);

    bool found = false;
    for (size_t i = 0; i < lsr.diags->count; i++) {
        if (lsr.diags->items[i].code == WGSL_DIAG_LOWER_UNSUPPORTED_BUILTIN) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "expected WGSL_DIAG_LOWER_UNSUPPORTED_BUILTIN";

    wgsl_diagnostic_list_free(lsr.diags);
    if (lsr.words) wgsl_lower_free(lsr.words);
}

/* Spec: Happy-path: wgsl_compile_to_ssir on each .wgsl sample returns
 * SW_OK and diags == NULL or diags->count == 0. */
#ifndef WGSL_SOURCE_DIR
#define WGSL_SOURCE_DIR "."
#endif

namespace {
struct SampleCase { const char *name; };
}  // namespace

class WgslSampleHappyPath : public ::testing::TestWithParam<SampleCase> {};

TEST_P(WgslSampleHappyPath, CompilesToSsirWithoutDiags) {
    const SampleCase &c = GetParam();
    std::string path = std::string(WGSL_SOURCE_DIR) + "/wgsl/" + c.name;
    std::string src = ReadFile(path);
    ASSERT_FALSE(src.empty()) << "could not read " << path;

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    WgslCompileResult cr = wgsl_compile_to_ssir(src.c_str(), &opts);

    if (cr.code != SW_OK) {
        if (cr.diags) {
            for (size_t i = 0; i < cr.diags->count; i++) {
                ADD_FAILURE() << "[" << c.name << "] "
                              << (cr.diags->items[i].message
                                      ? cr.diags->items[i].message
                                      : "(no message)");
            }
        }
    }
    EXPECT_EQ(cr.code, SW_OK) << "sample: " << c.name;
    EXPECT_TRUE(cr.diags == nullptr || cr.diags->count == 0)
        << "sample: " << c.name;

    wgsl_compile_free(&cr);
}

INSTANTIATE_TEST_SUITE_P(
    Samples, WgslSampleHappyPath,
    ::testing::Values(
        SampleCase{"compute_basic.wgsl"},
        SampleCase{"compute_minimal.wgsl"},
        SampleCase{"transitive_vertex.wgsl"},
        SampleCase{"vertex_fragment.wgsl"}),
    [](const ::testing::TestParamInfo<SampleCase> &info) {
        std::string n = info.param.name;
        for (auto &ch : n) {
            if (!isalnum((unsigned char)ch)) ch = '_';
        }
        return n;
    });
