#pragma once

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>

#ifdef _WIN32
#include <process.h>
#define POPEN _popen
#define PCLOSE _pclose
static inline int wgsl_getpid() { return _getpid(); }
#else
#include <unistd.h>
#define POPEN popen
#define PCLOSE pclose
static inline int wgsl_getpid() { return getpid(); }
#endif

extern "C" {
#include "simple_wgsl.h"
}

namespace wgsl_test {

class AstGuard {
  public:
    /* Ownership: AstGuard takes ownership of both the AST node and the
     * diagnostic list carried by the WgslParseResult. */
    explicit AstGuard(WgslParseResult pr) : pr_(pr) {}
    explicit AstGuard(WgslAstNode *ast)
        : pr_{SW_OK, ast, nullptr} {}
    ~AstGuard() {
        if (pr_.value) wgsl_free_ast(pr_.value);
        wgsl_diagnostic_list_free(pr_.diags);
    }
    WgslAstNode *get() { return pr_.value; }
    SwResult code() const { return pr_.code; }
    const WgslDiagnosticList *diags() const { return pr_.diags; }

  private:
    WgslParseResult pr_;
};

class ResolverGuard {
  public:
    explicit ResolverGuard(WgslResolveResult rr) : rr_(rr) {}
    explicit ResolverGuard(WgslResolver *r)
        : rr_{SW_OK, r, nullptr} {}
    ~ResolverGuard() {
        if (rr_.value) wgsl_resolver_free(rr_.value);
        wgsl_diagnostic_list_free(rr_.diags);
    }
    WgslResolver *get() { return rr_.value; }
    SwResult code() const { return rr_.code; }
    const WgslDiagnosticList *diags() const { return rr_.diags; }

  private:
    WgslResolveResult rr_;
};

class LowerGuard {
  public:
    explicit LowerGuard(WgslLowerResult lr) : lr_(lr) {}
    explicit LowerGuard(WgslLower *l)
        : lr_{SW_OK, l, nullptr} {}
    ~LowerGuard() {
        if (lr_.value) wgsl_lower_destroy(lr_.value);
        wgsl_diagnostic_list_free(lr_.diags);
    }
    WgslLower *get() { return lr_.value; }
    SwResult code() const { return lr_.code; }
    const WgslDiagnosticList *diags() const { return lr_.diags; }

  private:
    WgslLowerResult lr_;
};

class SpirvGuard {
  public:
    explicit SpirvGuard(uint32_t *words) : words_(words) {}
    ~SpirvGuard() {
        if (words_) wgsl_lower_free(words_);
    }
    uint32_t *get() { return words_; }

  private:
    uint32_t *words_;
};

inline std::string MakeTempSpvPath(const char *prefix) {
    static std::atomic<int> counter{0};
    int n = counter.fetch_add(1);
    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%d_%d.spv", prefix, wgsl_getpid(), n);
    return std::string(buf);
}

inline std::string ReadFile(const std::string &path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

// Write SPIR-V binary to file
inline bool WriteSpirvFile(const std::string &path, const uint32_t *words, size_t count) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char *>(words), count * sizeof(uint32_t));
    return f.good();
}

// Run an external command, capture stdout+stderr, return exit code
inline int RunCommand(const std::string &cmd, std::string *output) {
    FILE *pipe = POPEN(cmd.c_str(), "r");
    if (!pipe) return -1;
    if (output) output->clear();
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        if (output) *output += buffer;
    }
    return PCLOSE(pipe);
}

// Validate SPIR-V using spirv-val
inline bool ValidateSpirv(const uint32_t *words, size_t word_count, std::string *out_error = nullptr) {
    std::string spv_path = MakeTempSpvPath("wgsl_val");
    if (!WriteSpirvFile(spv_path, words, word_count)) {
        if (out_error) *out_error = "Failed to write temp SPIR-V file";
        return false;
    }

    std::string output;
    int ret = RunCommand("spirv-val --target-env vulkan1.3 " + spv_path + " 2>&1", &output);
    std::remove(spv_path.c_str());

    if (ret != 0) {
        if (out_error) *out_error = output;
        return false;
    }
    return true;
}

// Helper to compile and validate WGSL source
struct CompileResult {
    bool success;
    std::string error;
    std::vector<uint32_t> spirv;
};

inline CompileResult CompileWgsl(const char *source) {
    CompileResult result;
    result.success = false;

    WgslParseResult pr = wgsl_parse(source);
    WgslAstNode *ast = pr.value;
    if (pr.code != SW_OK || !ast) {
        std::string detail = "Parse failed";
        if (pr.diags) {
            for (size_t i = 0; i < pr.diags->count; i++) {
                detail += "\n  ";
                detail += pr.diags->items[i].message ? pr.diags->items[i].message : "(no message)";
            }
        }
        result.error = detail;
        wgsl_diagnostic_list_free(pr.diags);
        if (ast) wgsl_free_ast(ast);
        return result;
    }

    WgslResolveResult rr = wgsl_resolver_build(ast);
    WgslResolver *resolver = rr.value;
    if (rr.code != SW_OK || !resolver) {
        std::string detail = "Resolve failed";
        if (rr.diags) {
            for (size_t i = 0; i < rr.diags->count; i++) {
                detail += "\n  ";
                detail += rr.diags->items[i].message ? rr.diags->items[i].message : "(no message)";
            }
        }
        wgsl_free_ast(ast);
        wgsl_diagnostic_list_free(pr.diags);
        wgsl_diagnostic_list_free(rr.diags);
        if (resolver) wgsl_resolver_free(resolver);
        result.error = detail;
        return result;
    }

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerSpirvResult lsr = wgsl_lower_emit_spirv(ast, resolver, &opts);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    wgsl_diagnostic_list_free(pr.diags);
    wgsl_diagnostic_list_free(rr.diags);

    if (lsr.code != SW_OK) {
        std::string detail = "Lower failed";
        if (lsr.diags) {
            for (size_t i = 0; i < lsr.diags->count; i++) {
                detail += "\n  ";
                detail += lsr.diags->items[i].message ? lsr.diags->items[i].message : "(no message)";
            }
        }
        result.error = detail;
        wgsl_diagnostic_list_free(lsr.diags);
        if (lsr.words) wgsl_lower_free(lsr.words);
        return result;
    }

    result.spirv.assign(lsr.words, lsr.words + lsr.word_count);
    wgsl_lower_free(lsr.words);
    wgsl_diagnostic_list_free(lsr.diags);

    // Validate
    if (!ValidateSpirv(result.spirv.data(), result.spirv.size(), &result.error)) {
        return result;
    }

    result.success = true;
    return result;
}

inline CompileResult CompileGlsl(const char *source, WgslStage stage) {
    CompileResult result;
    result.success = false;

    WgslAstNode *ast = glsl_parse(source, NULL, stage, NULL);
    if (!ast) {
        result.error = "GLSL parse failed";
        return result;
    }

    WgslResolveResult rr = wgsl_resolver_build(ast);
    WgslResolver *resolver = rr.value;
    if (rr.code != SW_OK || !resolver) {
        wgsl_free_ast(ast);
        wgsl_diagnostic_list_free(rr.diags);
        if (resolver) wgsl_resolver_free(resolver);
        result.error = "Resolve failed";
        return result;
    }

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerSpirvResult lsr = wgsl_lower_emit_spirv(ast, resolver, &opts);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    wgsl_diagnostic_list_free(rr.diags);

    if (lsr.code != SW_OK) {
        result.error = "Lower failed";
        wgsl_diagnostic_list_free(lsr.diags);
        if (lsr.words) wgsl_lower_free(lsr.words);
        return result;
    }

    result.spirv.assign(lsr.words, lsr.words + lsr.word_count);
    wgsl_lower_free(lsr.words);
    wgsl_diagnostic_list_free(lsr.diags);

    if (!ValidateSpirv(result.spirv.data(), result.spirv.size(), &result.error)) {
        return result;
    }

    result.success = true;
    return result;
}

struct RaiseResult {
    bool success;
    std::string error;
    std::string wgsl;
};

inline RaiseResult RaiseSpirvToWgsl(const std::vector<uint32_t> &spirv) {
    RaiseResult result;
    result.success = false;

    char *wgsl = nullptr;
    char *error = nullptr;
    WgslRaiseOptions opts = {};
    opts.preserve_names = 1;

    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        spirv.data(), spirv.size(), &opts, &wgsl, &error);

    if (raise_result != WGSL_RAISE_SUCCESS) {
        result.error = error ? error : "Raise failed";
        if (error) wgsl_raise_free(error);
        if (wgsl) wgsl_raise_free(wgsl);
        return result;
    }

    result.wgsl = wgsl;
    wgsl_raise_free(wgsl);
    result.success = true;
    return result;
}

struct GlslRaiseResult {
    bool success;
    std::string error;
    std::string glsl;
};

inline GlslRaiseResult RaiseSsirToGlsl(const SsirModule *ssir, SsirStage stage) {
    GlslRaiseResult result;
    result.success = false;

    char *glsl = nullptr;
    char *error = nullptr;
    SsirToGlslOptions opts = {};
    opts.preserve_names = 1;

    SsirToGlslResult raise_result = ssir_to_glsl(ssir, stage, &opts, &glsl, &error);

    if (raise_result != SSIR_TO_GLSL_OK) {
        result.error = error ? error : "GLSL raise failed";
        if (error) ssir_to_glsl_free(error);
        if (glsl) ssir_to_glsl_free(glsl);
        return result;
    }

    result.glsl = glsl;
    ssir_to_glsl_free(glsl);
    result.success = true;
    return result;
}

} // namespace wgsl_test
