#pragma once

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

extern "C" {
#include "simple_wgsl.h"
}

namespace wgsl_test {

class AstGuard {
public:
    explicit AstGuard(WgslAstNode* ast) : ast_(ast) {}
    ~AstGuard() { if (ast_) wgsl_free_ast(ast_); }
    WgslAstNode* get() { return ast_; }
private:
    WgslAstNode* ast_;
};

class ResolverGuard {
public:
    explicit ResolverGuard(WgslResolver* r) : r_(r) {}
    ~ResolverGuard() { if (r_) wgsl_resolver_free(r_); }
    WgslResolver* get() { return r_; }
private:
    WgslResolver* r_;
};

class LowerGuard {
public:
    explicit LowerGuard(WgslLower* l) : l_(l) {}
    ~LowerGuard() { if (l_) wgsl_lower_destroy(l_); }
    WgslLower* get() { return l_; }
private:
    WgslLower* l_;
};

class SpirvGuard {
public:
    explicit SpirvGuard(uint32_t* words) : words_(words) {}
    ~SpirvGuard() { if (words_) wgsl_lower_free(words_); }
    uint32_t* get() { return words_; }
private:
    uint32_t* words_;
};

inline std::string ReadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

// Write SPIR-V binary to file
inline bool WriteSpirvFile(const std::string& path, const uint32_t* words, size_t count) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(words), count * sizeof(uint32_t));
    return f.good();
}

// Validate SPIR-V using spirv-val
// Returns true if valid, false otherwise
// If out_error is provided, fills it with error message on failure
inline bool ValidateSpirv(const uint32_t* words, size_t word_count, std::string* out_error = nullptr) {
    // Create temp file
    char temp_path[] = "/tmp/wgsl_test_XXXXXX.spv";
    int fd = mkstemps(temp_path, 4);
    if (fd < 0) {
        if (out_error) *out_error = "Failed to create temp file";
        return false;
    }

    // Write SPIR-V
    ssize_t written = write(fd, words, word_count * sizeof(uint32_t));
    close(fd);
    if (written != static_cast<ssize_t>(word_count * sizeof(uint32_t))) {
        unlink(temp_path);
        if (out_error) *out_error = "Failed to write SPIR-V";
        return false;
    }

    // Run spirv-val
    std::string cmd = "spirv-val --target-env vulkan1.3 " + std::string(temp_path) + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        unlink(temp_path);
        if (out_error) *out_error = "Failed to run spirv-val";
        return false;
    }

    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int ret = pclose(pipe);
    unlink(temp_path);

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

inline CompileResult CompileWgsl(const char* source) {
    CompileResult result;
    result.success = false;

    WgslAstNode* ast = wgsl_parse(source);
    if (!ast) {
        result.error = "Parse failed";
        return result;
    }

    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        result.error = "Resolve failed";
        return result;
    }

    uint32_t* spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerResult lower_result = wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &spirv_size);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lower_result != WGSL_LOWER_OK) {
        result.error = "Lower failed";
        return result;
    }

    result.spirv.assign(spirv, spirv + spirv_size);
    wgsl_lower_free(spirv);

    // Validate
    if (!ValidateSpirv(result.spirv.data(), result.spirv.size(), &result.error)) {
        return result;
    }

    result.success = true;
    return result;
}

inline CompileResult CompileGlsl(const char* source, WgslStage stage) {
    CompileResult result;
    result.success = false;

    WgslAstNode* ast = glsl_parse(source, stage);
    if (!ast) {
        result.error = "GLSL parse failed";
        return result;
    }

    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        result.error = "Resolve failed";
        return result;
    }

    uint32_t* spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerResult lower_result = wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &spirv_size);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lower_result != WGSL_LOWER_OK) {
        result.error = "Lower failed";
        return result;
    }

    result.spirv.assign(spirv, spirv + spirv_size);
    wgsl_lower_free(spirv);

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

inline RaiseResult RaiseSpirvToWgsl(const std::vector<uint32_t>& spirv) {
    RaiseResult result;
    result.success = false;

    char* wgsl = nullptr;
    char* error = nullptr;
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

} // namespace wgsl_test
