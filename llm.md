# LLM Context: simple_wgsl

## Project Overview

A feature-complete WGSL (WebGPU Shading Language) compiler library written in **pure C99**. Provides bidirectional conversion between WGSL, GLSL 450, SPIR-V binary, and an internal intermediate representation (SSIR).

## Build System

- **Build tool**: CMake + Ninja
- **Build directory**: `build/`
- **C standard**: C99 (library), C++17 (tests)
- **Testing**: Google Test via `ctest --test-dir build`

```bash
cmake -B build -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure
```

## Compilation Pipeline

```
WGSL source ──► wgsl_parse() ──► AST (WgslAstNode*)
GLSL source ──► glsl_parse() ──►    │
                                     ▼
                              wgsl_resolver_build() ──► WgslResolver*
                                     │
                                     ▼
                              wgsl_lower_create() ──► SSIR ──► SPIR-V binary (uint32_t[])
                                                                    │
                                     ┌──────────────────────────────┘
                                     ▼
                              wgsl_raise_to_wgsl() ──► WGSL source (reconstructed)
```

Alternative path through SSIR directly:
```
SPIR-V binary ──► spirv_to_ssir() ──► SsirModule* ──► ssir_to_spirv() ──► SPIR-V binary
                                           │
                                           ├──► ssir_to_wgsl() ──► WGSL source
                                           ├──► ssir_to_glsl() ──► GLSL 450 source
                                           ├──► ssir_to_msl()  ──► Metal Shading Language
                                           └──► ssir_to_hlsl() ──► HLSL source

MSL source    ──► msl_to_ssir()   ──► SsirModule*
```

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `simple_wgsl.h` | ~1320 | **Unified public API header** - all types, enums, and function declarations |
| `wgsl_parser.c` | ~1940 | WGSL lexer + recursive-descent parser → AST |
| `glsl_parser.c` | ~2140 | GLSL 450 lexer + parser → same AST |
| `wgsl_resolve.c` | ~1010 | Semantic analysis: symbol table, binding extraction, entrypoint detection |
| `wgsl_lower.c` | ~5660 | AST → SSIR → SPIR-V compilation (largest file) |
| `wgsl_raise.c` | ~2090 | SPIR-V → WGSL decompilation |
| `ssir.c` | ~2040 | SSIR module, type system, and builder API |
| `ssir_to_spirv.c` | ~2090 | SSIR → SPIR-V serialization |
| `ssir_to_wgsl.c` | ~1280 | SSIR → WGSL text emission |
| `ssir_to_glsl.c` | ~870 | SSIR → GLSL 450 text emission |
| `ssir_to_msl.c` | ~2000 | SSIR → Metal Shading Language (MSL) text emission |
| `ssir_to_hlsl.c` | ~1400 | SSIR → HLSL text emission |
| `spirv_to_ssir.c` | ~1990 | SPIR-V → SSIR deserialization |
| `msl_parser.c` | ~2200 | MSL lexer + parser → SSIR deserialization |

## Key Data Structures

**AST** (`simple_wgsl.h`): `WgslAstNode` is a tagged union with `WgslNodeType` discriminator. Node types include `PROGRAM`, `STRUCT`, `FUNCTION`, `BLOCK`, `VAR_DECL`, `IF`, `WHILE`, `FOR`, `BINARY`, `CALL`, `MEMBER`, `INDEX`, `LITERAL`, `IDENT`, etc. Each node carries `line`/`col` for source mapping.

**Resolver** (`simple_wgsl.h`): `WgslResolver` holds symbol table, identifier-to-symbol mappings, struct/function registries, and scope chain. `WgslSymbolInfo` has kind (GLOBAL/PARAM/LOCAL), name, group/binding indices, and declaration node.

**SSIR** (`simple_wgsl.h`): `SsirModule` contains arrays of `SsirType`, `SsirConstant`, `SsirGlobalVar`, `SsirFunction`, and `SsirEntryPoint`. Functions contain `SsirBlock` arrays, each with `SsirInst` arrays. All entities have unique `uint32_t` IDs. The IR is in SSA form.

**SPIR-V lowering** (`wgsl_lower.c`): Uses `SpvSections` to accumulate words into separate buffers (capabilities, types, globals, functions, etc.) then concatenates in correct order.

## Memory Management

All allocators are overridable via macros defined before including `simple_wgsl.h`:
- `NODE_ALLOC`, `NODE_MALLOC`, `NODE_REALLOC`, `NODE_FREE` for AST nodes
- `SSIR_MALLOC`, `SSIR_REALLOC`, `SSIR_FREE` for SSIR module

Default: standard `malloc`/`realloc`/`free`.

## Lowering Options

`WgslLowerOptions` controls SPIR-V output:
- `spirv_version` - target SPIR-V version
- `env` - Vulkan 1.1/1.2/1.3 or WebGPU
- `packing` - struct layout (std140/std430)
- `enable_debug_names` / `enable_line_info` - debug info
- `zero_initialize_vars` - safety default

## Test Structure

Tests live in `tests/` and use Google Test (fetched via CMake FetchContent).

| Test file | What it tests |
|-----------|--------------|
| `parser_test.cpp` | WGSL parsing: structs, functions, expressions, precedence |
| `glsl_parser_test.cpp` | GLSL 450 parsing (very large, ~43k lines) |
| `resolver_test.cpp` | Symbol resolution, bindings, entrypoints |
| `lower_test.cpp` | SPIR-V generation and validation |
| `raise_test.cpp` | SPIR-V → WGSL roundtrip |
| `integration_test.cpp` | Full pipeline end-to-end |
| `expression_test.cpp` | Discovers `expressions/**/*.wgsl` + `.expected.spvasm` pairs |
| `spirv_to_ssir_test.cpp` | SPIR-V deserialization |
| `ssir_raise_test.cpp` | SSIR → WGSL emission |
| `glsl_raise_test.cpp` | SSIR → GLSL 450 emission and WGSL→SSIR→GLSL→SPIR-V roundtrip |
| `msl_parser_test.cpp` | MSL parsing and WGSL→SSIR→MSL→SSIR→MSL roundtrip |
| `ssir_to_hlsl_test.cpp` | SSIR → HLSL text emission and structural tests |
| `ssir_to_msl_test.cpp` | SSIR → MSL text emission logic |
| `egl_compute_test.cpp` / `egl_graphics_test.cpp` | GPU execution of generated GLSL via local EGL/OpenGL |
| `vulkan_compute_test.cpp` | GPU compute execution (requires Vulkan) |
| `immediate_test.cpp` | Push constants: `var<immediate>` parsing, resolution, lowering, SPIR-V validation |
| `vulkan_graphics_test.cpp` | GPU graphics pipeline (requires Vulkan) |
| `vulkan_graphics_complex_test.cpp` | Complex graphics scenarios (~84k lines, requires Vulkan) |

Test utilities in `tests/test_utils.h` provide RAII guards (`AstGuard`, `ResolverGuard`, `LowerGuard`, `SpirvGuard`) and helpers (`CompileWgsl()`, `CompileGlsl()`, `RaiseSpirvToWgsl()`, `RaiseSsirToGlsl()`, `ValidateSpirv()`, `WriteSpirvFile()`).

Vulkan tests are conditionally compiled (`WGSL_HAS_VULKAN=1`) and skipped when Vulkan is unavailable.

## Verification Strategy (Generated spirv/glsl etc. correctness)

- **SPIR-V correctness:** Validated offline using `spirv-val` from SPIRV-Tools (`test_utils.h`'s `ValidateSpirv()`), which strictly checks against the requested Vulkan target environment logic. End-to-end operational execution correctness is assessed over actual graphics and compute hardware using `vulkan_compute_harness.cpp` and `vulkan_graphics_harness.cpp`.
- **GLSL correctness:** Syntactically, GLSL outputs are parsed by the local `glsl_parser.c`. However, behavioral and runtime validity is explicitly verified by tests leveraging `egl_compute_harness.cpp`; this passes the generated GLSL exactly into OpenGL API via EGL, validating compile/link, and functionally executing compute and render-to-texture pixels logic on native drivers.
- **Metal/MSL correctness:** Tested statically via Apple's native Metal framework APIs (`metal_compute.m`, `msl_roundtrip.m`) and functionally round-tripped (WGSL → MSL → SSIR → MSL) to assert fidelity across text representations via `msl_parser.c`.
- **HLSL correctness:** Functionally scrutinized in `ssir_to_hlsl_test.cpp` ensuring proper structural rendering, intrinsic mappings, buffer alignments, and entry points.

## Test Data

- `wgsl/` - full WGSL shader files (compute, vertex, fragment)
- `glsl/` - GLSL shader files
- `expressions/` - exhaustive expression test cases organized by category:
  - `binary/` (add, sub, mul, div, mod, comparisons, shifts)
  - `unary/` (negate, not, bitwise not)
  - `type_conv/`, `type_ctor/`, `bitcast/`
  - `splat/`, `swizzle/`, `index/`
  - `zero_init/`, `literals/`, `user_call/`

  Each test case has a `.wgsl` input and `.expected.spvasm` expected output.

## Dependencies

| Dependency | Version | Source |
|-----------|---------|--------|
| SPIR-V Headers (Khronos) | vulkan-sdk-1.4.341 | System or FetchContent |
| Google Test | v1.14.0 | FetchContent |
| Vulkan SDK | system | Optional, for GPU tests |
| `spirv-val` / `spirv-dis` | system | SPIRV-Tools, used by tests for validation |

## Important Patterns

- **Single-header API**: Everything public is in `simple_wgsl.h`. Implementation is split across `.c` files.
- **C99 core, C++17 tests**: The library itself is pure C. Tests are C++ (for gtest and RAII guards).
- **Recursive descent parsing**: Both WGSL and GLSL parsers use hand-written recursive descent with operator precedence climbing.
- **Shared AST**: GLSL parser produces the same `WgslAstNode` tree as the WGSL parser, allowing shared resolution and lowering.
- **Section-buffered SPIR-V emission**: Lowering accumulates words into separate section buffers, concatenated at the end in the order mandated by the SPIR-V spec.
- **SSA form in SSIR**: All values are single-assignment. Each gets a unique ID.
- **ID-based entity references**: SSIR, SPIR-V, and the resolver all use `uint32_t` IDs to cross-reference types, values, and variables.
- **Dual raising targets**: SSIR can be raised to both WGSL (`ssir_to_wgsl.c`) and GLSL 450 (`ssir_to_glsl.c`).

## Common Tasks

**Adding a new WGSL feature**: Extend the lexer/parser in `wgsl_parser.c`, add AST node types in `simple_wgsl.h`, handle in `wgsl_resolve.c` for semantics, lower in `wgsl_lower.c`, and raise in `wgsl_raise.c`. Add tests in `tests/`.

**Adding a new SSIR instruction**: Add opcode to `SsirOpcode` enum in `simple_wgsl.h`, implement builder in `ssir.c`, emit in `ssir_to_spirv.c`, parse in `spirv_to_ssir.c`, print in `ssir_to_wgsl.c` and `ssir_to_glsl.c`.

**Adding expression tests**: Create `expressions/<category>/<subcase>/test.wgsl` and `test.expected.spvasm`. The `expression_test.cpp` discovery mechanism picks them up automatically.

## GLSL Raising Details

`ssir_to_glsl()` converts an `SsirModule` to GLSL 450 text for a specified shader stage (`SsirStage`). Key WGSL→GLSL mappings:

| WGSL/SSIR | GLSL 450 |
|-----------|----------|
| `f32` / `i32` / `u32` | `float` / `int` / `uint` |
| `vec4<f32>` | `vec4` |
| `vec3<i32>` | `ivec3` |
| `mat4x4<f32>` | `mat4` |
| `@location(N) param` | `layout(location = N) in/out` |
| `@builtin(position)` | `gl_Position` / `gl_FragCoord` |
| `@group(G) @binding(B) var<uniform>` | `layout(std140, set = G, binding = B) uniform` |
| `@group(G) @binding(B) var<storage>` | `layout(std430, set = G, binding = B) buffer` |
| `@workgroup_size(X,Y,Z)` | `layout(local_size_x=X, ...) in;` |
| Entry point function | `void main()` |
| `inverseSqrt` | `inversesqrt` |
| `countOneBits` | `bitCount` |
| `dpdx`/`dpdy` | `dFdx`/`dFdy` |
| `textureSample(t,s,c)` | `texture(t, c)` |

Roundtrip tested via: WGSL → parse → resolve → lower → SSIR → `ssir_to_glsl()` → `glsl_parse()` → resolve → lower → SPIR-V → `spirv-val`.

## Push Constants via `var<immediate>`

WGSL lacks native push constants. The `var<immediate>` extension adds a WGSL-native syntax for push constants that the compiler automatically packs into per-entry-point push constant blocks.

### Extensions

Enable with WGSL `enable` directives:

| Extension | What it enables |
|---|---|
| `immediate_address_space` | `var<immediate>` with scalar, vector, matrix, and plain struct types |
| `immediate_arrays` | Additionally allows arrays (implies `immediate_address_space`) |

### Syntax

```wgsl
enable immediate_address_space;

var<immediate> scale: f32;
var<immediate> offset: vec2f;
var<immediate> transform: mat4x4f;

// Pointer support
fn apply(p: ptr<immediate, f32>) -> f32 { return *p; }

@compute @workgroup_size(1)
fn main() {
    let s = scale;           // direct access
    let r = apply(&offset);  // pointer passing
}
```

### Semantics

- **Address space**: `immediate` — uniform, read-only, no `@group`/`@binding`
- **Per-entry-point isolation**: Each entry point gets only the immediates it transitively uses
- **Declaration-order layout**: Packed in source order with std430 or scalar alignment
- **Pointer support**: `ptr<immediate, T>` is valid for function parameters

### Lowering

The compiler generates a synthetic push constant struct per entry point. Given:

```wgsl
var<immediate> a: f32;
var<immediate> b: u32;

@compute @workgroup_size(1) fn main() { let x = a + f32(b); }
```

The per-entry-point SPIR-V gets a push constant block:
```
OpDecorate %_PushConstants Block
OpMemberDecorate %_PushConstants 0 Offset 0    ; a: f32
OpMemberDecorate %_PushConstants 1 Offset 4    ; b: u32
%_PushConstants = OpTypeStruct %float %uint
%pc = OpVariable %_ptr_PushConstant__PushConstants PushConstant
```

Entry points that use no immediates get no push constant block.

### API

**Per-entry-point compilation** via `WgslLowerOptions`:
```c
WgslLowerOptions opts = {0};
opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
opts.entry_point = "main";                    // compile only this entry point
opts.immediate_layout = SSIR_LAYOUT_STD430;   // or SSIR_LAYOUT_SCALAR
```

**Reflection** — query immediates for an entry point:
```c
int count = 0;
const WgslImmediateInfo *imms = wgsl_resolver_entrypoint_immediates(
    resolver, "main", SSIR_LAYOUT_STD430, &count);
for (int i = 0; i < count; i++) {
    printf("%s: size=%d offset=%d align=%d\n",
           imms[i].name, imms[i].type_size, imms[i].offset, imms[i].alignment);
}
wgsl_resolve_free((void *)imms);
```

**Symbol enumeration** — `WGSL_SYM_IMMEDIATE` kind appears in `wgsl_resolver_globals()` and `wgsl_resolver_entrypoint_globals()`.

### Layout Rules

| Rule | Alignment | Notes |
|------|-----------|-------|
| `SSIR_LAYOUT_STD430` | Natural alignment, vec3→16B | Default, compatible with Vulkan push constants |
| `SSIR_LAYOUT_SCALAR` | 4-byte for all scalars | Requires `VK_EXT_scalar_block_layout` |

### Tests

`tests/immediate_test.cpp` — 26 tests covering:
- Parser: `enable` directives, `var<immediate>`, `ptr<immediate, T>`
- Resolver: symbol kind, per-entry-point isolation, transitive call graph, reflection API
- Lowering: SPIR-V validation, push constant block generation, offset decorations, layout modes
- Integration: multi-entry-point compilation, pointer passing, mixed bindings + immediates
