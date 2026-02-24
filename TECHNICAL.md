# Technical Reference

Complete technical documentation for simple_wgsl: architecture, intermediate representation, API surface, and internal design.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Compilation Pipeline](#compilation-pipeline)
  - [Forward Path (Source to Binary)](#forward-path-source-to-binary)
  - [Reverse Path (Binary to Source)](#reverse-path-binary-to-source)
  - [Cross-Compilation Path](#cross-compilation-path)
- [AST (Abstract Syntax Tree)](#ast-abstract-syntax-tree)
  - [Node Types](#node-types)
  - [AST Structure](#ast-structure)
  - [Source Location Tracking](#source-location-tracking)
- [Resolver](#resolver)
  - [Symbol Table](#symbol-table)
  - [Binding Extraction](#binding-extraction)
  - [Entry Point Detection](#entry-point-detection)
  - [Vertex Input Reflection](#vertex-input-reflection)
  - [Fragment Output Reflection](#fragment-output-reflection)
- [SSIR (Simple Shader Intermediate Representation)](#ssir-simple-shader-intermediate-representation)
  - [Design Principles](#design-principles)
  - [Module Structure](#module-structure)
  - [Type System](#type-system)
  - [Constants](#constants)
  - [Global Variables](#global-variables)
  - [Functions and Blocks](#functions-and-blocks)
  - [Instruction Set](#instruction-set)
  - [Built-in Functions](#built-in-functions)
  - [Entry Points](#entry-points)
  - [Validation](#validation)
- [Lowering (AST to SPIR-V)](#lowering-ast-to-spirv)
  - [Lowering Options](#lowering-options)
  - [Section-Buffered Emission](#section-buffered-emission)
  - [Two-Phase API](#two-phase-api)
  - [Introspection](#introspection)
- [Raising (SPIR-V to WGSL)](#raising-spirv-to-wgsl)
  - [Raise Options](#raise-options)
  - [Incremental API](#incremental-api)
- [SSIR Emitters](#ssir-emitters)
  - [SSIR to SPIR-V](#ssir-to-spirv)
  - [SSIR to WGSL](#ssir-to-wgsl)
  - [SSIR to GLSL](#ssir-to-glsl)
  - [SSIR to MSL](#ssir-to-msl)
  - [SSIR to HLSL](#ssir-to-hlsl)
- [SPIR-V to SSIR Deserialization](#spirv-to-ssir-deserialization)
- [MSL to SSIR Parser](#msl-to-ssir-parser)
- [PTX to SSIR Parser](#ptx-to-ssir-parser)
  - [Overview](#overview-1)
  - [Why PTX Bypasses the AST](#why-ptx-bypasses-the-ast)
  - [Supported PTX Features](#supported-ptx-features)
  - [API](#ptx-api)
  - [Architecture of ptx_parser.c](#architecture-of-ptx_parserc)
  - [PTX Type System Mapping](#ptx-type-system-mapping)
  - [Register Model](#register-model)
  - [Kernel Parameter Convention](#kernel-parameter-convention)
  - [Special Register Mapping](#special-register-mapping)
  - [Control Flow Translation](#control-flow-translation)
  - [Predicated Execution](#predicated-execution)
  - [Memory Operations](#ptx-memory-operations)
  - [Instruction Mapping Reference](#instruction-mapping-reference)
  - [Usage Examples](#ptx-usage-examples)
  - [Limitations and Future Work](#limitations-and-future-work)
- [Immediates (Push Constants)](#immediates-push-constants)
  - [Language Extensions](#language-extensions)
  - [Syntax and Semantics](#syntax-and-semantics)
  - [Resolver API](#resolver-api)
  - [Lowering Behavior](#lowering-behavior)
  - [Layout Rules](#layout-rules)
- [Memory Management](#memory-management)
- [Language Mapping Tables](#language-mapping-tables)
  - [WGSL to GLSL Mapping](#wgsl-to-glsl-mapping)
- [Error Handling](#error-handling)

---

## Architecture Overview

Simple WGSL is organized around a central intermediate representation called SSIR. All source languages converge to SSIR, and all output formats diverge from it.

```
                        ┌─────────────┐
    WGSL source ──────► │  wgsl_parse  │──► AST ──► Resolver ──► Lowering ──┐
    GLSL source ──────► │  glsl_parse  │──► AST ──────────────────────────┘ │
                        └─────────────┘                                     ▼
    MSL source  ──────► msl_to_ssir ────────────────────────────────► SsirModule
    PTX assembly ─────► ptx_to_ssir ────────────────────────────────►     │
    SPIR-V binary ───► spirv_to_ssir ──────────────────────────────►     │
                                                                         │
                        ┌────────────────────────────────────────────────┘
                        │
                        ├──► ssir_to_spirv ──► SPIR-V binary
                        ├──► ssir_to_wgsl  ──► WGSL source
                        ├──► ssir_to_glsl  ──► GLSL 450 source
                        ├──► ssir_to_msl   ──► Metal Shading Language
                        └──► ssir_to_hlsl  ──► HLSL source
```

Key architectural decisions:

- **Shared AST for WGSL and GLSL**: Both parsers produce the same `WgslAstNode` tree, so the resolver and lowering stages are shared between both input languages.
- **MSL and PTX bypass the AST**: The MSL parser and PTX parser each produce an `SsirModule` directly rather than going through the shared AST. MSL's C++-based syntax and PTX's flat assembly semantics differ too significantly from WGSL/GLSL to share their AST.
- **SPIR-V is both input and output**: SPIR-V can be deserialized into SSIR for decompilation, and SSIR can be serialized back to SPIR-V for compilation.
- **Pure C99 core**: The library itself uses no C++ features. Tests use C++17 for Google Test and RAII convenience.

---

## Compilation Pipeline

### Forward Path (Source to Binary)

The forward path compiles WGSL or GLSL source code to SPIR-V binary:

```
Source (WGSL/GLSL)  ──►  AST (WgslAstNode*)  ──►  Resolver  ──►  SPIR-V (uint32_t[])
                     parse                    resolve         lower
```

**Stage 1: Parsing** -- Hand-written recursive-descent parsers with operator-precedence climbing tokenize and parse the input into a shared AST. Both WGSL and GLSL produce the same `WgslAstNode` tree structure. The parser tracks line and column numbers for every node.

**Stage 2: Resolution** -- The resolver walks the AST and builds a symbol table with scope tracking. It extracts `@group`/`@binding` attributes for resource binding reflection, identifies entry points and their shader stages, and resolves identifier references to their declaration sites.

**Stage 3: Lowering** -- The lowering pass walks the resolved AST, builds an SSIR module internally, then serializes to SPIR-V. It uses section-buffered emission: SPIR-V words are accumulated into separate buffers (capabilities, extensions, types, globals, functions, etc.) and concatenated at the end in the order mandated by the SPIR-V specification.

### Reverse Path (Binary to Source)

The reverse path decompiles SPIR-V binary back to WGSL source:

```
SPIR-V (uint32_t[])  ──►  SsirModule  ──►  WGSL source (char*)
                     spirv_to_ssir     wgsl_raise / ssir_to_wgsl
```

The `wgsl_raise_to_wgsl()` convenience function wraps this two-step process. The incremental `WgslRaiser` API exposes each phase separately for more control.

### Cross-Compilation Path

To cross-compile between languages, chain the appropriate parser and emitter through SSIR:

```
WGSL ──► parse ──► resolve ──► lower ──► SSIR ──► ssir_to_glsl ──► GLSL 450
WGSL ──► parse ──► resolve ──► lower ──► SSIR ──► ssir_to_msl  ──► MSL
WGSL ──► parse ──► resolve ──► lower ──► SSIR ──► ssir_to_hlsl ──► HLSL
MSL  ──► msl_to_ssir ──► SSIR ──► ssir_to_spirv ──► SPIR-V
PTX  ──► ptx_to_ssir ──► SSIR ──► ssir_to_wgsl  ──► WGSL
PTX  ──► ptx_to_ssir ──► SSIR ──► ssir_to_spirv ──► SPIR-V
PTX  ──► ptx_to_ssir ──► SSIR ──► ssir_to_glsl  ──► GLSL 450
SPIR-V ──► spirv_to_ssir ──► SSIR ──► ssir_to_glsl ──► GLSL 450
```

---

## AST (Abstract Syntax Tree)

### Node Types

Every AST node is a `WgslAstNode`, a tagged union discriminated by `WgslNodeType`:

| Node Type | Enum | Description |
|-----------|------|-------------|
| Program | `WGSL_NODE_PROGRAM` | Root node, contains top-level declarations |
| Struct | `WGSL_NODE_STRUCT` | Struct type declaration |
| Struct Field | `WGSL_NODE_STRUCT_FIELD` | Individual field within a struct |
| Global Variable | `WGSL_NODE_GLOBAL_VAR` | Module-scope `var<>` declaration |
| Function | `WGSL_NODE_FUNCTION` | Function declaration with params, return type, and body |
| Parameter | `WGSL_NODE_PARAM` | Function parameter |
| Type | `WGSL_NODE_TYPE` | Type reference (e.g., `vec4<f32>`) |
| Attribute | `WGSL_NODE_ATTRIBUTE` | Attribute (e.g., `@group(0)`, `@vertex`) |
| Block | `WGSL_NODE_BLOCK` | Brace-delimited statement list `{ ... }` |
| Var Declaration | `WGSL_NODE_VAR_DECL` | Local `var`, `let`, or `const` declaration |
| Return | `WGSL_NODE_RETURN` | Return statement |
| Expression Statement | `WGSL_NODE_EXPR_STMT` | Expression used as a statement |
| If | `WGSL_NODE_IF` | If/else statement |
| While | `WGSL_NODE_WHILE` | While loop |
| For | `WGSL_NODE_FOR` | For loop (init; cond; cont) |
| Do-While | `WGSL_NODE_DO_WHILE` | Do-while loop (GLSL) |
| Switch | `WGSL_NODE_SWITCH` | Switch statement |
| Case | `WGSL_NODE_CASE` | Case clause (NULL expr = default) |
| Break | `WGSL_NODE_BREAK` | Break statement |
| Continue | `WGSL_NODE_CONTINUE` | Continue statement |
| Discard | `WGSL_NODE_DISCARD` | Fragment discard |
| Identifier | `WGSL_NODE_IDENT` | Variable/function reference |
| Literal | `WGSL_NODE_LITERAL` | Integer or float literal |
| Binary | `WGSL_NODE_BINARY` | Binary operator expression |
| Assign | `WGSL_NODE_ASSIGN` | Assignment (`=`, `+=`, etc.) |
| Call | `WGSL_NODE_CALL` | Function or type constructor call |
| Member | `WGSL_NODE_MEMBER` | Member access (`object.field`) |
| Index | `WGSL_NODE_INDEX` | Subscript (`object[index]`) |
| Unary | `WGSL_NODE_UNARY` | Unary operator (prefix or postfix) |
| Ternary | `WGSL_NODE_TERNARY` | Ternary conditional (GLSL `? :`) |

### AST Structure

```c
struct WgslAstNode {
    WgslNodeType type;     // discriminator
    int line;              // source line (1-based)
    int col;               // source column (1-based)
    union {
        Program program;
        Function function;
        StructDecl struct_decl;
        // ... one variant per node type
    };
};
```

Variable declarations distinguish between three kinds via `WgslDeclKind`:

| Kind | Enum | Semantics |
|------|------|-----------|
| `var` | `WGSL_DECL_VAR` | Mutable variable |
| `let` | `WGSL_DECL_LET` | Immutable binding |
| `const` | `WGSL_DECL_CONST` | Compile-time constant |

### Source Location Tracking

Every AST node stores its source `line` and `col` (both 1-based). The lowering pass can optionally emit `OpLine` instructions in SPIR-V for debugger source mapping when `enable_line_info` is set.

---

## Resolver

The resolver performs semantic analysis on the AST, producing a `WgslResolver` that stores symbol information, scope chains, and resource binding data.

### Symbol Table

```c
typedef struct {
    int id;                           // unique symbol ID
    WgslSymbolKind kind;              // GLOBAL, PARAM, or LOCAL
    const char *name;                 // identifier name
    int has_group, group_index;       // @group(N)
    int has_binding, binding_index;   // @binding(N)
    int has_min_binding_size;         // @min_binding_size(N)
    int min_binding_size;
    const WgslAstNode *decl_node;    // declaration AST node
    const WgslAstNode *function_node; // enclosing function (NULL for globals)
} WgslSymbolInfo;
```

Query functions:

| Function | Returns |
|----------|---------|
| `wgsl_resolver_all_symbols(r, &count)` | Every symbol in the program |
| `wgsl_resolver_globals(r, &count)` | Only module-scope symbols |
| `wgsl_resolver_binding_vars(r, &count)` | Only symbols with `@group`/`@binding` |
| `wgsl_resolver_ident_symbol_id(r, ident_node)` | Symbol ID for a specific identifier AST node |
| `wgsl_resolver_entrypoint_globals(r, "main", &count)` | Globals used by a specific entry point |
| `wgsl_resolver_entrypoint_binding_vars(r, "main", &count)` | Binding vars for a specific entry point |
| `wgsl_resolver_entrypoint_immediates(r, "main", layout, &count)` | Immediate vars with layout info for an entry point |

Symbol kinds include `WGSL_SYM_GLOBAL`, `WGSL_SYM_PARAM`, `WGSL_SYM_LOCAL`, and `WGSL_SYM_IMMEDIATE` (for `var<immediate>` declarations).

### Binding Extraction

The resolver extracts `@group(G)` and `@binding(B)` from global variable attributes. This data is available both for the whole program and per entry point, enabling descriptor set layout generation.

### Entry Point Detection

```c
typedef struct {
    const char *name;                   // function name
    WgslStage stage;                    // VERTEX, FRAGMENT, or COMPUTE
    const WgslAstNode *function_node;   // AST node
} WgslResolverEntrypoint;
```

Query with `wgsl_resolver_entrypoints(r, &count)`. Shader stage is determined from `@vertex`, `@fragment`, or `@compute` attributes.

### Vertex Input Reflection

```c
typedef struct {
    int location;
    int component_count;    // 1-4
    WgslNumericType numeric_type;  // F32, I32, U32, F16, BOOL
    int byte_size;
} WgslVertexSlot;
```

Call `wgsl_resolver_vertex_inputs(r, "main", &slots)` to get the vertex input layout for a vertex entry point. This is sufficient to construct `VkVertexInputAttributeDescription` arrays.

### Fragment Output Reflection

```c
typedef struct {
    int location;
    int component_count;
    WgslNumericType numeric_type;
} WgslFragmentOutput;
```

Call `wgsl_resolver_fragment_outputs(r, "main", &outputs)` to get fragment output locations and formats.

---

## SSIR (Simple Shader Intermediate Representation)

SSIR is a low-level, language-independent intermediate representation that serves as the canonical representation of shader semantics. Every conversion in the library flows through SSIR.

### Design Principles

- **ID-based references**: All types, values, constants, and variables use unique `uint32_t` IDs instead of pointers. This enables serialization, cross-references, and avoids pointer invalidation when arrays grow.
- **SSA form**: Every instruction result gets a unique ID. PHI nodes handle control-flow merges.
- **Explicit control flow**: Basic blocks with explicit branch/return terminators rather than implicit tree structure.
- **Complete type system**: Scalars, vectors, matrices, arrays, structs, pointers with address spaces, textures, and samplers.
- **No language bias**: SSIR does not privilege any source or target language. Mappings to SPIR-V, WGSL, GLSL, MSL, and HLSL are all done by separate emitters.

### Module Structure

```c
struct SsirModule {
    SsirType      *types;           // type array
    uint32_t       type_count;
    SsirConstant  *constants;       // constant array
    uint32_t       constant_count;
    SsirGlobalVar *globals;         // global variable array
    uint32_t       global_count;
    SsirFunction  *functions;       // function array
    uint32_t       function_count;
    SsirEntryPoint *entry_points;   // entry point array
    uint32_t       entry_point_count;
    SsirNameEntry  *names;          // debug name table
    uint32_t       name_count;
    uint32_t       next_id;         // next available ID
    SsirClipSpaceConvention clip_space;  // coordinate convention
};
```

```
SsirModule
├── Types[]          (type declarations: scalars, vectors, matrices, structs, pointers, textures, ...)
├── Constants[]      (compile-time values: bool, i32, u32, f32, f16, f64, composite, null, ...)
├── GlobalVars[]     (module-scope variables with bindings, locations, builtins)
├── Functions[]
│   ├── Params[]     (function parameters)
│   ├── Locals[]     (function-scope pointer variables)
│   └── Blocks[]     (basic blocks)
│       └── Insts[]  (SSA instructions)
├── EntryPoints[]    (shader stage entry points with interface variables)
└── Names[]          (debug name table)
```

### Type System

All types are stored in the module's type array and referenced by ID.

| Type Kind | Enum | Description |
|-----------|------|-------------|
| Void | `SSIR_TYPE_VOID` | Void type (function returns) |
| Bool | `SSIR_TYPE_BOOL` | Boolean |
| I32 | `SSIR_TYPE_I32` | Signed 32-bit integer |
| U32 | `SSIR_TYPE_U32` | Unsigned 32-bit integer |
| F32 | `SSIR_TYPE_F32` | 32-bit float |
| F16 | `SSIR_TYPE_F16` | 16-bit float (half) |
| F64 | `SSIR_TYPE_F64` | 64-bit float (double) |
| I8/U8/I16/U16/I64/U64 | `SSIR_TYPE_*` | Extended integer types |
| Vector | `SSIR_TYPE_VEC` | Vector (elem type + size 2/3/4) |
| Matrix | `SSIR_TYPE_MAT` | Matrix (column type + cols + rows) |
| Array | `SSIR_TYPE_ARRAY` | Fixed-length array (elem type + length + stride) |
| Runtime Array | `SSIR_TYPE_RUNTIME_ARRAY` | Variable-length array (storage buffers) |
| Struct | `SSIR_TYPE_STRUCT` | Struct with member types, offsets, names, layout rule |
| Pointer | `SSIR_TYPE_PTR` | Pointer (pointee type + address space) |
| Sampler | `SSIR_TYPE_SAMPLER` | Sampler |
| Comparison Sampler | `SSIR_TYPE_SAMPLER_COMPARISON` | Comparison sampler (depth) |
| Texture | `SSIR_TYPE_TEXTURE` | Sampled texture (dim + sampled type) |
| Storage Texture | `SSIR_TYPE_TEXTURE_STORAGE` | Storage texture (dim + format + access) |
| Depth Texture | `SSIR_TYPE_TEXTURE_DEPTH` | Depth texture (dim) |

**Address Spaces**:

| Space | Enum | Description |
|-------|------|-------------|
| Function | `SSIR_ADDR_FUNCTION` | Function-local variables |
| Private | `SSIR_ADDR_PRIVATE` | Module-scope private variables |
| Workgroup | `SSIR_ADDR_WORKGROUP` | Shared within workgroup |
| Uniform | `SSIR_ADDR_UNIFORM` | Uniform buffer |
| Uniform Constant | `SSIR_ADDR_UNIFORM_CONSTANT` | Textures and samplers |
| Storage | `SSIR_ADDR_STORAGE` | Storage buffer |
| Input | `SSIR_ADDR_INPUT` | Shader stage input |
| Output | `SSIR_ADDR_OUTPUT` | Shader stage output |
| Push Constant | `SSIR_ADDR_PUSH_CONSTANT` | Vulkan push constants |
| Physical Storage Buffer | `SSIR_ADDR_PHYSICAL_STORAGE_BUFFER` | Buffer device address |

**Layout Rules** (for struct member offsets):

| Rule | Enum |
|------|------|
| None | `SSIR_LAYOUT_NONE` |
| std140 | `SSIR_LAYOUT_STD140` |
| std430 | `SSIR_LAYOUT_STD430` |
| Scalar | `SSIR_LAYOUT_SCALAR` |

**Texture Dimensions**:

`SSIR_TEX_1D`, `SSIR_TEX_2D`, `SSIR_TEX_3D`, `SSIR_TEX_CUBE`, `SSIR_TEX_2D_ARRAY`, `SSIR_TEX_CUBE_ARRAY`, `SSIR_TEX_MULTISAMPLED_2D`, `SSIR_TEX_1D_ARRAY`, `SSIR_TEX_BUFFER`, `SSIR_TEX_MULTISAMPLED_2D_ARRAY`

**Clip Space Conventions**:

| Convention | Y Direction | Z Range | Enum |
|-----------|-------------|---------|------|
| Vulkan | Y-down | [0, 1] | `SSIR_CLIP_SPACE_VULKAN` |
| OpenGL | Y-up | [-1, 1] | `SSIR_CLIP_SPACE_OPENGL` |
| DirectX | Y-up | [0, 1] | `SSIR_CLIP_SPACE_DIRECTX` |
| Metal | Y-up | [0, 1] | `SSIR_CLIP_SPACE_METAL` |

### Constants

```c
struct SsirConstant {
    uint32_t id;
    uint32_t type;
    SsirConstantKind kind;       // BOOL, I32, U32, F32, F16, F64, I8, U8, ..., COMPOSITE, NULL
    const char *name;
    bool is_specialization;      // true for specialization constants
    uint32_t spec_id;            // specialization ID
    union { ... };               // kind-specific value
};
```

Supported constant kinds: `SSIR_CONST_BOOL`, `SSIR_CONST_I32`, `SSIR_CONST_U32`, `SSIR_CONST_F32`, `SSIR_CONST_F16`, `SSIR_CONST_F64`, `SSIR_CONST_I8`, `SSIR_CONST_U8`, `SSIR_CONST_I16`, `SSIR_CONST_U16`, `SSIR_CONST_I64`, `SSIR_CONST_U64`, `SSIR_CONST_COMPOSITE`, `SSIR_CONST_NULL`.

Specialization constants are supported with `ssir_const_spec_*()` functions that set `is_specialization = true` and store the `spec_id`.

### Global Variables

```c
struct SsirGlobalVar {
    uint32_t id;
    const char *name;
    uint32_t type;           // must be a pointer type
    bool has_group;          uint32_t group;
    bool has_binding;        uint32_t binding;
    bool has_location;       uint32_t location;
    SsirBuiltinVar builtin; // NONE, VERTEX_INDEX, POSITION, FRAG_DEPTH, ...
    SsirInterpolation interp;
    SsirInterpolationSampling interp_sampling;
    bool non_writable;
    bool invariant;
    bool has_initializer;    uint32_t initializer;
};
```

Global variables represent all module-scope declarations: uniform/storage buffers (`@group`/`@binding`), shader inputs/outputs (`@location`), and built-in variables (`@builtin`).

**Supported built-in variables**: `VERTEX_INDEX`, `INSTANCE_INDEX`, `POSITION`, `FRONT_FACING`, `FRAG_DEPTH`, `SAMPLE_INDEX`, `SAMPLE_MASK`, `LOCAL_INVOCATION_ID`, `LOCAL_INVOCATION_INDEX`, `GLOBAL_INVOCATION_ID`, `WORKGROUP_ID`, `NUM_WORKGROUPS`, `POINT_SIZE`, `CLIP_DISTANCE`, `CULL_DISTANCE`, `LAYER`, `VIEWPORT_INDEX`, `FRAG_COORD`, `HELPER_INVOCATION`, `PRIMITIVE_ID`, `BASE_VERTEX`, `BASE_INSTANCE`, `SUBGROUP_SIZE`, `SUBGROUP_INVOCATION_ID`, `SUBGROUP_ID`, `NUM_SUBGROUPS`

**Interpolation modes**: `PERSPECTIVE`, `LINEAR`, `FLAT`

**Interpolation sampling**: `CENTER`, `CENTROID`, `SAMPLE`

### Functions and Blocks

```c
struct SsirFunction {
    uint32_t id;
    const char *name;
    uint32_t return_type;
    SsirFunctionParam *params;    uint32_t param_count;
    SsirLocalVar *locals;         uint32_t local_count;
    SsirBlock *blocks;            uint32_t block_count;
};

struct SsirBlock {
    uint32_t id;
    const char *name;
    SsirInst *insts;              uint32_t inst_count;
};

struct SsirInst {
    SsirOpcode op;
    uint32_t result;              // result ID (0 if none)
    uint32_t type;                // result type ID (0 if none)
    uint32_t operands[SSIR_MAX_OPERANDS];  // up to 8 inline operands
    uint8_t operand_count;
    uint32_t *extra;              // overflow operands (e.g., phi sources)
    uint16_t extra_count;
};
```

Local variables are stored separately from instructions. They represent `OpVariable` with function storage class. Each local has a pointer type and an optional initializer.

### Instruction Set

SSIR supports 70+ opcodes organized into categories:

**Arithmetic**:
`ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `REM`, `NEG`

**Matrix**:
`MAT_MUL`, `MAT_TRANSPOSE`

**Bitwise**:
`BIT_AND`, `BIT_OR`, `BIT_XOR`, `BIT_NOT`, `SHL`, `SHR`, `SHR_LOGICAL`

**Comparison**:
`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`

**Logical**:
`AND`, `OR`, `NOT`

**Composite**:
`CONSTRUCT`, `EXTRACT`, `INSERT`, `SHUFFLE`, `SPLAT`, `EXTRACT_DYN`, `INSERT_DYN`

**Memory**:
`LOAD`, `STORE`, `ACCESS` (pointer access chain), `ARRAY_LEN`

**Control Flow**:
`BRANCH`, `BRANCH_COND`, `SWITCH`, `PHI`, `RETURN`, `RETURN_VOID`, `UNREACHABLE`, `LOOP_MERGE`, `SELECTION_MERGE`

**Invocation**:
`CALL`, `BUILTIN`, `CONVERT`, `BITCAST`

**Texture**:
`TEX_SAMPLE`, `TEX_SAMPLE_BIAS`, `TEX_SAMPLE_LEVEL`, `TEX_SAMPLE_GRAD`, `TEX_SAMPLE_CMP`, `TEX_SAMPLE_CMP_LEVEL`, `TEX_SAMPLE_OFFSET`, `TEX_SAMPLE_BIAS_OFFSET`, `TEX_SAMPLE_LEVEL_OFFSET`, `TEX_SAMPLE_GRAD_OFFSET`, `TEX_SAMPLE_CMP_OFFSET`, `TEX_GATHER`, `TEX_GATHER_CMP`, `TEX_GATHER_OFFSET`, `TEX_LOAD`, `TEX_STORE`, `TEX_SIZE`, `TEX_QUERY_LOD`, `TEX_QUERY_LEVELS`, `TEX_QUERY_SAMPLES`

**Synchronization**:
`BARRIER`, `ATOMIC`, `DISCARD`

### Built-in Functions

72+ built-in functions callable via `SSIR_OP_BUILTIN`:

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Exponential**: `exp`, `exp2`, `log`, `log2`, `pow`, `sqrt`, `inverseSqrt`

**Numeric**: `abs`, `sign`, `floor`, `ceil`, `round`, `trunc`, `fract`, `min`, `max`, `clamp`, `saturate`, `mix`, `step`, `smoothstep`, `fma`

**Geometry**: `dot`, `cross`, `length`, `distance`, `normalize`, `faceForward`, `reflect`, `refract`

**Logic**: `all`, `any`, `select`

**Bit Manipulation**: `countBits`, `reverseBits`, `firstLeadingBit`, `firstTrailingBit`, `extractBits`, `insertBits`

**Derivative**: `dpdx`, `dpdy`, `fwidth`, `dpdxCoarse`, `dpdyCoarse`, `dpdxFine`, `dpdyFine`

**Floating Point**: `isInf`, `isNan`, `degrees`, `radians`, `modf`, `frexp`, `ldexp`

**Matrix**: `determinant`, `transpose`

**Packing**: `pack4x8snorm`, `pack4x8unorm`, `pack2x16snorm`, `pack2x16unorm`, `pack2x16float`, `unpack4x8snorm`, `unpack4x8unorm`, `unpack2x16snorm`, `unpack2x16unorm`, `unpack2x16float`

**Subgroup**: `subgroupBallot`, `subgroupBroadcast`, `subgroupAdd`, `subgroupMin`, `subgroupMax`, `subgroupAll`, `subgroupAny`, `subgroupShuffle`, `subgroupPrefixAdd`

### Entry Points

```c
struct SsirEntryPoint {
    SsirStage stage;              // VERTEX, FRAGMENT, COMPUTE, GEOMETRY, TESS_CONTROL, TESS_EVAL
    uint32_t function;            // function ID
    const char *name;
    uint32_t *interface;          // interface variable IDs
    uint32_t interface_count;
    uint32_t workgroup_size[3];   // compute workgroup dimensions
    bool depth_replacing;         // writes gl_FragDepth
    bool origin_upper_left;       // fragment coord origin
    bool early_fragment_tests;    // early depth/stencil
};
```

### Validation

The SSIR validation API checks module consistency:

```c
SsirValidationResult *ssir_validate(SsirModule *mod);
```

Validates:
- Type references exist and are valid
- SSA property: no ID is defined twice
- Every block ends with a terminator (branch, return, unreachable)
- PHI nodes are only at the start of blocks
- Operand types match instruction requirements
- Address space consistency for pointer operations

Error codes: `SSIR_OK`, `SSIR_ERROR_OUT_OF_MEMORY`, `SSIR_ERROR_INVALID_TYPE`, `SSIR_ERROR_INVALID_ID`, `SSIR_ERROR_INVALID_OPERAND`, `SSIR_ERROR_TYPE_MISMATCH`, `SSIR_ERROR_INVALID_BLOCK`, `SSIR_ERROR_INVALID_FUNCTION`, `SSIR_ERROR_SSA_VIOLATION`, `SSIR_ERROR_TERMINATOR_MISSING`, `SSIR_ERROR_PHI_PLACEMENT`, `SSIR_ERROR_ADDRESS_SPACE`, `SSIR_ERROR_ENTRY_POINT`

---

## Lowering (AST to SPIR-V)

### Lowering Options

```c
typedef struct {
    uint32_t spirv_version;              // e.g., 0x00010500 for SPIR-V 1.5
    WgslLowerEnv env;                    // VULKAN_1_1, VULKAN_1_2, VULKAN_1_3, WEBGPU
    WgslLowerPacking packing;            // DEFAULT, STD430, STD140
    int enable_debug_names;              // emit OpName/OpMemberName
    int enable_line_info;                // emit OpLine for source mapping
    int zero_initialize_vars;            // zero-init all variables
    int relax_block_layout;              // relaxed block layout rules
    int use_khr_shader_draw_parameters;  // KHR_shader_draw_parameters extension
    uint32_t id_bound_hint;              // pre-allocate ID space
    const char *entry_point;             // NULL = all entry points; set to compile one
    SsirLayoutRule immediate_layout;     // layout for immediate/push constant packing
} WgslLowerOptions;
```

**Target environments**:

| Environment | Enum | Description |
|------------|------|-------------|
| Vulkan 1.1 | `WGSL_LOWER_ENV_VULKAN_1_1` | Minimum Vulkan support |
| Vulkan 1.2 | `WGSL_LOWER_ENV_VULKAN_1_2` | Recommended default |
| Vulkan 1.3 | `WGSL_LOWER_ENV_VULKAN_1_3` | Latest Vulkan features |
| WebGPU | `WGSL_LOWER_ENV_WEBGPU` | WebGPU-compatible SPIR-V |

### Section-Buffered Emission

Internally, lowering accumulates SPIR-V words into separate section buffers:

1. Capabilities
2. Extensions
3. ExtInstImport
4. Memory model
5. Entry points
6. Execution modes
7. Debug info (names, lines)
8. Annotations (decorations)
9. Type declarations
10. Global variables
11. Function declarations
12. Function bodies

These are concatenated in SPIR-V spec order at serialization time.

### Two-Phase API

The lowering API supports both one-shot and two-phase usage:

**One-shot** (parse + resolve + lower + serialize in one call):
```c
WgslLowerResult wgsl_lower_emit_spirv(program, resolver, &opts, &spirv, &word_count);
```

**Two-phase** (create lower context, inspect, then serialize):
```c
WgslLower *lower = wgsl_lower_create(program, resolver, &opts);
// ... inspect SSIR, query entry points, etc.
WgslLowerResult res = wgsl_lower_serialize(lower, &spirv, &word_count);
wgsl_lower_destroy(lower);
```

The two-phase API also provides `wgsl_lower_serialize_into()` for writing into a pre-allocated buffer.

### Introspection

After lowering, you can query:

| Function | Returns |
|----------|---------|
| `wgsl_lower_get_ssir(lower)` | The internal SSIR module (read-only) |
| `wgsl_lower_entrypoints(lower, &count)` | Entry point info with SPIR-V function IDs and interface variable IDs |
| `wgsl_lower_module_features(lower)` | Required capabilities and extensions |
| `wgsl_lower_last_error(lower)` | Last error message string |
| `wgsl_lower_node_result_id(lower, node)` | SPIR-V result ID for an AST node |
| `wgsl_lower_symbol_result_id(lower, sym_id)` | SPIR-V result ID for a resolver symbol |

---

## Raising (SPIR-V to WGSL)

### Raise Options

```c
typedef struct {
    int emit_debug_comments;  // add comments with SPIR-V IDs
    int preserve_names;       // use OpName debug info for identifiers
    int inline_constants;     // inline constant values instead of declaring them
} WgslRaiseOptions;
```

### Incremental API

The raiser can be used as a single convenience call or incrementally:

**One-shot**:
```c
WgslRaiseResult wgsl_raise_to_wgsl(spirv, word_count, &opts, &wgsl_out, &error);
```

**Incremental**:
```c
WgslRaiser *r = wgsl_raise_create(spirv, word_count);
wgsl_raise_parse(r);                          // deserialize SPIR-V
wgsl_raise_analyze(r);                        // reconstruct control flow
int n = wgsl_raise_entry_point_count(r);      // inspect before emitting
const char *name = wgsl_raise_entry_point_name(r, 0);
const char *wgsl = wgsl_raise_emit(r, &opts); // emit WGSL text
wgsl_raise_destroy(r);
```

---

## SSIR Emitters

### SSIR to SPIR-V

```c
SsirToSpirvResult ssir_to_spirv(const SsirModule *mod,
                                 const SsirToSpirvOptions *opts,
                                 uint32_t **out_words,
                                 size_t *out_count);
```

Options: `spirv_version`, `enable_debug_names`, `enable_line_info`.

### SSIR to WGSL

```c
SsirToWgslResult ssir_to_wgsl(const SsirModule *mod,
                               const SsirToWgslOptions *opts,
                               char **out_wgsl,
                               char **out_error);
```

Options: `preserve_names` (use debug names for identifiers).

### SSIR to GLSL

```c
SsirToGlslResult ssir_to_glsl(const SsirModule *mod,
                                SsirStage stage,
                                const SsirToGlslOptions *opts,
                                char **out_glsl,
                                char **out_error);
```

Options: `preserve_names`, `target_opengl` (suppress Vulkan-only qualifiers like `set = N`).

Requires a `stage` parameter because GLSL programs are per-stage (unlike WGSL modules which can contain multiple entry points).

### SSIR to MSL

```c
SsirToMslResult ssir_to_msl(const SsirModule *mod,
                              const SsirToMslOptions *opts,
                              char **out_msl,
                              char **out_error);
```

Options: `preserve_names`.

### SSIR to HLSL

```c
SsirToHlslResult ssir_to_hlsl(const SsirModule *mod,
                                SsirStage stage,
                                const SsirToHlslOptions *opts,
                                char **out_hlsl,
                                char **out_error);
```

Options: `preserve_names`, `shader_model_major`, `shader_model_minor`.

---

## SPIR-V to SSIR Deserialization

```c
SpirvToSsirResult spirv_to_ssir(const uint32_t *spirv,
                                 size_t word_count,
                                 const SpirvToSsirOptions *opts,
                                 SsirModule **out_module,
                                 char **out_error);
```

Options: `preserve_names` (import OpName debug info), `preserve_locations` (keep location decorations).

Parses a SPIR-V binary into a full `SsirModule`. Handles:
- All standard SPIR-V type instructions
- Constants and specialization constants
- Global variable decorations (group, binding, location, builtin, interpolation)
- Function bodies with full control flow
- Texture sampling and storage operations
- Atomic operations and barriers

---

## MSL to SSIR Parser

```c
MslToSsirResult msl_to_ssir(const char *msl_source,
                              const MslToSsirOptions *opts,
                              SsirModule **out_module,
                              char **out_error);
```

Parses Metal Shading Language source directly into an `SsirModule`, bypassing the AST. Options: `preserve_names`.

---

## PTX to SSIR Parser

### Overview

The PTX parser converts NVIDIA PTX (Parallel Thread Execution) assembly into SSIR, enabling the full pipeline:

```
PTX source → ptx_to_ssir() → SsirModule → WGSL / GLSL / MSL / HLSL / SPIR-V
```

This enables cross-compilation of CUDA compute kernels to run on non-NVIDIA GPUs via WebGPU, Vulkan, OpenGL, Metal, or DirectX.

### Why PTX Bypasses the AST

The shared `WgslAstNode` AST was designed for C-like shading languages (WGSL, GLSL) with structured control flow, expression trees, type declarations, and lexical scoping. PTX has none of these — it is a flat, register-based assembly language with:

- **Predicated execution** instead of structured `if`/`else`
- **No expression trees** — only single instructions with register operands
- **No type declarations** — types are instruction suffixes (`.f32`, `.u64`)
- **Explicit register allocation** with parameterized naming (`%r<100>`)
- **Labels and branches** instead of structured blocks
- **State-space annotations** on every memory operation (`.global`, `.shared`, `.param`)

Forcing PTX into the WGSL AST would require either losing PTX-specific semantics or bloating the AST with assembly-only node types. The MSL parser established the precedent for direct SSIR production, and the PTX parser follows the same pattern.

### Supported PTX Features

**Module structure:**
- `.version X.Y` — PTX ISA version (6.0+)
- `.target sm_XX` — minimum GPU architecture (sm_50 through sm_100)
- `.address_size 32|64` — pointer width

**Functions and entry points:**
- `.entry` — kernel entry points (map to `SSIR_STAGE_COMPUTE`)
- `.func` — device functions (with optional return values)
- `.visible` / `.extern` linkage modifiers
- `.maxntid` / `.reqntid` — workgroup size directives

**Register declarations:**
- `.reg .type name` — named registers
- `.reg .type name<N>` — parameterized registers (`%r0` through `%r(N-1)`)
- `.reg .v2/.v4 .type name` — vector registers
- All scalar types: `.pred`, `.b8`–`.b64`, `.u8`–`.u64`, `.s8`–`.s64`, `.f16`, `.f32`, `.f64`

**Memory:**
- `.global`, `.shared`, `.const`, `.local`, `.param` state spaces
- `ld.{space}.{type}` — loads (scalar and vector `.v2`/`.v4`)
- `st.{space}.{type}` — stores (scalar and vector)
- `cvta.to.{space}` — address space conversion
- Cache control modifiers (`.ca`, `.cg`, `.cs`, `.cv` — parsed but not mapped)

**Arithmetic:**
- `add`, `sub`, `mul`, `div`, `rem` — binary arithmetic
- `mad`, `fma` — fused multiply-add
- `neg`, `abs` — unary arithmetic
- `min`, `max` — binary min/max
- `.lo`/`.hi`/`.wide` modifiers, `.rn`/`.rz`/`.rm`/`.rp` rounding modes

**Bitwise:**
- `and`, `or`, `xor`, `not` — bitwise operations
- `shl`, `shr` — shifts (arithmetic for signed types, logical for unsigned)
- `cnot` — conditional NOT

**Comparison and selection:**
- `setp.{cmp}.{type}` — set predicate (with all comparison operators: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `lo`, `ls`, `hi`, `hs`, plus float unordered variants)
- `setp.{cmp}.{combine}.{type} %p1|%p2, ...` — dual-predicate with combine (`and`/`or`/`xor`)
- `selp.{type}` — select (ternary)
- `set.{cmp}.{type}` — set (boolean result as integer)

**Control flow:**
- Labels (`LABEL_NAME:`)
- `bra` / `bra.uni` — unconditional branch
- `@%p bra` / `@!%p bra` — conditional branch via predicate guard
- `ret` — return from `.func`
- `exit` — terminate thread in `.entry`
- `call` — function call (with optional return value)

**Special registers:**
- `%tid.x/y/z` — thread index within block
- `%ctaid.x/y/z` — block index within grid
- `%ntid.x/y/z` — block dimensions (workgroup size)
- `%nctaid.x/y/z` — grid dimensions
- `%laneid` — lane within warp (subgroup invocation ID)
- `%warpid` — warp index (computed as `tid.x / 32`)

**Type conversions:**
- `cvt.{rnd}.{dst_type}.{src_type}` — numeric conversion
- Rounding modes: `.rn` (nearest), `.rz` (zero), `.rm` (minus infinity), `.rp` (plus infinity)
- `.sat` saturation modifier
- `.ftz` flush-to-zero modifier

**Math functions:**
- `rcp` — reciprocal (emitted as `1.0 / x`)
- `sqrt`, `rsqrt` — square root, reciprocal square root
- `sin`, `cos` — trigonometric
- `lg2`, `ex2` — log base 2, 2^x
- `.approx` modifier (accepted but treated same as exact — GPU math is approximate anyway)

**Atomics:**
- `atom.{space}.{op}.{type}` — atomic operations
- Operations: `add`, `min`, `max`, `and`, `or`, `xor`, `exch`, `cas`, `inc`, `dec`
- Spaces: `.global`, `.shared`

**Synchronization:**
- `bar.sync N` — workgroup barrier
- `bar.arrive`, `bar.red` — partial barrier variants
- `membar.{scope}` — memory fence (`.cta`, `.gl`, `.sys`)

### PTX API

```c
typedef enum {
    PTX_TO_SSIR_OK = 0,
    PTX_TO_SSIR_PARSE_ERROR,
    PTX_TO_SSIR_UNSUPPORTED,
} PtxToSsirResult;

typedef struct {
    int preserve_names;   // keep PTX register names as SSIR debug names
    int strict_mode;      // reject .approx instructions
} PtxToSsirOptions;

PtxToSsirResult ptx_to_ssir(const char *ptx_source,
                             const PtxToSsirOptions *opts,
                             SsirModule **out_module,
                             char **out_error);

void ptx_to_ssir_free(char *str);

const char *ptx_to_ssir_result_string(PtxToSsirResult r);
```

Usage follows the same pattern as all other parsers:

```c
SsirModule *mod = NULL;
char *error = NULL;
PtxToSsirOptions opts = { .preserve_names = 1 };

PtxToSsirResult res = ptx_to_ssir(ptx_source, &opts, &mod, &error);
if (res != PTX_TO_SSIR_OK) {
    fprintf(stderr, "PTX error: %s\n", error);
    ptx_to_ssir_free(error);
    return;
}

// Use mod with any emitter:
ssir_to_wgsl(mod, &wgsl_opts, &wgsl_out, &wgsl_err);
ssir_to_spirv(mod, &spirv_opts, &words, &word_count);
ssir_to_glsl(mod, SSIR_STAGE_COMPUTE, &glsl_opts, &glsl_out, &glsl_err);

ssir_module_destroy(mod);
```

### Architecture of ptx_parser.c

The parser is organized into clearly separated sections (~2,200 lines total):

```
┌──────────────────────────────────────────────────┐
│  1. Lexer (~250 lines)                           │
│     - PtxTokType enum (25 token types)           │
│     - PtxToken struct (type, text, line, col,    │
│       parsed int/float values)                   │
│     - plx_next() — hand-written tokenizer        │
│     - Handles: //, /* */, dot-tokens, %-regs,    │
│       0fXXXXXXXX hex-float, 0x hex-int           │
├──────────────────────────────────────────────────┤
│  2. Parser context (~100 lines)                  │
│     - PtxParser struct (lexer, SSIR module,      │
│       register map, label map, function table,   │
│       predicate state, error state)              │
│     - PtxReg: PTX register → SSIR local var      │
│     - PtxLabel: label name → block ID            │
├──────────────────────────────────────────────────┤
│  3. Register management (~80 lines)              │
│     - pp_find_reg(), pp_add_reg()                │
│     - pp_load_reg(), pp_store_reg()              │
│     - Deferred SSA via load/store to locals      │
├──────────────────────────────────────────────────┤
│  4. Type/constant helpers (~100 lines)           │
│     - pp_ptx_type() — parse .f32/.u64/etc.       │
│     - pp_const_for_type() — create typed const   │
├──────────────────────────────────────────────────┤
│  5. Special register handling (~120 lines)       │
│     - pp_ensure_builtin_global() — lazy creation │
│     - pp_load_special_reg() — %tid, %ctaid, etc. │
├──────────────────────────────────────────────────┤
│  6. Module-level parsing (~150 lines)            │
│     - pp_parse_version/target/address_size()     │
│     - pp_parse_global_decl() — .global/.shared   │
├──────────────────────────────────────────────────┤
│  7. Register declaration parsing (~70 lines)     │
│     - pp_parse_reg_decl() — .reg with            │
│       parameterized naming, vector types          │
├──────────────────────────────────────────────────┤
│  8. Instruction handlers (~800 lines)            │
│     - pp_parse_arith/unary_arith/minmax/mad/fma  │
│     - pp_parse_bitwise/shift                     │
│     - pp_parse_setp/selp                         │
│     - pp_parse_mov/ld/st/cvta/cvt                │
│     - pp_parse_math_unary                        │
│     - pp_parse_atom/bar/membar                   │
│     - pp_parse_bra/ret/exit/call                 │
├──────────────────────────────────────────────────┤
│  9. Instruction dispatch (~120 lines)            │
│     - pp_parse_instruction() — predicate guard,  │
│       label detection, opcode dispatch            │
├──────────────────────────────────────────────────┤
│ 10. Function/entry parsing (~250 lines)          │
│     - pp_parse_entry() — .entry kernel           │
│     - pp_parse_func() — .func device function    │
│     - pp_parse_param_list()                      │
│     - pp_parse_function_body()                   │
├──────────────────────────────────────────────────┤
│ 11. Top-level + public API (~100 lines)          │
│     - pp_parse_toplevel()                        │
│     - ptx_to_ssir(), ptx_to_ssir_free()          │
│     - ptx_to_ssir_result_string()                │
└──────────────────────────────────────────────────┘
```

### PTX Type System Mapping

PTX types appear as dot-suffixes on instructions and register declarations. They map to SSIR types as follows:

| PTX Type | Width | SSIR Type | Description |
|----------|-------|-----------|-------------|
| `.pred` | 1 bit | `SSIR_TYPE_BOOL` | Predicate (boolean) |
| `.b8` | 8 bits | `SSIR_TYPE_U8` | Untyped 8-bit |
| `.b16` | 16 bits | `SSIR_TYPE_U16` | Untyped 16-bit |
| `.b32` | 32 bits | `SSIR_TYPE_U32` | Untyped 32-bit |
| `.b64` | 64 bits | `SSIR_TYPE_U64` | Untyped 64-bit |
| `.u8`–`.u64` | 8–64 bits | `SSIR_TYPE_U8`–`SSIR_TYPE_U64` | Unsigned integers |
| `.s8`–`.s64` | 8–64 bits | `SSIR_TYPE_I8`–`SSIR_TYPE_I64` | Signed integers |
| `.f16` | 16 bits | `SSIR_TYPE_F16` | IEEE 754 half |
| `.f32` | 32 bits | `SSIR_TYPE_F32` | IEEE 754 single |
| `.f64` | 64 bits | `SSIR_TYPE_F64` | IEEE 754 double |

Untyped `.bN` types map to unsigned integers of the same width. The instruction context determines whether signed or unsigned semantics apply (e.g., `shr.b32` uses logical shift).

Vector types (`.v2`, `.v4`) map to `SSIR_TYPE_VEC` with the appropriate element type and component count.

### Register Model

PTX uses virtual registers that can be freely redefined. SSIR uses SSA form where each value has a unique ID. The parser bridges this gap using **deferred SSA construction with load/store to local variables**:

1. Each PTX register declaration creates an `SsirLocalVar` with `SSIR_ADDR_FUNCTION` storage
2. Each register read emits `SSIR_OP_LOAD` from that local variable
3. Each register write emits `SSIR_OP_STORE` to that local variable
4. SSIR consumers (emitters) handle this load/store pattern — it's the same pattern that SPIR-V uses

```
PTX:                          SSIR:
.reg .f32 %a, %b, %c;        %var_a = local_var(ptr<function, f32>)
                              %var_b = local_var(ptr<function, f32>)
                              %var_c = local_var(ptr<function, f32>)

add.f32 %c, %a, %b;          %t1 = load %var_a
                              %t2 = load %var_b
                              %t3 = add.f32 %t1, %t2
                              store %var_c, %t3

mul.f32 %c, %c, %a;          %t4 = load %var_c    // reads new value of %c
                              %t5 = load %var_a
                              %t6 = mul.f32 %t4, %t5
                              store %var_c, %t6
```

**Parameterized registers** (`%r<100>`) expand to `%r0` through `%r99`, each getting its own local variable. They are allocated lazily when first referenced.

### Kernel Parameter Convention

PTX kernel parameters are flat values (pointers passed as `u64`, scalars passed by value). GPU shader APIs expect typed buffer bindings with group/binding decorations. The parser uses this convention:

**Pointer parameters** (`.param .u64`) become storage buffer bindings:
```
.param .u64 input_ptr  →  @group(0) @binding(0) var<storage> input_ptr: array<u8>
.param .u64 output_ptr →  @group(0) @binding(1) var<storage> output_ptr: array<u8>
```

**Scalar parameters** (`.param .u32`, `.param .f32`, etc.) are passed as function parameters and accessed via `ld.param` from local variables that hold the parameter value.

Binding indices are assigned sequentially starting from 0, all in descriptor set (group) 0. Each `.param .u64` gets its own storage buffer binding.

### Special Register Mapping

| PTX Special Register | SSIR Built-in | WGSL Equivalent |
|---------------------|---------------|-----------------|
| `%tid.x/y/z` | `SSIR_BUILTIN_LOCAL_INVOCATION_ID` + extract | `local_invocation_id.x/y/z` |
| `%ctaid.x/y/z` | `SSIR_BUILTIN_WORKGROUP_ID` + extract | `workgroup_id.x/y/z` |
| `%nctaid.x/y/z` | `SSIR_BUILTIN_NUM_WORKGROUPS` + extract | `num_workgroups.x/y/z` |
| `%ntid.x/y/z` | Constant (if `.reqntid` set) or `NUM_WORKGROUPS` | Workgroup size constant |
| `%laneid` | `SSIR_BUILTIN_SUBGROUP_INVOCATION_ID` | `subgroup_invocation_id` |
| `%warpid` | Computed: `tid.x / 32` | (no direct equivalent) |

Built-in globals are created lazily on first access. The `%ntid` register uses the workgroup size from `.reqntid`/`.maxntid` directives when available (emitted as constants), falling back to a built-in variable when the size is dynamic.

### Control Flow Translation

PTX has unstructured control flow (labels + branches). SSIR expects basic blocks with explicit terminator instructions.

**Block splitting**: The parser creates new basic blocks at:
- Label definitions (`LABEL_NAME:`) — the current block gets an implicit branch to the label's block
- Branch instructions (`bra`, `ret`, `exit`) — terminate the current block
- Instructions following a branch — start a new unreachable block

**Branch resolution**: Labels may be forward-referenced. The parser creates blocks for unknown labels on first reference via `pp_get_or_create_label()`, which uses a label-to-block-ID map.

**Structured control flow reconstruction is NOT needed**: SSIR basic blocks with `SSIR_OP_BRANCH` and `SSIR_OP_BRANCH_COND` terminators are sufficient. The downstream emitters (`ssir_to_wgsl`, `ssir_to_glsl`, etc.) already reconstruct structured loops and selections from unstructured CFGs.

### Predicated Execution

Any PTX instruction can be guarded by a predicate: `@%p inst ...` or `@!%p inst ...`.

**Conditional branches** (`@%p bra LABEL`) map directly to `SSIR_OP_BRANCH_COND`:
```
@%p1 bra TARGET;  →  branch_cond %p1 → TARGET, fallthrough_block
```

This is the most common predicated pattern and is handled with zero overhead.

**Non-branch predicated instructions** are currently handled at the PTX parsing level — the predicate guard is parsed but the instruction executes unconditionally in SSIR. For most compute kernel use cases, predicated non-branch instructions are rare; the common pattern is `setp` + `@%p bra`.

### PTX Memory Operations

**Load instructions** (`ld.{space}.{type}`):

| PTX | SSIR |
|-----|------|
| `ld.global.f32 %f, [%rd]` | `SSIR_OP_LOAD` with pointer in `SSIR_ADDR_STORAGE` |
| `ld.shared.f32 %f, [smem]` | `SSIR_OP_LOAD` with pointer in `SSIR_ADDR_WORKGROUP` |
| `ld.param.u64 %rd, [name]` | Load from local variable holding the parameter value |
| `ld.const.f32 %f, [cst]` | `SSIR_OP_LOAD` with pointer in `SSIR_ADDR_UNIFORM` |

**Store instructions** (`st.{space}.{type}`):

| PTX | SSIR |
|-----|------|
| `st.global.f32 [%rd], %f` | `SSIR_OP_STORE` with pointer in `SSIR_ADDR_STORAGE` |
| `st.shared.f32 [addr], %f` | `SSIR_OP_STORE` with pointer in `SSIR_ADDR_WORKGROUP` |

**Vector load/store**: `ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [addr]` loads a `vec4<f32>` then extracts components. `st.global.v4.f32 [addr], {%f1,...,%f4}` constructs a vector then stores.

**Address arithmetic**: PTX computes effective addresses with explicit arithmetic (`mul.lo.u64 %off, %idx, 4; add.u64 %addr, %base, %off`). The parser preserves this arithmetic as-is in SSIR rather than attempting to reconstruct typed array indexing.

### Instruction Mapping Reference

**Arithmetic:**

| PTX | SSIR |
|-----|------|
| `add.{type}` | `SSIR_OP_ADD` |
| `sub.{type}` | `SSIR_OP_SUB` |
| `mul[.lo].{type}` | `SSIR_OP_MUL` |
| `div.{type}` | `SSIR_OP_DIV` |
| `rem.{type}` | `SSIR_OP_REM` |
| `neg.{type}` | `SSIR_OP_NEG` |
| `abs.{type}` | `SSIR_OP_BUILTIN(ABS)` |
| `min.{type}` | `SSIR_OP_BUILTIN(MIN)` |
| `max.{type}` | `SSIR_OP_BUILTIN(MAX)` |
| `mad[.lo].{type}` | `SSIR_OP_MUL` + `SSIR_OP_ADD` (integer) or `SSIR_OP_BUILTIN(FMA)` (float) |
| `fma.{type}` | `SSIR_OP_BUILTIN(FMA)` |

**Bitwise:**

| PTX | SSIR |
|-----|------|
| `and.{type}` | `SSIR_OP_BIT_AND` (or `SSIR_OP_AND` for `.pred`) |
| `or.{type}` | `SSIR_OP_BIT_OR` (or `SSIR_OP_OR` for `.pred`) |
| `xor.{type}` | `SSIR_OP_BIT_XOR` |
| `not.{type}` | `SSIR_OP_BIT_NOT` (or `SSIR_OP_NOT` for `.pred`) |
| `shl.{type}` | `SSIR_OP_SHL` |
| `shr.{signed}` | `SSIR_OP_SHR` (arithmetic) |
| `shr.{unsigned}` | `SSIR_OP_SHR_LOGICAL` |

**Comparison:**

| PTX | SSIR |
|-----|------|
| `setp.eq` | `SSIR_OP_EQ` → store to predicate register |
| `setp.ne` | `SSIR_OP_NE` |
| `setp.lt` / `setp.lo` | `SSIR_OP_LT` |
| `setp.le` / `setp.ls` | `SSIR_OP_LE` |
| `setp.gt` / `setp.hi` | `SSIR_OP_GT` |
| `setp.ge` / `setp.hs` | `SSIR_OP_GE` |
| `selp.{type}` | `SSIR_OP_BUILTIN(SELECT)` |

**Math:**

| PTX | SSIR |
|-----|------|
| `rcp.{type}` | `SSIR_OP_DIV` (1.0 / x) |
| `sqrt.{type}` | `SSIR_OP_BUILTIN(SQRT)` |
| `rsqrt.{type}` | `SSIR_OP_BUILTIN(INVERSESQRT)` |
| `sin.{type}` | `SSIR_OP_BUILTIN(SIN)` |
| `cos.{type}` | `SSIR_OP_BUILTIN(COS)` |
| `lg2.{type}` | `SSIR_OP_BUILTIN(LOG2)` |
| `ex2.{type}` | `SSIR_OP_BUILTIN(EXP2)` |

**Atomics:**

| PTX | SSIR |
|-----|------|
| `atom.{space}.add.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_ADD)` |
| `atom.{space}.min.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_MIN)` |
| `atom.{space}.max.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_MAX)` |
| `atom.{space}.and.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_AND)` |
| `atom.{space}.or.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_OR)` |
| `atom.{space}.xor.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_XOR)` |
| `atom.{space}.exch.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_EXCHANGE)` |
| `atom.{space}.cas.{type}` | `SSIR_OP_ATOMIC(SSIR_ATOMIC_COMPARE_EXCHANGE)` |

**Synchronization:**

| PTX | SSIR |
|-----|------|
| `bar.sync N` | `SSIR_OP_BARRIER(SSIR_BARRIER_WORKGROUP)` |
| `membar.cta` | `SSIR_OP_BARRIER(SSIR_BARRIER_WORKGROUP)` |
| `membar.gl` | `SSIR_OP_BARRIER(SSIR_BARRIER_STORAGE)` |

**Type conversion:**

| PTX | SSIR |
|-----|------|
| `cvt.{dst}.{src}` | `SSIR_OP_CONVERT` |
| `cvta.to.{space}` | No-op (address identity in our model) |

### PTX Usage Examples

**PTX → WGSL:**
```c
PtxToSsirOptions ptx_opts = { .preserve_names = 1 };
SsirModule *mod = NULL;
char *err = NULL;

ptx_to_ssir(ptx_source, &ptx_opts, &mod, &err);

SsirToWgslOptions wgsl_opts = { .preserve_names = 1 };
char *wgsl = NULL;
ssir_to_wgsl(mod, &wgsl_opts, &wgsl, &err);

printf("%s\n", wgsl);
ssir_to_wgsl_free(wgsl);
ssir_module_destroy(mod);
```

**PTX → SPIR-V:**
```c
ptx_to_ssir(ptx_source, &opts, &mod, &err);

SsirToSpirvOptions spirv_opts = { .enable_debug_names = 1 };
uint32_t *words = NULL;
size_t count = 0;
ssir_to_spirv(mod, &spirv_opts, &words, &count);

// words is now a valid SPIR-V module
ssir_to_spirv_free(words);
ssir_module_destroy(mod);
```

**PTX → GLSL 450:**
```c
ptx_to_ssir(ptx_source, &opts, &mod, &err);

SsirToGlslOptions glsl_opts = { .preserve_names = 1 };
char *glsl = NULL;
ssir_to_glsl(mod, SSIR_STAGE_COMPUTE, &glsl_opts, &glsl, &err);

printf("%s\n", glsl);
ssir_to_glsl_free(glsl);
ssir_module_destroy(mod);
```

### Limitations and Future Work

**Current limitations:**
- **Pointer arithmetic preserved as-is**: PTX's raw pointer arithmetic (`base + index * sizeof(T)`) is not reconstructed into typed array indexing. Output is correct but more verbose.
- **No texture/surface instructions**: `tex`, `tld4`, `suld`, `sust` are not supported (requires SSIR texture infrastructure mapping).
- **No tensor core / WMMA instructions**: Matrix multiply-accumulate operations are out of scope.
- **No warp-level primitives**: `shfl`, `vote`, `match`, `redux` are not implemented.
- **No indirect calls**: Function pointers are not supported.
- **Predicated non-branch instructions**: Currently parsed but not wrapped in conditional blocks (the predicate guard is consumed, but the instruction executes unconditionally in SSIR).

**Planned future work:**
- **Typed array reconstruction**: Pattern-match `base + idx * sizeof(T)` → `array[idx]` for cleaner output
- **Full predicate lowering**: Insert conditional blocks for predicated non-branch instructions
- **Dead store elimination**: Remove redundant register stores
- **Constant folding**: Evaluate compile-time-known expressions
- **Warp-level primitives**: Map `shfl` to subgroup operations
- **`ssir_to_ptx` emitter**: Enable WGSL/GLSL → PTX compilation for CUDA targets

---

## Immediates (Push Constants)

WGSL lacks native push constant support. The `var<immediate>` extension provides a WGSL-native syntax for Vulkan push constants. The compiler automatically generates per-entry-point push constant blocks with correct layout and SPIR-V decorations.

### Language Extensions

Enable via WGSL `enable` directives, tracked as bitflags on the program AST node:

| Extension | Flag | What it enables |
|-----------|------|-----------------|
| `immediate_address_space` | `WGSL_EXT_IMMEDIATE_ADDRESS_SPACE` | `var<immediate>` with scalar, vector, matrix, and plain struct types |
| `immediate_arrays` | `WGSL_EXT_IMMEDIATE_ARRAYS` | Additionally allows arrays (implies `immediate_address_space`) |

### Syntax and Semantics

**Declaration**: Module-scope `var<immediate>` with no `@group`/`@binding`:

```wgsl
enable immediate_address_space;

var<immediate> scale: f32;
var<immediate> offset: vec2f;
var<immediate> transform: mat4x4f;
```

**Pointer support**: `ptr<immediate, T>` is valid in function parameters:

```wgsl
fn apply(p: ptr<immediate, f32>) -> f32 { return *p; }
```

**Semantics**:
- Address space is `immediate` — uniform, read-only
- No `@group`/`@binding` attributes
- Each entry point receives only the immediates it transitively uses (through its call graph)
- Members are packed in source declaration order

### Resolver API

**Symbol kind**: `var<immediate>` declarations appear as `WGSL_SYM_IMMEDIATE` in the symbol table. They show up in `wgsl_resolver_globals()` and `wgsl_resolver_entrypoint_globals()` but never in `wgsl_resolver_binding_vars()` (since they have no group/binding).

**Layout reflection**:

```c
typedef struct WgslImmediateInfo {
    const char *name;              // variable name
    int type_size;                 // size in bytes
    int offset;                    // byte offset in push constant block
    int alignment;                 // alignment in bytes
    const WgslAstNode *decl_node;  // AST declaration node
} WgslImmediateInfo;

const WgslImmediateInfo *wgsl_resolver_entrypoint_immediates(
    const WgslResolver *r,
    const char *entry_name,
    SsirLayoutRule layout,
    int *out_count);
```

The returned array contains only the immediate variables used (transitively) by the named entry point, with offsets computed according to the requested layout rule. Free with `wgsl_resolve_free()`.

**Per-entry-point isolation**: Given four entry points where `main1` calls `foo()` which uses `a`, `main3` calls `bar()` which uses `b`, and `main4` uses neither:
- `wgsl_resolver_entrypoint_immediates(r, "main1", ...)` returns `[{a, offset=0}]`
- `wgsl_resolver_entrypoint_immediates(r, "main3", ...)` returns `[{b, offset=0}]`
- `wgsl_resolver_entrypoint_immediates(r, "main4", ...)` returns count=0

### Lowering Behavior

When `opts.entry_point` is set, the lowering pass:

1. Walks the entry point's transitive call graph to find all referenced `var<immediate>` symbols
2. Creates a synthetic struct type with those members, applying the requested layout rule
3. Decorates the struct with `Block` and each member with its computed `Offset`
4. Emits an `OpVariable` with `PushConstant` storage class
5. Replaces all loads from immediate variables with `OpAccessChain` into the push constant struct

Entry points that use no immediates get no push constant variable at all.

Generated SPIR-V for `var<immediate> a: f32; var<immediate> b: u32;`:

```
OpDecorate %_PushConstants Block
OpMemberDecorate %_PushConstants 0 Offset 0    ; a: f32
OpMemberDecorate %_PushConstants 1 Offset 4    ; b: u32
%_PushConstants = OpTypeStruct %float %uint
%pc = OpVariable %_ptr_PushConstant__PushConstants PushConstant
```

### Layout Rules

| Rule | Enum | Alignment behavior | Notes |
|------|------|--------------------|-------|
| std430 | `SSIR_LAYOUT_STD430` | Natural alignment; vec3 aligns to 16 bytes | Default, compatible with Vulkan push constants |
| Scalar | `SSIR_LAYOUT_SCALAR` | 4-byte alignment for all scalar components | Requires `VK_EXT_scalar_block_layout` |

Example layout for `var<immediate> a: f32; var<immediate> b: vec3f;`:

| Member | std430 offset | Scalar offset |
|--------|--------------|---------------|
| `a` (f32) | 0 | 0 |
| `b` (vec3f) | 16 | 4 |

---

## Memory Management

All memory allocation in simple_wgsl goes through overridable macros. Define these before including `simple_wgsl.h` to use custom allocators:

**AST allocators** (used by parsers):
```c
#define NODE_ALLOC(T)      my_alloc(sizeof(T))   // typed allocation
#define NODE_MALLOC(SZ)    my_alloc(SZ)           // raw allocation
#define NODE_REALLOC(P,SZ) my_realloc(P, SZ)      // reallocation
#define NODE_FREE(P)       my_free(P)              // deallocation
```

**SSIR allocators** (used by SSIR module and all emitters):
```c
#define SSIR_MALLOC(sz)     my_alloc(sz)
#define SSIR_REALLOC(p,sz)  my_realloc(p, sz)
#define SSIR_FREE(p)        my_free(p)
```

Default: `calloc`/`realloc`/`free` from the C standard library. Note that `NODE_ALLOC` and `NODE_MALLOC` use `calloc` (zero-initialized) by default.

**Ownership rules**:
- `wgsl_parse()` returns an AST owned by the caller. Free with `wgsl_free_ast()`.
- `wgsl_resolver_build()` returns a resolver owned by the caller. Free with `wgsl_resolver_free()`.
- `wgsl_lower_create()` returns a lower context that owns its internal SSIR. Free with `wgsl_lower_destroy()`.
- `wgsl_lower_emit_spirv()` allocates the output SPIR-V buffer. Free with `wgsl_lower_free()`.
- `spirv_to_ssir()` allocates the output module. Free with `ssir_module_destroy()`.
- All `ssir_to_*()` output strings/buffers are freed with their respective `*_free()` functions.
- `wgsl_raise_to_wgsl()` output is freed with `wgsl_raise_free()`.

---

## Language Mapping Tables

### WGSL to GLSL Mapping

| WGSL / SSIR | GLSL 450 |
|-------------|----------|
| `f32` / `i32` / `u32` | `float` / `int` / `uint` |
| `vec4<f32>` | `vec4` |
| `vec3<i32>` | `ivec3` |
| `vec2<u32>` | `uvec2` |
| `mat4x4<f32>` | `mat4` |
| `mat3x3<f32>` | `mat3` |
| `@location(N) param` | `layout(location = N) in/out` |
| `@builtin(position)` (vertex out) | `gl_Position` |
| `@builtin(position)` (fragment in) | `gl_FragCoord` |
| `@builtin(vertex_index)` | `gl_VertexIndex` |
| `@builtin(instance_index)` | `gl_InstanceIndex` |
| `@builtin(frag_depth)` | `gl_FragDepth` |
| `@builtin(front_facing)` | `gl_FrontFacing` |
| `@builtin(global_invocation_id)` | `gl_GlobalInvocationID` |
| `@builtin(local_invocation_id)` | `gl_LocalInvocationID` |
| `@builtin(workgroup_id)` | `gl_WorkGroupID` |
| `@group(G) @binding(B) var<uniform>` | `layout(std140, set = G, binding = B) uniform` |
| `@group(G) @binding(B) var<storage>` | `layout(std430, set = G, binding = B) buffer` |
| `@workgroup_size(X,Y,Z)` | `layout(local_size_x=X, ...) in;` |
| Entry point function | `void main()` |
| `inverseSqrt` | `inversesqrt` |
| `countOneBits` | `bitCount` |
| `dpdx` / `dpdy` | `dFdx` / `dFdy` |
| `textureSample(t, s, c)` | `texture(t, c)` |
| `textureLoad(t, c, l)` | `texelFetch(t, c, l)` |

---

## Error Handling

Every major API function returns a result enum. The pattern is consistent across all modules:

```c
// Check the result code
if (result != XXX_OK) {
    // Get the error message (if the API provides one)
    const char *msg = xxx_last_error(context);
    fprintf(stderr, "Error: %s\n", msg);
}
```

Result enums follow the naming convention `<Module>Result` with values `*_OK = 0` for success and specific error codes for different failure modes.

All emitters provide a `*_result_string()` function that converts a result code to a human-readable string:

```c
ssir_to_spirv_result_string(result)
ssir_to_wgsl_result_string(result)
ssir_to_glsl_result_string(result)
ssir_to_msl_result_string(result)
ssir_to_hlsl_result_string(result)
spirv_to_ssir_result_string(result)
msl_to_ssir_result_string(result)
ptx_to_ssir_result_string(result)
```
