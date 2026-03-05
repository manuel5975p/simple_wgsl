# Simplifications: Repetitive Patterns in simple_wgsl

Comprehensive audit of duplicated code across the entire codebase. Organized by
impact (estimated lines saved). Each section includes exact file locations and
concrete extraction strategy.

---

## 1. SPIR-V Parser: ~600 lines duplicated between two files

`wgsl_raise.c` and `spirv_to_ssir.c` are parallel implementations of SPIR-V
binary -> internal representation. They share:

### 1a. Identical struct definitions (6 struct pairs)

| wgsl_raise.c | spirv_to_ssir.c |
|---|---|
| `SpvTypeInfo` (L49-100) | `VtsSpvTypeInfo` (L53-104) |
| `SpvDecorationEntry` (L102-107) | `VtsSpvDecorationEntry` (L106-111) |
| `SpvMemberDecoration` (L109-114) | `VtsSpvMemberDecoration` (L113-118) |
| `SpvIdInfo` (L116-140) | `VtsSpvIdInfo` (L120-146) |
| `SpvBasicBlock` (L142-148) | `VtsSpvBasicBlock` (L148-155) |
| `SpvFunction` (L150-163) | `VtsSpvFunction` (L157-170) |

Every field is identical, only the prefix differs.

### 1b. Identical helper functions (5 function pairs)

- `read_string` (L315-332) vs `vts_read_string` (L260-277): same SPIR-V
  string decoding, byte-for-byte identical algorithm
- `add_decoration` (L444-481) vs `vts_add_decoration` (L280-315): same
  grow-and-append
- `add_member_decoration` / `has_decoration` / `get_decoration`: duplicated
- `add_function` (L504-519) vs `vts_add_function` (L361-376)
- `add_block` (L522-537) vs `vts_add_block` (L379-394)
- `add_block_instr` (L538-548) vs `vts_add_block_instr` (L395-410)

### 1c. Identical SPIR-V opcode dispatch (~500 lines)

The main parse loop (`wgsl_raise_parse` L550-1146 vs `parse_spirv` L421-1116)
handles every type opcode (SpvOpTypeVoid through SpvOpTypeSampler), constants,
variables, entry points, execution modes with identical switch arms. Example:

```c
// wgsl_raise.c L648-660 -- identical in spirv_to_ssir.c L508-516
case SpvOpTypeVoid:
    if (operand_count >= 1) {
        uint32_t id = operands[0];
        if (id < r->id_bound) {
            r->ids[id].kind = SPV_ID_TYPE;
            r->ids[id].type_info.kind = SPV_TYPE_VOID;
        }
    }
    break;
```

### Extraction strategy

Create `spv_reader_internal.h` with:
- Shared struct definitions (parameterized prefix via typedef or just one name)
- Shared helper functions
- A `spv_parse_types_and_decorations()` that populates the shared structs
- Both consumers call this, then do their own IR-specific conversion pass

Estimated savings: ~600 lines removed, ~200 shared lines added.

---

## 2. Text Backend String Buffer: ~240 lines (4x copy-paste)

All four text emitters (WGSL, GLSL, MSL, HLSL) implement the same growable
`char` buffer with identical `reserve`, `append`, `appendf`, `indent`, `init`,
`free` functions. Only the struct name and allocator macro differ.

| File | Struct | Prefix | Lines |
|---|---|---|---|
| ssir_to_wgsl.c | `StwStringBuffer` | `stw_sb_` | L32-86 |
| ssir_to_glsl.c | `GlslBuf` | `gb_` | L33-83 |
| ssir_to_msl.c | `MslBuf` | `mb_` | L33-83 |
| ssir_to_hlsl.c | `HlslBuf` | `hb_` | L32-82 |

All have `{ char *data; size_t len; size_t cap; int indent; }` and the same
six functions. The `appendf` uses `char buf[1024]` + `vsnprintf` in all four.

### Extraction strategy

A single `ssir_textbuf.h` static-inline header parameterized by allocator
macros (defaulting to `malloc/realloc/free`). Each backend `#include`s it after
setting its allocator macros if non-default.

---

## 3. Text Backend Context: ~160 lines (4x copy-paste)

All four text emitters have near-identical context structs and init/free:

```c
typedef struct {
    const SsirModule *mod;
    SsirToXOptions opts;
    XBuf sb;
    char **id_names;
    uint32_t id_names_cap;
    const SsirFunction *current_func;
    uint32_t *use_counts;
    SsirInst **inst_map;
    uint32_t inst_map_cap;
    char last_error[256];
} XCtx;
```

The init (memset + malloc id_names) and free (loop-free id_names + free
use_counts + free inst_map + free sb) are identical across WGSL (L96-139),
GLSL (L91-127), MSL (L105-150), HLSL (L90-126).

### Supporting duplications that follow from the shared context

- `find_param` / `find_local`: 4x identical (WGSL L545-563, GLSL L538-551,
  MSL L578-591, HLSL L536-549)
- `get_id_name` fallback: 4x identical static-buf pattern
- Inst-map + use-count build: 5x identical (4 backends + HLSL entry-point
  path at L1470-1492)
- Block emission state (`BlockEmitState` struct + `emit_block` function):
  3x identical (WGSL L1209-1374, GLSL L1113-1276, MSL L1300-1494)
- Name assignment loop (globals -> `_g%u`, params -> `_p%u`, locals -> `_l%u`):
  3x identical (WGSL L1562-1600, GLSL L1585-1640, HLSL L1280-1318)
- Reserved-word check + name mangling: 3x identical (GLSL L136-163, MSL
  L159-186, HLSL L141-167)

### Extraction strategy

A shared `ssir_backend_common.h` providing:
- `SsirTextCtx` base struct with common fields
- `ssir_text_ctx_init()` / `ssir_text_ctx_free()`
- `ssir_text_find_param()` / `ssir_text_find_local()`
- `ssir_text_build_inst_map(ctx, fn)` (the ~20-line block that appears 5x)
- `ssir_text_assign_default_names(ctx, mod, set_name_cb)`

Each backend embeds `SsirTextCtx` as first member and adds backend-specific
fields.

---

## 4. Identical `emit_expr` Cases Across 4 Backends: ~400+ lines

These SSIR ops have token-for-token identical handling in all four text backends
(only the emit function prefix differs):

| Op | Lines per backend | Backends |
|---|---|---|
| `SSIR_OP_ACCESS` (chain traversal) | ~55 | 4 (WGSL L790, GLSL L774, MSL L853, HLSL L764) |
| `SSIR_OP_CONSTRUCT` | ~17 | 4 (WGSL L709, GLSL L699, MSL L778, HLSL L735) |
| `SSIR_OP_EXTRACT` (swizzle) | ~12 | 4 (WGSL L727, GLSL L716, MSL L795, HLSL L752) |
| `SSIR_OP_SHUFFLE` | ~20 | 3 (WGSL L745, GLSL L733, MSL L812) |
| `SSIR_OP_CALL` | ~16 | 3 (WGSL L854, GLSL L837, MSL L916) |
| `SSIR_OP_SPLAT` | ~5 | 3 |
| `SSIR_OP_EXTRACT_DYN` | ~5 | 3 |
| `SSIR_OP_SHR_LOGICAL` (signed cast) | ~25 | 4 (differs only in cast syntax) |
| `SSIR_OP_INSERT_DYN` (in emit_stmt) | ~12 | 4 |
| `SSIR_OP_STORE` (in emit_stmt) | ~6 | 4 |
| Default materialization | ~25 | 3 |

Total: ~400 lines duplicated 3-4 times = ~1200 lines of redundancy.

### Extraction strategy

Factor the shared logic into callbacks. For example, `SSIR_OP_ACCESS` needs
only two callbacks: `emit_expr(ctx, id, buf)` and `append(buf, str)`. A shared
`ssir_emit_access_chain(mod, inst, emit_expr_cb, append_cb, user_ctx)` could
handle all four backends.

For simpler ops (CONSTRUCT, EXTRACT, SPLAT, EXTRACT_DYN), a macro-based
approach works since they only call `emit_type`, `emit_expr`, and `append`.

---

## 5. Builtin Function Name Tables: ~320 lines (4x parallel switch)

All four text backends have a `builtin_func_to_XX()` function mapping
`SsirBuiltinId` to target-language name strings. They cover the same ~60 cases
in the same order:

- WGSL: L186-277
- GLSL: L213-304
- MSL: L245-336
- HLSL: L400-484

### Extraction strategy

A single `SsirBuiltinNameTable` struct with four string columns
(`wgsl_name, glsl_name, msl_name, hlsl_name`) indexed by `SsirBuiltinId`.
One table definition, four `const char *` lookups. Also enables runtime backend
selection.

---

## 6. Parser Lexer Infrastructure: ~400 lines (4x copy-paste)

The four parsers (wgsl, glsl, msl, ptx) each implement the same lexer skeleton:

### 6a. `strndup` / `strdup` (4x identical, ~12 lines each)

- wgsl_parser.c L10-22, glsl_parser.c L25-37, msl_parser.c L37-49,
  ptx_parser.c L27-38

### 6b. Lexer struct + advance + skip_ws_comments (~80 lines each)

All have `{ const char *src; size_t pos; int line; int col; }` and identical
newline-tracking advance. The comment-skipping handles `//` and `/* */` the
same way (glsl/msl add `#` preprocessor lines).

- wgsl L131-200, glsl L144-212, msl L121-194, ptx L64-102

### 6c. `is_ident_start` / `is_ident_part` (4x identical)

### 6d. `make_token` (3x identical, ptx inlines it)

### 6e. `check` / `match` / `expect` / `parse_error` (4x identical)

- wgsl L526-567, glsl L511-552, msl L500-541, ptx L297-319

### 6f. Token string comparison (4x identical)

`tok_eq` / `check_ident_text` / `mp_match_ident` / `dot_eq`: all do
`length == strlen(s) && strncmp(start, s, length) == 0`.

### Extraction strategy

A shared `shader_lexer.h` providing:
- `ShaderLexer` struct + `shader_lexer_advance` / `shader_lexer_skip_ws`
- `shader_tok_eq(token, str)`
- Parameterize comment style (C-style only vs also `#`)
- Each parser creates its own token types but shares the lexer core

---

## 7. Expression Parsing Ladder: ~600 lines (2x in wgsl/glsl)

The 10-level binary precedence chain (logical_or -> logical_and -> bitwise_or
-> xor -> bitwise_and -> equality -> relational -> shift -> additive ->
multiplicative) is duplicated between `wgsl_parser.c` L1206-1502 and
`glsl_parser.c` L1722-2023. Each level is a `while(match(op)) { build
BINARY node; }` loop. All 10 functions are structurally identical, producing
the same AST node types.

The unary/postfix chain (8 unary ops + `.`/`[]`/`()` postfix) is also
duplicated: wgsl L1505-1641 vs glsl L2025-2145.

### Extraction strategy

Replace the 10-function ladder with a precedence-climbing parser (as
`msl_parser.c` already does at L1560-1639 in ~80 lines). A table
`{ token_type, op_string, precedence, assoc }` drives a single `parse_expr`
function.

---

## 8. `ssir_const_T` and `ssir_type_T` Families: ~250 lines in ssir.c

### 8a. 12 scalar constant constructors (L762-966)

All follow the same pattern: find matching type, linear-scan dedup, add:

```c
uint32_t ssir_const_T(SsirModule *mod, ctype val) {
    uint32_t type_id = ssir_type_T(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_T &&
            mod->constants[i].T_val == val) return mod->constants[i].id;
    }
    SsirConstant c = {.type = type_id, .kind = SSIR_CONST_T, .T_val = val};
    return ssir_add_constant(mod, &c);
}
```

12 copies (i32, u32, f32, f16, f64, i8, u8, i16, u16, i64, u64, bool).

### 8b. 13 scalar type constructors (L357-472)

All follow: `find_type(SSIR_TYPE_T) -> if found return -> add_type(T)`.

### Extraction strategy

An X-macro table:

```c
#define SSIR_SCALAR_TYPES(X) \
    X(void,  SSIR_TYPE_VOID,  /* no const */) \
    X(bool,  SSIR_TYPE_BOOL,  SSIR_CONST_BOOL, bool_val, bool) \
    X(i32,   SSIR_TYPE_I32,   SSIR_CONST_I32,  i32_val,  int32_t) \
    X(u32,   SSIR_TYPE_U32,   SSIR_CONST_U32,  u32_val,  uint32_t) \
    ...
```

Generates both `ssir_type_T` and `ssir_const_T` from one definition.

---

## 9. Dynamic Array Growth: ~25 inline sites

`ssir.c` has `ssir_grow_array` (L16-27) but no other file uses it. Every other
file re-implements `cap ? cap * 2 : N` inline:

- ptx_lower.c: 11 sites
- wgsl_lower.c: 10+ sites
- spirv_to_ssir.c: ~8 sites
- wgsl_raise.c: ~6 sites
- msl_parser.c: ~5 inline sites

### Extraction strategy

A shared macro in `simple_wgsl.h` or a new `sw_util.h`:

```c
#define SW_GROW(ptr, count, cap, T, alloc_fn) do { \
    if ((count) >= (cap)) { \
        uint32_t nc = (cap) ? (cap) * 2 : 8; \
        (ptr) = (T *)alloc_fn((ptr), nc * sizeof(T)); \
        (cap) = nc; \
    } \
} while(0)
```

---

## 10. SpvBuiltIn <-> SsirBuiltinVar Mapping: 3x encoding

The same bijection between SPIR-V builtins and SSIR builtins is encoded three
times:

- `spirv_to_ssir.c` `spv_builtin_to_ssir` (L1492-1519): 20-case switch
  SpvBuiltIn -> SsirBuiltinVar
- `ssir_to_spirv.c` (L1057-1082): 20-case switch SsirBuiltinVar -> SpvBuiltIn
  (inverse)
- `wgsl_lower.c` (L6191-6202): struct array `{name, SpvBuiltIn, SsirBuiltinVar}`

### Extraction strategy

One table `{ const char *name, SpvBuiltIn spv, SsirBuiltinVar ssir }[]` in
`ssir.c` with two lookup functions: `ssir_builtin_from_spv()` and
`ssir_builtin_to_spv()`. Also eliminates the wgsl_lower array.

---

## 11. SpvStorageClass <-> SsirAddressSpace Mapping: 2x encoding

- `wgsl_lower.c` `spv_sc_to_ssir_addr` (L499-512): 10-case switch
- `spirv_to_ssir.c` `storage_class_to_addr_space` (L1152-1166): 11-case switch
  (has one extra case: `PhysicalStorageBuffer`)

### Extraction strategy

One function in `ssir.c`. The wgsl_lower version is missing the
`PhysicalStorageBuffer` case which is probably a bug.

---

## 12. SpvDim <-> SsirTextureDim Mapping: 3x encoding

- `spirv_to_ssir.c` `spv_dim_to_ssir` (L1168-1178)
- `wgsl_lower.c` inlined in `spv_type_to_ssir` (L604-633): partial coverage
- `ssir_to_spirv.c` (L657-665): reverse direction

One table drives forward and reverse lookup.

---

## 13. `(mod, func_id, block_id)` Argument Repetition: hundreds of call sites

Both `wgsl_lower.c` and `ptx_lower.c` pass the same three arguments to every
`ssir_build_*` call:

```c
// ptx_lower.c -- every single ssir_build call looks like this:
result = ssir_build_add(p->mod, p->func_id, p->block_id, type, a, b);
result = ssir_build_sub(p->mod, p->func_id, p->block_id, type, a, b);
result = ssir_build_mul(p->mod, p->func_id, p->block_id, type, a, b);
```

### Extraction strategy

An `SsirBuildCtx` struct holding `{ SsirModule *mod; uint32_t func_id;
uint32_t block_id; }` and wrapper macros or inline functions:

```c
static inline uint32_t ssir_ctx_build_add(SsirBuildCtx *c, uint32_t t,
    uint32_t a, uint32_t b) {
    return ssir_build_add(c->mod, c->func_id, c->block_id, t, a, b);
}
```

Or a macro `#define BCTX p->mod, p->func_id, p->block_id` used at call sites.
Either approach eliminates ~3 repeated arguments per call across hundreds of
call sites.

---

## 14. Test Infrastructure Duplication: ~300 lines

### 14a. `SsirCompileResult` + `CompileToSsir()` + `SsirCompileGuard`: 5x copy

This ~45-line block (struct + function + RAII guard) is duplicated in:
- wgsl_roundtrip_test.cpp L17-65
- glsl_roundtrip_test.cpp L17-61
- raise_test.cpp L267-311
- ssir_raise_test.cpp L22-80
- glsl_raise_test.cpp L10+

### 14b. `SsirModuleGuard`: 4 names for the same class

`SsirModuleGuard`, `SsirGuard`, `SsirGuard2`, `ModGuard` appear in 8 test
files. All wrap `SsirModule *` + `ssir_module_destroy` in destructor.

### 14c. `CompileWgsl` / `CompileGlsl` in test_utils.h: 93% identical

Both (test_utils.h L141 and L184) do the same parse -> resolve -> lower ->
emit -> validate pipeline, differing only in the parse call.

### Extraction strategy

- Move `SsirCompileResult` / `CompileToSsir` / `SsirCompileGuard` into
  `test_utils.h`
- Add `SsirModuleGuard` to `test_utils.h`
- Unify `CompileWgsl` / `CompileGlsl` with a parse-function parameter

---

## 15. Public API Header (`simple_wgsl.h`) Repetition

### 15a. Six identical result enums

`WgslLowerResult`, `SsirToSpirvResult`, `SsirToWgslResult`,
`SsirToGlslResult`, `SsirToMslResult`, `SsirToHlslResult` all have the same
five values: `OK, ERR_INVALID_INPUT, ERR_UNSUPPORTED, ERR_INTERNAL, ERR_OOM`.

Could be a single `SsirBackendResult` enum. However, this changes public API,
so may need typedefs for backwards compatibility.

### 15b. Repeated Options struct field `preserve_names`

Present in `SsirToWgslOptions`, `SsirToGlslOptions`, `SsirToMslOptions`,
`SsirToHlslOptions`, `SpirvToSsirOptions`, `MslToSsirOptions`,
`PtxToSsirOptions`. Seven structs share this field. Could have a base
`SsirBackendOptions` struct embedded.

### 15c. Repeated `_free(void *)` / `_result_string()` declarations

Six `_free` functions and six `_result_string` functions, all with the same
body. A single `ssir_backend_free(void *)` and
`ssir_result_string(SsirBackendResult)` would suffice.

### 15d. `SsirModule` typed array triplets

Six `(T *items, uint32_t count, uint32_t capacity)` triplets. A generic
`SsirArray` macro or struct template would factor these, though in C99 this
requires macros.

---

## 16. Allocator Macro Groups: 7 parallel sets

Each subsystem declares its own `XX_MALLOC / XX_REALLOC / XX_FREE` group
(NODE_, SSIR_, WGSL_, PTX_, MSL_, STG_, STM_). All default to
`calloc/realloc/free`. A single parameterized macro:

```c
#define DEFINE_ALLOCATORS(PREFIX) \
    #ifndef PREFIX##_MALLOC        \
    #define PREFIX##_MALLOC(sz) calloc(1, (sz)) \
    #endif                         \
    ...
```

---

## 17. cuvk_runtime Patterns

### 17a. Context guard: 17 occurrences

```c
struct CUctx_st *ctx = g_cuvk.current_ctx;
if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
```

Replace with `CUVK_GET_CTX_OR_RETURN()` macro.

### 17b. `cuMemsetD8_v2` / `cuMemsetD16_v2`: ~87 lines each, same algorithm

Both (cuvk_memory.c L544-630, L639-731) do: lookup alloc, fill pattern, fast
path via `vkCmdFillBuffer`, slow path via staging buffer + `vkCmdCopyBuffer`.

Replace with `cuvk_memset_impl(dptr, pattern, elem_size, count)`.

### 17c. `cuCtxCreate_v4` staircase cleanup (cuvk_init.c L481-593)

Six levels of repeated cleanup on error. Replace with `goto cleanup` pattern
with labels.

### 17d. VkSubmitInfo + fence wait: 2 copies

`cuvk_oneshot_end` (cuvk_stream.c L56-100) and
`cuvk_stream_submit_and_wait` (L106-149) share the same submit+wait logic.

### 17e. VkCommandBufferAllocateInfo: 3 copies

Same four-field struct initialization in cuvk_init.c L560, cuvk_stream.c L27,
cuvk_stream.c L173.

---

## 18. Constant Emission in Text Backends: ~80 lines (4x)

The `SSIR_CONST_F32` formatting logic (check if integer, use `%.1f` vs `%g`)
and `SSIR_CONST_COMPOSITE` emission (type + parens + comma-separated recursive
emit) are identical across all four backends:

- SSIR_CONST_F32: WGSL L461, GLSL L463, MSL L503, HLSL L331
- SSIR_CONST_COMPOSITE: WGSL L502, GLSL L505, MSL L545, HLSL L373

---

## Summary: Estimated Total Redundancy

| Category | Redundant lines | Priority |
|---|---|---|
| SPIR-V parser (wgsl_raise + spirv_to_ssir) | ~600 | High |
| Expression ladder (wgsl + glsl parsers) | ~600 | Medium |
| Text backend shared ops (emit_expr cases) | ~400 | High |
| Builtin name tables (4 backends) | ~320 | Medium |
| Parser lexer infrastructure (4 parsers) | ~400 | Medium |
| ssir_const_T / ssir_type_T (ssir.c) | ~250 | Low |
| Text backend string buffer (4 backends) | ~240 | High |
| Text backend context (4 backends) | ~160 | High |
| Dynamic array growth (25 sites) | ~150 | Medium |
| Test infrastructure | ~300 | Low |
| API header repetition | ~100 | Low |
| cuvk patterns | ~200 | Low |
| Enum/mapping duplication | ~150 | Medium |
| (mod,func_id,block_id) repetition | ~diffuse | Medium |

Approximate total: **~3900 lines** of duplicated code that could be collapsed
into shared infrastructure without changing behavior.
