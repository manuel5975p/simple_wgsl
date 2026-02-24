# PTX Input Support — Implementation Plan

## 1. Overview

Add PTX (Parallel Thread Execution) input support to simple_wgsl, enabling:

```
PTX source → ptx_to_ssir() → SsirModule → WGSL / GLSL / MSL / HLSL / SPIR-V
```

This follows the same pattern as the existing MSL parser (`msl_parser.c`), which bypasses the shared AST and produces SSIR directly. PTX's assembly-like syntax and flat register-based semantics are too different from WGSL/GLSL to share their AST, so a direct PTX → SSIR path is the right approach.

### Why bypass the AST?

The shared `WgslAstNode` AST was designed for C-like shading languages (WGSL, GLSL) with:
- Structured control flow (if/else/for/while blocks)
- Expression trees with operator precedence
- Type declarations and struct definitions
- Function-scoped variables with lexical scoping

PTX has none of this. It is a flat, register-based assembly language with:
- Predicated execution instead of structured control flow
- No expression trees — only single instructions with register operands
- No type declarations — types are instruction suffixes (`.f32`, `.u64`)
- Explicit register allocation with parameterized naming (`%r<100>`)
- Labels and branches instead of structured blocks
- State-space annotations on every memory operation

Forcing PTX into the WGSL AST would require either (a) losing PTX-specific semantics or (b) bloating the AST with assembly-only node types that no other parser uses. The MSL parser already established the precedent for this: `msl_parser.c` produces `SsirModule` directly, and `ptx_parser.c` should do the same.

---

## 2. Scope

### In scope (initial implementation)
- PTX module parsing: `.version`, `.target`, `.address_size`
- Kernel entry points (`.entry`) and device functions (`.func`)
- Register declarations (`.reg`) for all scalar types
- Shared memory (`.shared`), global memory (`.global`), constant memory (`.const`)
- Parameter space (`.param`) for kernel arguments
- Core integer arithmetic: `add`, `sub`, `mul`, `mad`, `div`, `rem`, `abs`, `neg`, `min`, `max`
- Core floating-point arithmetic: `add`, `sub`, `mul`, `mad`, `fma`, `div`, `abs`, `neg`, `min`, `max`
- Math functions: `rcp`, `sqrt`, `rsqrt`, `sin`, `cos`, `lg2`, `ex2`
- Bitwise operations: `and`, `or`, `xor`, `not`, `shl`, `shr`
- Comparison and selection: `setp`, `selp`, `set`
- Data movement: `mov`, `ld`, `st`, `cvt`
- Address conversion: `cvta`
- Control flow: `bra`, `call`, `ret`, `exit`, labels
- Predicated execution: `@p` and `@!p` guards
- Synchronization: `bar.sync`, `bar.arrive`, `bar.red`
- Atomic operations: `atom.{add,min,max,and,or,xor,cas,exch}`
- Special registers: `%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%laneid`, `%warpid`
- Vector types: `.v2` and `.v4` register packing/unpacking
- Type suffixes: `.b8`-`.b64`, `.u8`-`.u64`, `.s8`-`.s64`, `.f16`, `.f32`, `.f64`, `.pred`

### Out of scope (future work)
- Texture/surface instructions (`tex`, `tld4`, `suld`, `sust`)
- Tensor Core / WMMA / MMA instructions
- Cooperative groups and cluster operations
- Video instructions (`vadd`, `vsub`, etc.)
- `.surref`, `.texref`, `.samplerref` opaque types
- Inline PTX within CUDA (only standalone `.ptx` files)
- PTX-specific performance hints (`.pragma`, cache control modifiers beyond basic `.ca`/`.cg`)
- Multiple return values from functions
- Indirect calls via function pointers

---

## 3. PTX Language Reference

This section documents the PTX constructs we need to parse, with exact syntax.

### 3.1 Module Structure

Every PTX file has this top-level structure:

```ptx
.version 7.8
.target sm_80
.address_size 64

// Optional: global/shared/constant declarations
.global .align 4 .b8 output[1024];
.shared .align 4 .f32 smem[256];
.const .align 4 .f32 constants[16];

// Kernel entry points
.visible .entry kernel_name(.param .u64 param0, .param .u32 param1) {
    // body
}

// Device functions
.visible .func (.reg .f32 retval) helper(.reg .f32 arg0, .reg .u32 arg1) {
    // body
}

// Device functions with no return value
.func void_helper(.reg .f32 arg0) {
    // body
}
```

**Key rules:**
- `.version X.Y` must appear first — specifies PTX ISA version
- `.target sm_XX` specifies minimum GPU architecture (sm_50, sm_60, sm_70, sm_80, sm_89, sm_90, sm_100)
- `.address_size 32|64` specifies pointer width (virtually always 64 in modern code)
- Module-level declarations appear before any functions/kernels
- `.visible` makes symbols externally visible; `.extern` imports external symbols

### 3.2 Type System

PTX types appear as suffixes on instructions and in register/variable declarations:

| PTX Type | Width | SSIR Mapping | Description |
|----------|-------|-------------|-------------|
| `.pred` | 1 bit | `SSIR_TYPE_BOOL` | Predicate (boolean) |
| `.b8` | 8 bits | `SSIR_TYPE_U8` | Untyped 8-bit |
| `.b16` | 16 bits | `SSIR_TYPE_U16` | Untyped 16-bit |
| `.b32` | 32 bits | `SSIR_TYPE_U32` | Untyped 32-bit |
| `.b64` | 64 bits | `SSIR_TYPE_U64` | Untyped 64-bit |
| `.u8` | 8 bits | `SSIR_TYPE_U8` | Unsigned 8-bit |
| `.u16` | 16 bits | `SSIR_TYPE_U16` | Unsigned 16-bit |
| `.u32` | 32 bits | `SSIR_TYPE_U32` | Unsigned 32-bit |
| `.u64` | 64 bits | `SSIR_TYPE_U64` | Unsigned 64-bit |
| `.s8` | 8 bits | `SSIR_TYPE_I8` | Signed 8-bit |
| `.s16` | 16 bits | `SSIR_TYPE_I16` | Signed 16-bit |
| `.s32` | 32 bits | `SSIR_TYPE_I32` | Signed 32-bit |
| `.s64` | 64 bits | `SSIR_TYPE_I64` | Signed 64-bit |
| `.f16` | 16 bits | `SSIR_TYPE_F16` | IEEE 754 half |
| `.f32` | 32 bits | `SSIR_TYPE_F32` | IEEE 754 single |
| `.f64` | 64 bits | `SSIR_TYPE_F64` | IEEE 754 double |

**Untyped `.bN` types**: PTX uses `.b8`/`.b16`/`.b32`/`.b64` for bit-manipulation operations where signedness doesn't matter. In SSIR, these map to unsigned integer types of the same width, and the instruction context determines whether signed or unsigned semantics apply. For example, `and.b32` maps to `SSIR_OP_BIT_AND` on `SSIR_TYPE_U32`.

**Vector types**: `.v2` and `.v4` prefixes create vector registers:
```ptx
.reg .v2 .f32 vec2reg;    // 2x f32
.reg .v4 .f32 vec4reg;    // 4x f32
```
These map to `SSIR_TYPE_VEC` with the appropriate element type and component count.

### 3.3 Register Declarations

```ptx
.reg .pred p, q, r;              // named predicate registers
.reg .pred p<10>;                 // parameterized: p0..p9
.reg .f32 f<100>;                // f0..f99 float registers
.reg .u32 r0, r1, r2;           // named integer registers
.reg .u64 rd<50>;               // rd0..rd49 64-bit registers
.reg .v4 .f32 color;            // single vector register
```

**Parameterized naming** (`%r<N>`) declares N registers named `%r0` through `%r(N-1)`. The `%` prefix is used when referencing registers in instructions. Named registers like `p, q, r` are referenced as `%p`, `%q`, `%r`.

### 3.4 State Spaces and Memory Declarations

| State Space | SSIR Address Space | Scope | Description |
|------------|-------------------|-------|-------------|
| `.reg` | (not memory) | Per-thread | Registers — fast, unlimited virtual |
| `.param` | `SSIR_ADDRESS_SPACE_UNIFORM` (kernel params) | Per-kernel-launch | Kernel parameters, read-only in kernel body |
| `.local` | `SSIR_ADDRESS_SPACE_FUNCTION` | Per-thread | Thread-private stack memory |
| `.shared` | `SSIR_ADDRESS_SPACE_WORKGROUP` | Per-CTA (workgroup) | On-chip shared memory, visible to all threads in block |
| `.global` | `SSIR_ADDRESS_SPACE_STORAGE` | Per-device | Device memory (VRAM), visible to all threads |
| `.const` | `SSIR_ADDRESS_SPACE_UNIFORM` | Per-device | Read-only constant memory with cache |

**Memory declaration syntax:**
```ptx
.global .align 16 .b8 buffer[4096];     // 4KB global buffer, 16-byte aligned
.shared .align 4 .f32 tile[32][32];     // 32x32 shared float array
.const .align 4 .f32 weights[256];      // constant memory array
.local .align 8 .b8 stack[512];         // per-thread local memory
```

**Kernel parameter syntax:**
```ptx
.entry my_kernel(
    .param .u64 input_ptr,        // pointer (passed as u64)
    .param .u64 output_ptr,       // pointer
    .param .u32 count             // scalar value
) { ... }
```

Kernel parameters are accessed via `ld.param`:
```ptx
ld.param.u64 %rd0, [input_ptr];
ld.param.u32 %r0, [count];
```

### 3.5 Instruction Format

General form:
```
[@predicate] opcode[.modifier]* [.type] destination, source1 [, source2 [, source3]];
```

Examples:
```ptx
add.s32       %r3, %r1, %r2;          // r3 = r1 + r2 (signed 32-bit)
mul.lo.s32    %r3, %r1, %r2;          // r3 = low 32 bits of r1 * r2
mad.lo.s32    %r4, %r1, %r2, %r3;     // r4 = r1*r2 + r3 (low 32 bits)
add.f32       %f3, %f1, %f2;          // floating-point add
fma.rn.f32    %f3, %f1, %f2, %f0;     // fused multiply-add, round-to-nearest
setp.lt.f32   %p1, %f1, %f2;          // p1 = (f1 < f2)
@%p1 bra      LABEL;                  // conditional branch
ld.global.f32 %f1, [%rd1];            // load from global memory
st.global.f32 [%rd2], %f1;            // store to global memory
cvt.rn.f32.s32 %f1, %r1;             // convert s32 to f32, round-to-nearest
bar.sync      0;                       // barrier synchronize
```

**Rounding modifiers** (floating-point):
- `.rn` — round to nearest even
- `.rz` — round toward zero
- `.rm` — round toward minus infinity
- `.rp` — round toward plus infinity

**Saturation modifier**: `.sat` clamps result to [0.0, 1.0] for floats or to type range for integers.

**Comparison operators** (used with `setp`, `set`):
- Integer: `.eq`, `.ne`, `.lt`, `.le`, `.gt`, `.ge`
- Unsigned: `.lo`, `.ls`, `.hi`, `.hs` (below, below-or-same, above, above-or-same)
- Float: `.eq`, `.ne`, `.lt`, `.le`, `.gt`, `.ge`, `.equ`, `.neu`, `.ltu`, `.leu`, `.gtu`, `.geu`, `.num`, `.nan`
  (The `u` suffixed variants are unordered comparisons for NaN handling; `.num` = both ordered, `.nan` = either NaN)

### 3.6 Predicated Execution

Any instruction can be predicated:
```ptx
setp.lt.f32  %p1, %f1, %f2;     // set predicate: p1 = (f1 < f2)
@%p1 add.f32 %f3, %f1, %f2;     // execute only if p1 is true
@!%p1 mov.f32 %f3, 0f00000000;  // execute only if p1 is false
```

**Predicate combination** (setp can set two predicates):
```ptx
setp.lt.and.f32 %p1|%p2, %f1, %f2, %p3;
// p1 = (f1 < f2) AND p3
// p2 = !(f1 < f2) AND p3
```

**SSIR mapping**: Predicated instructions translate to `SSIR_OP_BRANCH_COND` + separate basic blocks. A predicated `add` becomes:
```
block_check:
  branch_cond %p1 → block_true, block_false
block_true:
  %result_true = add %f1, %f2
  branch → block_merge
block_false:
  %result_false = mov 0.0
  branch → block_merge
block_merge:
  %f3 = phi [%result_true, block_true], [%result_false, block_false]
```

For simple cases (single predicated instruction with no else), the phi is not needed — only the conditional block and a merge.

### 3.7 Control Flow

**Labels**: Any identifier followed by `:` is a label:
```ptx
LOOP_START:
    // ...
    bra LOOP_START;
```

**Unconditional branch**:
```ptx
bra TARGET_LABEL;
```

**Conditional branch** (via predicate):
```ptx
setp.ge.u32 %p1, %r1, %r2;
@%p1 bra DONE;
```

**Function calls**:
```ptx
// Call with return value:
call (%retval), helper_func, (%arg0, %arg1);

// Call without return value:
call void_func, (%arg0);
```

**Return and exit**:
```ptx
ret;              // return from .func
exit;             // terminate thread (in .entry)
```

**SSIR mapping for control flow reconstruction**: PTX uses unstructured control flow (labels + branches), but SSIR expects structured basic blocks. The parser must:
1. Split instructions at labels and branches to form basic blocks
2. Build a CFG (control flow graph) from branch targets
3. Optionally reconstruct structured control flow (loops, if/else) — but this is not required since SSIR supports unstructured CFG via basic blocks with branch/branch_cond terminators

### 3.8 Memory Operations

**Load**:
```ptx
ld.global.f32    %f1, [%rd1];           // load f32 from global ptr
ld.shared.f32    %f1, [smem + 16];      // load from shared + offset
ld.param.u64     %rd1, [param_name];    // load kernel parameter
ld.const.f32     %f1, [constants + 4];  // load from constant memory
ld.local.b32     %r1, [%rd1];           // load from local memory
ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1]; // vector load
```

**Store**:
```ptx
st.global.f32    [%rd2], %f1;           // store f32 to global ptr
st.shared.f32    [smem + 16], %f1;      // store to shared memory
st.global.v4.f32 [%rd2], {%f1,%f2,%f3,%f4}; // vector store
```

**Address arithmetic** (usually done via `add.u64` or `mad.lo.u64`):
```ptx
// Compute address: base + index * 4
mul.lo.u64  %rd3, %rd_index, 4;
add.u64     %rd4, %rd_base, %rd3;
ld.global.f32 %f1, [%rd4];
```

**Address space conversion**:
```ptx
cvta.to.global.u64 %rd1, %rd0;    // generic → global
cvta.to.shared.u64 %rd1, %rd0;    // generic → shared
```

**SSIR mapping**:
- `ld.global` → `SSIR_OP_LOAD` with pointer in `SSIR_ADDRESS_SPACE_STORAGE`
- `ld.shared` → `SSIR_OP_LOAD` with pointer in `SSIR_ADDRESS_SPACE_WORKGROUP`
- `st.global` → `SSIR_OP_STORE` with pointer in `SSIR_ADDRESS_SPACE_STORAGE`
- Vector loads/stores → `SSIR_OP_LOAD`/`SSIR_OP_STORE` on vector type, may need `SSIR_OP_CONSTRUCT`/`SSIR_OP_EXTRACT`

### 3.9 Special Registers

Read-only special registers accessed via `mov`:

| PTX Special Register | SSIR Built-in | Description |
|---------------------|---------------|-------------|
| `%tid.x/y/z` | `SSIR_BUILTIN_LOCAL_INVOCATION_ID` | Thread index within block |
| `%ntid.x/y/z` | `SSIR_BUILTIN_WORKGROUP_SIZE` (if available, else constant) | Block dimensions |
| `%ctaid.x/y/z` | `SSIR_BUILTIN_WORKGROUP_ID` | Block index within grid |
| `%nctaid.x/y/z` | `SSIR_BUILTIN_NUM_WORKGROUPS` | Grid dimensions |
| `%laneid` | `SSIR_BUILTIN_SUBGROUP_INVOCATION_ID` | Lane within warp |
| `%warpid` | (compute from tid / 32) | Warp index within block |
| `%clock` | (no direct mapping) | Cycle counter — drop or warn |
| `%smid` | (no direct mapping) | SM identifier — drop or warn |

**Access pattern**:
```ptx
mov.u32 %r1, %tid.x;      // r1 = threadIdx.x
mov.u32 %r2, %ctaid.x;    // r2 = blockIdx.x
mov.u32 %r3, %ntid.x;     // r3 = blockDim.x
```

**SSIR mapping**: Special register reads become loads from built-in global variables decorated with `SSIR_BUILTIN_*`. The `.x/.y/.z` component access maps to `SSIR_OP_EXTRACT` on a `vec3<u32>`.

### 3.10 Atomic Operations

```ptx
atom.global.add.u32  %r1, [%rd1], %r2;    // r1 = atomicAdd(ptr, r2)
atom.shared.min.s32  %r1, [addr], %r2;    // atomicMin in shared memory
atom.global.cas.b32  %r1, [%rd1], %r2, %r3;  // compare-and-swap
atom.global.exch.b32 %r1, [%rd1], %r2;    // exchange
```

**SSIR mapping**: `SSIR_OP_ATOMIC` with corresponding `SsirAtomicOp`:
- `atom.add` → `SSIR_ATOMIC_ADD`
- `atom.min` → `SSIR_ATOMIC_MIN`
- `atom.max` → `SSIR_ATOMIC_MAX`
- `atom.and` → `SSIR_ATOMIC_AND`
- `atom.or` → `SSIR_ATOMIC_OR`
- `atom.xor` → `SSIR_ATOMIC_XOR`
- `atom.exch` → `SSIR_ATOMIC_EXCHANGE`
- `atom.cas` → `SSIR_ATOMIC_COMPARE_EXCHANGE`

### 3.11 Synchronization

```ptx
bar.sync 0;                // all threads in CTA hit barrier 0
bar.sync 0, %r1;           // partial barrier (r1 threads participate)
membar.cta;                // memory fence within CTA
membar.gl;                 // memory fence global
membar.sys;                // memory fence system-wide
```

**SSIR mapping**:
- `bar.sync` → `SSIR_OP_BARRIER` with `SSIR_SCOPE_WORKGROUP`
- `membar.cta` → `SSIR_OP_BARRIER` (memory-only variant if supported)
- `membar.gl` → `SSIR_OP_BARRIER` with `SSIR_SCOPE_DEVICE`

### 3.12 Type Conversions

```ptx
cvt.rn.f32.s32  %f1, %r1;     // signed int → float (round nearest)
cvt.rzi.s32.f32 %r1, %f1;     // float → signed int (round toward zero, integer)
cvt.f64.f32     %fd1, %f1;    // float → double (exact)
cvt.rn.f32.f64  %f1, %fd1;    // double → float (round nearest)
cvt.u32.u16     %r1, %rh1;    // zero-extend u16 → u32
cvt.s32.s16     %r1, %rh1;    // sign-extend s16 → s32
```

**SSIR mapping**: `SSIR_OP_CONVERT` (for numeric conversions) or `SSIR_OP_BITCAST` (for `.b` type reinterpretation).

### 3.13 Math Functions

```ptx
rcp.rn.f32   %f1, %f2;     // reciprocal (1/x)
sqrt.rn.f32  %f1, %f2;     // square root
rsqrt.approx.f32 %f1, %f2; // reciprocal square root (fast approx)
sin.approx.f32   %f1, %f2; // sine (fast approx)
cos.approx.f32   %f1, %f2; // cosine (fast approx)
lg2.approx.f32   %f1, %f2; // log base 2 (fast approx)
ex2.approx.f32   %f1, %f2; // 2^x (fast approx)
abs.f32      %f1, %f2;     // absolute value
neg.f32      %f1, %f2;     // negation
min.f32      %f1, %f2, %f3;  // minimum
max.f32      %f1, %f2, %f3;  // maximum
```

**SSIR mapping**: `SSIR_OP_BUILTIN` with:
- `rcp` → `SSIR_BUILTIN_RECIP` (or emit as `1.0 / x` if no direct builtin)
- `sqrt` → `SSIR_BUILTIN_SQRT`
- `rsqrt` → `SSIR_BUILTIN_INVERSE_SQRT`
- `sin` → `SSIR_BUILTIN_SIN`
- `cos` → `SSIR_BUILTIN_COS`
- `lg2` → `SSIR_BUILTIN_LOG2`
- `ex2` → `SSIR_BUILTIN_EXP2`
- `abs` → `SSIR_BUILTIN_ABS`
- `min` → `SSIR_BUILTIN_MIN`
- `max` → `SSIR_BUILTIN_MAX`
- `neg` → `SSIR_OP_NEG`

---

## 4. Architecture

### 4.1 New Files

| File | Purpose | Approx Lines |
|------|---------|-------------|
| `ptx_parser.c` | Lexer + parser + SSIR builder | ~3,000-4,000 |
| `tests/ptx_parser_test.cpp` | Unit tests | ~1,500-2,000 |
| `tests/ptx/` | PTX test shader files | ~20-30 files |

### 4.2 API Surface (additions to `simple_wgsl.h`)

```c
/* ── PTX → SSIR ───────────────────────────────────────────── */

typedef enum {
    PTX_TO_SSIR_OK = 0,
    PTX_TO_SSIR_PARSE_ERROR,
    PTX_TO_SSIR_UNSUPPORTED,
} PtxToSsirResult;

typedef struct {
    int preserve_names;        // keep PTX register names as debug names
    int strict_mode;           // reject .approx instructions (exact only)
} PtxToSsirOptions;

PtxToSsirResult ptx_to_ssir(const char *ptx_source,
                             const PtxToSsirOptions *opts,
                             SsirModule **out_module,
                             char **out_error);

void ptx_to_ssir_free(SsirModule *module, char *error);
```

This mirrors the `msl_to_ssir()` API exactly. Once the PTX is converted to an `SsirModule`, all existing output backends work automatically:

```c
// PTX → WGSL
ptx_to_ssir(ptx_src, &opts, &module, &error);
ssir_to_wgsl(module, &wgsl_opts, &wgsl_out, &wgsl_error);

// PTX → SPIR-V
ptx_to_ssir(ptx_src, &opts, &module, &error);
ssir_to_spirv(module, &spirv_opts, &words, &word_count);

// PTX → GLSL
ptx_to_ssir(ptx_src, &opts, &module, &error);
ssir_to_glsl(module, SSIR_STAGE_COMPUTE, &glsl_opts, &glsl_out, &glsl_error);
```

### 4.3 Internal Structure of `ptx_parser.c`

The file is organized into these sections (following the pattern of `msl_parser.c`):

```
┌──────────────────────────────────────────────┐
│  1. Token types and lexer                     │
│     - PtxTokenType enum                       │
│     - PtxToken struct (type, text, line, col) │
│     - ptx_next_token() — hand-written lexer   │
├──────────────────────────────────────────────┤
│  2. Parser context                            │
│     - PtxCtx struct (token stream, SSIR       │
│       module, register map, label map,        │
│       current function, error state)          │
├──────────────────────────────────────────────┤
│  3. Module-level parsing                      │
│     - parse_version()                         │
│     - parse_target()                          │
│     - parse_address_size()                    │
│     - parse_global_declaration()              │
│     - parse_entry() — .entry kernel           │
│     - parse_func()  — .func device function   │
├──────────────────────────────────────────────┤
│  4. Function body parsing                     │
│     - parse_reg_declaration()                 │
│     - parse_instruction() — big dispatch      │
│     - parse_label()                           │
│     - parse_predicate_guard()                 │
├──────────────────────────────────────────────┤
│  5. Instruction handlers                      │
│     - parse_arith_inst() — add/sub/mul/etc.   │
│     - parse_mad_inst()                        │
│     - parse_setp_inst()                       │
│     - parse_selp_inst()                       │
│     - parse_mov_inst()                        │
│     - parse_ld_inst()                         │
│     - parse_st_inst()                         │
│     - parse_cvt_inst()                        │
│     - parse_bra_inst()                        │
│     - parse_call_inst()                       │
│     - parse_bar_inst()                        │
│     - parse_atom_inst()                       │
│     - parse_math_inst() — sin/cos/sqrt/etc.   │
├──────────────────────────────────────────────┤
│  6. CFG construction                          │
│     - split_blocks_at_labels_and_branches()   │
│     - resolve_branch_targets()                │
│     - build_cfg_edges()                       │
├──────────────────────────────────────────────┤
│  7. Predicate lowering                        │
│     - lower_predicated_instructions()         │
│     - insert_conditional_blocks()             │
│     - insert_phi_nodes()                      │
├──────────────────────────────────────────────┤
│  8. Register allocation / SSA construction    │
│     - build_ssa_from_registers()              │
│     - insert_phi_for_register_redefs()        │
│     - map_ptx_registers_to_ssir_ids()         │
├──────────────────────────────────────────────┤
│  9. Public API                                │
│     - ptx_to_ssir()                           │
│     - ptx_to_ssir_free()                      │
└──────────────────────────────────────────────┘
```

---

## 5. Detailed Design

### 5.1 Lexer

The PTX lexer is straightforward because PTX has a simple, regular token structure:

**Token types:**
```c
typedef enum {
    // Structural
    PTX_TOK_EOF,
    PTX_TOK_NEWLINE,
    PTX_TOK_SEMICOLON,      // ;
    PTX_TOK_COMMA,           // ,
    PTX_TOK_LBRACE,          // {
    PTX_TOK_RBRACE,          // }
    PTX_TOK_LBRACKET,        // [
    PTX_TOK_RBRACKET,        // ]
    PTX_TOK_LPAREN,          // (
    PTX_TOK_RPAREN,          // )
    PTX_TOK_COLON,           // : (after labels)
    PTX_TOK_PIPE,            // | (predicate output separator)
    PTX_TOK_AT,              // @ (predicate guard)
    PTX_TOK_BANG,            // ! (negated predicate)
    PTX_TOK_PLUS,            // + (address offset)
    PTX_TOK_MINUS,           // - (negative immediates)

    // Directives (start with .)
    PTX_TOK_DOT_VERSION,     // .version
    PTX_TOK_DOT_TARGET,      // .target
    PTX_TOK_DOT_ADDRESS_SIZE,// .address_size
    PTX_TOK_DOT_ENTRY,       // .entry
    PTX_TOK_DOT_FUNC,        // .func
    PTX_TOK_DOT_REG,         // .reg
    PTX_TOK_DOT_PARAM,       // .param
    PTX_TOK_DOT_LOCAL,       // .local
    PTX_TOK_DOT_SHARED,      // .shared
    PTX_TOK_DOT_GLOBAL,      // .global
    PTX_TOK_DOT_CONST,       // .const
    PTX_TOK_DOT_VISIBLE,     // .visible
    PTX_TOK_DOT_EXTERN,      // .extern
    PTX_TOK_DOT_ALIGN,       // .align
    PTX_TOK_DOT_MAXNTID,     // .maxntid
    PTX_TOK_DOT_REQNTID,     // .reqntid
    PTX_TOK_DOT_PRAGMA,      // .pragma

    // Type suffixes (start with .)
    PTX_TOK_DOT_PRED,
    PTX_TOK_DOT_B8, PTX_TOK_DOT_B16, PTX_TOK_DOT_B32, PTX_TOK_DOT_B64,
    PTX_TOK_DOT_U8, PTX_TOK_DOT_U16, PTX_TOK_DOT_U32, PTX_TOK_DOT_U64,
    PTX_TOK_DOT_S8, PTX_TOK_DOT_S16, PTX_TOK_DOT_S32, PTX_TOK_DOT_S64,
    PTX_TOK_DOT_F16, PTX_TOK_DOT_F32, PTX_TOK_DOT_F64,

    // Vector suffixes
    PTX_TOK_DOT_V2,         // .v2
    PTX_TOK_DOT_V4,         // .v4

    // Modifiers
    PTX_TOK_DOT_RN, PTX_TOK_DOT_RZ, PTX_TOK_DOT_RM, PTX_TOK_DOT_RP,  // rounding
    PTX_TOK_DOT_SAT,        // saturation
    PTX_TOK_DOT_APPROX,     // approximate math
    PTX_TOK_DOT_FTZ,        // flush to zero
    PTX_TOK_DOT_LO, PTX_TOK_DOT_HI, PTX_TOK_DOT_WIDE,  // multiply width

    // Comparison operators
    PTX_TOK_DOT_EQ, PTX_TOK_DOT_NE,
    PTX_TOK_DOT_LT, PTX_TOK_DOT_LE, PTX_TOK_DOT_GT, PTX_TOK_DOT_GE,
    // Float-specific unordered comparisons
    PTX_TOK_DOT_EQU, PTX_TOK_DOT_NEU,
    PTX_TOK_DOT_LTU, PTX_TOK_DOT_LEU, PTX_TOK_DOT_GTU, PTX_TOK_DOT_GEU,
    PTX_TOK_DOT_NUM, PTX_TOK_DOT_NAN_CMP,

    // Boolean combiners for setp
    PTX_TOK_DOT_AND, PTX_TOK_DOT_OR, PTX_TOK_DOT_XOR,

    // Memory modifiers
    PTX_TOK_DOT_TO,         // cvta.to.{space}
    PTX_TOK_DOT_CA, PTX_TOK_DOT_CG, PTX_TOK_DOT_CS, PTX_TOK_DOT_CV,  // cache ops

    // Atomic operations
    PTX_TOK_DOT_ADD, PTX_TOK_DOT_MIN, PTX_TOK_DOT_MAX,
    PTX_TOK_DOT_INC, PTX_TOK_DOT_DEC,
    PTX_TOK_DOT_CAS, PTX_TOK_DOT_EXCH,

    // Barrier modifiers
    PTX_TOK_DOT_SYNC, PTX_TOK_DOT_ARRIVE, PTX_TOK_DOT_RED,

    // Memory fence scope
    PTX_TOK_DOT_CTA, PTX_TOK_DOT_GL, PTX_TOK_DOT_SYS,

    // Opcodes (plain identifiers that are recognized as instructions)
    PTX_TOK_OP_ADD, PTX_TOK_OP_SUB, PTX_TOK_OP_MUL, PTX_TOK_OP_MAD,
    PTX_TOK_OP_FMA, PTX_TOK_OP_DIV, PTX_TOK_OP_REM, PTX_TOK_OP_ABS,
    PTX_TOK_OP_NEG, PTX_TOK_OP_MIN, PTX_TOK_OP_MAX,
    PTX_TOK_OP_AND, PTX_TOK_OP_OR, PTX_TOK_OP_XOR, PTX_TOK_OP_NOT,
    PTX_TOK_OP_SHL, PTX_TOK_OP_SHR,
    PTX_TOK_OP_SETP, PTX_TOK_OP_SELP, PTX_TOK_OP_SET,
    PTX_TOK_OP_MOV, PTX_TOK_OP_LD, PTX_TOK_OP_ST,
    PTX_TOK_OP_CVT, PTX_TOK_OP_CVTA,
    PTX_TOK_OP_BRA, PTX_TOK_OP_CALL, PTX_TOK_OP_RET, PTX_TOK_OP_EXIT,
    PTX_TOK_OP_BAR, PTX_TOK_OP_MEMBAR, PTX_TOK_OP_ATOM,
    PTX_TOK_OP_RCP, PTX_TOK_OP_SQRT, PTX_TOK_OP_RSQRT,
    PTX_TOK_OP_SIN, PTX_TOK_OP_COS, PTX_TOK_OP_LG2, PTX_TOK_OP_EX2,
    PTX_TOK_OP_CNOT,

    // General tokens
    PTX_TOK_IDENT,           // identifier (register name, label, function name)
    PTX_TOK_SPECIAL_REG,     // %tid, %ntid, %ctaid, etc. (with .x/.y/.z)
    PTX_TOK_INT_LIT,         // integer literal (decimal or 0x hex)
    PTX_TOK_FLOAT_LIT,       // float literal (decimal or 0fXXXXXXXX hex-float)
} PtxTokenType;
```

**Lexer specifics:**
- PTX uses `//` for single-line comments (C-style)
- PTX uses `/* */` for multi-line comments
- Dot-prefixed tokens: The lexer sees `.` and looks ahead to classify (directive, type, modifier, or comparison). A table-driven lookup on the following identifier handles this.
- Register references start with `%` — the lexer reads `%identifier` as a single token
- Hex float literals use `0f` or `0d` prefix (e.g., `0f3F800000` = 1.0f, `0d3FF0000000000000` = 1.0)
- Labels are identifiers followed by `:` — the colon is a separate token; the parser recognizes `IDENT COLON` as a label definition

### 5.2 Parser Context

```c
typedef struct {
    // Source
    const char *source;
    const char *cursor;
    int line, col;

    // Current token
    PtxToken current;
    PtxToken peek;

    // SSIR module being built
    SsirModule *module;

    // PTX module info
    int ptx_version_major, ptx_version_minor;
    int address_size;  // 32 or 64
    char target[32];   // "sm_80" etc.

    // Register map: PTX register name → SSIR ID
    // Uses a simple hash table or linear array (registers are numbered)
    struct {
        char name[64];
        uint32_t ssir_id;
        uint32_t ssir_type;
        bool is_predicate;
    } *registers;
    int register_count, register_cap;

    // Label map: label name → block ID (for branch resolution)
    struct {
        char name[64];
        uint32_t block_id;
    } *labels;
    int label_count, label_cap;

    // Unresolved forward branches (patched after function body)
    struct {
        uint32_t branch_inst_index;  // instruction to patch
        uint32_t block_index;        // which block it's in
        char target_label[64];
    } *unresolved_branches;
    int unresolved_count, unresolved_cap;

    // Current function being built
    uint32_t current_func_id;
    uint32_t current_block_id;
    bool is_entry_point;

    // Predicate state
    uint32_t pending_predicate;      // SSIR ID of predicate register
    bool pending_predicate_negated;
    bool has_pending_predicate;

    // Error handling
    char error_buf[1024];
    bool had_error;

    // Options
    PtxToSsirOptions opts;
} PtxCtx;
```

### 5.3 Register Management

PTX uses virtual registers that can be redefined. SSIR uses SSA form where each value has a unique ID. We need to bridge this gap.

**Strategy: Deferred SSA construction with load/store to local variables**

This is the simplest and most reliable approach (same as what `spirv_to_ssir` does with `OpVariable`):

1. Each PTX register declaration creates a local variable (`SsirLocalVar`) with `SSIR_ADDRESS_SPACE_FUNCTION`
2. Each register read emits `SSIR_OP_LOAD` from that local variable
3. Each register write (instruction result) emits `SSIR_OP_STORE` to that local variable
4. SSIR consumers already handle this pattern — it's how SPIR-V works

This avoids needing explicit SSA construction (phi-node insertion, dominance frontiers, etc.) at the cost of more load/store instructions in the IR. The downstream SPIR-V emitter already expects this.

**Example**:
```ptx
.reg .f32 a, b, c;
add.f32 %c, %a, %b;    // c = a + b
mul.f32 %c, %c, %a;    // c = c * a   (register reuse!)
```

Becomes:
```
%var_a = OpVariable Function %ptr_f32
%var_b = OpVariable Function %ptr_f32
%var_c = OpVariable Function %ptr_f32

%t1 = load %var_a
%t2 = load %var_b
%t3 = add.f32 %t1, %t2
store %var_c, %t3

%t4 = load %var_c
%t5 = load %var_a
%t6 = mul.f32 %t4, %t5
store %var_c, %t6
```

**Parameterized registers** (`%r<100>`) expand to `%r0` through `%r99`, each getting its own local variable. We allocate them lazily — only when first referenced in an instruction.

### 5.4 Predicate Lowering

Predicated instructions are the trickiest part of PTX → SSIR translation because SSIR has no instruction-level predication.

**Strategy:**

For each predicated instruction `@%p inst ...`:

1. End the current basic block with `SSIR_OP_BRANCH_COND` testing `%p`
2. Create a "true" block containing the instruction
3. Create a "merge" block that both paths converge to
4. If the instruction writes a register, insert a `SSIR_OP_PHI` in the merge block choosing between the new value (from true block) and the old value (from the implicit false path)

For `@!%p inst ...`, swap the true/false branches.

**Optimization for common patterns:**

When we see `@%p bra LABEL` (conditional branch), this directly maps to `SSIR_OP_BRANCH_COND` without needing extra blocks:
```
branch_cond %p → LABEL, fallthrough
```

This is the most common predicated pattern in real PTX code, so handling it efficiently is important.

**Consecutive predicated instructions** with the same predicate (`@%p inst1; @%p inst2; @%p inst3`) can be grouped into a single conditional block instead of creating separate blocks for each.

### 5.5 Control Flow Graph Construction

PTX has unstructured control flow (labels + goto). SSIR expects basic blocks with explicit terminator instructions. The translation works in two passes:

**Pass 1: Block splitting**

Scan the instruction stream and split into basic blocks at:
- Label definitions (start a new block)
- Branch instructions (end current block)
- Instructions following a branch (start a new block — fallthrough target)

**Pass 2: Branch resolution**

- For each `bra LABEL` → look up the label in the label map → set `SSIR_OP_BRANCH` target to that block ID
- For each `@%p bra LABEL` → `SSIR_OP_BRANCH_COND` with true-target = label's block, false-target = next block (fallthrough)
- For each block with no terminator → add `SSIR_OP_BRANCH` to the next sequential block (implicit fallthrough)
- Handle forward references: if a label hasn't been seen yet, record it in `unresolved_branches` and patch after the function is fully parsed

**No structured control flow reconstruction needed**: SSIR basic blocks with branch/branch_cond terminators are sufficient. The downstream emitters (`ssir_to_wgsl`, `ssir_to_glsl`, etc.) already handle unstructured CFGs by reconstructing loops and selections during emission.

### 5.6 Kernel Parameters → SSIR Interface

A PTX kernel entry point:
```ptx
.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
```

Maps to an SSIR compute entry point with storage buffer parameters:

```
// Global variables for parameters
%a_ptr = SsirGlobalVar { address_space: STORAGE, type: ptr<storage, array<u8>> }
%b_ptr = SsirGlobalVar { address_space: STORAGE, type: ptr<storage, array<u8>> }
%c_ptr = SsirGlobalVar { address_space: STORAGE, type: ptr<storage, array<u8>> }

// Scalar params become push constants or a uniform buffer:
%params = SsirGlobalVar {
    address_space: UNIFORM,  // or PUSH_CONSTANT
    type: struct { n: u32 }
}

// Entry point
SsirEntryPoint {
    stage: COMPUTE,
    function: %main_func,
    interface: [%a_ptr, %b_ptr, %c_ptr, %params, %builtin_global_id, ...]
}
```

**The mapping challenge**: PTX kernel parameters are flat (pointers passed as u64 values, scalars passed by value). SSIR/SPIR-V/WGSL expect typed buffer bindings with group/binding decorations.

**Strategy**: We need a convention. The simplest approach:
- `.param .u64` parameters are assumed to be buffer pointers → generate `@group(0) @binding(N) var<storage, read_write> paramN: array<u8>` (byte-addressable storage buffer). Binding N increments per pointer parameter.
- `.param .u32/.s32/.f32/etc.` scalar parameters are packed into a uniform buffer → generate `@group(0) @binding(K) var<uniform> params: ParamsStruct` where K is the next binding after all pointer params.

This convention is documented in the API and can be overridden via options in future work.

### 5.7 Shared Memory → SSIR Workgroup Variables

```ptx
.shared .align 4 .f32 smem[256];
```

Maps to:
```
SsirGlobalVar {
    name: "smem",
    type: array<f32, 256>,
    address_space: SSIR_ADDRESS_SPACE_WORKGROUP
}
```

Accesses via `ld.shared`/`st.shared` with address arithmetic map to `SSIR_OP_ACCESS` (pointer offset) + `SSIR_OP_LOAD`/`SSIR_OP_STORE`.

### 5.8 Global Memory Access Pattern

PTX does pointer arithmetic explicitly:
```ptx
ld.param.u64  %rd1, [input_ptr];          // load base pointer
mov.u32       %r1, %tid.x;                // thread ID
mul.lo.u64    %rd2, %r1, 4;               // byte offset = tid * sizeof(f32)
add.u64       %rd3, %rd1, %rd2;           // effective address
ld.global.f32 %f1, [%rd3];                // load element
```

In SSIR, this pattern is recognized and converted to typed array access:
```
%base = access %input_buffer, [0]          // base of storage buffer
%elem = access %base, [%tid_x]            // index into array
%val = load %elem                          // typed load
```

**However**, perfect reconstruction of typed array indexing from raw pointer arithmetic is a hard decompilation problem. For the initial implementation, we use byte-addressable storage buffers and preserve the pointer arithmetic as-is:
```
%base_ptr = load %param_var                // load u64 from param
%offset = mul %tid_x, 4                   // byte offset
%addr = add %base_ptr, %offset            // effective address
%val = load [%addr]                        // byte-addressed load
```

This is correct but produces less readable output. Typed array reconstruction can be added as an optimization pass later.

### 5.9 Workgroup Size

PTX kernels specify workgroup size via performance directives:
```ptx
.maxntid 256, 1, 1     // maximum threads per block
.reqntid 256, 1, 1     // required threads per block (exact)
```

Or it may be absent (dynamically specified at launch time).

**SSIR mapping**:
- `.reqntid X, Y, Z` → `entry_point.workgroup_size = {X, Y, Z}`
- `.maxntid X, Y, Z` → same (treat as workgroup size)
- Neither present → use `{1, 1, 1}` as default (caller must override via specialization constants or pipeline state)

---

## 6. Implementation Steps

### Step 1: Lexer (~400 lines)

Implement the PTX tokenizer:
- `ptx_next_token()` — advances cursor, returns next token
- Handle: comments (`//`, `/* */`), dot-prefixed keywords/types, `%`-prefixed registers, integer/float literals (including `0fXXXXXXXX` hex-floats), identifiers, punctuation
- Dot-token disambiguation: `.version` is a directive, `.f32` is a type, `.rn` is a modifier — use a sorted lookup table
- Special register recognition: `%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%laneid`, `%warpid` (with `.x`/`.y`/`.z` component)

**Test**: Tokenize a simple PTX file and verify all tokens are correct.

### Step 2: Module-Level Parsing (~300 lines)

Parse the PTX module header and global declarations:
- `.version X.Y` → store in context
- `.target sm_XX` → store in context (may influence feature availability)
- `.address_size 32|64` → store in context (determines pointer width)
- `.global`, `.shared`, `.const` declarations → create `SsirGlobalVar` entries
- `.visible`, `.extern` linkage modifiers

**Test**: Parse a PTX module with global declarations, verify SSIR globals are created with correct types and address spaces.

### Step 3: Kernel/Function Signature Parsing (~400 lines)

Parse `.entry` and `.func` declarations:
- Parameter lists with types
- Return value declaration for `.func`
- Create `SsirFunction` with parameters
- For `.entry`: create `SsirEntryPoint` with `SSIR_STAGE_COMPUTE`
- Handle `.maxntid`/`.reqntid` directives for workgroup size

**Test**: Parse kernel and function signatures, verify SSIR functions have correct parameter types and entry points are registered.

### Step 4: Register Declarations and Basic Instructions (~600 lines)

Parse function body register declarations and simple arithmetic:
- `.reg` declarations → create `SsirLocalVar` entries
- Parameterized registers (`%r<N>`) → expand and create lazily
- `add`, `sub`, `mul`, `div`, `rem`, `neg`, `abs`, `min`, `max` → `SSIR_OP_ADD/SUB/MUL/DIV/MOD/NEG` + `SSIR_OP_BUILTIN` for abs/min/max
- `mad`/`fma` → `SSIR_OP_MUL` + `SSIR_OP_ADD` (or `SSIR_OP_BUILTIN_FMA` if available)
- `and`, `or`, `xor`, `not`, `shl`, `shr` → `SSIR_OP_BIT_AND/OR/XOR/NOT/SHL/SHR`
- `mov` → `SSIR_OP_STORE` (copy value to target register's local var)
- Type suffix parsing for all instructions

**Test**: Parse a function with arithmetic instructions, verify SSIR instructions are generated with correct opcodes and operand types.

### Step 5: Memory Operations (~400 lines)

Parse load/store instructions:
- `ld.{space}.{type}` → `SSIR_OP_LOAD` with appropriate address space
- `st.{space}.{type}` → `SSIR_OP_STORE`
- `ld.param` → load from parameter variable
- Address expressions: `[%reg]`, `[%reg + offset]`, `[symbol]`, `[symbol + offset]`
- Vector load/store: `ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [addr]` → load vec4, then extract components
- `cvta` (address space conversion) → `SSIR_OP_BITCAST` or `SSIR_OP_CONVERT`

**Test**: Parse loads and stores from different address spaces. Verify correct SSIR memory operations.

### Step 6: Comparison, Selection, and Predicates (~400 lines)

- `setp.{cmp}.{type} %p, %a, %b` → `SSIR_OP_LT/LE/GT/GE/EQ/NE` producing a boolean, stored to predicate register's local var
- `selp.{type} %d, %a, %b, %p` → `SSIR_OP_SELECT` (or emit as branch_cond + phi)
- `set.{cmp}.{type}.{type2} %d, %a, %b` → comparison + type convert
- Predicate combination (`setp.lt.and.f32 %p1|%p2, ...`) → comparison + `SSIR_OP_AND` + negation for the complement predicate

**Test**: Parse setp/selp instructions. Verify boolean comparisons produce correct SSIR.

### Step 7: Control Flow (~500 lines)

- Label parsing: `LABEL_NAME:` → record block start
- `bra LABEL` → `SSIR_OP_BRANCH`
- `@%p bra LABEL` → `SSIR_OP_BRANCH_COND`
- `ret` → `SSIR_OP_RETURN`
- `exit` → `SSIR_OP_RETURN` (for entry points)
- `call` → `SSIR_OP_CALL`
- Block splitting at labels and branches
- Forward reference resolution (patch unresolved branch targets)
- Implicit fallthrough → explicit branch to next block

**Test**: Parse a function with branches, labels, and loops. Verify the CFG is correct with proper block structure and branch targets.

### Step 8: Predicated Execution Lowering (~400 lines)

Transform predicated non-branch instructions into conditional blocks:
- `@%p add.f32 %c, %a, %b` → split into conditional blocks with phi nodes
- `@!%p mov.f32 %c, 0.0` → negated condition
- Consecutive instructions with same predicate → group into one conditional block
- Handle the common `setp` + `@%p bra` pattern efficiently (direct branch_cond, no extra blocks)

**Test**: Parse predicated instructions, verify they produce correct conditional block structure in SSIR.

### Step 9: Special Registers (~200 lines)

Map PTX special register reads to SSIR built-in variable loads:
- `mov.u32 %r, %tid.x` → load from `SSIR_BUILTIN_LOCAL_INVOCATION_ID`, extract `.x`
- `mov.u32 %r, %ctaid.x` → load from `SSIR_BUILTIN_WORKGROUP_ID`, extract `.x`
- `mov.u32 %r, %ntid.x` → workgroup size (constant or built-in)
- `mov.u32 %r, %nctaid.x` → load from `SSIR_BUILTIN_NUM_WORKGROUPS`, extract `.x`
- `mov.u32 %r, %laneid` → load from `SSIR_BUILTIN_SUBGROUP_INVOCATION_ID`
- Create built-in global input variables decorated with the appropriate built-in kind
- Add these variables to the entry point's interface list

**Test**: Parse special register accesses, verify built-in variables are created and loaded correctly.

### Step 10: Type Conversions and Math (~300 lines)

- `cvt.{rnd}.{dst_type}.{src_type}` → `SSIR_OP_CONVERT` (rounding mode stored as operand or ignored if not representable)
- `rcp`, `sqrt`, `rsqrt`, `sin`, `cos`, `lg2`, `ex2` → `SSIR_OP_BUILTIN` with appropriate built-in function
- `.approx` modifier → note in SSIR (or just treat as regular — all GPU math is approximate anyway)
- `.sat` modifier → emit `SSIR_OP_BUILTIN_CLAMP` with 0.0, 1.0 bounds after the instruction

**Test**: Parse type conversions and math instructions. Verify correct SSIR built-in calls.

### Step 11: Atomics and Barriers (~200 lines)

- `atom.{space}.{op}.{type}` → `SSIR_OP_ATOMIC`
- `bar.sync N` → `SSIR_OP_BARRIER` with `SSIR_SCOPE_WORKGROUP`
- `membar.{scope}` → `SSIR_OP_BARRIER` (memory-only)

**Test**: Parse atomic and barrier instructions. Verify SSIR atomics have correct operation, scope, and memory semantics.

### Step 12: Kernel Parameter Convention (~300 lines)

Implement the parameter-to-binding mapping:
- Scan `.entry` parameter list
- Pointer params (`.u64`) → storage buffer bindings at `@group(0) @binding(0..N-1)`
- Scalar params → pack into uniform buffer or push constant struct at next binding
- Create `SsirGlobalVar` entries with decorations (`SSIR_DECORATION_BINDING`, `SSIR_DECORATION_DESCRIPTOR_SET`)
- Replace `ld.param` instructions with loads from these SSIR global variables

**Test**: Parse a kernel with mixed pointer and scalar parameters. Verify correct binding assignments and parameter access lowering.

### Step 13: Integration Testing (~500 lines of test code)

End-to-end tests:
- PTX → SSIR → WGSL: roundtrip produces valid WGSL
- PTX → SSIR → GLSL: produces valid GLSL 450 compute shader
- PTX → SSIR → SPIR-V: produces valid SPIR-V (validate with `spirv-val`)
- Real-world PTX samples: compile simple CUDA kernels to PTX with `nvcc -ptx`, feed through parser
- Compare output correctness against CUDA execution results (manual verification)

**Test files** to create in `tests/ptx/`:
```
vector_add.ptx          — basic kernel: C[i] = A[i] + B[i]
saxpy.ptx               — SAXPY: y[i] = a*x[i] + y[i]
reduce_sum.ptx          — parallel reduction with shared memory
dot_product.ptx         — dot product with shared mem + atomic
matmul_naive.ptx        — naive matrix multiply (nested loops)
control_flow.ptx        — branches, labels, predicates
type_convert.ptx        — various cvt instructions
atomic_ops.ptx          — atomic add/min/max/cas
math_functions.ptx      — sin/cos/sqrt/rsqrt/lg2/ex2
device_function.ptx     — .func call from .entry
predicated_exec.ptx     — heavy predicated instruction usage
vector_types.ptx        — .v2/.v4 register operations
special_regs.ptx        — %tid/%ctaid/%ntid usage patterns
barrier_sync.ptx        — bar.sync and shared memory patterns
multi_kernel.ptx        — multiple .entry in one module
```

### Step 14: CMake Integration

- Add `ptx_parser.c` to the library sources in `CMakeLists.txt`
- Add `tests/ptx_parser_test.cpp` to the test targets
- Copy `tests/ptx/*.ptx` test files

---

## 7. Key Challenges and Solutions

### 7.1 Unstructured → Structured Control Flow

**Challenge**: PTX has arbitrary `goto`s; WGSL/GLSL require structured control flow (if/else/for/while).

**Solution**: SSIR already supports unstructured CFGs (basic blocks + branch/branch_cond). The existing emitters (`ssir_to_wgsl.c`, `ssir_to_glsl.c`) already have CFG structurization logic for SPIR-V input (which is also unstructured at the CFG level). We leverage this entirely. No new structurization code needed.

### 7.2 Register Reuse (non-SSA → SSA)

**Challenge**: PTX reuses registers (`%r1 = ...; %r1 = ...`). SSIR is SSA.

**Solution**: Model PTX registers as local variables (SSIR `OpVariable` with `Function` storage class). Every register write → `store`, every register read → `load`. This is exactly what SPIR-V does before `mem2reg`, and SSIR consumers already handle it. This generates more instructions than necessary but is correct and simple.

### 7.3 Predicated Execution

**Challenge**: Any PTX instruction can be predicated. SSIR has no instruction-level predication.

**Solution**: Lower predicated instructions to conditional blocks (section 5.4). The common case (`@%p bra LABEL`) maps directly to `branch_cond` with zero overhead. Predicated non-branch instructions (less common in real code) get the block-splitting treatment.

### 7.4 Pointer Arithmetic → Typed Access

**Challenge**: PTX does raw pointer arithmetic. WGSL/GLSL want typed array indexing.

**Solution**: For the initial implementation, use byte-addressable storage buffers and preserve arithmetic as-is. This produces correct but verbose output. A future optimization pass could pattern-match `base + index * sizeof(T)` → `array[index]`.

### 7.5 Kernel Parameters → Descriptor Bindings

**Challenge**: PTX passes flat parameters (u64 pointers, scalar values). GPU shader APIs use descriptor sets with bindings.

**Solution**: Use a simple convention (section 5.6): pointer params → storage buffer bindings, scalar params → uniform buffer. Document the convention. Allow override via options in future work.

### 7.6 Half-precision and Double-precision

**Challenge**: SSIR may not support all PTX type widths in all backends.

**Solution**: SSIR already has `SSIR_TYPE_F16`, `SSIR_TYPE_F64`, `SSIR_TYPE_I8`, `SSIR_TYPE_U8`, etc. These map to WGSL `f16` (with `enable f16`), GLSL `float16_t`/`double`, HLSL `half`/`double`. For backends that don't support a type, the emitter already handles widening/narrowing as needed.

### 7.7 PTX Version Compatibility

**Challenge**: Different PTX versions introduce new instructions/features.

**Solution**: Parse the `.version` directive and store it. For unsupported instructions, emit a clear error: `"PTX instruction 'wmma.load' requires tensor core support (not yet implemented)"`. The initial implementation targets PTX 6.0+ features (sm_60+), which covers the vast majority of real-world PTX code.

---

## 8. Estimated Size

| Component | Lines |
|-----------|-------|
| Lexer | ~400 |
| Parser context & utilities | ~200 |
| Module-level parsing | ~300 |
| Kernel/function signatures | ~400 |
| Register management | ~200 |
| Instruction parsing (all categories) | ~1,200 |
| Memory operations | ~400 |
| Control flow & CFG construction | ~500 |
| Predicate lowering | ~400 |
| Special registers & built-ins | ~200 |
| Parameter convention | ~300 |
| **Total ptx_parser.c** | **~4,500** |
| Test code (ptx_parser_test.cpp) | ~2,000 |
| Test PTX files | ~1,000 |
| API additions to simple_wgsl.h | ~30 |
| CMake changes | ~10 |
| **Grand total** | **~7,500** |

This is comparable to the existing parsers: `wgsl_parser.c` is 2,452 lines, `glsl_parser.c` is 2,466 lines, and `msl_parser.c` is 2,432 lines. PTX is larger because it requires CFG construction and predicate lowering that the other parsers don't need (they parse structured languages).

---

## 9. SSIR Additions Required

The existing SSIR type system and instruction set should cover nearly all PTX needs. However, a few additions may be needed:

### 9.1 Potentially Needed

- **`SSIR_BUILTIN_RECIP`**: PTX `rcp` (reciprocal). If not present, emit as `1.0 / x`.
- **Rounding mode on convert**: PTX `cvt.rn.f32.s32` specifies rounding. SSIR `CONVERT` may not carry rounding info. If not, the rounding mode is implicit (round-to-nearest-even is the default for GPU hardware, which matches PTX `.rn`).

### 9.2 Probably Not Needed

The following SSIR features already exist and handle PTX constructs:
- Atomic operations (all variants)
- Barrier with scope
- All comparison operators
- Vector construction/extraction
- Type conversion (int↔float, widening, narrowing)
- All arithmetic and bitwise operations
- Built-in math functions (sin, cos, sqrt, rsqrt, log2, exp2, abs, min, max)
- Multiple address spaces (function, workgroup, storage, uniform)

---

## 10. Future Work (Beyond Initial Scope)

### 10.1 Optimization Passes
- **Typed array reconstruction**: Pattern-match `base + idx * sizeof(T)` → `array[idx]`
- **Dead store elimination**: Remove redundant register stores
- **Predicate simplification**: Merge consecutive predicated blocks
- **Constant folding**: Evaluate compile-time-known expressions

### 10.2 Extended Instruction Support
- Texture/surface operations (requires SSIR texture infrastructure)
- Tensor Core / WMMA operations (requires matrix type support)
- Warp-level primitives (`shfl`, `vote`, `match`, `redux`)
- Video instructions (niche, low priority)

### 10.3 SSIR → PTX Emitter (Output Direction)
Once input works, adding an `ssir_to_ptx.c` emitter enables:
```
WGSL/GLSL/MSL → SSIR → PTX → cubin (via ptxas)
```
This would allow compiling web shaders to CUDA-compatible PTX.

### 10.4 Bidirectional Roundtrip
With both `ptx_to_ssir` and `ssir_to_ptx`, we can test roundtrip fidelity:
```
PTX → SSIR → PTX'  (verify semantic equivalence)
```
