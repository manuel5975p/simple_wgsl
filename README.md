## Simple WGSL parser and tools
- C99 (fast compile times)
- Aims to be feature complete
- Includes parsing, lowering, resolution, and raising passes

### API

**Parsing** (`wgsl_parser.h`)
```c
const char *source = "@vertex fn main() -> @builtin(position) vec4f { return vec4f(0); }";
WgslAstNode *ast = wgsl_parse(source);
printf("Parsed %d declarations\n", ast->program.decl_count);
wgsl_debug_print(ast, 0);  // pretty-print the AST
wgsl_free_ast(ast);
```

**Resolution** (`wgsl_resolve.h`)
```c
WgslResolver *r = wgsl_resolver_build(ast);
int count;
const WgslSymbolInfo *syms = wgsl_resolver_globals(r, &count);
for (int i = 0; i < count; i++) {
    printf("Global: %s (group=%d, binding=%d)\n",
           syms[i].name, syms[i].group_index, syms[i].binding_index);
}
wgsl_resolver_free(r);
```

**Lower to SPIR-V** (`wgsl_lower.h`)
```c
WgslLowerOptions opts = {0};
opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
opts.enable_debug_names = 1;

uint32_t *spirv;
size_t word_count;
WgslLowerResult res = wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &word_count);
if (res == WGSL_LOWER_OK) {
    printf("Generated %zu SPIR-V words\n", word_count);
}
wgsl_lower_free(spirv);
```

**Raise from SPIR-V** (`wgsl_raise.h`)
```c
WgslRaiseOptions opts = {0};
opts.preserve_names = 1;

char *wgsl_out;
char *error;
WgslRaiseResult res = wgsl_raise_to_wgsl(spirv, word_count, &opts, &wgsl_out, &error);
if (res == WGSL_RAISE_SUCCESS) {
    printf("Generated WGSL:\n%s\n", wgsl_out);
}
wgsl_raise_free(wgsl_out);
```

### Building

```bash
cmake -B build -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure
```

Or compile directly (parser only):
```bash
cc -c wgsl_parser.c -o wgsl_parser.o
```
