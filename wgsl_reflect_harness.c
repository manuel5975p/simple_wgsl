// FILE: wgsl_reflect_harness.c
#include "simple_wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        tests_failed++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } else { \
        tests_passed++; \
    } \
} while (0)

static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* s = (char*)malloc((size_t)n + 1);
    if (!s) { fclose(f); return NULL; }
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0;
    fclose(f);
    return s;
}

static int starts_with(const char* s, const char* pfx) { return strncmp(s, pfx, strlen(pfx)) == 0; }
static void trim(char* s) {
    size_t L = strlen(s);
    while (L && (unsigned char)s[L-1] <= ' ') s[--L] = 0;
    size_t i = 0; while (s[i] && (unsigned char)s[i] <= ' ') i++;
    if (i) memmove(s, s + i, strlen(s + i) + 1);
}
static char* strdup_range(const char* a, const char* b) {
    size_t n = (size_t)(b - a);
    char* s = (char*)malloc(n + 1);
    memcpy(s, a, n);
    s[n] = 0;
    return s;
}

/* expectations */
typedef struct {
    char* name;
    WgslStage stage;
} ExpEntry;

typedef struct {
    char* func;
    char** names; int* grp; int* bind; int count, cap;
} ExpBindings;

typedef struct {
    char* func;
    int* loc; int* comps; int count, cap;
} ExpVtx;

typedef struct {
    char** names; int count, cap;
} ExpGlobals;

typedef struct {
    ExpEntry* entries; int entries_n, entries_cap;
    ExpBindings* binds; int binds_n, binds_cap;
    ExpVtx* vslots; int vslots_n, vslots_cap;
    ExpGlobals globals;
} Expectations;

static void vec_grow(void** p, int* cap, size_t esz) {
    *cap = *cap ? (*cap * 2) : 8;
    *p = realloc(*p, (size_t)(*cap) * esz);
}

/* parse lines like:
   // EXPECT: entrypoints=vs_main:vertex,fs_main:fragment
   // EXPECT: bindings.vs_main=Perspective_View@0:0,modelMatrix@0:3
   // EXPECT: vertex_inputs.vs_main=loc0:3,loc1:2,loc2:3,loc3:4
   // EXPECT: globals_contains=konschtante,lights2
*/
static void parse_expectations(const char* src, Expectations* E) {
    memset(E, 0, sizeof(*E));
    const char* s = src;
    while (*s) {
        const char* line = s;
        const char* nl = strchr(s, '\n');
        if (!nl) nl = s + strlen(s);
        if (starts_with(line, "// EXPECT:")) {
            char* L = strdup_range(line + 10, nl);
            trim(L);
            char* eq = strchr(L, '=');
            if (eq) {
                *eq = 0; trim(L); char* key = L; char* val = eq + 1; trim(val);
                if (strcmp(key, "entrypoints") == 0) {
                    /* vs:vertex,fs:fragment */
                    char* p = val;
                    while (*p) {
                        char* comma = strchr(p, ','); if (!comma) comma = p + strlen(p);
                        char* pair = strdup_range(p, comma);
                        char* c2 = strchr(pair, ':');
                        if (c2) {
                            *c2 = 0; trim(pair); trim(c2 + 1);
                            if (E->entries_n >= E->entries_cap) vec_grow((void**)&E->entries, &E->entries_cap, sizeof(ExpEntry));
                            E->entries[E->entries_n].name = strdup(pair);
                            WgslStage st = WGSL_STAGE_UNKNOWN;
                            if (strcmp(c2 + 1, "vertex") == 0) st = WGSL_STAGE_VERTEX;
                            if (strcmp(c2 + 1, "fragment") == 0) st = WGSL_STAGE_FRAGMENT;
                            if (strcmp(c2 + 1, "compute") == 0) st = WGSL_STAGE_COMPUTE;
                            E->entries[E->entries_n].stage = st;
                            E->entries_n++;
                        }
                        free(pair);
                        if (*comma == 0) break;
                        p = comma + 1;
                        while (*p == ' ' || *p == '\t') p++;
                    }
                } else if (starts_with(key, "bindings.")) {
                    const char* fn = key + 9;
                    char* p = val;
                    ExpBindings ex = {0};
                    ex.func = strdup(fn);
                    while (*p) {
                        char* comma = strchr(p, ','); if (!comma) comma = p + strlen(p);
                        char* item = strdup_range(p, comma);
                        char* at = strchr(item, '@');
                        char* colon = at ? strchr(at + 1, ':') : NULL;
                        if (at && colon) {
                            *at = 0; *colon = 0;
                            trim(item); trim(at + 1); trim(colon + 1);
                            if (ex.count >= ex.cap) {
                                ex.cap = ex.cap ? ex.cap * 2 : 4;
                                ex.names = (char**)realloc(ex.names, sizeof(char*) * ex.cap);
                                ex.grp = (int*)realloc(ex.grp, sizeof(int) * ex.cap);
                                ex.bind = (int*)realloc(ex.bind, sizeof(int) * ex.cap);
                            }
                            ex.names[ex.count] = strdup(item);
                            ex.grp[ex.count] = atoi(at + 1);
                            ex.bind[ex.count] = atoi(colon + 1);
                            ex.count++;
                        }
                        free(item);
                        if (*comma == 0) break;
                        p = comma + 1;
                        while (*p == ' ' || *p == '\t') p++;
                    }
                    if (E->binds_n >= E->binds_cap) vec_grow((void**)&E->binds, &E->binds_cap, sizeof(ExpBindings));
                    E->binds[E->binds_n++] = ex;
                } else if (starts_with(key, "vertex_inputs.")) {
                    const char* fn = key + 14;
                    char* p = val;
                    ExpVtx ex = {0};
                    ex.func = strdup(fn);
                    while (*p) {
                        char* comma = strchr(p, ','); if (!comma) comma = p + strlen(p);
                        char* item = strdup_range(p, comma);
                        char* colon = strchr(item, ':');
                        if (colon) {
                            *colon = 0; trim(item); trim(colon + 1);
                            int loc = -1, comps = atoi(colon + 1);
                            if (starts_with(item, "loc")) loc = atoi(item + 3);
                            if (ex.count >= ex.cap) {
                                ex.cap = ex.cap ? ex.cap * 2 : 4;
                                ex.loc = (int*)realloc(ex.loc, sizeof(int) * ex.cap);
                                ex.comps = (int*)realloc(ex.comps, sizeof(int) * ex.cap);
                            }
                            ex.loc[ex.count] = loc;
                            ex.comps[ex.count] = comps;
                            ex.count++;
                        }
                        free(item);
                        if (*comma == 0) break;
                        p = comma + 1;
                        while (*p == ' ' || *p == '\t') p++;
                    }
                    if (E->vslots_n >= E->vslots_cap) vec_grow((void**)&E->vslots, &E->vslots_cap, sizeof(ExpVtx));
                    E->vslots[E->vslots_n++] = ex;
                } else if (strcmp(key, "globals_contains") == 0) {
                    char* p = val;
                    while (*p) {
                        char* comma = strchr(p, ','); if (!comma) comma = p + strlen(p);
                        char* item = strdup_range(p, comma);
                        trim(item);
                        if (E->globals.count >= E->globals.cap) {
                            E->globals.cap = E->globals.cap ? E->globals.cap * 2 : 8;
                            E->globals.names = (char**)realloc(E->globals.names, sizeof(char*) * E->globals.cap);
                        }
                        E->globals.names[E->globals.count++] = item;
                        if (*comma == 0) break;
                        p = comma + 1;
                        while (*p == ' ' || *p == '\t') p++;
                    }
                }
            }
            free(L);
        }
        s = (*nl) ? nl + 1 : nl;
    }
}

static const WgslResolverEntrypoint* find_ep(const WgslResolverEntrypoint* eps, int n, const char* name) {
    for (int i = 0; i < n; i++) if (eps[i].name && strcmp(eps[i].name, name) == 0) return &eps[i];
    return NULL;
}
static int has_binding(const WgslSymbolInfo* syms, int n, const char* name, int g, int b) {
    for (int i = 0; i < n; i++)
        if (syms[i].name && strcmp(syms[i].name, name) == 0 &&
            syms[i].has_group && syms[i].group_index == g &&
            syms[i].has_binding && syms[i].binding_index == b) return 1;
    return 0;
}
static int has_global(const WgslSymbolInfo* syms, int n, const char* name) {
    for (int i = 0; i < n; i++) if (syms[i].name && strcmp(syms[i].name, name) == 0) return 1;
    return 0;
}

static void free_expectations(Expectations* E) {
    if (!E) return;

    for (int i = 0; i < E->entries_n; i++) {
        free(E->entries[i].name);
    }
    free(E->entries);
    E->entries = NULL; E->entries_n = E->entries_cap = 0;

    for (int i = 0; i < E->binds_n; i++) {
        ExpBindings* b = &E->binds[i];
        free(b->func);
        if (b->names) {
            for (int k = 0; k < b->count; k++) free(b->names[k]);
        }
        free(b->names);
        free(b->grp);
        free(b->bind);
        b->names = NULL; b->grp = NULL; b->bind = NULL; b->count = b->cap = 0;
    }
    free(E->binds);
    E->binds = NULL; E->binds_n = E->binds_cap = 0;

    for (int i = 0; i < E->vslots_n; i++) {
        ExpVtx* v = &E->vslots[i];
        free(v->func);
        free(v->loc);
        free(v->comps);
        v->loc = NULL; v->comps = NULL; v->count = v->cap = 0;
    }
    free(E->vslots);
    E->vslots = NULL; E->vslots_n = E->vslots_cap = 0;

    if (E->globals.names) {
        for (int i = 0; i < E->globals.count; i++) free(E->globals.names[i]);
    }
    free(E->globals.names);
    E->globals.names = NULL; E->globals.count = E->globals.cap = 0;
}

static void run_one(const char* path) {
    char* src = read_file(path);
    CHECK(src != NULL);
    Expectations E; parse_expectations(src, &E);

    WgslAstNode* ast = wgsl_parse(src);
    CHECK(ast != NULL);
    WgslResolver* r = wgsl_resolver_build(ast);
    CHECK(r != NULL);

    int epn = 0;
    const WgslResolverEntrypoint* eps = wgsl_resolver_entrypoints(r, &epn);
    CHECK(eps != NULL || E.entries_n == 0);
    char spvPath[256] = {0};
    snprintf(spvPath, 128, "%s.spv", path);
    WgslLowerOptions lowerOptions = {
        .env = WGSL_LOWER_ENV_VULKAN_1_3
    };
    uint32_t* outWords;
    size_t wordCount;
    wgsl_lower_emit_spirv(ast, r, &lowerOptions, &outWords, &wordCount);
    FILE* f = fopen(spvPath, "w");
    fwrite(outWords, sizeof(uint32_t), wordCount, f);
    fclose(f);
    wgsl_lower_free(outWords);
    //WgslLower* lowered = wgsl_lower_create(ast, r, &lowerOptions);
    //wgsl_lower_destroy(lowered);
    
    for (int i = 0; i < E.entries_n; i++) {
        const WgslResolverEntrypoint* ep = eps ? find_ep(eps, epn, E.entries[i].name) : NULL;
        CHECK(ep != NULL);
        CHECK(ep && ep->stage == E.entries[i].stage);
    }

    for (int i = 0; i < E.binds_n; i++) {
        int rc = 0;
        const WgslSymbolInfo* b = wgsl_resolver_entrypoint_binding_vars(r, E.binds[i].func, &rc);
        CHECK(b != NULL || E.binds[i].count == 0);
        for (int k = 0; k < E.binds[i].count; k++)
            CHECK(has_binding(b, rc, E.binds[i].names[k], E.binds[i].grp[k], E.binds[i].bind[k]));
        wgsl_resolve_free((void*)b);
    }

    for (int i = 0; i < E.vslots_n; i++) {
        WgslVertexSlot* slots = NULL;
        int n = wgsl_resolver_vertex_inputs(r, E.vslots[i].func, &slots);
        for (int k = 0; k < E.vslots[i].count; k++) {
            int loc = E.vslots[i].loc[k], want = E.vslots[i].comps[k], got = -1;
            for (int j = 0; j < n; j++) if (slots[j].location == loc) got = slots[j].component_count;
            CHECK(got == want);
        }
        wgsl_resolve_free(slots);
    }

    if (E.globals.count > 0) {
        int gn = 0;
        const WgslSymbolInfo* globals = wgsl_resolver_globals(r, &gn);
        CHECK(globals != NULL);
        for (int i = 0; i < E.globals.count; i++)
            CHECK(has_global(globals, gn, E.globals.names[i]));
        wgsl_resolve_free((void*)globals);
    }

    wgsl_resolve_free((void*)eps);
    wgsl_resolver_free(r);
    wgsl_free_ast(ast);
    free_expectations(&E);
    free(src);
}


int main(void) {
    const char* files[] = {
        "wgsl/vertex_fragment.wgsl",
        "wgsl/compute_basic.wgsl",
        "wgsl/compute_minimal.wgsl",
        "wgsl/transitive_vertex.wgsl",
    };
    for(int runs = 0; runs < 1;runs++){
        for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); i++){
            run_one(files[i]);
        }
    }
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    return tests_failed ? 1 : 0;
}
