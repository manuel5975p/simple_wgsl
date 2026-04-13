// BEGIN FILE wgsl_diagnostics.c
//
// Implementation of the WgslDiagnosticList type and the internal emission
// helpers shared by wgsl_parser.c / wgsl_resolve.c / wgsl_lower.c.
//
// Ownership contract (see docs/superpowers/specs/2026-04-13-wgsl-ssir-diagnostics-design.md):
//   - Each diagnostic's `message` is heap-allocated via SW_MALLOC and owned
//     by the containing WgslDiagnosticList.
//   - `code_section_begin` / `code_section_end` alias the caller's source
//     buffer; the list does NOT own them.
//   - wgsl_diagnostic_list_free frees every message, the items array, and
//     the list struct itself. Safe to call with NULL.

#include "simple_wgsl_internal.h"

void wgsl_diagnostic_list_free(WgslDiagnosticList *list) {
    if (!list) return;
    for (int i = 0; i < list->count; ++i) {
        SW_FREE((void *)list->items[i].message);
    }
    SW_FREE(list->items);
    SW_FREE(list);
}

const char *wgsl_diagnostic_severity_string(WgslDiagnosticSeverity s) {
    switch (s) {
        case WGSL_DIAG_ERROR:   return "error";
        case WGSL_DIAG_WARNING: return "warning";
        case WGSL_DIAG_NOTE:    return "note";
    }
    return "unknown";
}

const char *wgsl_diagnostic_code_string(WgslDiagnosticCode c) {
    switch (c) {
        case WGSL_DIAG_CODE_NONE:                    return "none";
        case WGSL_DIAG_PARSE_UNEXPECTED_TOKEN:       return "parse.unexpected_token";
        case WGSL_DIAG_PARSE_EXPECTED_IDENT:         return "parse.expected_ident";
        case WGSL_DIAG_PARSE_EXPECTED_TYPE:          return "parse.expected_type";
        case WGSL_DIAG_PARSE_EXPECTED_EXPRESSION:    return "parse.expected_expression";
        case WGSL_DIAG_PARSE_MISSING_INITIALIZER:    return "parse.missing_initializer";
        case WGSL_DIAG_PARSE_INVALID_ATTRIBUTE:      return "parse.invalid_attribute";
        case WGSL_DIAG_PARSE_INVALID_LITERAL:        return "parse.invalid_literal";
        case WGSL_DIAG_RESOLVE_UNDEFINED_SYMBOL:     return "resolve.undefined_symbol";
        case WGSL_DIAG_RESOLVE_DUPLICATE_SYMBOL:     return "resolve.duplicate_symbol";
        case WGSL_DIAG_RESOLVE_TYPE_MISMATCH:        return "resolve.type_mismatch";
        case WGSL_DIAG_RESOLVE_INVALID_ENTRY_POINT:  return "resolve.invalid_entry_point";
        case WGSL_DIAG_RESOLVE_INVALID_BINDING:      return "resolve.invalid_binding";
        case WGSL_DIAG_LOWER_UNSUPPORTED_BUILTIN:    return "lower.unsupported_builtin";
        case WGSL_DIAG_LOWER_UNSUPPORTED_TYPE:       return "lower.unsupported_type";
        case WGSL_DIAG_LOWER_INVALID_ENTRY_POINT:    return "lower.invalid_entry_point";
        case WGSL_DIAG_LOWER_INTERNAL:               return "lower.internal";
    }
    return "unknown";
}

WgslDiagnosticList *wgsl_diag_list_new(void) {
    return (WgslDiagnosticList *)SW_MALLOC(sizeof(WgslDiagnosticList));
}

void wgsl_diag_list_append(WgslDiagnosticList *list,
                           WgslDiagnosticSeverity sev,
                           WgslDiagnosticCode code,
                           const char *owned_message,
                           const char *begin,
                           const char *end) {
    if (!list) {
        /* No list: the message would be leaked; drop it safely. */
        SW_FREE((void *)owned_message);
        return;
    }
    if (list->count >= list->capacity) {
        int nc = list->capacity ? list->capacity * 2 : 8;
        WgslDiagnostic *np = (WgslDiagnostic *)SW_REALLOC(
            list->items, (size_t)nc * sizeof(WgslDiagnostic));
        if (!np) {
            SW_FREE((void *)owned_message);
            return;
        }
        list->items = np;
        list->capacity = nc;
    }
    WgslDiagnostic *d = &list->items[list->count++];
    d->severity = sev;
    d->code = code;
    d->message = owned_message;
    d->code_section_begin = begin;
    d->code_section_end = end;
}

WgslDiagnosticList *wgsl_diag_list_concat(WgslDiagnosticList *a,
                                          WgslDiagnosticList *b) {
    if (!b) return a;
    if (!a) return b;
    int need = a->count + b->count;
    if (need > a->capacity) {
        int nc = a->capacity ? a->capacity : 8;
        while (nc < need) nc *= 2;
        WgslDiagnostic *np = (WgslDiagnostic *)SW_REALLOC(
            a->items, (size_t)nc * sizeof(WgslDiagnostic));
        if (!np) {
            /* OOM: free b's messages + struct, return a as-is. */
            for (int i = 0; i < b->count; ++i) SW_FREE((void *)b->items[i].message);
            SW_FREE(b->items);
            SW_FREE(b);
            return a;
        }
        a->items = np;
        a->capacity = nc;
    }
    for (int i = 0; i < b->count; ++i) {
        a->items[a->count++] = b->items[i];
    }
    SW_FREE(b->items);
    SW_FREE(b);
    return a;
}

char *wgsl_diag_vformat(const char *fmt, va_list ap) {
    if (!fmt) return NULL;
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);
    if (n < 0) return NULL;
    char *buf = (char *)SW_MALLOC((size_t)n + 1);
    if (!buf) return NULL;
    vsnprintf(buf, (size_t)n + 1, fmt, ap);
    return buf;
}
// END FILE wgsl_diagnostics.c
