/*
 * fft_strbuf.h - Shared string builder for FFT WGSL shader generators
 *
 * Provides StrBuf (growable string buffer), sb_float (WGSL-safe float
 * literals), and sb_cmul (inline complex multiply code generation).
 *
 * Include this header in exactly the .c files that need it — all functions
 * are static, so each translation unit gets its own copy with no link
 * conflicts.
 */

#ifndef FFT_STRBUF_H
#define FFT_STRBUF_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef SW_UNUSED
#if defined(__GNUC__) || defined(__clang__)
#  define SW_UNUSED __attribute__((unused))
#else
#  define SW_UNUSED
#endif
#endif

typedef struct {
  char *buf;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_init_cap(StrBuf *sb, size_t initial_cap) {
  sb->cap = initial_cap;
  sb->buf = (char *)malloc(sb->cap);
  if (!sb->buf) { sb->cap = 0; sb->len = 0; return; }
  sb->buf[0] = '\0';
  sb->len = 0;
}

static SW_UNUSED void sb_init(StrBuf *sb) { sb_init_cap(sb, 4096); }

__attribute__((format(printf, 2, 3)))
static void sb_printf(StrBuf *sb, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int needed = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (needed < 0) return;
  while (sb->len + (size_t)needed + 1 > sb->cap) {
    sb->cap *= 2;
    char *new_buf = (char *)realloc(sb->buf, sb->cap);
    if (!new_buf) return;
    sb->buf = new_buf;
  }
  va_start(ap, fmt);
  vsnprintf(sb->buf + sb->len, sb->cap - sb->len, fmt, ap);
  va_end(ap);
  sb->len += (size_t)needed;
}

static SW_UNUSED char *sb_finish(StrBuf *sb) { return sb->buf; }

/*
 * sb_float: emit a float literal that WGSL will parse as f32, not i32.
 *
 * %.17g can format 1.0 as "1" and 0.0 as "0". WGSL treats these as integer
 * literals. When multiplied with f32, the compiler emits OpIMul on float bit
 * patterns instead of OpFMul. We append ".0" when the formatted string has
 * no decimal point and no exponent notation.
 */
static void sb_float(StrBuf *sb, double v) {
  char tmp[64];
  snprintf(tmp, sizeof(tmp), "%.17g", v);
  sb_printf(sb, "%s", tmp);
  if (!strchr(tmp, '.') && !strchr(tmp, 'e') && !strchr(tmp, 'E'))
    sb_printf(sb, ".0");
}

/*
 * sb_cmul: emit inline complex multiply WGSL expression.
 * Result: vec2<f32>(a.x*br - a.y*bi, a.x*bi + a.y*br)
 * where 'a' is a variable name and (br, bi) is a baked constant.
 */
__attribute__((unused))
static void sb_cmul(StrBuf *sb, const char *a, double br, double bi) {
  sb_printf(sb, "vec2<f32>(%s.x*", a);
  sb_float(sb, br);
  sb_printf(sb, " - %s.y*", a);
  sb_float(sb, bi);
  sb_printf(sb, ", %s.x*", a);
  sb_float(sb, bi);
  sb_printf(sb, " + %s.y*", a);
  sb_float(sb, br);
  sb_printf(sb, ")");
}

#endif /* FFT_STRBUF_H */
