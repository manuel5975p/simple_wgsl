/* wgsl_parser.h */
#ifndef WGSL_PARSER_H
#define WGSL_PARSER_H

/* Pure C99 WGSL parser that returns an AST only.
   Skybrain: you can swap allocators via the macros below. */

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef NODE_ALLOC
#  define NODE_ALLOC(T)   (T*)calloc(1, sizeof(T))
#endif
#ifndef NODE_MALLOC
#  define NODE_MALLOC(SZ) calloc(1, (SZ))
#endif
#ifndef NODE_REALLOC
#  define NODE_REALLOC(P,SZ) realloc((P),(SZ))
#endif
#ifndef NODE_FREE
#  define NODE_FREE(P)    free((P))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum WgslNodeType {
    /* top-level */
    WGSL_NODE_PROGRAM = 1,

    /* declarations */
    WGSL_NODE_STRUCT,
    WGSL_NODE_STRUCT_FIELD,
    WGSL_NODE_GLOBAL_VAR,
    WGSL_NODE_FUNCTION,
    WGSL_NODE_PARAM,

    /* types & attributes */
    WGSL_NODE_TYPE,
    WGSL_NODE_ATTRIBUTE,

    /* statements */
    WGSL_NODE_BLOCK,
    WGSL_NODE_VAR_DECL,
    WGSL_NODE_RETURN,
    WGSL_NODE_EXPR_STMT,
    WGSL_NODE_IF,        /* <-- NEW */
    WGSL_NODE_WHILE,     /* <-- NEW */
    WGSL_NODE_FOR,       /* <-- NEW */

    /* expressions */
    WGSL_NODE_IDENT,
    WGSL_NODE_LITERAL,
    WGSL_NODE_BINARY,
    WGSL_NODE_ASSIGN,
    WGSL_NODE_CALL,
    WGSL_NODE_MEMBER,
    WGSL_NODE_INDEX
} WgslNodeType;


/* Forward so sub-structs can point to nodes */
typedef struct WgslAstNode WgslAstNode;

/* ==== Sub-node structs (by value inside the big union) ==== */

typedef struct Attribute {
    char *name;                 /* e.g. "location" or "builtin" */
    int   arg_count;            /* list of expression nodes */
    WgslAstNode **args;
} Attribute;

typedef struct IfStmt {
    WgslAstNode *cond;        /* expression */
    WgslAstNode *then_branch; /* BLOCK */
    WgslAstNode *else_branch; /* NULL, BLOCK, or nested IF */
} IfStmt;

typedef struct WhileStmt {
    WgslAstNode *cond;        /* expression */
    WgslAstNode *body;        /* BLOCK */
} WhileStmt;

typedef struct ForStmt {
    WgslAstNode *init;        /* NULL or VAR_DECL or EXPR_STMT */
    WgslAstNode *cond;        /* NULL or expression node */
    WgslAstNode *cont;        /* NULL or expression node */
    WgslAstNode *body;        /* BLOCK */
} ForStmt;


typedef struct TypeNode {
    char *name;                 /* e.g. "vec4f", "array", "texture_2d", "u32" */
    int   type_arg_count;       /* generic type args: array<T,N> puts T here */
    WgslAstNode **type_args;    /* each must be WGSL_NODE_TYPE */
    int   expr_arg_count;       /* generic non-type args: array<T, N> puts N here */
    WgslAstNode **expr_args;    /* expression nodes (usually literals/idents) */
} TypeNode;

typedef struct StructField {
    int   attr_count;
    WgslAstNode **attrs;        /* WGSL_NODE_ATTRIBUTE* */
    char *name;
    WgslAstNode *type;          /* WGSL_NODE_TYPE */
} StructField;

typedef struct StructDecl {
    int   attr_count;
    WgslAstNode **attrs;        /* not used in sample but supported */
    char *name;
    int   field_count;
    WgslAstNode **fields;       /* WGSL_NODE_STRUCT_FIELD* */
} StructDecl;

typedef struct GlobalVar {
    int   attr_count;
    WgslAstNode **attrs;        /* @group, @binding, etc. */
    char *address_space;        /* e.g. "uniform", "storage" (from var<...>) or NULL */
    char *name;
    WgslAstNode *type;          /* WGSL_NODE_TYPE */
} GlobalVar;

typedef struct Param {
    int   attr_count;
    WgslAstNode **attrs;        /* e.g. @builtin(instance_index) */
    char *name;
    WgslAstNode *type;          /* WGSL_NODE_TYPE */
} Param;

typedef struct Block {
    int   stmt_count;
    WgslAstNode **stmts;        /* mixture of *_STMT nodes */
} Block;

typedef struct VarDecl {
    char *name;
    WgslAstNode *type;          /* WGSL_NODE_TYPE or NULL for inference (not used here) */
    WgslAstNode *init;          /* optional initializer expr or NULL */
} VarDecl;

typedef struct ReturnNode {
    WgslAstNode *expr;          /* may be NULL */
} ReturnNode;

typedef struct ExprStmt {
    WgslAstNode *expr;
} ExprStmt;

typedef struct Function {
    int   attr_count;
    WgslAstNode **attrs;        /* @vertex / @fragment / etc. */
    char *name;
    int   param_count;
    WgslAstNode **params;       /* WGSL_NODE_PARAM* */
    int   ret_attr_count;
    WgslAstNode **ret_attrs;    /* attributes on return value (e.g. @location(0)) */
    WgslAstNode *return_type;   /* WGSL_NODE_TYPE or NULL for void */
    WgslAstNode *body;          /* WGSL_NODE_BLOCK */
} Function;

/* expressions */
typedef struct Ident {
    char *name;
} Ident;

typedef enum WgslLiteralKind {
    WGSL_LIT_INT,
    WGSL_LIT_FLOAT
} WgslLiteralKind;

typedef struct Literal {
    WgslLiteralKind kind;
    char *lexeme;               /* original text, e.g. "1.0f" */
} Literal;

typedef struct Binary {
    char *op;                   /* "*", "+", etc. (here we only use "*") */
    WgslAstNode *left;
    WgslAstNode *right;
} Binary;

typedef struct Assign {
    WgslAstNode *lhs;           /* lvalue expression */
    WgslAstNode *rhs;
} Assign;

typedef struct Call {
    WgslAstNode *callee;        /* IDENT or MEMBER etc. */
    int   arg_count;
    WgslAstNode **args;
} Call;

typedef struct Member {
    WgslAstNode *object;
    char *member;               /* field or swizzle (e.g. "xyz", "rgba") */
} Member;

typedef struct Index {
    WgslAstNode *object;
    WgslAstNode *index;
} Index;

typedef struct Program {
    int   decl_count;
    WgslAstNode **decls;        /* mixture: STRUCT, GLOBAL_VAR, FUNCTION */
} Program;

/* ==== The single generic node with a big union ==== */

typedef struct WgslAstNode {
    WgslNodeType type;
    int line;
    int col;
    union {
        Program      program;
        Attribute    attribute;
        TypeNode     type_node;

        StructDecl   struct_decl;
        StructField  struct_field;
        GlobalVar    global_var;
        Function     function;
        Param        param;

        Block        block;
        VarDecl      var_decl;
        ReturnNode   return_stmt;
        ExprStmt     expr_stmt;
        IfStmt       if_stmt;
        WhileStmt    while_stmt;
        ForStmt      for_stmt;

        Ident        ident;
        Literal      literal;
        Binary       binary;
        Assign       assign;
        Call         call;
        Member       member;
        Index        index;
    };
} WgslAstNode;

/* ==== API ==== */

/* Parse WGSL source into an AST (WGSL_NODE_PROGRAM as the root). */
WgslAstNode* wgsl_parse(const char *source);

/* Recursively free the AST (works with the allocation macros). */
void wgsl_free_ast(WgslAstNode *node);

/* Helper: get a human-readable node type name. */
const char* wgsl_node_type_name(WgslNodeType t);

/* Optional: debug print the AST (indent-based). */
void wgsl_debug_print(const WgslAstNode *node, int indent);

#ifdef __cplusplus
}
#endif

#endif /* WGSL_PARSER_H */
