// BEGIN FILE wgsl_parser.h
#ifndef WGSL_PARSER_H
#define WGSL_PARSER_H

#ifndef NODE_ALLOC
#define NODE_ALLOC(T) (T *)calloc(1, sizeof(T))
#endif
#ifndef NODE_MALLOC
#define NODE_MALLOC(SZ) calloc(1, (SZ))
#endif
#ifndef NODE_REALLOC
#define NODE_REALLOC(P, SZ) realloc((P), (SZ))
#endif
#ifndef NODE_FREE
#define NODE_FREE(P) free((P))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum WgslNodeType {
    WGSL_NODE_PROGRAM = 1,
    WGSL_NODE_STRUCT,
    WGSL_NODE_STRUCT_FIELD,
    WGSL_NODE_GLOBAL_VAR,
    WGSL_NODE_FUNCTION,
    WGSL_NODE_PARAM,
    WGSL_NODE_TYPE,
    WGSL_NODE_ATTRIBUTE,
    WGSL_NODE_BLOCK,
    WGSL_NODE_VAR_DECL,
    WGSL_NODE_RETURN,
    WGSL_NODE_EXPR_STMT,
    WGSL_NODE_IF,
    WGSL_NODE_WHILE,
    WGSL_NODE_FOR,
    WGSL_NODE_IDENT,
    WGSL_NODE_LITERAL,
    WGSL_NODE_BINARY,
    WGSL_NODE_ASSIGN,
    WGSL_NODE_CALL,
    WGSL_NODE_MEMBER,
    WGSL_NODE_INDEX,
    WGSL_NODE_UNARY,
    WGSL_NODE_TERNARY
} WgslNodeType;

typedef struct WgslAstNode WgslAstNode;

typedef struct Attribute {
    char *name;
    int arg_count;
    WgslAstNode **args;
} Attribute;

typedef struct TypeNode {
    char *name;
    int type_arg_count;
    WgslAstNode **type_args;
    int expr_arg_count;
    WgslAstNode **expr_args;
} TypeNode;

typedef struct StructField {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    WgslAstNode *type;
} StructField;

typedef struct StructDecl {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    int field_count;
    WgslAstNode **fields;
} StructDecl;

typedef struct GlobalVar {
    int attr_count;
    WgslAstNode **attrs;
    char *address_space;
    char *name;
    WgslAstNode *type;
} GlobalVar;

typedef struct Param {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    WgslAstNode *type;
} Param;

typedef struct Block {
    int stmt_count;
    WgslAstNode **stmts;
} Block;

typedef struct VarDecl {
    char *name;
    WgslAstNode *type;
    WgslAstNode *init;
} VarDecl;

typedef struct ReturnNode {
    WgslAstNode *expr;
} ReturnNode;

typedef struct ExprStmt {
    WgslAstNode *expr;
} ExprStmt;

typedef struct Function {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    int param_count;
    WgslAstNode **params;
    int ret_attr_count;
    WgslAstNode **ret_attrs;
    WgslAstNode *return_type;
    WgslAstNode *body;
} Function;

typedef struct IfStmt {
    WgslAstNode *cond;
    WgslAstNode *then_branch;
    WgslAstNode *else_branch;
} IfStmt;

typedef struct WhileStmt {
    WgslAstNode *cond;
    WgslAstNode *body;
} WhileStmt;

typedef struct ForStmt {
    WgslAstNode *init;
    WgslAstNode *cond;
    WgslAstNode *cont;
    WgslAstNode *body;
} ForStmt;

typedef struct Ident {
    char *name;
} Ident;

typedef enum WgslLiteralKind { WGSL_LIT_INT, WGSL_LIT_FLOAT } WgslLiteralKind;

typedef struct Literal {
    WgslLiteralKind kind;
    char *lexeme;
} Literal;

typedef struct Binary {
    char *op;
    WgslAstNode *left;
    WgslAstNode *right;
} Binary;

typedef struct Assign {
    WgslAstNode *lhs;
    WgslAstNode *rhs;
} Assign;

typedef struct Call {
    WgslAstNode *callee;
    int arg_count;
    WgslAstNode **args;
} Call;

typedef struct Member {
    WgslAstNode *object;
    char *member;
} Member;

typedef struct Index {
    WgslAstNode *object;
    WgslAstNode *index;
} Index;

typedef struct Unary {
    char *op;
    int is_postfix;
    WgslAstNode *expr;
} Unary;

typedef struct Ternary {
    WgslAstNode *cond;
    WgslAstNode *then_expr;
    WgslAstNode *else_expr;
} Ternary;

typedef struct Program {
    int decl_count;
    WgslAstNode **decls;
} Program;

typedef struct WgslAstNode {
    WgslNodeType type;
    int line;
    int col;
    union {
        Program program;
        Attribute attribute;
        TypeNode type_node;
        StructDecl struct_decl;
        StructField struct_field;
        GlobalVar global_var;
        Function function;
        Param param;
        Block block;
        VarDecl var_decl;
        ReturnNode return_stmt;
        ExprStmt expr_stmt;
        IfStmt if_stmt;
        WhileStmt while_stmt;
        ForStmt for_stmt;
        Ident ident;
        Literal literal;
        Binary binary;
        Assign assign;
        Call call;
        Member member;
        Index index;
        Unary unary;
        Ternary ternary;
    };
} WgslAstNode;

WgslAstNode *wgsl_parse(const char *source);
void wgsl_free_ast(WgslAstNode *node);
const char *wgsl_node_type_name(WgslNodeType t);
void wgsl_debug_print(const WgslAstNode *node, int indent);

#ifdef __cplusplus
}
#endif
#endif
// END FILE wgsl_parser.h