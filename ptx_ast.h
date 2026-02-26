#ifndef PTX_AST_H
#define PTX_AST_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
  PTX_TYPE_PRED,
  PTX_TYPE_B8, PTX_TYPE_B16, PTX_TYPE_B32, PTX_TYPE_B64,
  PTX_TYPE_U8, PTX_TYPE_U16, PTX_TYPE_U32, PTX_TYPE_U64,
  PTX_TYPE_S8, PTX_TYPE_S16, PTX_TYPE_S32, PTX_TYPE_S64,
  PTX_TYPE_F16, PTX_TYPE_F32, PTX_TYPE_F64,
  PTX_TYPE_NONE = -1,
} PtxDataType;

typedef enum {
  PTX_SPACE_NONE = 0,
  PTX_SPACE_GLOBAL, PTX_SPACE_SHARED, PTX_SPACE_LOCAL,
  PTX_SPACE_CONST, PTX_SPACE_PARAM,
} PtxMemSpace;

typedef enum {
  PTX_OP_ADD, PTX_OP_SUB, PTX_OP_MUL, PTX_OP_MUL24,
  PTX_OP_DIV, PTX_OP_REM,
  PTX_OP_NEG, PTX_OP_ABS, PTX_OP_NOT, PTX_OP_CNOT,
  PTX_OP_MIN, PTX_OP_MAX,
  PTX_OP_MAD, PTX_OP_FMA,
  PTX_OP_AND, PTX_OP_OR, PTX_OP_XOR,
  PTX_OP_SHL, PTX_OP_SHR,
  PTX_OP_SETP, PTX_OP_SET, PTX_OP_SELP,
  PTX_OP_MOV,
  PTX_OP_LD, PTX_OP_ST,
  PTX_OP_CVT, PTX_OP_CVTA,
  PTX_OP_BRA, PTX_OP_RET, PTX_OP_EXIT, PTX_OP_CALL,
  PTX_OP_BAR, PTX_OP_MEMBAR,
  PTX_OP_ATOM,
  PTX_OP_RCP, PTX_OP_SQRT, PTX_OP_RSQRT,
  PTX_OP_SIN, PTX_OP_COS, PTX_OP_LG2, PTX_OP_EX2,
  PTX_OP_TEX, PTX_OP_TLD4, PTX_OP_SULD, PTX_OP_SUST,
  PTX_OP_TXQ, PTX_OP_SUQ,
  PTX_OP_POPC, PTX_OP_BREV, PTX_OP_CLZ, PTX_OP_BFIND,
  PTX_OP_SHF, PTX_OP_BFI, PTX_OP_PRMT,
  PTX_OP_COPYSIGN,
  PTX_OP_UNKNOWN,
} PtxOpcode;

typedef enum {
  PTX_CMP_EQ, PTX_CMP_NE, PTX_CMP_LT, PTX_CMP_LE,
  PTX_CMP_GT, PTX_CMP_GE,
  PTX_CMP_EQU, PTX_CMP_NEU, PTX_CMP_LTU, PTX_CMP_LEU,
  PTX_CMP_GTU, PTX_CMP_GEU,
  PTX_CMP_LO, PTX_CMP_LS, PTX_CMP_HI, PTX_CMP_HS,
  PTX_CMP_NONE,
} PtxCmpOp;

typedef enum {
  PTX_ATOMIC_ADD, PTX_ATOMIC_MIN, PTX_ATOMIC_MAX,
  PTX_ATOMIC_AND, PTX_ATOMIC_OR, PTX_ATOMIC_XOR,
  PTX_ATOMIC_EXCH, PTX_ATOMIC_CAS,
  PTX_ATOMIC_INC, PTX_ATOMIC_DEC,
} PtxAtomicOp;

typedef enum {
  PTX_TEX_1D, PTX_TEX_2D, PTX_TEX_3D,
  PTX_TEX_A1D, PTX_TEX_A2D, PTX_TEX_CUBE,
} PtxTexGeom;

typedef enum {
  PTX_MIP_NONE, PTX_MIP_LEVEL, PTX_MIP_GRAD,
} PtxMipMode;

typedef enum {
  PTX_MEMBAR_CTA, PTX_MEMBAR_GL, PTX_MEMBAR_SYS,
} PtxMembarScope;

enum {
  PTX_MOD_WIDE    = 1 << 0,  PTX_MOD_HI      = 1 << 1,
  PTX_MOD_LO      = 1 << 2,  PTX_MOD_SAT     = 1 << 3,
  PTX_MOD_FTZ     = 1 << 4,  PTX_MOD_APPROX  = 1 << 5,
  PTX_MOD_UNI     = 1 << 6,
  PTX_MOD_RN      = 1 << 7,  PTX_MOD_RZ      = 1 << 8,
  PTX_MOD_RM      = 1 << 9,  PTX_MOD_RP      = 1 << 10,
  PTX_MOD_RNI     = 1 << 11, PTX_MOD_RZI     = 1 << 12,
  PTX_MOD_RMI     = 1 << 13, PTX_MOD_RPI     = 1 << 14,
  PTX_MOD_SHIFTAMT= 1 << 15,
  PTX_MOD_LEFT    = 1 << 16, PTX_MOD_RIGHT   = 1 << 17,
  PTX_MOD_WRAP    = 1 << 18, PTX_MOD_CLAMP   = 1 << 19,
  PTX_MOD_TO      = 1 << 20,
};

typedef enum {
  PTX_OPER_NONE = 0,
  PTX_OPER_REG, PTX_OPER_IMM_INT, PTX_OPER_IMM_FLT,
  PTX_OPER_ADDR, PTX_OPER_LABEL, PTX_OPER_VEC,
} PtxOperKind;

typedef struct {
  PtxOperKind kind;
  char name[80];
  int64_t ival;
  double fval;
  char base[80];
  int64_t offset;
  char regs[4][80];
  int vec_count;
  bool negated;
} PtxOperand;

typedef struct {
  PtxOpcode opcode;
  PtxDataType type;
  PtxDataType type2;
  uint32_t modifiers;
  PtxCmpOp cmp_op;
  PtxMemSpace space;
  int vec_width;
  PtxAtomicOp atomic_op;
  PtxTexGeom tex_geom;
  PtxMipMode mip_mode;
  PtxMembarScope membar_scope;
  int tex_gather_comp;
  int bar_id;
  char combine[8];
  bool has_combine;
  char pred[80];
  bool pred_negated;
  bool has_pred;
  PtxOperand dst;
  PtxOperand dst2;
  bool has_dst2;
  PtxOperand src[8];
  int src_count;
  int line, col;
} PtxInst;

typedef enum { PTX_STMT_LABEL, PTX_STMT_INST } PtxStmtKind;

typedef struct {
  PtxStmtKind kind;
  union {
    char label[80];
    PtxInst inst;
  };
} PtxStmt;

typedef struct {
  char name[80];
  PtxDataType type;
  int vec_width;
  int count;
  bool is_parameterized;
} PtxRegDecl;

typedef struct {
  char name[80];
  PtxDataType type;
  uint32_t alignment;
  uint32_t array_size;
} PtxLocalDecl;

typedef struct {
  char name[80];
  PtxDataType type;
  uint32_t alignment;
  uint32_t array_size;
} PtxParam;

typedef struct {
  char name[80];
  PtxParam *params;
  int param_count, param_cap;
  PtxRegDecl *reg_decls;
  int reg_decl_count, reg_decl_cap;
  PtxLocalDecl *local_decls;
  int local_decl_count, local_decl_cap;
  PtxStmt *body;
  int body_count, body_cap;
  uint32_t wg_size[3];
} PtxEntry;

typedef struct {
  char name[80];
  PtxDataType return_type;
  bool has_return;
  char return_reg[80];
  PtxParam *params;
  int param_count, param_cap;
  PtxRegDecl *reg_decls;
  int reg_decl_count, reg_decl_cap;
  PtxLocalDecl *local_decls;
  int local_decl_count, local_decl_cap;
  PtxStmt *body;
  int body_count, body_cap;
  bool is_decl_only;
} PtxFunc;

typedef struct {
  char name[80];
  PtxDataType type;
  PtxMemSpace space;
  uint32_t alignment;
  uint32_t array_size;
  uint64_t init_vals[1024];
  int init_count;
  bool has_init;
} PtxGlobalDecl;

typedef enum {
  PTX_REF_TEXREF, PTX_REF_SAMPLERREF, PTX_REF_SURFREF
} PtxRefKind;

typedef struct {
  char name[80];
  PtxRefKind kind;
} PtxRefDecl;

struct PtxModule {
  int version_major, version_minor;
  int address_size;
  char target[32];
  PtxGlobalDecl *globals;
  int global_count, global_cap;
  PtxRefDecl *refs;
  int ref_count, ref_cap;
  PtxEntry *entries;
  int entry_count, entry_cap;
  PtxFunc *functions;
  int func_count, func_cap;
};

#endif /* PTX_AST_H */
