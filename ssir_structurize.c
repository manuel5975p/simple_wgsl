/*
 * ssir_structurize.c - Authoritative CFG structurizer for SPIR-V compliance
 *
 * This pass runs AFTER the inline PTX lowering. It performs:
 *  -1. Collapse bridge chain blocks (single OpBranch, non-loop blocks)
 *   0. Fix duplicate merge targets with trampoline blocks
 *   1. Add missing selection merges for uncovered branches using ipdom analysis
 *
 * Loop merges from the lowerer are preserved (they accompany physical CFG
 * surgery that must not be undone).
 */

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#ifndef STRUCT_MALLOC
#define STRUCT_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef STRUCT_REALLOC
#define STRUCT_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef STRUCT_FREE
#define STRUCT_FREE(p) free((p))
#endif

#define SC_MAX 4096

typedef struct {
    SsirModule *mod;
    uint32_t func_id;
    SsirFunction *func;
    int n;
    uint32_t ids[SC_MAX];

    int succ[SC_MAX][140];
    int nsuc[SC_MAX];
    int pred[SC_MAX][140];
    int npred[SC_MAX];

    int idom[SC_MAX];
    int ipdom[SC_MAX];

    int rpo[SC_MAX];
    int rpo_num[SC_MAX];
    int rpo_n;

    int rrpo[SC_MAX];
    int rrpo_num[SC_MAX];
    int rrpo_n;

    bool reachable[SC_MAX];
    bool is_merge_target[SC_MAX];
} SC;

static int sc_idx(SC *c, uint32_t id) {
    for (int i = 0; i < c->n; i++)
        if (c->ids[i] == id) return i;
    return -1;
}

/* ============================================================================
 * Build CFG
 * ============================================================================ */
static void sc_add_edge(SC *c, int f, int t) {
    if (f < 0 || t < 0) return;
    bool found = false;
    for (int i = 0; i < c->nsuc[f]; i++) if (c->succ[f][i] == t) { found = true; break; }
    if (!found && c->nsuc[f] < 140) c->succ[f][c->nsuc[f]++] = t;
    found = false;
    for (int i = 0; i < c->npred[t]; i++) if (c->pred[t][i] == f) { found = true; break; }
    if (!found && c->npred[t] < 140) c->pred[t][c->npred[t]++] = f;
}

static void sc_build_cfg(SC *c) {
    memset(c->nsuc, 0, sizeof(int) * c->n);
    memset(c->npred, 0, sizeof(int) * c->n);
    for (int bi = 0; bi < c->n; bi++) {
        SsirBlock *b = &c->func->blocks[bi];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirInst *s = &b->insts[ii];
            if (s->op == SSIR_OP_BRANCH)
                sc_add_edge(c, bi, sc_idx(c, s->operands[0]));
            else if (s->op == SSIR_OP_BRANCH_COND) {
                sc_add_edge(c, bi, sc_idx(c, s->operands[1]));
                sc_add_edge(c, bi, sc_idx(c, s->operands[2]));
            } else if (s->op == SSIR_OP_SWITCH) {
                sc_add_edge(c, bi, sc_idx(c, s->operands[1]));
                for (uint32_t ei = 1; ei < s->extra_count; ei += 2)
                    sc_add_edge(c, bi, sc_idx(c, s->extra[ei]));
            }
        }
    }
}

/* ============================================================================
 * RPO
 * ============================================================================ */
static void sc_dfs(SC *c, int node, bool *vis, int *stk, int *sp) {
    vis[node] = true;
    for (int i = 0; i < c->nsuc[node]; i++) {
        int s = c->succ[node][i];
        if (s >= 0 && !vis[s]) sc_dfs(c, s, vis, stk, sp);
    }
    stk[(*sp)++] = node;
}

static void sc_compute_rpo(SC *c) {
    bool vis[SC_MAX]; int stk[SC_MAX], sp = 0;
    memset(vis, 0, sizeof(bool) * c->n);
    memset(c->reachable, 0, sizeof(bool) * c->n);
    if (c->n > 0) sc_dfs(c, 0, vis, stk, &sp);
    c->rpo_n = 0;
    for (int i = sp - 1; i >= 0; i--) {
        c->rpo[c->rpo_n] = stk[i];
        c->rpo_num[stk[i]] = c->rpo_n;
        c->reachable[stk[i]] = true;
        c->rpo_n++;
    }
    for (int i = 0; i < c->n; i++)
        if (!vis[i]) c->rpo_num[i] = c->n + i;
}

static void sc_rdfs(SC *c, int node, bool *vis, int *stk, int *sp) {
    vis[node] = true;
    for (int i = 0; i < c->npred[node]; i++) {
        int p = c->pred[node][i];
        if (p >= 0 && !vis[p]) sc_rdfs(c, p, vis, stk, sp);
    }
    stk[(*sp)++] = node;
}

static void sc_compute_rrpo(SC *c) {
    bool vis[SC_MAX]; int stk[SC_MAX], sp = 0;
    memset(vis, 0, sizeof(bool) * c->n);
    for (int i = 0; i < c->n; i++) {
        if (!c->reachable[i]) continue;
        bool is_exit = (c->nsuc[i] == 0);
        SsirBlock *b = &c->func->blocks[i];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirOpcode op = b->insts[ii].op;
            if (op == SSIR_OP_RETURN || op == SSIR_OP_RETURN_VOID || op == SSIR_OP_UNREACHABLE)
                is_exit = true;
        }
        if (is_exit && !vis[i]) sc_rdfs(c, i, vis, stk, &sp);
    }
    c->rrpo_n = 0;
    for (int i = sp - 1; i >= 0; i--) {
        c->rrpo[c->rrpo_n] = stk[i];
        c->rrpo_num[stk[i]] = c->rrpo_n;
        c->rrpo_n++;
    }
    for (int i = 0; i < c->n; i++)
        if (!vis[i]) c->rrpo_num[i] = c->n + i;
}

/* ============================================================================
 * Dominators (Cooper-Harvey-Kennedy)
 * ============================================================================ */
static int sc_intersect(int *doms, int *num, int a, int b) {
    int steps = 0;
    while (a != b && steps < SC_MAX * 2) {
        while (num[a] > num[b] && steps < SC_MAX * 2) { a = doms[a]; steps++; }
        while (num[b] > num[a] && steps < SC_MAX * 2) { b = doms[b]; steps++; }
    }
    return (a == b) ? a : -1;
}

static void sc_compute_dom(SC *c) {
    for (int i = 0; i < c->n; i++) c->idom[i] = -1;
    c->idom[0] = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int ri = 0; ri < c->rpo_n; ri++) {
            int b = c->rpo[ri];
            if (b == 0 || !c->reachable[b]) continue;
            int nd = -1;
            for (int pi = 0; pi < c->npred[b]; pi++) {
                int p = c->pred[b][pi];
                if (c->idom[p] == -1) continue;
                nd = (nd < 0) ? p : sc_intersect(c->idom, c->rpo_num, nd, p);
            }
            if (nd >= 0 && nd != c->idom[b]) { c->idom[b] = nd; changed = true; }
        }
    }
}

static void sc_compute_pdom(SC *c) {
    for (int i = 0; i < c->n; i++) c->ipdom[i] = -1;
    for (int i = 0; i < c->n; i++) {
        if (!c->reachable[i]) continue;
        bool is_exit = (c->nsuc[i] == 0);
        SsirBlock *b = &c->func->blocks[i];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirOpcode op = b->insts[ii].op;
            if (op == SSIR_OP_RETURN || op == SSIR_OP_RETURN_VOID || op == SSIR_OP_UNREACHABLE)
                is_exit = true;
        }
        if (is_exit) c->ipdom[i] = i;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int ri = 0; ri < c->rrpo_n; ri++) {
            int b = c->rrpo[ri];
            if (!c->reachable[b] || c->ipdom[b] == b) continue;
            int nd = -1;
            for (int si = 0; si < c->nsuc[b]; si++) {
                int s = c->succ[b][si];
                if (c->ipdom[s] == -1) continue;
                nd = (nd < 0) ? s : sc_intersect(c->ipdom, c->rrpo_num, nd, s);
            }
            if (nd >= 0 && nd != c->ipdom[b]) { c->ipdom[b] = nd; changed = true; }
        }
    }
}

static bool sc_dom(SC *c, int a, int b) {
    if (a < 0 || b < 0) return false;
    int x = b, steps = 0;
    while (x >= 0 && steps < c->n + 1) {
        if (x == a) return true;
        if (x == c->idom[x]) break;
        x = c->idom[x]; steps++;
    }
    return false;
}

static void sc_collect_merge_targets(SC *c) {
    memset(c->is_merge_target, 0, sizeof(bool) * c->n);
    for (int bi = 0; bi < c->n; bi++) {
        SsirBlock *b = &c->func->blocks[bi];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirInst *s = &b->insts[ii];
            int mi = -1;
            if (s->op == SSIR_OP_LOOP_MERGE) mi = sc_idx(c, s->operands[0]);
            if (s->op == SSIR_OP_SELECTION_MERGE) mi = sc_idx(c, s->operands[0]);
            if (s->op == SSIR_OP_BRANCH_COND && s->operand_count >= 4 && s->operands[3])
                mi = sc_idx(c, s->operands[3]);
            if (mi >= 0) c->is_merge_target[mi] = true;
        }
    }
}

static void sc_collect_body(SC *c, int header, int latch, bool *body) {
    memset(body, 0, sizeof(bool) * c->n);
    body[header] = true;
    if (header == latch) return;
    body[latch] = true;
    int wl[SC_MAX], wn = 0;
    wl[wn++] = latch;
    while (wn > 0) {
        int node = wl[--wn];
        for (int i = 0; i < c->npred[node]; i++) {
            int p = c->pred[node][i];
            if (p >= 0 && !body[p] && c->reachable[p]) {
                body[p] = true; wl[wn++] = p;
            }
        }
    }
}

static void sc_insert_at(SsirBlock *b, uint32_t pos, SsirInst *inst) {
    if (b->inst_count >= b->inst_capacity) {
        uint32_t nc = b->inst_capacity ? b->inst_capacity * 2 : 8;
        b->insts = (SsirInst *)STRUCT_REALLOC(b->insts, nc * sizeof(SsirInst));
        b->inst_capacity = nc;
    }
    if (pos < b->inst_count)
        memmove(&b->insts[pos + 1], &b->insts[pos], (b->inst_count - pos) * sizeof(SsirInst));
    b->insts[pos] = *inst;
    b->inst_count++;
}

typedef struct { int header; bool body[SC_MAX]; int merge; int cont; } LoopInfo;

static bool sc_is_trivial_branch_bridge(const SsirBlock *b) {
    if (!b || b->inst_count == 0) return false;
    SsirInst *term = &b->insts[b->inst_count - 1];
    if (term->op != SSIR_OP_BRANCH || term->operand_count < 1) return false;
    for (uint32_t ii = 0; ii + 1 < b->inst_count; ii++) {
        SsirOpcode op = b->insts[ii].op;
        if (op != SSIR_OP_SELECTION_MERGE && op != SSIR_OP_LOOP_MERGE)
            return false;
    }
    return true;
}

static bool sc_is_plain_branch_bridge(const SsirBlock *b) {
    if (!b || b->inst_count != 1) return false;
    const SsirInst *term = &b->insts[0];
    return term->op == SSIR_OP_BRANCH && term->operand_count >= 1;
}

static uint32_t sc_follow_plain_branch(SsirFunction *func, uint32_t block_id) {
    if (!func || !block_id) return block_id;
    uint32_t cur = block_id;
    for (int steps = 0; steps < 32; steps++) {
        SsirBlock *b = NULL;
        for (uint32_t bi = 0; bi < func->block_count; bi++) {
            if (func->blocks[bi].id == cur) {
                b = &func->blocks[bi];
                break;
            }
        }
        if (!sc_is_plain_branch_bridge(b))
            break;
        uint32_t next = b->insts[0].operands[0];
        if (!next || next == cur)
            break;
        cur = next;
    }
    return cur;
}

static uint32_t sc_follow_trivial_branch(SsirFunction *func, uint32_t block_id) {
    if (!func || !block_id) return block_id;
    uint32_t cur = block_id;
    for (int steps = 0; steps < 32; steps++) {
        SsirBlock *b = NULL;
        for (uint32_t bi = 0; bi < func->block_count; bi++) {
            if (func->blocks[bi].id == cur) {
                b = &func->blocks[bi];
                break;
            }
        }
        if (!b || b->inst_count == 0) break;
        SsirInst *term = &b->insts[b->inst_count - 1];
        if (!sc_is_trivial_branch_bridge(b)) break;
        if (term->operands[0] == cur) break;
        cur = term->operands[0];
    }
    return cur;
}

static bool sc_is_trivial_loop_header_bridge(const SsirBlock *b,
                                             uint32_t *merge_id,
                                             uint32_t *cont_id,
                                             uint32_t *body_id) {
    if (!b || b->inst_count < 2) return false;
    uint32_t lm_merge = 0, lm_cont = 0, br_tgt = 0;
    bool saw_loop_merge = false;
    for (uint32_t ii = 0; ii < b->inst_count; ii++) {
        const SsirInst *inst = &b->insts[ii];
        bool is_last = (ii + 1 == b->inst_count);
        if (is_last) {
            if (inst->op != SSIR_OP_BRANCH || inst->operand_count < 1)
                return false;
            br_tgt = inst->operands[0];
            continue;
        }
        if (inst->op != SSIR_OP_LOOP_MERGE || inst->operand_count < 2 || saw_loop_merge)
            return false;
        lm_merge = inst->operands[0];
        lm_cont = inst->operands[1];
        saw_loop_merge = true;
    }
    if (!saw_loop_merge || !lm_merge || !lm_cont || !br_tgt)
        return false;
    if (merge_id) *merge_id = lm_merge;
    if (cont_id) *cont_id = lm_cont;
    if (body_id) *body_id = br_tgt;
    return true;
}

static void sc_rewrite_nonmerge_targets(SsirFunction *func,
                                        uint32_t from_id,
                                        uint32_t to_id) {
    if (!func || !from_id || !to_id || from_id == to_id) return;
    for (uint32_t bi = 0; bi < func->block_count; bi++) {
        SsirBlock *b = &func->blocks[bi];
        if (b->inst_count == 0) continue;
        SsirInst *term = &b->insts[b->inst_count - 1];
        if (term->op == SSIR_OP_BRANCH &&
            term->operand_count >= 1 &&
            term->operands[0] == from_id) {
            term->operands[0] = to_id;
        } else if (term->op == SSIR_OP_BRANCH_COND) {
            if (term->operand_count >= 2 && term->operands[1] == from_id)
                term->operands[1] = to_id;
            if (term->operand_count >= 3 && term->operands[2] == from_id)
                term->operands[2] = to_id;
        } else if (term->op == SSIR_OP_SWITCH) {
            if (term->operand_count >= 2 && term->operands[1] == from_id)
                term->operands[1] = to_id;
            for (uint32_t ei = 1; ei < term->extra_count; ei += 2)
                if (term->extra[ei] == from_id)
                    term->extra[ei] = to_id;
        }
    }
}

static void sc_canonicalize_duplicate_loop_headers(SsirFunction *func,
                                                   bool debug_phase3) {
    if (!func) return;

    typedef struct {
        uint32_t merge_id;
        uint32_t cont_id;
        uint32_t body_id;
        uint32_t canonical_id;
        int canonical_idx;
    } LoopHeaderGroup;

    LoopHeaderGroup groups[SC_MAX];
    int ngroups = 0;

    for (uint32_t bi = 0; bi < func->block_count; bi++) {
        uint32_t merge_id = 0, cont_id = 0, body_id = 0;
        SsirBlock *b = &func->blocks[bi];
        if (!sc_is_trivial_loop_header_bridge(b, &merge_id, &cont_id, &body_id))
            continue;

        int gi = -1;
        for (int i = 0; i < ngroups; i++) {
            if (groups[i].merge_id == merge_id &&
                groups[i].cont_id == cont_id &&
                groups[i].body_id == body_id) {
                gi = i;
                break;
            }
        }
        if (gi < 0) {
            if (ngroups >= SC_MAX) continue;
            groups[ngroups].merge_id = merge_id;
            groups[ngroups].cont_id = cont_id;
            groups[ngroups].body_id = body_id;
            groups[ngroups].canonical_id = b->id;
            groups[ngroups].canonical_idx = (int)bi;
            ngroups++;
            continue;
        }

        SsirBlock *canon = &func->blocks[groups[gi].canonical_idx];
        bool prefer_cur = false;
        if (!canon->name && b->name)
            prefer_cur = true;
        else if ((!canon->name && !b->name) || (canon->name && b->name))
            prefer_cur = ((int)bi < groups[gi].canonical_idx);

        if (prefer_cur) {
            groups[gi].canonical_id = b->id;
            groups[gi].canonical_idx = (int)bi;
        }
    }

    for (uint32_t bi = 0; bi < func->block_count; bi++) {
        uint32_t merge_id = 0, cont_id = 0, body_id = 0;
        SsirBlock *b = &func->blocks[bi];
        if (!sc_is_trivial_loop_header_bridge(b, &merge_id, &cont_id, &body_id))
            continue;

        int gi = -1;
        for (int i = 0; i < ngroups; i++) {
            if (groups[i].merge_id == merge_id &&
                groups[i].cont_id == cont_id &&
                groups[i].body_id == body_id) {
                gi = i;
                break;
            }
        }
        if (gi < 0) continue;
        if ((int)bi == groups[gi].canonical_idx) continue;

        uint32_t canonical_id = groups[gi].canonical_id;
        sc_rewrite_nonmerge_targets(func, b->id, canonical_id);

        if (debug_phase3) {
            const char *dup_name = b->name ? b->name : "<anon>";
            const char *canon_name = func->blocks[groups[gi].canonical_idx].name
                ? func->blocks[groups[gi].canonical_idx].name : "<anon>";
            fprintf(stderr,
                    "[ssir_struct] dedup loop-header dup=%s(%u) canon=%s(%u) merge=%u cont=%u body=%u\n",
                    dup_name, b->id, canon_name, canonical_id,
                    merge_id, cont_id, body_id);
        }

        if (b->insts[b->inst_count - 1].extra) {
            STRUCT_FREE(b->insts[b->inst_count - 1].extra);
            b->insts[b->inst_count - 1].extra = NULL;
            b->insts[b->inst_count - 1].extra_count = 0;
        }
        memset(&b->insts[0], 0, sizeof(SsirInst) * b->inst_count);
        b->insts[0].op = SSIR_OP_BRANCH;
        b->insts[0].operands[0] = canonical_id;
        b->insts[0].operand_count = 1;
        b->inst_count = 1;
    }
}

static void sc_split_duplicate_merge_targets(SsirModule *mod, SsirFunction *func) {
    if (!mod || !func) return;
    int n = (int)func->block_count;
    uint32_t dup_ids[SC_MAX];
    int dup_counts[SC_MAX];
    int ndups = 0;
    for (int bi = 0; bi < n; bi++) {
        SsirBlock *b = &func->blocks[bi];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirInst *s = &b->insts[ii];
            uint32_t mid = 0;
            if (s->op == SSIR_OP_SELECTION_MERGE && s->operand_count >= 1)
                mid = s->operands[0];
            else if (s->op == SSIR_OP_BRANCH_COND && s->operand_count >= 4 && s->operands[3])
                mid = s->operands[3];
            else if (s->op == SSIR_OP_LOOP_MERGE && s->operand_count >= 1)
                mid = s->operands[0];
            else continue;
            if (!mid) continue;
            int found = -1;
            for (int j = 0; j < ndups; j++)
                if (dup_ids[j] == mid) { found = j; break; }
            if (found >= 0) dup_counts[found]++;
            else if (ndups < SC_MAX) { dup_ids[ndups] = mid; dup_counts[ndups] = 1; ndups++; }
        }
    }
    for (int mi = 0; mi < ndups; mi++) {
        if (dup_counts[mi] <= 1) continue;
        uint32_t did = dup_ids[mi];
        bool loop_claims = false;
        for (int bi2 = 0; bi2 < n && !loop_claims; bi2++) {
            SsirBlock *b2 = &func->blocks[bi2];
            for (uint32_t ii2 = 0; ii2 < b2->inst_count; ii2++)
                if (b2->insts[ii2].op == SSIR_OP_LOOP_MERGE &&
                    b2->insts[ii2].operand_count >= 1 &&
                    b2->insts[ii2].operands[0] == did) { loop_claims = true; break; }
        }
        bool first = loop_claims;
        for (int bi2 = 0; bi2 < (int)func->block_count; bi2++) {
            SsirBlock *b = &func->blocks[bi2];
            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                SsirInst *s = &b->insts[ii];
                uint32_t *mp = NULL;
                if (s->op == SSIR_OP_SELECTION_MERGE && s->operand_count >= 1 && s->operands[0] == did)
                    mp = &s->operands[0];
                else if (s->op == SSIR_OP_BRANCH_COND && s->operand_count >= 4 && s->operands[3] == did)
                    mp = &s->operands[3];
                if (!mp) continue;
                if (!first) { first = true; continue; }
                SsirBlock tramp;
                memset(&tramp, 0, sizeof(tramp));
                tramp.id = mod->next_id++;
                tramp.inst_capacity = 2;
                tramp.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
                if (!tramp.insts) continue;
                memset(tramp.insts, 0, 2 * sizeof(SsirInst));
                tramp.insts[0].op = SSIR_OP_BRANCH;
                tramp.insts[0].operands[0] = did;
                tramp.insts[0].operand_count = 1;
                tramp.inst_count = 1;
                if (func->block_count >= func->block_capacity) {
                    uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
                    SsirBlock *np = (SsirBlock *)STRUCT_REALLOC(func->blocks, nc * sizeof(SsirBlock));
                    if (!np) continue;
                    func->blocks = np;
                    func->block_capacity = nc;
                    b = &func->blocks[bi2]; s = &b->insts[ii];
                    mp = (s->op == SSIR_OP_SELECTION_MERGE) ? &s->operands[0] : &s->operands[3];
                }
                func->blocks[func->block_count++] = tramp;
                *mp = tramp.id;
            }
        }
    }
}

static int sc_loop_of(LoopInfo *loops, int nloops, SC *c, int bi) {
    int best = -1, best_size = SC_MAX + 1;
    for (int li = 0; li < nloops; li++) {
        if (!loops[li].body[bi]) continue;
        int sz = 0; for (int k = 0; k < c->n; k++) if (loops[li].body[k]) sz++;
        if (sz < best_size) { best = li; best_size = sz; }
    }
    return best;
}

static void sc_ensure_term(SC *c) {
    for (int bi = 0; bi < c->n; bi++) {
        SsirBlock *b = &c->func->blocks[bi];
        if (b->inst_count == 0) goto add_unreach;
        { SsirInst *last = &b->insts[b->inst_count - 1];
          if (last->op == SSIR_OP_BRANCH || last->op == SSIR_OP_BRANCH_COND ||
              last->op == SSIR_OP_SWITCH || last->op == SSIR_OP_RETURN ||
              last->op == SSIR_OP_RETURN_VOID || last->op == SSIR_OP_UNREACHABLE)
              continue;
        }
        add_unreach:
        if (b->inst_count >= b->inst_capacity) {
            uint32_t nc = b->inst_capacity ? b->inst_capacity * 2 : 4;
            SsirInst *np = (SsirInst *)STRUCT_REALLOC(b->insts, nc * sizeof(SsirInst));
            if (!np) continue;
            b->insts = np;
            b->inst_capacity = nc;
        }
        memset(&b->insts[b->inst_count], 0, sizeof(SsirInst));
        b->insts[b->inst_count].op = SSIR_OP_UNREACHABLE;
        b->inst_count++;
    }
}


/* ============================================================================
 * Public API
 * ============================================================================ */
void ssir_structurize_function(SsirModule *mod, uint32_t func_id) {
    if (!mod) return;
    SsirFunction *func = ssir_get_function(mod, func_id);
    if (!func || func->block_count <= 1 || func->block_count > SC_MAX) return;

    const char *debug_env = getenv("SSIR_STRUCT_DEBUG");
    bool debug_phase3 = debug_env && debug_env[0] != '\0' && debug_env[0] != '0';

    /* Phase -1 intentionally left empty -- bridge collapse is not safe for
     * all patterns. Instead, the structurizer works purely additively. */

    /* Phase 0: Fix duplicate merge targets with trampoline blocks */
    sc_split_duplicate_merge_targets(mod, func);

    /* Normalize merge/continue annotations through pure bridge blocks so the
     * later CFG passes operate on the actual target blocks. */
    for (uint32_t bi = 0; bi < func->block_count; bi++) {
        SsirBlock *b = &func->blocks[bi];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirInst *s = &b->insts[ii];
            if (s->op == SSIR_OP_SELECTION_MERGE && s->operand_count >= 1) {
                s->operands[0] = sc_follow_trivial_branch(func, s->operands[0]);
            } else if (s->op == SSIR_OP_BRANCH_COND &&
                       s->operand_count >= 4 && s->operands[3] != 0) {
                s->operands[3] = sc_follow_trivial_branch(func, s->operands[3]);
            } else if (s->op == SSIR_OP_LOOP_MERGE && s->operand_count >= 2) {
                s->operands[0] = sc_follow_trivial_branch(func, s->operands[0]);
                s->operands[1] = sc_follow_trivial_branch(func, s->operands[1]);
            }
        }
    }
    sc_split_duplicate_merge_targets(mod, func);

    /* The CFG pass is also responsible for fixing explicit merge conflicts
     * (for example, a selection merge reused as a loop continue). That means
     * we must run it for any function with real multi-way control flow, not
     * only when the lowerer omitted a merge annotation. */
    bool has_multi_succ = false;
    for (uint32_t bi = 0; bi < func->block_count; bi++) {
        SsirBlock *b = &func->blocks[bi];
        if (b->inst_count == 0) continue;
        SsirInst *last = &b->insts[b->inst_count - 1];
        if (last->op == SSIR_OP_BRANCH_COND) {
            has_multi_succ = true;
        }
        if (last->op == SSIR_OP_SWITCH) {
            has_multi_succ = true;
        }
    }
    if (!has_multi_succ) return;

    SC *c = (SC *)STRUCT_MALLOC(sizeof(SC));
    if (!c) return;
    memset(c, 0, sizeof(*c));
    c->mod = mod; c->func_id = func_id; c->func = func;
    c->n = (int)func->block_count;
    for (int i = 0; i < c->n; i++) c->ids[i] = func->blocks[i].id;

    sc_ensure_term(c);
    sc_build_cfg(c);
    sc_compute_rpo(c);
    sc_compute_rrpo(c);
    sc_compute_dom(c);
    sc_compute_pdom(c);
    sc_collect_merge_targets(c);

    /* === Phase 1: Build loop info === */
    typedef struct { int header; int latch; } BackEdge;
    BackEdge back_edges[1024];
    int nbe = 0;
    for (int bi = 0; bi < c->n; bi++) {
        if (!c->reachable[bi]) continue;
        for (int si = 0; si < c->nsuc[bi]; si++) {
            int t = c->succ[bi][si];
            if (t >= 0 && sc_dom(c, t, bi) && nbe < 1024) {
                back_edges[nbe].header = t;
                back_edges[nbe].latch = bi;
                nbe++;
            }
        }
    }

    LoopInfo *loops = NULL;
    int nloops = 0;
    for (int i = 0; i < nbe; i++) {
        int h = back_edges[i].header;
        int existing = -1;
        for (int j = 0; j < nloops; j++)
            if (loops[j].header == h) { existing = j; break; }
        if (existing >= 0) {
            bool body[SC_MAX];
            sc_collect_body(c, h, back_edges[i].latch, body);
            for (int k = 0; k < c->n; k++)
                if (body[k]) loops[existing].body[k] = true;
        } else {
            LoopInfo *np = (LoopInfo *)STRUCT_REALLOC(loops, (nloops + 1) * sizeof(LoopInfo));
            if (!np) continue;
            loops = np;
            memset(&loops[nloops], 0, sizeof(LoopInfo));
            loops[nloops].header = h;
            sc_collect_body(c, h, back_edges[i].latch, loops[nloops].body);
            nloops++;
        }
    }

    for (int li = 0; li < nloops; li++) {
        int h = loops[li].header;
        bool *body = loops[li].body;
        int merge = c->ipdom[h];
        int steps = 0;
        while (merge >= 0 && body[merge] && merge != c->ipdom[merge] && steps < c->n) {
            merge = c->ipdom[merge]; steps++;
        }
        if (merge < 0 || body[merge]) {
            int best = -1;
            for (int j = 0; j < c->n; j++) {
                if (!body[j]) continue;
                for (int si = 0; si < c->nsuc[j]; si++) {
                    int s = c->succ[j][si];
                    if (s >= 0 && !body[s] && c->reachable[s])
                        if (best < 0 || c->rpo_num[s] < c->rpo_num[best]) best = s;
                }
            }
            merge = best;
        }
        loops[li].merge = merge;

        int cont = -1;
        for (int j = 0; j < c->n; j++) {
            if (!body[j]) continue;
            for (int si = 0; si < c->nsuc[j]; si++) {
                if (c->succ[j][si] == h)
                    if (cont < 0 || c->rpo_num[j] > c->rpo_num[cont]) cont = j;
            }
        }
        loops[li].cont = cont;
    }

    /* Loop merge annotation */
    for (int li = 0; li < nloops; li++) {
        int h = loops[li].header;
        SsirBlock *hb = &c->func->blocks[h];
        bool has_lm = false;
        bool has_sm = false;
        for (uint32_t ii = 0; ii < hb->inst_count; ii++) {
            if (hb->insts[ii].op == SSIR_OP_LOOP_MERGE) { has_lm = true; break; }
            if (hb->insts[ii].op == SSIR_OP_SELECTION_MERGE) has_sm = true;
        }
        if (has_lm) continue;

        int merge = loops[li].merge;
        int cont = loops[li].cont;
        if (merge < 0 || cont < 0) continue;
        if (hb->inst_count == 0) continue;

        if (has_sm) {
            /* Header already has SelectionMerge — can't add LoopMerge in
             * the same block.  Create a trivial bridge block as the new
             * loop header and redirect back-edges to it. */
            SsirBlock bridge;
            memset(&bridge, 0, sizeof(bridge));
            bridge.id = c->mod->next_id++;
            bridge.inst_capacity = 4;
            bridge.insts = (SsirInst *)STRUCT_MALLOC(4 * sizeof(SsirInst));
            if (!bridge.insts) continue;
            memset(bridge.insts, 0, 4 * sizeof(SsirInst));
            bridge.insts[0].op = SSIR_OP_LOOP_MERGE;
            bridge.insts[0].operands[0] = c->ids[merge];
            bridge.insts[0].operands[1] = c->ids[cont];
            bridge.insts[0].operand_count = 2;
            bridge.insts[1].op = SSIR_OP_BRANCH;
            bridge.insts[1].operands[0] = hb->id;
            bridge.insts[1].operand_count = 1;
            bridge.inst_count = 2;
            if (func->block_count >= func->block_capacity) {
                uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
                func->blocks = (SsirBlock *)STRUCT_REALLOC(func->blocks, nc * sizeof(SsirBlock));
                func->block_capacity = nc;
                hb = &func->blocks[h];
            }
            func->blocks[func->block_count++] = bridge;
            /* Redirect latch back-edges to the new bridge header */
            for (int bi = 0; bi < (int)func->block_count; bi++) {
                SsirBlock *b = &func->blocks[bi];
                if (b->inst_count == 0) continue;
                if (!loops[li].body[bi < c->n ? bi : -1]) {
                    /* Block outside loop body or new block — skip unless
                     * it's a forward entry that also needs redirection. */
                }
                SsirInst *term = &b->insts[b->inst_count - 1];
                if (term->op == SSIR_OP_BRANCH &&
                    term->operand_count >= 1 &&
                    term->operands[0] == hb->id) {
                    /* Only redirect if this is a back-edge (block is in loop body) */
                    if (bi < c->n && loops[li].body[bi])
                        term->operands[0] = bridge.id;
                } else if (term->op == SSIR_OP_BRANCH_COND) {
                    if (bi < c->n && loops[li].body[bi]) {
                        if (term->operand_count >= 2 && term->operands[1] == hb->id)
                            term->operands[1] = bridge.id;
                        if (term->operand_count >= 3 && term->operands[2] == hb->id)
                            term->operands[2] = bridge.id;
                    }
                }
            }
        } else {
            /* No conflicting merge — insert OpLoopMerge before terminator */
            SsirInst lm;
            memset(&lm, 0, sizeof(lm));
            lm.op = SSIR_OP_LOOP_MERGE;
            lm.operands[0] = c->ids[merge];
            lm.operands[1] = c->ids[cont];
            lm.operand_count = 2;
            sc_insert_at(hb, hb->inst_count - 1, &lm);
        }
        c->is_merge_target[merge] = true;
    }

    /* === Phase 2: Add selection merges for uncovered branches === */
    for (int ri = 0; ri < c->rpo_n; ri++) {
        int bi = c->rpo[ri];
        if (!c->reachable[bi]) continue;
        SsirBlock *b = &c->func->blocks[bi];
        if (b->inst_count == 0) continue;

        /* Skip blocks that already have merge annotations */
        { bool has_merge = false;
          for (uint32_t ii = 0; ii < b->inst_count; ii++) {
              if (b->insts[ii].op == SSIR_OP_LOOP_MERGE ||
                  b->insts[ii].op == SSIR_OP_SELECTION_MERGE) { has_merge = true; break; }
              if (b->insts[ii].op == SSIR_OP_BRANCH_COND &&
                  b->insts[ii].operand_count >= 4 && b->insts[ii].operands[3] != 0)
                  { has_merge = true; break; }
          }
          if (has_merge) continue;
        }

        SsirInst *term = &b->insts[b->inst_count - 1];
        if (term->op != SSIR_OP_BRANCH_COND && term->op != SSIR_OP_SWITCH) continue;

        /* Find merge: immediate post-dominator */
        int merge = c->ipdom[bi];
        if (merge == bi || merge < 0) continue;
        if (c->rpo_num[merge] <= c->rpo_num[bi]) continue;

        /* Loop containment: if inside a loop (not header), merge must stay in body */
        int li = sc_loop_of(loops, nloops, c, bi);
        if (li >= 0 && loops[li].header != bi) {
            if (!loops[li].body[merge]) {
                int lm = loops[li].merge;
                if (lm >= 0 && c->rpo_num[lm] > c->rpo_num[bi])
                    merge = lm;
                else
                    continue;
            }
        }

        /* SPIR-V rule: merge must NOT be a branch target of the header */
        if (term->op == SSIR_OP_BRANCH_COND) {
            uint32_t mid_check = c->ids[merge];
            if (mid_check == term->operands[1] || mid_check == term->operands[2]) {
                /* Walk ipdom chain for a non-target merge */
                int alt = c->ipdom[merge];
                int steps = 0;
                bool found_alt = false;
                while (alt >= 0 && alt != merge && steps < c->n) {
                    uint32_t alt_id = c->ids[alt];
                    if (alt_id != term->operands[1] && alt_id != term->operands[2] &&
                        !c->is_merge_target[alt] && c->rpo_num[alt] > c->rpo_num[bi]) {
                        merge = alt; found_alt = true; break;
                    }
                    int next = c->ipdom[alt];
                    if (next == alt) break;
                    alt = next; steps++;
                }
                if (!found_alt) {
                    /* Create trampoline as merge */
                    SsirBlock tramp;
                    memset(&tramp, 0, sizeof(tramp));
                    tramp.id = c->mod->next_id++;
                    tramp.inst_capacity = 2;
                    tramp.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
                    if (!tramp.insts) continue;
                    memset(tramp.insts, 0, 2 * sizeof(SsirInst));
                    tramp.insts[0].op = SSIR_OP_BRANCH;
                    tramp.insts[0].operands[0] = mid_check;
                    tramp.insts[0].operand_count = 1;
                    tramp.inst_count = 1;
                    if (c->func->block_count >= c->func->block_capacity) {
                        uint32_t nc = c->func->block_capacity ? c->func->block_capacity * 2 : 16;
                        c->func->blocks = (SsirBlock *)STRUCT_REALLOC(c->func->blocks,
                                                                       nc * sizeof(SsirBlock));
                        c->func->block_capacity = nc;
                        b = &c->func->blocks[bi];
                        term = &b->insts[b->inst_count - 1];
                    }
                    c->func->blocks[c->func->block_count] = tramp;
                    c->func->block_count++;
                    if (c->n < SC_MAX) {
                        c->ids[c->n] = tramp.id;
                        c->reachable[c->n] = true;
                        c->is_merge_target[c->n] = true;
                        c->rpo_num[c->n] = c->rpo_num[bi] + 1;
                        merge = c->n;
                        c->n++;
                    } else {
                        continue;
                    }
                }
            }
        }

        /* Duplicate merge target resolution */
        if (c->is_merge_target[merge]) {
            int alt = c->ipdom[merge];
            int steps = 0;
            bool found_alt = false;
            while (alt >= 0 && alt != merge && steps < c->n) {
                if (!c->is_merge_target[alt] && c->rpo_num[alt] > c->rpo_num[bi]) {
                    bool ok = true;
                    if (li >= 0 && loops[li].header != bi && !loops[li].body[alt])
                        ok = false;
                    if (ok) { merge = alt; found_alt = true; break; }
                }
                int next = c->ipdom[alt];
                if (next == alt) break;
                alt = next; steps++;
            }
            if (!found_alt) {
                SsirBlock tramp;
                memset(&tramp, 0, sizeof(tramp));
                tramp.id = c->mod->next_id++;
                tramp.inst_capacity = 2;
                tramp.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
                if (!tramp.insts) continue;
                memset(tramp.insts, 0, 2 * sizeof(SsirInst));
                tramp.insts[0].op = SSIR_OP_BRANCH;
                tramp.insts[0].operands[0] = c->ids[merge];
                tramp.insts[0].operand_count = 1;
                tramp.inst_count = 1;
                if (c->func->block_count >= c->func->block_capacity) {
                    uint32_t nc = c->func->block_capacity ? c->func->block_capacity * 2 : 16;
                    c->func->blocks = (SsirBlock *)STRUCT_REALLOC(c->func->blocks,
                                                                   nc * sizeof(SsirBlock));
                    c->func->block_capacity = nc;
                    b = &c->func->blocks[bi];
                    term = &b->insts[b->inst_count - 1];
                }
                c->func->blocks[c->func->block_count] = tramp;
                c->func->block_count++;
                if (c->n < SC_MAX) {
                    c->ids[c->n] = tramp.id;
                    c->reachable[c->n] = true;
                    c->is_merge_target[c->n] = true;
                    c->rpo_num[c->n] = c->rpo_num[bi] + 1;
                    merge = c->n;
                    c->n++;
                } else {
                    continue;
                }
            }
        }

        c->is_merge_target[merge] = true;
        uint32_t mid = c->ids[merge];

        if (term->op == SSIR_OP_BRANCH_COND) {
            term->operands[3] = mid;
            if (term->operand_count < 4) term->operand_count = 4;
        } else if (term->op == SSIR_OP_SWITCH) {
            SsirInst sm;
            memset(&sm, 0, sizeof(sm));
            sm.op = SSIR_OP_SELECTION_MERGE;
            sm.operands[0] = mid;
            sm.operand_count = 1;
            sc_insert_at(b, b->inst_count - 1, &sm);
        }
    }

    /* === Phase 2.5: Split continue targets reused as selection merges === */
    {
        uint32_t split_conts[SC_MAX];
        int nsplit = 0;
        for (uint32_t bi = 0; bi < (uint32_t)c->n; bi++) {
            SsirBlock *b = &func->blocks[bi];
            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                SsirInst *s = &b->insts[ii];
                if (s->op != SSIR_OP_LOOP_MERGE || s->operand_count < 2)
                    continue;

                uint32_t cont_id = s->operands[1];
                int ci = sc_idx(c, cont_id);
                if (ci < 0)
                    continue;

                bool already_split = false;
                for (int si = 0; si < nsplit; si++) {
                    if (split_conts[si] == cont_id) {
                        already_split = true;
                        break;
                    }
                }
                if (already_split)
                    continue;

                bool used_as_selection_merge = false;
                for (uint32_t hj = 0; hj < (uint32_t)c->n && !used_as_selection_merge; hj++) {
                    SsirBlock *hb = &func->blocks[hj];
                    for (uint32_t ji = 0; ji < hb->inst_count; ji++) {
                        SsirInst *hi = &hb->insts[ji];
                        if (hi->op == SSIR_OP_SELECTION_MERGE &&
                            hi->operand_count >= 1 &&
                            hi->operands[0] == cont_id) {
                            used_as_selection_merge = true;
                            break;
                        }
                        if (hi->op == SSIR_OP_BRANCH_COND &&
                            hi->operand_count >= 4 &&
                            hi->operands[3] == cont_id) {
                            used_as_selection_merge = true;
                            break;
                        }
                    }
                }
                if (!used_as_selection_merge)
                    continue;

                bool cont_is_header = false;
                {
                    SsirBlock *cb = &func->blocks[ci];
                    for (uint32_t ti = 0; ti < cb->inst_count; ti++) {
                        SsirOpcode op = cb->insts[ti].op;
                        if (op == SSIR_OP_LOOP_MERGE || op == SSIR_OP_SELECTION_MERGE) {
                            cont_is_header = true;
                            break;
                        }
                        if (op == SSIR_OP_BRANCH_COND &&
                            cb->insts[ti].operand_count >= 4 &&
                            cb->insts[ti].operands[3] != 0) {
                            cont_is_header = true;
                            break;
                        }
                    }
                }
                if (cont_is_header)
                    continue;

                SsirBlock tramp;
                memset(&tramp, 0, sizeof(tramp));
                tramp.id = mod->next_id++;
                tramp.inst_capacity = 2;
                tramp.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
                if (!tramp.insts)
                    continue;
                memset(tramp.insts, 0, 2 * sizeof(SsirInst));
                SsirBlock *cont_blk = &func->blocks[ci];
                if (cont_blk->inst_count == 0)
                    continue;
                SsirInst moved_term = cont_blk->insts[cont_blk->inst_count - 1];
                if (moved_term.extra && moved_term.extra_count > 0) {
                    uint32_t bytes = moved_term.extra_count * sizeof(uint32_t);
                    uint32_t *copy = (uint32_t *)STRUCT_MALLOC(bytes);
                    if (copy) {
                        memcpy(copy, moved_term.extra, bytes);
                        moved_term.extra = copy;
                    } else {
                        moved_term.extra = NULL;
                        moved_term.extra_count = 0;
                    }
                }
                tramp.insts[0] = moved_term;
                tramp.inst_count = 1;
                if (func->block_count >= func->block_capacity) {
                    uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
                    func->blocks = (SsirBlock *)STRUCT_REALLOC(func->blocks,
                                                               nc * sizeof(SsirBlock));
                    func->block_capacity = nc;
                    b = &func->blocks[bi];
                    s = &b->insts[ii];
                    cont_blk = &func->blocks[ci];
                }
                func->blocks[func->block_count++] = tramp;

                /* The old continue block becomes ordinary code that flows into
                 * the new continue header, which owns the original loop-back
                 * terminator. */
                if (cont_blk->insts[cont_blk->inst_count - 1].extra) {
                    STRUCT_FREE(cont_blk->insts[cont_blk->inst_count - 1].extra);
                    cont_blk->insts[cont_blk->inst_count - 1].extra = NULL;
                    cont_blk->insts[cont_blk->inst_count - 1].extra_count = 0;
                }
                memset(&cont_blk->insts[cont_blk->inst_count - 1], 0, sizeof(SsirInst));
                cont_blk->insts[cont_blk->inst_count - 1].op = SSIR_OP_BRANCH;
                cont_blk->insts[cont_blk->inst_count - 1].operands[0] = tramp.id;
                cont_blk->insts[cont_blk->inst_count - 1].operand_count = 1;

                int rewritten_loop_merges = 0;
                for (uint32_t rk = 0; rk < (uint32_t)c->n; rk++) {
                    SsirBlock *rb = &func->blocks[rk];
                    for (uint32_t ri = 0; ri < rb->inst_count; ri++) {
                        SsirInst *inst = &rb->insts[ri];
                        if (inst->op == SSIR_OP_LOOP_MERGE &&
                            inst->operand_count >= 2 &&
                            inst->operands[1] == cont_id) {
                            inst->operands[1] = tramp.id;
                            rewritten_loop_merges++;
                        }
                    }
                }

                if (debug_phase3) {
                    fprintf(stderr,
                            "[ssir_struct] continue split func=%s cont=%u tramp=%u loop_rewrites=%d branch_rewrites=%d\n",
                            func->name ? func->name : "<anon-func>",
                            cont_id, tramp.id, rewritten_loop_merges, 0);
                }

                if (nsplit < SC_MAX)
                    split_conts[nsplit++] = cont_id;
            }
        }
    }
    sc_split_duplicate_merge_targets(mod, func);

    /* === Phase 3: Split problematic selection merges through bridge blocks === */
    uint32_t cfg_block_count = (uint32_t)c->n;
    for (uint32_t bi = 0; bi < cfg_block_count; bi++) {
        SsirBlock *b = &func->blocks[bi];
        if (b->inst_count == 0) continue;

        SsirInst *term = &b->insts[b->inst_count - 1];
        if (term->op != SSIR_OP_BRANCH_COND && term->op != SSIR_OP_SWITCH) continue;

        uint32_t merge_id = 0;
        SsirInst *merge_inst = NULL;
        if (term->op == SSIR_OP_BRANCH_COND &&
            term->operand_count >= 4 && term->operands[3] != 0) {
            merge_id = term->operands[3];
        } else {
            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                if (b->insts[ii].op == SSIR_OP_SELECTION_MERGE &&
                    b->insts[ii].operand_count >= 1) {
                    merge_inst = &b->insts[ii];
                    merge_id = merge_inst->operands[0];
                    break;
                }
            }
        }
        if (!merge_id) continue;

        int mi = sc_idx(c, merge_id);
        if (mi < 0) continue;

        bool merge_is_continue = false;
        for (uint32_t hi = 0; hi < cfg_block_count && !merge_is_continue; hi++) {
            SsirBlock *hb = &func->blocks[hi];
            for (uint32_t ii = 0; ii < hb->inst_count; ii++) {
                if (hb->insts[ii].op == SSIR_OP_LOOP_MERGE &&
                    hb->insts[ii].operand_count >= 2 &&
                    hb->insts[ii].operands[1] == merge_id) {
                    merge_is_continue = true;
                    break;
                }
            }
        }

        bool merge_is_header = false;
        bool merge_is_bridge = sc_is_trivial_branch_bridge(&func->blocks[mi]);
        bool merge_is_branch_target = false;
        uint32_t merge_exit_id = 0;
        bool split_after_merge_tail = false;
        {
            SsirBlock *mb = &func->blocks[mi];
            for (uint32_t ii = 0; ii < mb->inst_count; ii++) {
                SsirOpcode op = mb->insts[ii].op;
                if (op == SSIR_OP_LOOP_MERGE || op == SSIR_OP_SELECTION_MERGE) {
                    merge_is_header = true;
                    break;
                }
                if (op == SSIR_OP_BRANCH_COND &&
                    mb->insts[ii].operand_count >= 4 && mb->insts[ii].operands[3] != 0) {
                    merge_is_header = true;
                    break;
                }
            }
            if (term->op == SSIR_OP_BRANCH_COND) {
                if ((term->operand_count >= 2 && term->operands[1] == merge_id) ||
                    (term->operand_count >= 3 && term->operands[2] == merge_id))
                    merge_is_branch_target = true;
            } else if (term->op == SSIR_OP_SWITCH) {
                if (term->operand_count >= 2 && term->operands[1] == merge_id)
                    merge_is_branch_target = true;
                for (uint32_t ei = 1; ei < term->extra_count && !merge_is_branch_target; ei += 2)
                    if (term->extra[ei] == merge_id)
                        merge_is_branch_target = true;
            }
            if (!merge_is_header && !merge_is_continue &&
                !merge_is_bridge &&
                mb->inst_count > 0) {
                SsirInst *mb_term = &mb->insts[mb->inst_count - 1];
                if (mb_term->op == SSIR_OP_BRANCH &&
                    mb_term->operand_count >= 1 &&
                    mb_term->operands[0] != merge_id) {
                    merge_exit_id = mb_term->operands[0];
                    split_after_merge_tail = true;
                }
            }
        }

        /* Selection merges that are shared with structurizer bridge blocks
         * need a private trampoline as well. Otherwise later branches can
         * reuse the bridge-as-merge block directly, which turns the merge
         * into ordinary CFG and breaks structured lowering. */
        if (!merge_is_header &&
            !merge_is_continue &&
            !merge_is_bridge &&
            !split_after_merge_tail) continue;
        if (debug_phase3) {
            const char *hname = b->name ? b->name : "<anon>";
            const char *mname = func->blocks[mi].name ? func->blocks[mi].name : "<anon>";
            fprintf(stderr,
                    "[ssir_struct] phase3 split func=%s header=%s(%u) merge=%s(%u) continue=%d header=%d bridge=%d direct=%d tail=%d exit=%u\n",
                    func->name ? func->name : "<anon-func>",
                    hname, b->id, mname, merge_id,
                    merge_is_continue ? 1 : 0,
                    merge_is_header ? 1 : 0,
                    merge_is_bridge ? 1 : 0,
                    merge_is_branch_target ? 1 : 0,
                    split_after_merge_tail ? 1 : 0,
                    merge_exit_id);
        }

        SsirBlock tramp;
        memset(&tramp, 0, sizeof(tramp));
        tramp.id = mod->next_id++;
        tramp.inst_capacity = 2;
        tramp.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
        if (!tramp.insts) continue;
        memset(tramp.insts, 0, 2 * sizeof(SsirInst));
        tramp.insts[0].op = SSIR_OP_BRANCH;
        tramp.insts[0].operands[0] = split_after_merge_tail ? merge_exit_id : merge_id;
        tramp.insts[0].operand_count = 1;
        tramp.inst_count = 1;
        if (func->block_count >= func->block_capacity) {
            uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
            func->blocks = (SsirBlock *)STRUCT_REALLOC(func->blocks, nc * sizeof(SsirBlock));
            func->block_capacity = nc;
            b = &func->blocks[bi];
            term = &b->insts[b->inst_count - 1];
            if (merge_inst)
                merge_inst = NULL;
        }
        func->blocks[func->block_count++] = tramp;

        if (term->op == SSIR_OP_BRANCH_COND &&
            term->operand_count >= 4 && term->operands[3] == merge_id) {
            term->operands[3] = tramp.id;
        } else {
            if (!merge_inst) {
                for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                    if (b->insts[ii].op == SSIR_OP_SELECTION_MERGE &&
                        b->insts[ii].operand_count >= 1) {
                        merge_inst = &b->insts[ii];
                        break;
                    }
                }
            }
            if (merge_inst) merge_inst->operands[0] = tramp.id;
        }

        if (merge_is_branch_target) {
            if (term->op == SSIR_OP_BRANCH_COND) {
                if (term->operand_count >= 2 && term->operands[1] == merge_id)
                    term->operands[1] = tramp.id;
                if (term->operand_count >= 3 && term->operands[2] == merge_id)
                    term->operands[2] = tramp.id;
            } else if (term->op == SSIR_OP_SWITCH) {
                if (term->operand_count >= 2 && term->operands[1] == merge_id)
                    term->operands[1] = tramp.id;
                for (uint32_t ei = 1; ei < term->extra_count; ei += 2)
                    if (term->extra[ei] == merge_id)
                        term->extra[ei] = tramp.id;
            }
        }

        if (split_after_merge_tail) {
            SsirBlock *merge_blk = &func->blocks[mi];
            if (merge_blk->inst_count > 0) {
                SsirInst *merge_term = &merge_blk->insts[merge_blk->inst_count - 1];
                if (merge_term->op == SSIR_OP_BRANCH &&
                    merge_term->operand_count >= 1 &&
                    merge_term->operands[0] == merge_exit_id) {
                    merge_term->operands[0] = tramp.id;
                }
            }
        }

        uint32_t redirected_id = split_after_merge_tail ? merge_exit_id : merge_id;
        for (uint32_t bj = 0; bj < cfg_block_count; bj++) {
            if ((int)bj == mi) continue;
            if ((int)bj == (int)bi) continue;
            if (!sc_dom(c, (int)bi, (int)bj)) continue;

            int walk = (int)bj;
            bool postdominated = false;
            int steps = 0;
            while (walk >= 0 && steps < c->n + 1) {
                if (walk == mi) { postdominated = true; break; }
                if (walk == c->ipdom[walk]) break;
                walk = c->ipdom[walk];
                steps++;
            }
            if (!postdominated) continue;

            SsirBlock *body_blk = &func->blocks[bj];
            if (body_blk->inst_count == 0) continue;
            SsirInst *body_term = &body_blk->insts[body_blk->inst_count - 1];
            if (body_term->op == SSIR_OP_BRANCH &&
                body_term->operand_count >= 1 &&
                body_term->operands[0] == redirected_id) {
                body_term->operands[0] = tramp.id;
            } else if (body_term->op == SSIR_OP_BRANCH_COND) {
                if (body_term->operand_count >= 2 && body_term->operands[1] == redirected_id)
                    body_term->operands[1] = tramp.id;
                if (body_term->operand_count >= 3 && body_term->operands[2] == redirected_id)
                    body_term->operands[2] = tramp.id;
            } else if (body_term->op == SSIR_OP_SWITCH) {
                if (body_term->operand_count >= 2 && body_term->operands[1] == redirected_id)
                    body_term->operands[1] = tramp.id;
                for (uint32_t ei = 1; ei < body_term->extra_count; ei += 2)
                    if (body_term->extra[ei] == redirected_id)
                        body_term->extra[ei] = tramp.id;
            }
        }
    }

    /* Some lowerer/structurizer paths leave behind anonymous loop-entry
     * bridges that duplicate the real loop header's merge/continue pair.
     * Canonicalize those before any later selection-target splitting. */
    sc_canonicalize_duplicate_loop_headers(func, debug_phase3);

    /* === Phase 4: Privatize shared selection-entry targets === */
    for (uint32_t bi = 0; bi < cfg_block_count; bi++) {
        SsirBlock *b = &func->blocks[bi];
        if (b->inst_count == 0) continue;

        SsirInst *term = &b->insts[b->inst_count - 1];
        if (term->op != SSIR_OP_BRANCH_COND && term->op != SSIR_OP_SWITCH) continue;

        uint32_t merge_id = 0;
        if (term->op == SSIR_OP_BRANCH_COND &&
            term->operand_count >= 4 && term->operands[3] != 0) {
            merge_id = term->operands[3];
        } else {
            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                if (b->insts[ii].op == SSIR_OP_SELECTION_MERGE &&
                    b->insts[ii].operand_count >= 1) {
                    merge_id = b->insts[ii].operands[0];
                    break;
                }
            }
        }

        uint32_t tramp_src[130];
        uint32_t tramp_dst[130];
        uint32_t tramp_count = 0;

        uint32_t *targets[130];
        uint32_t target_count = 0;
        if (term->op == SSIR_OP_BRANCH_COND) {
            if (term->operand_count >= 2) targets[target_count++] = &term->operands[1];
            if (term->operand_count >= 3) targets[target_count++] = &term->operands[2];
        } else {
            if (term->operand_count >= 2) targets[target_count++] = &term->operands[1];
            for (uint32_t ei = 1; ei < term->extra_count && target_count < 130; ei += 2)
                targets[target_count++] = &term->extra[ei];
        }

        for (uint32_t tii = 0; tii < target_count; tii++) {
            uint32_t *tp = targets[tii];
            uint32_t target_id = *tp;
            if (!target_id || target_id == merge_id) continue;

            bool reused = false;
            for (uint32_t ci = 0; ci < tramp_count; ci++) {
                if (tramp_src[ci] == target_id) {
                    *tp = tramp_dst[ci];
                    reused = true;
                    break;
                }
            }
            if (reused) continue;

            bool shared_outside_construct = false;
            int tgt_idx = sc_idx(c, target_id);
            if (tgt_idx >= 0 && !sc_dom(c, (int)bi, tgt_idx))
                shared_outside_construct = true;

            for (uint32_t pj = 0; pj < func->block_count && !shared_outside_construct; pj++) {
                SsirBlock *pred = &func->blocks[pj];
                if (pred->inst_count == 0 || pred->id == b->id) continue;
                SsirInst *pred_term = &pred->insts[pred->inst_count - 1];
                bool targets_tgt = false;
                if (pred_term->op == SSIR_OP_BRANCH &&
                    pred_term->operand_count >= 1 &&
                    pred_term->operands[0] == target_id) {
                    targets_tgt = true;
                } else if (pred_term->op == SSIR_OP_BRANCH_COND) {
                    if ((pred_term->operand_count >= 2 && pred_term->operands[1] == target_id) ||
                        (pred_term->operand_count >= 3 && pred_term->operands[2] == target_id))
                        targets_tgt = true;
                } else if (pred_term->op == SSIR_OP_SWITCH) {
                    if (pred_term->operand_count >= 2 && pred_term->operands[1] == target_id)
                        targets_tgt = true;
                    for (uint32_t ei = 1; ei < pred_term->extra_count && !targets_tgt; ei += 2)
                        if (pred_term->extra[ei] == target_id)
                            targets_tgt = true;
                }
                if (!targets_tgt) continue;

                int pred_idx = sc_idx(c, pred->id);
                if (pred_idx < 0 || !sc_dom(c, (int)bi, pred_idx))
                    shared_outside_construct = true;
            }
            if (!shared_outside_construct) continue;

            SsirBlock *target_blk = ssir_get_block(mod, func_id, target_id);
            uint32_t private_target_id = target_id;
            bool collapsed_plain_bridge = target_blk && sc_is_plain_branch_bridge(target_blk);
            if (collapsed_plain_bridge)
                private_target_id = sc_follow_plain_branch(func, target_id);

            SsirBlock priv;
            memset(&priv, 0, sizeof(priv));
            priv.id = mod->next_id++;
            priv.inst_capacity = 2;
            priv.insts = (SsirInst *)STRUCT_MALLOC(
                priv.inst_capacity * sizeof(SsirInst));
            if (!priv.insts) continue;
            memset(priv.insts, 0, priv.inst_capacity * sizeof(SsirInst));
            priv.insts[0].op = SSIR_OP_BRANCH;
            priv.insts[0].operands[0] = private_target_id;
            priv.insts[0].operand_count = 1;
            priv.inst_count = 1;

            if (func->block_count >= func->block_capacity) {
                uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
                func->blocks = (SsirBlock *)STRUCT_REALLOC(
                    func->blocks, nc * sizeof(SsirBlock));
                func->block_capacity = nc;
                b = &func->blocks[bi];
                term = &b->insts[b->inst_count - 1];
                tp = targets[tii];
            }
            func->blocks[func->block_count++] = priv;
            *tp = priv.id;

            if (debug_phase3) {
                const char *hname = b->name ? b->name : "<anon>";
                fprintf(stderr,
                        "[ssir_struct] phase4 split func=%s header=%s(%u) target=%u tramp=%u final=%u\n",
                        func->name ? func->name : "<anon-func>",
                        hname, b->id, target_id, priv.id, private_target_id);
            }

            if (tramp_count < 130) {
                tramp_src[tramp_count] = target_id;
                tramp_dst[tramp_count] = priv.id;
                tramp_count++;
            }
        }
    }

    STRUCT_FREE(loops);

    /* === Final pass: rebuild CFG and detect back-edges created by
     * earlier structurization transforms.  Add OpLoopMerge for any
     * back-edge target that doesn't have one. === */
    {
        int fn = (int)func->block_count;
        if (fn > SC_MAX) fn = SC_MAX;
        SC *c2 = (SC *)STRUCT_MALLOC(sizeof(SC));
        if (c2 && fn > 1) {
            memset(c2, 0, sizeof(*c2));
            c2->mod = mod; c2->func_id = func_id; c2->func = func;
            c2->n = fn;
            for (int i = 0; i < fn; i++) c2->ids[i] = func->blocks[i].id;
            sc_build_cfg(c2);
            sc_compute_rpo(c2);
            /* Also visit unreachable components so dominator
             * computation covers the entire function. */
            {
                bool vis2[SC_MAX]; int stk2[SC_MAX], sp2 = 0;
                memset(vis2, 0, sizeof(bool) * fn);
                for (int i = 0; i < c2->rpo_n; i++) vis2[c2->rpo[i]] = true;
                for (int i = 0; i < fn; i++) {
                    if (vis2[i]) continue;
                    sc_dfs(c2, i, vis2, stk2, &sp2);
                }
                /* Append to RPO */
                for (int i = sp2 - 1; i >= 0; i--) {
                    int blk = stk2[i];
                    c2->rpo[c2->rpo_n] = blk;
                    c2->rpo_num[blk] = c2->rpo_n;
                    c2->reachable[blk] = true;
                    c2->rpo_n++;
                }
            }
            sc_compute_dom(c2);
            /* Seed idom for unreachable component roots and
             * rerun dominator computation to cover them. */
            {
                bool uc = true;
                while (uc) {
                    uc = false;
                    for (int i = 0; i < fn; i++) {
                        if (c2->idom[i] != -1) continue;
                        /* Check if this block has any predecessor with idom set */
                        bool has_known = false;
                        for (int pi = 0; pi < c2->npred[i]; pi++) {
                            int p = c2->pred[i][pi];
                            if (p >= 0 && c2->idom[p] != -1)
                                { has_known = true; break; }
                        }
                        if (!has_known) {
                            /* Component root — self-dominator */
                            c2->idom[i] = i;
                            uc = true;
                        }
                    }
                    /* Rerun dominator computation with new seeds */
                    if (uc) {
                        bool dc = true;
                        while (dc) {
                            dc = false;
                            for (int ri = 0; ri < c2->rpo_n; ri++) {
                                int b = c2->rpo[ri];
                                if (c2->idom[b] == b) continue; /* root */
                                if (c2->idom[b] != -1 && b < fn) {
                                    /* Already computed, but might need update */
                                }
                                int nd = -1;
                                for (int pi = 0; pi < c2->npred[b]; pi++) {
                                    int p = c2->pred[b][pi];
                                    if (c2->idom[p] == -1) continue;
                                    nd = (nd < 0) ? p : sc_intersect(c2->idom, c2->rpo_num, nd, p);
                                }
                                if (nd >= 0 && nd != c2->idom[b]) {
                                    c2->idom[b] = nd;
                                    dc = true;
                                }
                            }
                        }
                    }
                }
            }
            sc_compute_pdom(c2);

            /* Fix dangling merge refs: if a reachable block's
             * merge/continue target is unreachable, create a
             * dummy merge block so unreachable blocks can be
             * safely omitted from the SPIR-V binary. */
            for (int bi = 0; bi < fn; bi++) {
                if (!c2->reachable[bi]) continue;
                SsirBlock *bk = &func->blocks[bi];
                for (uint32_t ii = 0; ii < bk->inst_count; ii++) {
                    SsirInst *si = &bk->insts[ii];
                    if (si->op != SSIR_OP_SELECTION_MERGE &&
                        si->op != SSIR_OP_LOOP_MERGE) continue;
                    for (uint32_t oi = 0; oi < si->operand_count; oi++) {
                        int tgt = sc_idx(c2, si->operands[oi]);
                        if (tgt < 0 || c2->reachable[tgt]) continue;
                        /* Target is unreachable — create dummy merge */
                        SsirBlock dummy;
                        memset(&dummy, 0, sizeof(dummy));
                        dummy.id = mod->next_id++;
                        dummy.inst_capacity = 2;
                        dummy.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
                        if (!dummy.insts) continue;
                        memset(dummy.insts, 0, 2 * sizeof(SsirInst));
                        dummy.insts[0].op = SSIR_OP_UNREACHABLE;
                        dummy.insts[0].operand_count = 0;
                        dummy.inst_count = 1;
                        if (func->block_count >= func->block_capacity) {
                            uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
                            func->blocks = (SsirBlock *)STRUCT_REALLOC(func->blocks, nc * sizeof(SsirBlock));
                            func->block_capacity = nc;
                            bk = &func->blocks[bi];
                            si = &bk->insts[ii];
                        }
                        func->blocks[func->block_count++] = dummy;
                        si->operands[oi] = dummy.id;
                    }
                }
            }

            /* Find back-edges in the new CFG */
            for (int bi = 0; bi < fn; bi++) {
                if (!c2->reachable[bi]) continue;
                for (int si = 0; si < c2->nsuc[bi]; si++) {
                    int h = c2->succ[bi][si];
                    if (h < 0 || !sc_dom(c2, h, bi)) continue;
                    /* bi->h is a back-edge. Check if h has OpLoopMerge. */
                    SsirBlock *hb = &func->blocks[h];
                    bool has_lm = false;
                    for (uint32_t ii = 0; ii < hb->inst_count; ii++)
                        if (hb->insts[ii].op == SSIR_OP_LOOP_MERGE)
                            { has_lm = true; break; }
                    if (has_lm) continue;

                    /* Compute merge and continue for this loop */
                    bool body[SC_MAX];
                    memset(body, 0, sizeof(body));
                    sc_collect_body(c2, h, bi, body);
                    int merge = c2->ipdom[h];
                    if (merge >= 0 && merge < fn && body[merge])
                        merge = -1;
                    if (merge < 0) {
                        int best = fn + 1;
                        for (int j = 0; j < fn; j++) {
                            if (!body[j]) continue;
                            for (int sj = 0; sj < c2->nsuc[j]; sj++) {
                                int s = c2->succ[j][sj];
                                if (s >= 0 && s < fn && !body[s] &&
                                    c2->rpo_num[s] < best) {
                                    best = c2->rpo_num[s];
                                    merge = s;
                                }
                            }
                        }
                    }
                    if (merge < 0) continue;
                    int cont = -1;
                    int best_rpo = -1;
                    for (int j = 0; j < fn; j++) {
                        if (!body[j]) continue;
                        for (int sj = 0; sj < c2->nsuc[j]; sj++) {
                            if (c2->succ[j][sj] == h &&
                                c2->rpo_num[j] > best_rpo) {
                                best_rpo = c2->rpo_num[j];
                                cont = j;
                            }
                        }
                    }
                    if (cont < 0) continue;

                    /* Check merge target isn't already used */
                    bool merge_in_use = false;
                    for (uint32_t fi = 0; fi < func->block_count; fi++) {
                        SsirBlock *fb = &func->blocks[fi];
                        for (uint32_t ii = 0; ii < fb->inst_count; ii++) {
                            if ((fb->insts[ii].op == SSIR_OP_LOOP_MERGE ||
                                 fb->insts[ii].op == SSIR_OP_SELECTION_MERGE) &&
                                fb->insts[ii].operand_count >= 1 &&
                                fb->insts[ii].operands[0] == c2->ids[merge]) {
                                merge_in_use = true;
                                break;
                            }
                        }
                        if (merge_in_use) break;
                    }
                    if (merge_in_use) {
                        /* Create a trampoline as unique merge target */
                        SsirBlock tramp;
                        memset(&tramp, 0, sizeof(tramp));
                        tramp.id = mod->next_id++;
                        tramp.inst_capacity = 2;
                        tramp.insts = (SsirInst *)STRUCT_MALLOC(2 * sizeof(SsirInst));
                        if (!tramp.insts) continue;
                        memset(tramp.insts, 0, 2 * sizeof(SsirInst));
                        tramp.insts[0].op = SSIR_OP_BRANCH;
                        tramp.insts[0].operands[0] = c2->ids[merge];
                        tramp.insts[0].operand_count = 1;
                        tramp.inst_count = 1;
                        if (func->block_count >= func->block_capacity) {
                            uint32_t nc = func->block_capacity ? func->block_capacity * 2 : 16;
                            func->blocks = (SsirBlock *)STRUCT_REALLOC(func->blocks, nc * sizeof(SsirBlock));
                            func->block_capacity = nc;
                            hb = &func->blocks[h];
                        }
                        func->blocks[func->block_count++] = tramp;
                        merge = -1; /* Use tramp ID below */
                        /* Find the tramp's ID */
                        uint32_t merge_id = tramp.id;
                        uint32_t cont_id = c2->ids[cont];
                        /* Check for existing SelectionMerge to replace */
                        bool replaced = false;
                        for (uint32_t ii = 0; ii < hb->inst_count; ii++) {
                            if (hb->insts[ii].op == SSIR_OP_SELECTION_MERGE) {
                                hb->insts[ii].op = SSIR_OP_LOOP_MERGE;
                                hb->insts[ii].operands[0] = merge_id;
                                hb->insts[ii].operands[1] = cont_id;
                                hb->insts[ii].operand_count = 2;
                                replaced = true;
                                break;
                            }
                        }
                        if (!replaced && hb->inst_count > 0) {
                            SsirInst lm;
                            memset(&lm, 0, sizeof(lm));
                            lm.op = SSIR_OP_LOOP_MERGE;
                            lm.operands[0] = merge_id;
                            lm.operands[1] = cont_id;
                            lm.operand_count = 2;
                            sc_insert_at(hb, hb->inst_count - 1, &lm);
                        }
                    } else {
                        /* Merge target is available — use it directly */
                        bool replaced = false;
                        for (uint32_t ii = 0; ii < hb->inst_count; ii++) {
                            if (hb->insts[ii].op == SSIR_OP_SELECTION_MERGE) {
                                hb->insts[ii].op = SSIR_OP_LOOP_MERGE;
                                hb->insts[ii].operands[0] = c2->ids[merge];
                                hb->insts[ii].operands[1] = c2->ids[cont];
                                hb->insts[ii].operand_count = 2;
                                replaced = true;
                                break;
                            }
                        }
                        if (!replaced && hb->inst_count > 0) {
                            SsirInst lm;
                            memset(&lm, 0, sizeof(lm));
                            lm.op = SSIR_OP_LOOP_MERGE;
                            lm.operands[0] = c2->ids[merge];
                            lm.operands[1] = c2->ids[cont];
                            lm.operand_count = 2;
                            sc_insert_at(hb, hb->inst_count - 1, &lm);
                        }
                    }
                    break; /* One loop annotation per header is enough */
                }
            }
        }
        if (c2) STRUCT_FREE(c2);
    }

    STRUCT_FREE(c);
}
