"""
MST, its ablations, and adapted baselines on one candidate pool.

All quantities follow the paper (Section 3): U and P use the product-union
form, interpretation cost is normalized by the complete candidate pool I_t
(including prohibited items), and K = |S| / |I_t|. Every subset of the pool is
enumerated once per instance with a bit mask; products and sums are accumulated
in item order so that the values match the reference implementation
(mst/mst_core.py) to the last bit.

Each method returns a selection outcome: either a disclosure (a bit mask) or
"INFEASIBLE". Methods that have no notion of infeasibility always return a
disclosure, as they would in practice.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

INFEASIBLE = -1

SOFT_PENALTY_PER_PROHIBITED_ITEM = 1.0   # larger than the whole range of J in [0, 1]
UTILITY_REWARD_WEIGHT = 1.0              # weight of U(S) when sufficiency is a reward
PARSIMONY_K = 3                          # fixed explanation length of the parsimony baseline
COVERAGE_BUDGET = 0.25                   # normalized interpretation-cost budget
GREEDY_EPSILON = 1e-12


@lru_cache(maxsize=None)
def bit_matrix(n: int) -> np.ndarray:
    masks = np.arange(1 << n, dtype=np.int64)
    return ((masks[:, None] >> np.arange(n)) & 1).astype(bool)


class Instance:
    """All-subset quantities of one pool; independent of tau and weights."""

    def __init__(self, ids, u, p, c, admissible):
        self.ids = tuple(ids)
        self.u = np.asarray(u, float)
        self.p = np.asarray(p, float)
        self.c = np.asarray(c, float)
        self.adm = np.asarray(admissible, bool)
        n = self.n = len(self.ids)
        B = self.B = bit_matrix(n)
        prod_u = np.ones(len(B))
        prod_p = np.ones(len(B))
        csum = np.zeros(len(B))
        for i in range(n):          # item order, as in mst_core
            col = B[:, i]
            prod_u = np.where(col, prod_u * (1.0 - self.u[i]), prod_u)
            prod_p = np.where(col, prod_p * (1.0 - self.p[i]), prod_p)
            csum = np.where(col, csum + self.c[i], csum)
        self.U = 1.0 - prod_u
        self.P = 1.0 - prod_p
        total_cost = 0
        for v in self.c:
            total_cost += v
        self.Cn = csum / total_cost if total_cost != 0.0 else np.zeros(len(B))
        self.card = B.sum(axis=1)
        self.K = self.card / n
        self.n_prohibited = (B & ~self.adm).sum(axis=1)
        self.all_adm = self.n_prohibited == 0
        self.adm_mask = int(sum(1 << i for i in range(n) if self.adm[i]))
        self.full_mask = (1 << n) - 1
        self._id_rank = None

    # ---- objective -------------------------------------------------------
    def J(self, w):
        lp, lc, lk = w
        return lp * self.P + lc * self.Cn + lk * self.K

    def id_tuple(self, mask):
        return tuple(self.ids[i] for i in range(self.n) if mask >> i & 1)

    def lexmin(self, candidates, keys):
        """Index of the lexicographically smallest candidate under keys, then ids."""
        idx = np.flatnonzero(candidates)
        if idx.size == 0:
            return INFEASIBLE
        order = np.lexsort(tuple(k[idx] for k in reversed(keys)))
        first = idx[order[0]]
        ties = [j for j in idx[order] if all(k[j] == k[first] for k in keys)]
        if len(ties) > 1:
            first = min(ties, key=self.id_tuple)
        return int(first)

    def admissible_utility(self):
        return self.U[self.adm_mask]


# ---------------------------------------------------------------------------
# MST and ablations
# ---------------------------------------------------------------------------

def mst_exact(inst, tau, w, J=None):
    J = inst.J(w) if J is None else J
    cand = inst.all_adm & (inst.U >= tau)
    return inst.lexmin(cand, (J, inst.P, inst.Cn, inst.card))


def ablation_soft_admissibility(inst, tau, w, J=None):
    """Hard admissibility replaced by a penalty per prohibited item."""
    J = inst.J(w) if J is None else J
    penalized = J + SOFT_PENALTY_PER_PROHIBITED_ITEM * inst.n_prohibited
    cand = inst.U >= tau
    return inst.lexmin(cand, (penalized, inst.P, inst.Cn, inst.card))


def ablation_utility_reward(inst, tau, w, J=None):
    """Sufficiency constraint replaced by a utility reward in the objective."""
    J = inst.J(w) if J is None else J
    score = J - UTILITY_REWARD_WEIGHT * inst.U
    return inst.lexmin(inst.all_adm, (score, inst.P, inst.Cn, inst.card))


def ablation_min_cardinality(inst, tau, w, J=None):
    """Disclosure-cost objective replaced by the number of items only."""
    cand = inst.all_adm & (inst.U >= tau)
    return inst.lexmin(cand, (inst.card, -inst.U))


def ablation_no_infeasibility(inst, tau, w, J=None):
    """When no admissible subset is sufficient, disclose every admissible item."""
    sel = mst_exact(inst, tau, w, J)
    return inst.adm_mask if sel == INFEASIBLE else sel


def ablation_greedy(inst, tau, w, J=None):
    """Dynamic Greedy-MST: largest marginal utility per marginal objective."""
    J = inst.J(w) if J is None else J
    if inst.U[inst.adm_mask] < tau:
        return INFEASIBLE
    sel = 0
    while inst.U[sel] < tau:
        best, best_score = None, -np.inf
        for i in range(inst.n):
            if not inst.adm[i] or sel >> i & 1:
                continue
            nxt = sel | (1 << i)
            du = inst.U[nxt] - inst.U[sel]
            if du <= 0.0:
                continue
            score = du / (J[nxt] - J[sel] + GREEDY_EPSILON)
            if score > best_score or (abs(score - best_score) <= 1e-12 and best is not None and i < best):
                best, best_score = i, score
        if best is None:
            return INFEASIBLE
        sel |= 1 << best
    return sel


# ---------------------------------------------------------------------------
# Adapted baselines (one per family in the positioning table)
# ---------------------------------------------------------------------------

def baseline_full_disclosure(inst, tau, w, J=None):
    return inst.full_mask


def baseline_policy_filter(inst, tau, w, J=None):
    return inst.adm_mask


def baseline_parsimony_topk(inst, tau, w, J=None):
    order = sorted(range(inst.n), key=lambda i: (-inst.u[i], i))[:PARSIMONY_K]
    return int(sum(1 << i for i in order))


def baseline_sufficient_reason(inst, tau, w, J=None):
    """Smallest sufficient subset, without admissibility or disclosure cost."""
    cand = inst.U >= tau
    sel = inst.lexmin(cand, (inst.card, -inst.U))
    return inst.full_mask if sel == INFEASIBLE else sel


def baseline_greedy_cover(inst, tau, w, J=None):
    """Wolsey-style greedy cover: marginal utility per item cost until tau."""
    sel = 0
    while inst.U[sel] < tau and sel != inst.full_mask:
        best, best_score = None, -np.inf
        for i in range(inst.n):
            if sel >> i & 1:
                continue
            score = (inst.U[sel | (1 << i)] - inst.U[sel]) / inst.c[i]
            if score > best_score:
                best, best_score = i, score
        sel |= 1 << best
    return sel


def baseline_budgeted_coverage(inst, tau, w, J=None):
    """Largest utility within a fixed interpretation-cost budget."""
    cand = inst.Cn <= COVERAGE_BUDGET
    return inst.lexmin(cand, (-inst.U, inst.card))


METHODS = {
    # name: (function, group, short label)
    "MST": (mst_exact, "MST", "MST (exact)"),
    "A1_soft_admissibility": (ablation_soft_admissibility, "ablation", "without hard admissibility"),
    "A2_utility_reward": (ablation_utility_reward, "ablation", "without sufficiency threshold"),
    "A3_min_cardinality": (ablation_min_cardinality, "ablation", "without disclosure-cost objective"),
    "A4_no_infeasibility": (ablation_no_infeasibility, "ablation", "without infeasibility reporting"),
    "A5_greedy": (ablation_greedy, "ablation", "without exact search (Dynamic Greedy-MST)"),
    "B1_full_disclosure": (baseline_full_disclosure, "baseline", "full disclosure"),
    "B2_policy_filter": (baseline_policy_filter, "baseline", "policy filter only"),
    "B3_parsimony_top3": (baseline_parsimony_topk, "baseline", "parsimonious top-3"),
    "B4_sufficient_reason": (baseline_sufficient_reason, "baseline", "minimal sufficient subset"),
    "B5_greedy_cover": (baseline_greedy_cover, "baseline", "greedy submodular cover"),
    "B6_budgeted_coverage": (baseline_budgeted_coverage, "baseline", "budgeted coverage"),
}


def repair(inst, sel, tau, J):
    """Make a disclosure valid at least cost of effort: drop prohibited items,
    then add admissible items by the Dynamic Greedy-MST rule until sufficient.
    Returns the repaired mask, or INFEASIBLE when no admissible repair exists."""
    if sel == INFEASIBLE:
        return INFEASIBLE
    sel &= inst.adm_mask
    while inst.U[sel] < tau:
        best, best_score = None, -np.inf
        for i in range(inst.n):
            if not inst.adm[i] or sel >> i & 1:
                continue
            nxt = sel | (1 << i)
            du = inst.U[nxt] - inst.U[sel]
            if du <= 0.0:
                continue
            score = du / (J[nxt] - J[sel] + GREEDY_EPSILON)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return INFEASIBLE
        sel |= 1 << best
    return sel
