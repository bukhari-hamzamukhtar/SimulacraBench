"""SimulacraBench submission: a self-tuning Super Learner fitted inside predict().

A held-out respondent shows only their GIVEN block, and the score is a sum of
per-cell log scores over i.i.d. rows. The expected score is therefore maximised
by P(answer to item j | GIVEN block), one item at a time: calibrated conditional
density estimation from categorical features ("cold-start" completion). We
cannot see the data and get one scored run, so the submission chooses its own
models on the data it is handed:

  1. every candidate estimator is fitted with 5-fold cross-validation on the
     visible rows, giving honest out-of-fold (OOF) predictions; held-out rows
     get a full-data refit (or, for networks, a refit blended with the folds);
  2. for each item, a convex mixture of the candidates is fitted to the OOF
     predictions by EM (concave in the weights, so EM reaches the global
     optimum), shrunk toward weights pooled over items by a strength chosen on
     a split of the OOF rows. The crowd marginal is always a candidate. This is
     the Super Learner (van der Laan, Polley and Hubbard, 2007); the grader's
     probability floor bounds the loss, which its oracle inequality needs;
  3. a one-parameter temperature is fitted per item on the same OOF data;
  4. hot-deck donor correction: visible respondents whose GIVEN block differs in
     at most two (or, weighted down, four) columns pull the stacked belief toward
     their own answers, P = (votes + alpha * B) / (weight + alpha), with the
     distance scale, cut-off and alpha chosen per item by OOF likelihood and left
     off where it does not help. When the GIVEN block has a survey-round column,
     a second set of donors comes from the other round only: a household
     interviewed in both rounds has its other interview there, and a same-round
     lookalike must not outvote it;
  5. the grader mixes every vector with a flat one before scoring, so reporting
     the belief directly is not optimal. We report the exact optimum (a
     water-filling transform) whenever it scores better OOF.

Candidates, in the order they run:
  marginal      the crowd baseline, the zero of the skill scale
  backoff       lookup tables: empirical-Bayes hierarchical Dirichlet smoothing
                over GIVEN columns ranked by mutual information, smoothing
                strength per item and level by leave-one-out likelihood
  knn           hot-deck donors within Hamming distance 2 or 4, adjusting a
                lookup estimate (the classic survey imputation method)
  ridge         multinomial logistic regression under Gaussian priors: main
                effects; main effects on each item's 4 most informative columns;
                main effects plus all pairwise interactions. All items at once
                with L-BFGS
  net           multitask networks: a shared trunk over embeddings of the GIVEN
                block, one softmax per item, several widths and seeds
  pnet          per-item networks: shared embeddings, a hidden layer of its own
                for every item (one batched einsum)
  gbdt          gradient-boosted trees, when the time budget allows

Skip logic: a gated item's answer is the sentinel when the respondent was never
asked. Lookups, logistic models and networks also come in a two-part ("hurdle")
form for gated items, P(k) = P(asked) * P(k | asked), so the columns that decide
the gate and the columns that decide the answer do not compete. Gates are
learned, never imposed, because the real gates hold only approximately.

A wall-clock governor keeps the whole run well inside the grader's 60 minutes
and drops whatever does not fit; every candidate can fail without failing the
submission.
"""

import math
import os
import re
import time

import numpy as np
import pandas as pd

T_IMPORT = time.time()

FLOOR = 1e-3                       # scoring.floor in the grader's config
SEED = 20261120
N_FOLDS = 5
TOTAL_BUDGET = float(os.environ.get("SB_TOTAL_BUDGET", "2700"))   # ours; the grader allows 3600
VERBOSE = os.environ.get("SB_VERBOSE", "1") == "1"
# Hot-deck donors (the knn candidate and the post-stack donor correction).
USE_DONORS = os.environ.get("SB_DONORS", "1") == "1"

# Share of the total budget each instrument may plan on. Unused time carries
# over to the instruments that come later; the order they arrive in is unknown.
SHARE = {"unhcr": 0.46, "world_bank": 0.30, "unicef": 0.24, "other": 0.30}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
except Exception:                                   # pragma: no cover
    HAVE_TORCH = False
    DEVICE = None

STATE = {"done": {}, "spent": 0.0}


def log(*args):
    if VERBOSE:
        print("[sb %6.1fs]" % (time.time() - T_IMPORT), *args, flush=True)


# =============================================================================
# Data
# =============================================================================

_NUM = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _numeric_value(v):
    """Lower bound of a count, band or numbered label; None if not numeric."""
    s = str(v)
    m = _NUM.match(s)
    if not m:
        return None
    return float(m.group(1))


class Instrument:
    """Everything predict() needs, encoded once."""

    def __init__(self, frame, schema):
        items = schema["items"]
        names = [n for n, r in items.items() if r["class"] in ("GIVEN", "PREDICT")]
        self.given = [n for n in names if items[n]["class"] == "GIVEN"]
        self.targets = [n for n in names if items[n]["class"] == "PREDICT"]
        gated_value = schema.get("gated_value", "NA_GATED")
        self.options, self.gated = [], []
        for t in self.targets:
            rec = items[t]
            self.gated.append(bool(rec.get("gate")))
            self.options.append(list(rec["values"]) + ([gated_value] if rec.get("gate") else []))
        self.K = np.array([len(o) for o in self.options], dtype=np.int64)
        self.J = len(self.targets)
        self.offsets = np.concatenate([[0], np.cumsum(self.K)]).astype(np.int64)
        self.D = int(self.offsets[-1])

        nan = frame[self.targets].isna().to_numpy() if self.J else np.zeros((len(frame), 0), bool)
        self.nan = nan
        self.n_rows = len(frame)
        self.vis = np.where(~nan.any(axis=1))[0] if self.J else np.arange(len(frame))
        self.hid = np.where(nan.any(axis=1))[0] if self.J else np.zeros(0, np.int64)

        # ---- GIVEN block, as integer codes with one extra "unknown" level --
        cols, cards, numeric = [], [], []
        for g in self.given:
            values = list(items[g]["values"] or [])
            lut = {v: i for i, v in enumerate(values)}
            lut_s = {str(v): i for i, v in enumerate(values)}
            unknown = len(values)
            raw = frame[g].to_numpy(dtype=object)
            codes = np.fromiter(
                (unknown if (v is None or (isinstance(v, float) and math.isnan(v)))
                 else lut.get(v, lut_s.get(str(v), unknown)) for v in raw),
                dtype=np.int64, count=len(raw))
            cols.append(codes)
            cards.append(unknown + 1)
            parsed = [_numeric_value(v) for v in values]
            ok = [p for p in parsed if p is not None]
            if len(values) >= 3 and len(ok) >= 0.7 * len(values):
                ranks = {p: r for r, p in enumerate(sorted(set(ok)))}
                top = max(1, len(ranks) - 1)
                lut_num = np.array([ranks[p] / top if p is not None else np.nan for p in parsed]
                                   + [np.nan], dtype=float)
                numeric.append((g, lut_num[codes], parsed))
        X = np.stack(cols, axis=1) if cols else np.zeros((len(frame), 0), np.int64)

        # ---- derived: sums of count-like columns that share a prefix --------
        # (the UNHCR roster: s2_3_boys_0_5 ... s2_3_women_60plus -> household size)
        groups = {}
        for g, _, parsed in numeric:
            prefix = "_".join(g.split("_")[:2]) if "_" in g else g
            groups.setdefault(prefix, []).append(g)
        derived_cat, derived_num = [], []
        for prefix, members in groups.items():
            if len(members) < 3:
                continue
            total = np.zeros(len(frame))
            for g in members:
                c = self.given.index(g)
                values = list(items[g]["values"])
                lb = np.array([(_numeric_value(v) or 0.0) for v in values] + [0.0])
                total += lb[X[:, c]]
            cat = np.clip(np.round(total), 0, 15).astype(np.int64)
            derived_cat.append(cat)
            derived_num.append(np.clip(total, 0, 30) / 30.0)
            log("derived household-size feature from %d columns (%s*)" % (len(members), prefix))
        if derived_cat:
            X = np.concatenate([X, np.stack(derived_cat, axis=1)], axis=1)
            cards += [16] * len(derived_cat)
        self.X = X
        self.cards = np.array(cards, dtype=np.int64)

        num = [v for _, v, _ in numeric] + derived_num
        if num:
            Z = np.stack(num, axis=1)
            mu = np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.5)
            Z = np.where(np.isfinite(Z), Z, mu)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        else:
            Z = np.zeros((len(frame), 0))
        self.Z = Z.astype(np.float32)

        # ---- targets of the visible rows, as option indices (-1 = unusable) --
        Y = np.full((len(self.vis), self.J), -1, dtype=np.int64)
        for j, t in enumerate(self.targets):
            lut = {v: i for i, v in enumerate(self.options[j])}
            lut_s = {str(v): i for i, v in enumerate(self.options[j])}
            raw = frame[t].to_numpy(dtype=object)[self.vis]
            Y[:, j] = np.fromiter((lut.get(v, lut_s.get(str(v), -1)) for v in raw),
                                  dtype=np.int64, count=len(raw))
        self.Y = Y
        self.Xv = self.X[self.vis]
        self.Xh = self.X[self.hid]
        self.Zv = self.Z[self.vis]
        self.Zh = self.Z[self.hid]

        self.kind = {213: "unhcr", 76: "world_bank", 12: "unicef"}.get(self.J, "other")


def marginal_vector(y, K):
    """Counts plus half a count on every option: exactly the baseline's vector."""
    y = y[y >= 0]
    return (np.bincount(y, minlength=K)[:K] + 0.5) / (len(y) + 0.5 * K)


# =============================================================================
# Candidates. Each fit(inst, tr, ev_list) returns, for every item j, a list of
# probability arrays, one per entry of ev_list (row indices into the visible
# block, or the string "hid" for the held-out rows).
# =============================================================================

def _rows(inst, which):
    return (inst.Xh, inst.Zh) if isinstance(which, str) else (inst.Xv[which], inst.Zv[which])


def cand_marginal(inst, tr, evs, budget):
    out = []
    for j in range(inst.J):
        m = marginal_vector(inst.Y[tr, j], inst.K[j])
        out.append([np.tile(m, (len(_rows(inst, e)[0]), 1)) for e in evs])
    return {"marginal": out}


def _mutual_information(x, y, cx, cy):
    joint = np.bincount(x * cy + y, minlength=cx * cy).reshape(cx, cy).astype(float)
    n = joint.sum()
    if n <= 0:
        return 0.0
    joint /= n
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


ALPHAS = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024], dtype=float)


def _loo_alpha(C, P):
    """Smoothing strength that maximises the leave-one-out likelihood.

    C: counts per cell [cells, K]; P: the parent estimate for each cell.
    The leave-one-out predictive for one observation of option k in a cell is
    (C_k - 1 + a P_k) / (n - 1 + a).
    """
    n = C.sum(1, keepdims=True)
    mask = C > 0
    best, best_ll = ALPHAS[0], -np.inf
    for a in ALPHAS:
        num = np.maximum(C - 1 + a * P, 1e-12)
        ll = (C * np.log(num))[mask].sum() - (n * np.log(np.maximum(n - 1 + a, 1e-12))).sum()
        if ll > best_ll:
            best, best_ll = a, ll
    return best


def _chain(Xo, yo, K, ev_X, cards, depth):
    """Empirical-Bayes backoff along the most informative columns.

    Returns, for depth 1..depth, the estimate for every evaluation set."""
    root = marginal_vector(yo, K)
    mis = [_mutual_information(Xo[:, c], yo, int(cards[c]), K)
           for c in range(Xo.shape[1])] if len(yo) else [0.0] * Xo.shape[1]
    order = list(np.argsort(mis)[::-1][:depth])
    cur_tr = np.tile(root, (len(yo), 1))
    cur_ev = [np.tile(root, (len(x), 1)) for x in ev_X]
    key_tr = np.zeros(len(yo), np.int64)
    key_ev = [np.zeros(len(x), np.int64) for x in ev_X]
    levels = []
    for c in order:
        card = int(cards[c])
        key_tr = key_tr * card + Xo[:, c]
        key_ev = [k * card + x[:, c] for k, x in zip(key_ev, ev_X)]
        if len(yo):
            uniq, inv = np.unique(key_tr, return_inverse=True)
            C = np.zeros((len(uniq), K))
            np.add.at(C, (inv, yo), 1.0)
            # parent estimate of each cell: every row of a cell shares it
            first = np.zeros(len(uniq), np.int64)
            first[inv[::-1]] = np.arange(len(inv))[::-1]
            P = cur_tr[first]
            a = _loo_alpha(C, P)
            cell_est = (C + a * P) / (C.sum(1, keepdims=True) + a)
            cur_tr = cell_est[inv]
            new_ev = []
            for k, cur in zip(key_ev, cur_ev):
                pos = np.searchsorted(uniq, k)
                pos_c = np.minimum(pos, len(uniq) - 1)
                hit = uniq[pos_c] == k
                est = cur.copy()
                if hit.any():
                    Cc = C[pos_c[hit]]
                    est[hit] = (Cc + a * cur[hit]) / (Cc.sum(1, keepdims=True) + a)
                new_ev.append(est)
            cur_ev = new_ev
        levels.append([e.copy() for e in cur_ev])
    while len(levels) < depth:
        levels.append([e.copy() for e in cur_ev])
    return levels


def cand_backoff(inst, tr, evs, budget, depth=4, hurdle=False):
    """Empirical-Bayes backoff along the most informative GIVEN columns.

    With hurdle=True it also returns a two-part version for gated items:
    P(asked) from one chain fitted to the asked/skipped indicator, times
    P(answer | asked) from a second chain fitted to the asked rows only. The
    two chains pick their own columns, so the columns that decide the gate and
    the columns that decide the answer do not compete for the same depth."""
    out = {"backoff%d" % d: [] for d in range(1, depth + 1)}
    if hurdle:
        out.update({"backoffh%d" % d: [] for d in range(1, depth + 1)})
    Xtr = inst.Xv[tr]
    ev_X = [_rows(inst, e)[0] for e in evs]
    for j in range(inst.J):
        K = int(inst.K[j])
        y = inst.Y[tr, j]
        ok = y >= 0
        Xo, yo = Xtr[ok], y[ok]
        plain = _chain(Xo, yo, K, ev_X, inst.cards, depth)
        for d in range(depth):
            out["backoff%d" % (d + 1)].append(plain[d])
        if not hurdle:
            continue
        if not inst.gated[j] or K < 3:
            for d in range(depth):
                out["backoffh%d" % (d + 1)].append([e.copy() for e in plain[d]])
            continue
        sent = K - 1
        asked = (yo != sent).astype(np.int64)
        lv_a = _chain(Xo, asked, 2, ev_X, inst.cards, depth)
        m = yo != sent
        lv_c = _chain(Xo[m], yo[m], K - 1, ev_X, inst.cards, depth)
        for d in range(depth):
            comb = []
            for pa, pc in zip(lv_a[d], lv_c[d]):
                p_ask = pa[:, 1:2]
                comb.append(np.concatenate([p_ask * pc, 1.0 - p_ask], axis=1))
            out["backoffh%d" % (d + 1)].append(comb)
    return out


def cand_knn(inst, tr, evs, budget, k=64,
             variants=((0.5, 0.25, "knn_sharp", 2), (1.5, 1.0, "knn_soft", 2),
                       (0.4, 0.25, "knn_wide", 4), (1.0, 1.0, "knn_wide_soft", 4))):
    """Nearest-neighbour hot-deck donors on top of a lookup-table estimate.

    Distance is the Hamming distance over the whole GIVEN block (one matrix
    product of one-hot codes). Donors within d_max mismatching columns vote with
    weight exp(-lam * distance) and adjust a depth-3 lookup estimate of the same
    respondent, which carries weight alpha:

        P = (sum of donor votes + alpha * P_lookup) / (sum of weights + alpha).

    A respondent with no close donor gets exactly the lookup estimate, so the
    candidate costs nothing where donors carry no information. Fitted inside each
    fold like every other candidate."""
    if not HAVE_TORCH:
        raise RuntimeError("torch unavailable")
    dev = DEVICE
    C = inst.X.shape[1]
    col_off = np.concatenate([[0], np.cumsum(inst.cards)[:-1]]).astype(np.int64)
    n_oh = int(inst.cards.sum())

    def onehot(X):
        H = torch.zeros((len(X), n_oh), device=dev)
        H.scatter_(1, torch.as_tensor(X + col_off[None, :], device=dev), 1.0)
        return H

    # the base estimate: the depth-3 lookup chain of every item, per evaluation set
    Xtr_np = inst.Xv[tr]
    ev_X = [_rows(inst, e)[0] for e in evs]
    base = [[] for _ in evs]
    for j in range(inst.J):
        y = inst.Y[tr, j]
        ok = y >= 0
        lv = _chain(Xtr_np[ok], y[ok], int(inst.K[j]), ev_X, inst.cards, 3)[2]
        for i in range(len(evs)):
            base[i].append(lv[i])
    base = [torch.as_tensor(np.concatenate(b, axis=1), device=dev, dtype=torch.float32) for b in base]

    Htr = onehot(Xtr_np)
    Ytr = torch.as_tensor(inst.Y[tr], device=dev)
    valid = (Ytr >= 0)
    slot = torch.where(valid, Ytr + torch.as_tensor(inst.offsets[:-1], device=dev)[None, :], 0)
    seg = torch.as_tensor(np.repeat(np.arange(inst.J), inst.K), device=dev)
    kk = int(min(k, len(tr)))
    per_eval = {name: [] for _, _, name, _ in variants}
    for i, Xe in enumerate(ev_X):
        res = {name: [] for _, _, name, _ in variants}
        for a in range(0, len(Xe), 2048):
            He = onehot(Xe[a:a + 2048])
            dist = C - He @ Htr.T                                   # mismatching columns
            d, idx = torch.topk(-dist, kk, dim=1)
            d = -d
            ys = slot[idx]                                          # rows x k x J
            vs = valid[idx].float()
            b = base[i][a:a + 2048]
            for lam, alpha, name, d_max in variants:
                w = torch.exp(-lam * d) * (d <= d_max).float()      # rows x k
                wv = w.unsqueeze(2) * vs                            # rows x k x J
                cnt = torch.zeros((len(He), inst.D), device=dev)
                cnt.scatter_add_(1, ys.reshape(len(He), -1), wv.reshape(len(He), -1))
                tot = wv.sum(1)                                     # rows x J
                p = (cnt + alpha * b) / (tot[:, seg] + alpha)
                res[name].append(p.cpu().numpy())
            if time.time() > budget:
                raise TimeoutError("knn over budget")
        for _, _, name, _ in variants:
            per_eval[name].append(np.concatenate(res[name], axis=0))
    return {name: [[pe[:, inst.offsets[j]:inst.offsets[j + 1]] for pe in per_eval[name]] for j in range(inst.J)]
            for _, _, name, _ in variants}


# ---------------------------------------------------------------- torch utils

def _seg_log_softmax(logits, seg, J):
    """Log-softmax inside each item's slice of the concatenated output."""
    n = logits.shape[0]
    idx = seg.unsqueeze(0).expand(n, -1)
    m = torch.full((n, J), -1e30, device=logits.device, dtype=logits.dtype)
    m = m.scatter_reduce(1, idx, logits.detach(), reduce="amax", include_self=True)
    ex = torch.exp(logits - m.gather(1, idx))
    s = torch.zeros((n, J), device=logits.device, dtype=logits.dtype).scatter_add(1, idx, ex)
    return logits - (m + torch.log(s)).gather(1, idx)


def _seg_log_hurdle(logits, tz, J):
    """Two-part output for gated items: P(skipped) = sigmoid(s) from the
    item's sentinel slot, and P(option k) = sigmoid(-s) * softmax over the real
    options. Items that are never gated get the ordinary softmax."""
    n = logits.shape[0]
    idx = tz.seg.unsqueeze(0).expand(n, -1)
    real = logits.masked_fill(tz.sent_mask.unsqueeze(0), -1e30)
    m = torch.full((n, J), -1e30, device=logits.device, dtype=logits.dtype)
    m = m.scatter_reduce(1, idx, real.detach(), reduce="amax", include_self=True)
    ex = torch.exp(real - m.gather(1, idx))
    ssum = torch.zeros((n, J), device=logits.device, dtype=logits.dtype).scatter_add(1, idx, ex)
    lp_real = real - (m + torch.log(ssum)).gather(1, idx)
    s = logits[:, tz.sent_slot]
    add = torch.where(tz.gated_t.unsqueeze(0), F.logsigmoid(-s), torch.zeros_like(s))
    lp = lp_real + add.gather(1, idx)
    return torch.where(tz.sent_mask.unsqueeze(0), F.logsigmoid(s).gather(1, idx), lp)


def _logp(logits, tz, J, hurdle):
    return _seg_log_hurdle(logits, tz, J) if hurdle else _seg_log_softmax(logits, tz.seg, J)


def _bias0(inst, tr, hurdle):
    """Output bias that reproduces the crowd marginal, in either parameterisation."""
    out = []
    for j in range(inst.J):
        m = marginal_vector(inst.Y[tr, j], inst.K[j])
        if hurdle and inst.gated[j] and inst.K[j] >= 3:
            ps = m[-1]
            out.append(np.concatenate([np.log(m[:-1] / m[:-1].sum()), [np.log(ps / (1.0 - ps))]]))
        else:
            out.append(np.log(m))
    return np.concatenate(out)


class _Torchy:
    """Per-instrument tensors shared by the torch candidates."""

    def __init__(self, inst):
        self.seg = torch.as_tensor(np.repeat(np.arange(inst.J), inst.K), device=DEVICE)
        gated = np.array([bool(g) and k >= 3 for g, k in zip(inst.gated, inst.K)], dtype=bool)
        sent_slot = np.where(gated, inst.offsets[1:] - 1, 0)
        sent_mask = np.zeros(inst.D, dtype=bool)
        sent_mask[sent_slot[gated]] = True
        self.gated_t = torch.as_tensor(gated, device=DEVICE)
        self.sent_slot = torch.as_tensor(sent_slot, device=DEVICE)
        self.sent_mask = torch.as_tensor(sent_mask, device=DEVICE)
        Y = inst.Y
        self.tcol_all = np.where(Y >= 0, inst.offsets[:-1][None, :] + np.maximum(Y, 0), 0)
        self.mask_all = (Y >= 0).astype(np.float32)

    def targets(self, rows):
        return (torch.as_tensor(self.tcol_all[rows], device=DEVICE),
                torch.as_tensor(self.mask_all[rows], device=DEVICE))


def _split_items(inst, logp_np):
    return [logp_np[:, inst.offsets[j]:inst.offsets[j + 1]] for j in range(inst.J)]


def _to_probs(inst, logp):
    p = torch.exp(logp).float().cpu().numpy()
    return _split_items(inst, p)


# ---------------------------------------------------------------- ridge

def _pairs(inst, cap=1200):
    p = []
    C = inst.X.shape[1]
    for a in range(C):
        for b in range(a + 1, C):
            if inst.cards[a] * inst.cards[b] <= cap:
                p.append((a, b))
    return p


def _ridge_index(inst, X, pairs):
    """Row of the concatenated weight table each active feature selects."""
    cols = [X[:, c] + off for c, off in enumerate(np.concatenate([[0], np.cumsum(inst.cards)[:-1]]))]
    base = int(inst.cards.sum())
    offs = []
    for a, b in pairs:
        offs.append(base)
        base += int(inst.cards[a] * inst.cards[b])
    for (a, b), off in zip(pairs, offs):
        cols.append(X[:, a] * inst.cards[b] + X[:, b] + off)
    return np.stack(cols, axis=1).astype(np.int64), base


def cand_ridge(inst, tr, evs, budget, sigma_main=2.0, sigma_pair=None, max_iter=120, name="ridge", hurdle=False,
               select=None):
    if not HAVE_TORCH:
        raise RuntimeError("torch unavailable")
    pairs = _pairs(inst) if sigma_pair else []
    tz = inst._torchy
    Xall = inst.X
    idx_all, n_feat = _ridge_index(inst, Xall, pairs)
    n_main = int(inst.cards.sum())
    vis_idx = idx_all[inst.vis]
    hid_idx = idx_all[inst.hid]

    n = len(tr)
    lam_main = 1.0 / (2.0 * n * sigma_main ** 2)
    pen = np.full(n_feat, lam_main, dtype=np.float32)
    if pairs:
        pen[n_main:] = 1.0 / (2.0 * n * sigma_pair ** 2)
    pen_t = torch.as_tensor(pen, device=DEVICE)[:, None]

    W = torch.zeros((n_feat, inst.D), device=DEVICE, requires_grad=True)
    M = None
    if select:
        # Sparsity by selection: an item that depends on two or three columns
        # pays variance for every irrelevant column a dense ridge keeps.
        col_off = np.concatenate([[0], np.cumsum(inst.cards)[:-1]])
        Mn = np.zeros((n_feat, inst.D), dtype=np.float32)
        Xtr = inst.Xv[tr]
        for j in range(inst.J):
            y = inst.Y[tr, j]
            ok = y >= 0
            if not ok.any():
                continue
            mis = [_mutual_information(Xtr[ok, c], y[ok], int(inst.cards[c]), int(inst.K[j]))
                   for c in range(Xtr.shape[1])]
            for c in np.argsort(mis)[::-1][:select]:
                Mn[col_off[c]:col_off[c] + inst.cards[c], inst.offsets[j]:inst.offsets[j + 1]] = 1.0
        M = torch.as_tensor(Mn, device=DEVICE)
    # bias starts at the log marginal, so the first iterate is the baseline
    b0 = _bias0(inst, tr, hurdle)
    b = torch.tensor(b0, device=DEVICE, dtype=torch.float32, requires_grad=True)
    I = torch.as_tensor(vis_idx[tr], device=DEVICE)
    tcol, mask = tz.targets(tr)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=10,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7,
                            tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        logits = F.embedding_bag(I, W if M is None else W * M, mode="sum") + b
        lp = _logp(logits, tz, inst.J, hurdle)
        nll = -(lp.gather(1, tcol) * mask).sum() / n
        loss = nll + (pen_t * W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    res = []
    with torch.no_grad():
        for e in evs:
            Ie = torch.as_tensor(hid_idx if isinstance(e, str) else vis_idx[e], device=DEVICE)
            lp = _logp(F.embedding_bag(Ie, W if M is None else W * M, mode="sum") + b, tz, inst.J, hurdle)
            res.append(_to_probs(inst, lp))
    return {name: [[r[j] for r in res] for j in range(inst.J)]}


# ---------------------------------------------------------------- network

class _Net(nn.Module if HAVE_TORCH else object):
    def __init__(self, cards, n_num, D, hidden=256, drop=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(int(c), int(min(16, max(2, round(1.6 * c ** 0.56)))))
                                   for c in cards])
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.05)
        d_in = sum(e.embedding_dim for e in self.embs) + n_num
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, D)

    def forward(self, x, z):
        h = torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)] + [z], dim=1)
        return self.head(self.body(h))


def cand_net(inst, tr, evs, budget, seeds=(0, 1), hidden=256, drop=0.2, name="net", hurdle=False):
    if not HAVE_TORCH:
        raise RuntimeError("torch unavailable")
    tz = inst._torchy
    rng = np.random.default_rng(SEED + 7)
    perm = rng.permutation(len(tr))
    n_val = max(64, int(0.12 * len(tr)))
    va, fit = tr[perm[:n_val]], tr[perm[n_val:]]
    Xv = torch.as_tensor(inst.Xv, device=DEVICE)
    Zv = torch.as_tensor(inst.Zv, device=DEVICE)
    Xh = torch.as_tensor(inst.Xh, device=DEVICE)
    Zh = torch.as_tensor(inst.Zh, device=DEVICE)
    tcol_f, mask_f = tz.targets(fit)
    tcol_v, mask_v = tz.targets(va)
    fit_t = torch.as_tensor(fit, device=DEVICE)
    va_t = torch.as_tensor(va, device=DEVICE)
    b0 = _bias0(inst, tr, hurdle)

    acc = None
    for s in seeds:
        torch.manual_seed(SEED + 101 * s)
        net = _Net(inst.cards, inst.Z.shape[1], inst.D, hidden, drop).to(DEVICE)
        with torch.no_grad():
            net.head.bias.copy_(torch.as_tensor(b0, dtype=torch.float32))
            net.head.weight.mul_(0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        bs = 256 if len(fit) < 8000 else 512
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(SEED + s)
        for epoch in range(200):
            net.train()
            order = torch.randperm(len(fit), generator=g).to(DEVICE)
            for k in range(0, len(fit), bs):
                bi = order[k:k + bs]
                rows = fit_t[bi]
                lp = _logp(net(Xv[rows], Zv[rows]), tz, inst.J, hurdle)
                loss = -(lp.gather(1, tcol_f[bi]) * mask_f[bi]).sum() / len(bi)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                lp = _logp(net(Xv[va_t], Zv[va_t]), tz, inst.J, hurdle)
                vl = float(-(lp.gather(1, tcol_v) * mask_v).sum() / len(va))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
            if time.time() > budget:
                raise TimeoutError("net over budget")
        net.load_state_dict(best_state)
        net.eval()
        res = []
        with torch.no_grad():
            for e in evs:
                if isinstance(e, str):
                    lp = _logp(net(Xh, Zh), tz, inst.J, hurdle)
                else:
                    et = torch.as_tensor(e, device=DEVICE)
                    lp = _logp(net(Xv[et], Zv[et]), tz, inst.J, hurdle)
                res.append(torch.exp(lp).float().cpu().numpy())
        acc = res if acc is None else [a + r for a, r in zip(acc, res)]
    acc = [a / len(seeds) for a in acc]
    return {name: [[a[:, inst.offsets[j]:inst.offsets[j + 1]] for a in acc] for j in range(inst.J)]}


class _PNet(nn.Module if HAVE_TORCH else object):
    """Shared embeddings of the GIVEN block, then a hidden layer of its own for
    every item. One trunk shared by all items suits a few common latent traits;
    item-specific rules (a lookup on two columns, a threshold on a sum of
    counts) need capacity of their own, and this gives each item that capacity
    in one batched einsum."""

    def __init__(self, cards, n_num, J, Kmax, hidden=24, drop=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(int(c), int(min(16, max(2, round(1.6 * c ** 0.56)))))
                                   for c in cards])
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.05)
        d_in = sum(e.embedding_dim for e in self.embs) + n_num
        self.W1 = nn.Parameter(torch.randn(J, d_in, hidden) / math.sqrt(d_in))
        self.b1 = nn.Parameter(torch.zeros(J, hidden))
        self.W2 = nn.Parameter(torch.randn(J, hidden, Kmax) * (0.1 / math.sqrt(hidden)))
        self.b2 = nn.Parameter(torch.zeros(J, Kmax))
        self.drop = nn.Dropout(drop)

    def forward(self, x, z, flat):
        h = torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)] + [z], dim=1)
        a = self.drop(F.gelu(torch.einsum("nd,jdh->njh", h, self.W1) + self.b1))
        o = torch.einsum("njh,jhk->njk", a, self.W2) + self.b2
        return o.reshape(o.shape[0], -1)[:, flat]


def cand_pnet(inst, tr, evs, budget, seeds=(0,), hidden=24, drop=0.1, name="pnet", hurdle=False):
    if not HAVE_TORCH:
        raise RuntimeError("torch unavailable")
    tz = inst._torchy
    Kmax = int(inst.K.max())
    flat = torch.as_tensor(np.concatenate([j * Kmax + np.arange(inst.K[j]) for j in range(inst.J)]),
                           device=DEVICE)
    rng = np.random.default_rng(SEED + 7)
    perm = rng.permutation(len(tr))
    n_val = max(64, int(0.12 * len(tr)))
    va, fit = tr[perm[:n_val]], tr[perm[n_val:]]
    Xv = torch.as_tensor(inst.Xv, device=DEVICE)
    Zv = torch.as_tensor(inst.Zv, device=DEVICE)
    Xh = torch.as_tensor(inst.Xh, device=DEVICE)
    Zh = torch.as_tensor(inst.Zh, device=DEVICE)
    tcol_f, mask_f = tz.targets(fit)
    tcol_v, mask_v = tz.targets(va)
    fit_t = torch.as_tensor(fit, device=DEVICE)
    va_t = torch.as_tensor(va, device=DEVICE)
    b0 = _bias0(inst, tr, hurdle)
    b2 = np.zeros((inst.J, Kmax), dtype=np.float32)
    for j in range(inst.J):
        b2[j, :inst.K[j]] = b0[inst.offsets[j]:inst.offsets[j + 1]]

    acc = None
    for s in seeds:
        torch.manual_seed(SEED + 101 * s)
        net = _PNet(inst.cards, inst.Z.shape[1], inst.J, Kmax, hidden, drop).to(DEVICE)
        with torch.no_grad():
            net.b2.copy_(torch.as_tensor(b2))
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        bs = 256 if len(fit) < 8000 else 512
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(SEED + s)
        for epoch in range(200):
            net.train()
            order = torch.randperm(len(fit), generator=g).to(DEVICE)
            for k in range(0, len(fit), bs):
                bi = order[k:k + bs]
                rows = fit_t[bi]
                lp = _logp(net(Xv[rows], Zv[rows], flat), tz, inst.J, hurdle)
                loss = -(lp.gather(1, tcol_f[bi]) * mask_f[bi]).sum() / len(bi)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                lp = _logp(net(Xv[va_t], Zv[va_t], flat), tz, inst.J, hurdle)
                vl = float(-(lp.gather(1, tcol_v) * mask_v).sum() / len(va))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
            if time.time() > budget:
                raise TimeoutError("pnet over budget")
        net.load_state_dict(best_state)
        net.eval()
        res = []
        with torch.no_grad():
            for e in evs:
                if isinstance(e, str):
                    lp = _logp(net(Xh, Zh, flat), tz, inst.J, hurdle)
                else:
                    et = torch.as_tensor(e, device=DEVICE)
                    lp = _logp(net(Xv[et], Zv[et], flat), tz, inst.J, hurdle)
                res.append(torch.exp(lp).float().cpu().numpy())
        acc = res if acc is None else [a + r for a, r in zip(acc, res)]
    acc = [a / len(seeds) for a in acc]
    return {name: [[a[:, inst.offsets[j]:inst.offsets[j + 1]] for a in acc] for j in range(inst.J)]}


# ---------------------------------------------------------------- gbdt

def _gbdt_item(Xfit, yfit, Xval, yval, ev_X, K, m, max_iter=200):
    """One item: boosting with early stopping chosen on our own inner split.

    sklearn's built-in early stopping draws a stratified split and fails when an
    answer occurs once; choosing the stopping round from staged predictions on
    a split we control has no such failure mode.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=max_iter, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=np.ones(Xfit.shape[1], dtype=bool),
        early_stopping=False, random_state=SEED)
    clf.fit(Xfit, yfit)
    cls = clf.classes_

    def full(pr):
        out = np.zeros((len(pr), K))
        out[:, cls] = pr
        return 0.995 * out + 0.005 * m          # an option unseen in this fold keeps a sliver

    best_it, best_ll, it = 0, np.inf, 0
    rows = np.arange(len(yval))
    for it, pr in enumerate(clf.staged_predict_proba(Xval), start=1):
        ll = -np.log(full(pr)[rows, yval]).mean()
        if ll < best_ll - 1e-7:
            best_ll, best_it = ll, it
    res = []
    for x in ev_X:
        for it, pr in enumerate(clf.staged_predict_proba(x), start=1):
            if it == best_it:
                break
        res.append(full(pr))
    return res


def cand_gbdt(inst, tr, evs, budget, name="gbdt"):
    Xtr = inst.Xv[tr]
    ev_X = [_rows(inst, e)[0] for e in evs]
    rng = np.random.default_rng(SEED + 3)
    perm = rng.permutation(len(tr))
    is_val = np.zeros(len(tr), dtype=bool)
    is_val[perm[:max(50, int(0.12 * len(tr)))]] = True
    out = []
    t_first = time.time()
    for j in range(inst.J):
        if j == 3 and not getattr(inst, "_gbdt_checked", False):
            # Pilot: boosting fits one tree per class, so cost grows with the
            # option count; scale the first three items' time by it, and never
            # let boosting take more than a third of what this instrument has left.
            inst._gbdt_checked = True
            per_class = (time.time() - t_first) / max(1, int(inst.K[:3].sum()))
            projected = per_class * int(inst.K.sum()) * N_FOLDS
            if projected > (budget - time.time()) / 3:
                raise TimeoutError("projected %.0fs for gbdt" % projected)
        K = int(inst.K[j])
        y = inst.Y[tr, j]
        m = marginal_vector(y, K)
        ok = y >= 0
        fit_rows, val_rows = ok & ~is_val, ok & is_val
        try:
            if len(np.unique(y[fit_rows])) < 2 or val_rows.sum() < 10:
                raise ValueError("degenerate item")
            res = _gbdt_item(Xtr[fit_rows], y[fit_rows], Xtr[val_rows], y[val_rows], ev_X, K, m)
        except Exception:
            res = [np.tile(m, (len(x), 1)) for x in ev_X]    # this item only falls back
        out.append(res)
        if time.time() > budget:
            raise TimeoutError("gbdt over budget")
    return {name: out}


def ordinal_variant(inst, probs):
    """Spread each answer's probability onto its neighbours in option order.

    Survey options are usually ordered (Likert scales, wage bands, age bands);
    for those, neighbouring options should get neighbouring probabilities. The
    sentinel slot is left alone. Applied to a candidate's predictions it makes a
    new candidate, and the stacker decides per item whether it helps.
    """
    out = []
    for j, p in enumerate(probs):
        K = int(inst.K[j])
        k_real = K - 1 if inst.gated[j] else K
        if k_real < 4:
            out.append(p)
            continue
        r = p[:, :k_real]
        left = np.concatenate([r[:, :1], r[:, :-1]], axis=1)
        right = np.concatenate([r[:, 1:], r[:, -1:]], axis=1)
        s = 0.5 * r + 0.25 * left + 0.25 * right
        s *= r.sum(1, keepdims=True) / np.maximum(s.sum(1, keepdims=True), 1e-300)
        out.append(np.concatenate([s, p[:, k_real:]], axis=1))
    return out


# =============================================================================
# Stacking, calibration and the floor-aware report
# =============================================================================

def _em_weights(L, iters=300):
    """Mixture weights maximising sum_i log sum_m w_m L[i, m] (concave in w)."""
    M = L.shape[1]
    w = np.full(M, 1.0 / M)
    L = np.maximum(L, 1e-300)
    prev = -np.inf
    for _ in range(iters):
        mix = L @ w
        ll = np.log(mix).sum()
        w = w * (L / mix[:, None]).mean(0)
        w = np.maximum(w, 0)
        w /= w.sum()
        if ll - prev < 1e-9 * max(1.0, abs(ll)):
            break
        prev = ll
    return w


def _temper(P, t):
    lp = np.log(np.maximum(P, 1e-300)) / t
    lp -= lp.max(1, keepdims=True)
    q = np.exp(lp)
    return q / q.sum(1, keepdims=True)


def _fit_temperature(P, y):
    """1-D concave fit of a temperature on held-out log likelihood."""
    rows = np.arange(len(y))

    def ll(t):
        return np.log(np.maximum(_temper(P, t)[rows, y], 1e-300)).sum()
    lo, hi = 0.6, 1.6
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = ll(a), ll(b)
    for _ in range(30):
        if fa > fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = ll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = ll(b)
    t = (lo + hi) / 2
    return t if ll(t) > ll(1.0) else 1.0


def floor_optimal(B, eps=FLOOR):
    """The report q whose floored version is the best response to belief B.

    The grader scores p = eps + (1 - K eps) q / sum(q). Maximising
    sum_k B_k log p_k over that set gives p_k = max(eps, lam B_k) with lam set
    so p sums to one (water-filling); q is p mapped back through the floor.
    """
    n, K = B.shape
    if K == 1:
        return np.ones_like(B)
    P = np.maximum(B, 0)
    P = P / P.sum(1, keepdims=True)
    active = np.ones_like(P, dtype=bool)
    for _ in range(K + 1):
        s_act = np.where(active, P, 0).sum(1, keepdims=True)
        n_inact = (~active).sum(1, keepdims=True)
        lam = (1.0 - eps * n_inact) / np.maximum(s_act, 1e-300)
        new_active = lam * P > eps
        if (new_active == active).all():
            break
        active = new_active
    p = np.where(active, lam * P, eps)
    p = p / p.sum(1, keepdims=True)
    q = np.maximum(p - eps, 0) / (1.0 - K * eps)
    bad = q.sum(1) <= 0
    if bad.any():
        q[bad] = 1.0 / K
    return q


def graded(Q, y, eps=FLOOR):
    """Mean log score the grader would give these reports."""
    K = Q.shape[1]
    Q = Q / Q.sum(1, keepdims=True)
    p = eps + (1 - K * eps) * Q[np.arange(len(y)), y]
    return np.log(p)


# =============================================================================
# Orchestration
# =============================================================================

REFIT = {"marginal", "backoff", "ridge", "ridge_pairs", "ridge_pairs_w", "ridge_h", "ridge_pairs_h",
         "ridge_sel", "ridge_sel_h", "knn"}
REFIT_BLEND = {"net", "net_wide", "net_narrow", "net_h", "pnet_h"}


def _plan(inst):
    """Candidates in the order they run, with rough cost weights."""
    plan = [("marginal", cand_marginal, {}),
            ("backoff", cand_backoff, {"depth": 4, "hurdle": True})]
    if HAVE_TORCH:
        plan += [("knn", cand_knn, {})] if USE_DONORS else []
        plan += [("ridge", cand_ridge, {"sigma_main": 2.0, "sigma_pair": None, "name": "ridge_main"}),
                 ("ridge_h", cand_ridge, {"sigma_main": 2.0, "sigma_pair": None, "name": "ridge_main_h", "hurdle": True}),
                 ("ridge_sel", cand_ridge, {"sigma_main": 2.0, "sigma_pair": None, "name": "ridge_sel", "select": 4}),
                 ("ridge_sel_h", cand_ridge, {"sigma_main": 2.0, "sigma_pair": None, "name": "ridge_sel_h", "hurdle": True, "select": 4}),
                 ("ridge_pairs", cand_ridge, {"sigma_main": 2.0, "sigma_pair": 0.5, "name": "ridge_pairs"}),
                 ("ridge_pairs_h", cand_ridge, {"sigma_main": 2.0, "sigma_pair": 0.5, "name": "ridge_pairs_h", "hurdle": True}),
                 ("net", cand_net, {"seeds": (0, 1, 2, 5), "name": "net"}),
                 ("net_h", cand_net, {"seeds": (10, 11, 12), "name": "net_h", "hurdle": True}),
                 ("pnet_h", cand_pnet, {"seeds": (20, 21, 22), "hidden": 24, "drop": 0.1, "name": "pnet_h", "hurdle": True}),
                 ("ridge_pairs_w", cand_ridge, {"sigma_main": 3.0, "sigma_pair": 1.0, "name": "ridge_pairs_w"}),
                 ("net_wide", cand_net, {"seeds": (3, 4, 6), "hidden": 512, "drop": 0.3, "name": "net_wide"}),
                 ("net_narrow", cand_net, {"seeds": (7, 8, 9), "hidden": 64, "drop": 0.1, "name": "net_narrow"})]
    plan += [("gbdt", cand_gbdt, {"name": "gbdt"})]
    return plan


def _run_candidates(inst, deadline):
    rng = np.random.default_rng(SEED)
    n = len(inst.vis)
    fold = rng.permutation(n) % N_FOLDS
    folds = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(N_FOLDS)]
    if HAVE_TORCH:
        inst._torchy = _Torchy(inst)

    oof, test, times = {}, {}, {}
    for label, fn, kw in _plan(inst):
        t0 = time.time()
        if t0 > deadline:
            log("  skip %-11s (budget spent)" % label)
            continue
        try:
            got_oof, got_test = None, None
            for f, (tr, va) in enumerate(folds):
                res = fn(inst, tr, [va, "hid"], deadline, **kw)
                if got_oof is None:
                    got_oof = {k: [np.zeros((n, inst.K[j])) for j in range(inst.J)] for k in res}
                    got_test = {k: [np.zeros((len(inst.hid), inst.K[j])) for j in range(inst.J)] for k in res}
                for k, per_item in res.items():
                    for j, (p_va, p_h) in enumerate(per_item):
                        got_oof[k][j][va] = p_va
                        got_test[k][j] += p_h / N_FOLDS
                if f == 0:
                    est = (time.time() - t0) * N_FOLDS
                    if t0 + est > deadline:
                        raise TimeoutError("projected %.0fs exceeds the budget" % est)
                if time.time() > deadline:
                    raise TimeoutError("over budget")
            if (label in REFIT or label in REFIT_BLEND) and time.time() < deadline:
                # Super Learner practice: weights come from the folds, but the
                # held-out rows are best predicted by models that saw every
                # visible row. Deterministic candidates are simply refitted; a
                # network refitted once is blended 50/50 with its five-fold bag,
                # keeping the bag's variance reduction and adding the data.
                try:
                    full = fn(inst, np.arange(n), ["hid"], deadline, **kw)
                    for k, per_item in full.items():
                        if label in REFIT:
                            got_test[k] = [per_item[j][0] for j in range(inst.J)]
                        else:
                            got_test[k] = [0.5 * got_test[k][j] + 0.5 * per_item[j][0]
                                           for j in range(inst.J)]
                except Exception as exc:
                    log("  refit of %s skipped (%s)" % (label, type(exc).__name__))
            for k in got_oof:
                oof[k], test[k] = got_oof[k], got_test[k]
            times[label] = time.time() - t0
            log("  %-11s %6.1fs  -> %s" % (label, times[label], ", ".join(got_oof)))
        except Exception as exc:                      # a candidate may fail; the stack goes on
            log("  drop %-11s (%s: %s)" % (label, type(exc).__name__, str(exc)[:120]))

    # ordinal-smoothed twins of the strongest candidates (cheap, post hoc)
    for k in ("net", "net_wide", "ridge_main", "ridge_pairs"):
        if k in oof and inst.K.max() >= 4:
            oof[k + "_ord"] = ordinal_variant(inst, oof[k])
            test[k + "_ord"] = ordinal_variant(inst, test[k])
    return oof, test


DONOR_CONFIGS = tuple((lam, dm) for dm in (2, 4) for lam in (0.3, 0.7, 1.5, 4.0))   # (lambda, largest distance)
CROSS_CONFIGS = tuple((lam, dm) for dm in (2, 4) for lam in (0.3, 1.5, 4.0))        # donors from the other round only
DONOR_ALPHAS = (0.002, 0.006, 0.02, 0.05, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0)
ROUND_NAMES = ("wave", "round", "survey_round", "survey_wave")


def _round_column(inst):
    """Index of a GIVEN column that labels the survey round, or None. One row is
    one household per round, so a household seen in both rounds has its other
    interview among the donors of the OTHER round, never its own."""
    for c, g in enumerate(inst.given):
        if str(g).strip().lower() in ROUND_NAMES and 2 <= int(inst.cards[c]) - 1 <= 6:
            return c
    return None


def _donor_votes(inst, configs=DONOR_CONFIGS, k=64):
    """Hot-deck donor votes: for every visible row, from the visible rows of the
    other folds; for every held-out row, from all visible rows. A donor is a
    visible respondent whose GIVEN block differs in at most d_max columns (2 or 4)
    (Hamming distance, one matrix product of one-hot codes); it votes for its
    own answer with weight exp(-lam * distance). When the instrument has a survey
    round column, a second set of configurations draws donors from the other
    round only, so a same-round lookalike cannot outvote a household's own
    interview in the other round."""
    n = len(inst.vis)
    fold = np.random.default_rng(SEED).permutation(n) % N_FOLDS
    dev = DEVICE
    C = inst.X.shape[1]
    col_off = np.concatenate([[0], np.cumsum(inst.cards)[:-1]]).astype(np.int64)
    n_oh = int(inst.cards.sum())
    rc = _round_column(inst)
    passes = [(False, configs)] + ([(True, CROSS_CONFIGS)] if rc is not None else [])
    n_cfg = sum(len(c) for _, c in passes)

    def onehot(X):
        H = torch.zeros((len(X), n_oh), device=dev)
        H.scatter_(1, torch.as_tensor(X + col_off[None, :], device=dev), 1.0)
        return H

    Yall = torch.as_tensor(inst.Y, device=dev)
    valid_all = Yall >= 0
    slot_all = torch.where(valid_all, Yall + torch.as_tensor(inst.offsets[:-1], device=dev)[None, :], 0)

    def votes(eval_X, tr_rows):
        tr_t = torch.as_tensor(tr_rows, device=dev)
        Htr = onehot(inst.Xv[tr_rows])
        slot, valid = slot_all[tr_t], valid_all[tr_t]
        kk = int(min(k, len(tr_rows)))
        cnts = [[] for _ in range(n_cfg)]
        tots = [[] for _ in range(n_cfg)]
        wave_tr = torch.as_tensor(inst.Xv[tr_rows, rc], device=dev) if rc is not None else None
        for a in range(0, len(eval_X), 2048):
            He = onehot(eval_X[a:a + 2048])
            dist = C - He @ Htr.T
            li = 0
            for cross, cfgs in passes:
                if cross:   # same-round donors are pushed out of reach
                    we = torch.as_tensor(eval_X[a:a + 2048, rc], device=dev)
                    dd = dist + 1000.0 * (we[:, None] == wave_tr[None, :]).float()
                else:
                    dd = dist
                d, idx = torch.topk(-dd, kk, dim=1)
                d = -d
                ys = slot[idx]
                vs = valid[idx].float()
                for lam, d_max in cfgs:
                    wv = (torch.exp(-lam * d.clamp_max(50.0)) * (d <= d_max).float()).unsqueeze(2) * vs
                    cnt = torch.zeros((len(He), inst.D), device=dev)
                    cnt.scatter_add_(1, ys.reshape(len(He), -1), wv.reshape(len(He), -1))
                    cnts[li].append(cnt.cpu().numpy())
                    tots[li].append(wv.sum(1).cpu().numpy())
                    li += 1
        return [np.concatenate(c) for c in cnts], [np.concatenate(t) for t in tots]

    oof_cnt = [np.zeros((n, inst.D), np.float32) for _ in range(n_cfg)]
    oof_tot = [np.zeros((n, inst.J), np.float32) for _ in range(n_cfg)]
    for f in range(N_FOLDS):
        va, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        c, t = votes(inst.Xv[va], tr)
        for li in range(n_cfg):
            oof_cnt[li][va], oof_tot[li][va] = c[li], t[li]
    hid_cnt, hid_tot = votes(inst.Xh, np.arange(n))
    return oof_cnt, oof_tot, hid_cnt, hid_tot


def _em_batched(L, mask, prior=None, c=0.0, iters=500):
    """Mixture weights for many items at once, optionally shrunk toward a prior.

    L[g, i, m]: probability candidate m gave the true answer of row i in group g;
    mask[g, i]: whether that row counts. With a prior w0 and strength c this is
    the MAP-EM update w_m <- (sum_i r_im + c w0_m) / (n_g + c), the posterior
    mode under a Dirichlet prior worth c rows centred on w0.
    """
    if HAVE_TORCH:
        Lt = torch.as_tensor(L, device=DEVICE, dtype=torch.float64).clamp_min(1e-300)
        mt = torch.as_tensor(mask, device=DEVICE, dtype=torch.float64)
        G, n, M = Lt.shape
        w = torch.full((G, M), 1.0 / M, device=DEVICE, dtype=torch.float64)
        ng = mt.sum(1, keepdim=True)
        pr = None if prior is None else torch.as_tensor(prior, device=DEVICE, dtype=torch.float64)[None, :]
        for _ in range(iters):
            mix = torch.einsum("gnm,gm->gn", Lt, w).clamp_min(1e-300)
            s = torch.einsum("gnm,gn->gm", Lt, mt / mix) * w
            new = s / ng.clamp_min(1.0) if (pr is None or c == 0) else (s + c * pr) / (ng + c)
            new = torch.where(ng > 0, new, w)
            new = new / new.sum(1, keepdim=True)
            done = float((new - w).abs().max()) < 1e-8
            w = new
            if done:
                break
        return w.cpu().numpy()
    out = []
    for g in range(L.shape[0]):
        keep = mask[g] > 0
        out.append(_em_weights(L[g][keep]) if keep.any() else np.full(L.shape[2], 1.0 / L.shape[2]))
    return np.stack(out)


def _stack(inst, oof, test, donors=None):
    names = [k for k in oof if k != "marginal"] + ["marginal"]
    M, J, n = len(names), inst.J, len(inst.vis)
    final, summary = [], {"model_ll": 0.0, "marginal_ll": 0.0, "n": 0}
    weight_report = {k: 0.0 for k in names}

    # ---- mixture weights: pooled over items, then shrunk per item ---------
    valid = inst.Y >= 0
    L = np.ones((J, n, M), dtype=np.float64)
    for j in range(J):
        rows = np.where(valid[:, j])[0]
        if len(rows):
            yo = inst.Y[rows, j]
            for m, k in enumerate(names):
                L[j, rows, m] = oof[k][j][rows, yo]
    mask = valid.T.astype(np.float64)
    w_pool = _em_batched(L.reshape(1, J * n, M), mask.reshape(1, -1))[0]
    half = np.random.default_rng(SEED + 11).random(n) < 0.5
    best_c, best_ll = None, -np.inf
    for c in (0.0, 25.0, 100.0, 400.0, None):            # None: the pooled weights only
        ll = 0.0
        for a, b in ((half, ~half), (~half, half)):
            W = (np.tile(w_pool, (J, 1)) if c is None
                 else _em_batched(L[:, a], mask[:, a], prior=w_pool, c=c))
            mix = np.einsum("jnm,jm->jn", L[:, b], W)
            ll += float((np.log(np.maximum(mix, 1e-300)) * mask[:, b]).sum())
        if ll > best_ll:
            best_c, best_ll = c, ll
    W_all = (np.tile(w_pool, (J, 1)) if best_c is None
             else _em_batched(L, mask, prior=w_pool, c=best_c))
    log("  stacking: shrinkage toward pooled weights c=%s" % ("pooled" if best_c is None else "%g" % best_c))
    del L

    for j in range(inst.J):
        y = inst.Y[:, j]
        ok = y >= 0
        K = int(inst.K[j])
        if ok.sum() == 0:
            final.append(np.full((len(inst.hid), K), 1.0 / K))
            continue
        yo = y[ok]
        w = W_all[j]
        for k, wk in zip(names, w):
            weight_report[k] += wk / inst.J
        B_oof = sum(wk * oof[k][j][ok] for k, wk in zip(names, w))
        B_hid = sum(wk * test[k][j] for k, wk in zip(names, w))
        t = _fit_temperature(B_oof, yo) if K > 1 else 1.0
        B_oof, B_hid = _temper(B_oof, t), _temper(B_hid, t)
        if donors is not None and K > 1:
            # Hot-deck correction: near-identical donors pull the stacked belief
            # toward their own answers, P = (votes + alpha * B) / (weight + alpha).
            # Rows without a close donor keep B exactly. (lam, alpha) per item
            # by out-of-fold likelihood; left off when it does not help.
            oc, ot, hc, ht = donors
            sl = slice(int(inst.offsets[j]), int(inst.offsets[j + 1]))
            best_ll, cfg = graded(B_oof, yo).sum(), None
            for li in range(len(oc)):
                cnt_o, tot_o = oc[li][ok][:, sl], ot[li][ok][:, j:j + 1]
                if tot_o.max() <= 0:
                    continue
                for alpha in DONOR_ALPHAS:
                    ll = graded((cnt_o + alpha * B_oof) / (tot_o + alpha), yo).sum()
                    if ll > best_ll + 1e-6:
                        best_ll, cfg = ll, (li, alpha)
            if cfg is not None:
                li, alpha = cfg
                B_oof = (oc[li][ok][:, sl] + alpha * B_oof) / (ot[li][ok][:, j:j + 1] + alpha)
                B_hid = (hc[li][:, sl] + alpha * B_hid) / (ht[li][:, j:j + 1] + alpha)
                summary["donor_items"] = summary.get("donor_items", 0) + 1
                if li >= len(DONOR_CONFIGS):
                    summary["cross_items"] = summary.get("cross_items", 0) + 1
        plain = graded(B_oof, yo).sum()
        water = graded(floor_optimal(B_oof), yo).sum()
        if water > plain:
            Q = floor_optimal(B_hid)
            best = water
        else:
            Q = B_hid
            best = plain
        final.append(Q)
        summary["model_ll"] += best
        summary["marginal_ll"] += graded(oof["marginal"][j][ok], yo).sum()
        summary["n"] += len(yo)
    if summary["n"]:
        skill = 1 - summary["model_ll"] / summary["marginal_ll"]
        log("  OOF skill estimate %.4f   donor correction on %d of %d items (%d from the other round)   mean weights: %s" % (
            skill, summary.get("donor_items", 0), inst.J, summary.get("cross_items", 0),
            ", ".join("%s %.2f" % (k, v) for k, v in weight_report.items() if v > 0.005)))
    return final


def predict(frame, schema):
    t_start = time.time()
    inst = Instrument(frame, schema)
    if len(inst.hid) == 0 or inst.J == 0:
        return []
    log("instrument %s: %d visible, %d held out, %d items, %d GIVEN(+derived) cols, device %s" % (
        inst.kind, len(inst.vis), len(inst.hid), inst.J, inst.X.shape[1], DEVICE))

    # ---- time governor --------------------------------------------------
    # Each instrument plans on its share of the total; time a finished
    # instrument did not use carries over; instruments still to come keep at
    # least half of their share in reserve whatever happens here.
    remaining_total = TOTAL_BUDGET - (time.time() - T_IMPORT)
    pending = [k for k in ("unhcr", "world_bank", "unicef")
               if k not in STATE["done"] and k != inst.kind]
    reserve = 0.5 * TOTAL_BUDGET * sum(SHARE[k] for k in pending)
    carry = max(0.0, TOTAL_BUDGET * sum(SHARE.get(k, 0.3) for k in STATE["done"]) - STATE["spent"])
    mine = SHARE.get(inst.kind, 0.3) * TOTAL_BUDGET + carry
    mine = max(30.0, min(mine, remaining_total - reserve))
    deadline = t_start + mine
    log("  budget %.0fs (total remaining %.0fs)" % (mine, remaining_total))

    marginal = [marginal_vector(inst.Y[:, j], inst.K[j]) for j in range(inst.J)]
    final = None
    try:
        oof, test = _run_candidates(inst, deadline)
        donors = None
        if HAVE_TORCH and USE_DONORS:
            try:
                t0 = time.time()
                donors = _donor_votes(inst)
                log("  donor votes   %6.1fs" % (time.time() - t0))
            except Exception as exc:
                log("  donor votes skipped (%s: %s)" % (type(exc).__name__, str(exc)[:120]))
        if "marginal" in oof and len(inst.hid):
            try:
                final = _stack(inst, oof, test, donors)
            except Exception as exc:
                if donors is None:
                    raise
                log("  stack with donors failed (%s); stacking without" % type(exc).__name__)
                final = _stack(inst, oof, test)
    except Exception as exc:
        log("  stack failed (%s: %s); returning the marginal" % (type(exc).__name__, exc))
        final = None

    out = _emit(inst, final, marginal)
    STATE["done"][inst.kind] = time.time() - t_start
    STATE["spent"] += time.time() - t_start
    log("  done in %.1fs" % (time.time() - t_start))
    return out


def _emit(inst, final, marginal):
    """One vector per blank cell, rows top to bottom, items in schema order."""
    hid_pos = np.full(inst.n_rows, -1, dtype=np.int64)
    hid_pos[inst.hid] = np.arange(len(inst.hid))
    out = []
    for r in inst.hid:
        i = hid_pos[r]
        for j in range(inst.J):
            if not inst.nan[r, j]:
                continue
            v = None
            if final is not None:
                v = np.asarray(final[j][i], dtype=float)
                if v.shape != (inst.K[j],) or not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
                    v = None
            out.append(v if v is not None else marginal[j])
    return out
