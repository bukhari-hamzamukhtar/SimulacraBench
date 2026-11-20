"""A harder sandbox: answers that really do follow from the GIVEN block.

    python stress.py SUBMISSION_DIR|oracle INSTRUMENT [SEED]

The official sandbox ties every answer to one hidden trait, so its ceiling is
low (about 0.3 to 0.4) and every method looks the same near it. The real UNHCR
leaderboard sits near 0.88, so the real data carry far more structure. This
generator keeps the sandbox's GIVEN columns and replaces every PREDICT column
with answers driven by the GIVEN block in four ways:

  A  near-deterministic in one or two GIVEN columns (a lookup with a little noise)
  B  additive effects of two or three GIVEN columns (a logistic model)
  C  thresholds on a sum of count columns (household-size style)
  D  noise only (the crowd marginal is the truth)

Gates follow the schema, and for UNHCR they hold only approximately (the
schema says agreement has a median of 0.83). Every item is drawn independently
given the GIVEN block and its gate parent, so P(answer | GIVEN) is exact by
marginalising over the parent, and the oracle below is the true ceiling.

Scoring is the grader's own: sample_rows, check, floored, marginal_reference
and score, after the same JSON round trip the hosted driver applies.
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(HERE, "public")
sys.path.insert(0, PUBLIC)

import score as S                                                       # noqa: E402
from make_sandbox import (ROLE_COLUMN, assign_roles, generated_items,  # noqa: E402
                          load_config, load_schema, make_sandbox, options_for,
                          _generation_order)

NUM = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def stress_frame(schema, config, seed, soft):
    base = make_sandbox(schema, config, seed=seed)
    rng = np.random.default_rng([seed, 4242])
    recs = schema["items"]
    items = generated_items(schema)
    given = [k for k in items if recs[k]["class"] == "GIVEN"]
    n = len(base)

    codes, cards, dist = {}, {}, {}
    for k in given:
        opts = options_for(schema, k)
        idx = {v: i for i, v in enumerate(opts)}
        codes[k] = np.array([idx[v] for v in base[k]])
        cards[k] = len(opts)
        dist[k] = np.eye(len(opts))[codes[k]]

    numeric = []
    for k in given:
        vals = recs[k]["values"]
        hits = [NUM.match(str(v)) for v in vals]
        if len(vals) >= 3 and sum(h is not None for h in hits) >= 0.7 * len(vals):
            lv = np.array([float(h.group(1)) if h else 0.0 for h in hits] + [0.0] * (cards[k] - len(vals)))
            numeric.append((k, lv))

    kinds = {}
    frame = base.copy()
    for name in _generation_order(schema, items):
        rec = recs[name]
        if rec["class"] != "PREDICT":
            continue
        values = list(rec["values"])
        K = len(values)
        logits = np.tile(np.log(rng.dirichlet(np.ones(K)) + 1e-9), (n, 1))
        u = rng.random()
        if u < 0.30:
            kind = "A"
            cols = list(rng.choice(given, size=min(len(given), int(rng.integers(1, 3))), replace=False))
            key = np.zeros(n, dtype=np.int64)
            for c in cols:
                key = key * cards[c] + codes[c]
            table = rng.integers(K, size=int(key.max()) + 1)
            logits += rng.uniform(3, 7) * np.eye(K)[table[key]]
        elif u < 0.65:
            kind = "B"
            cols = list(rng.choice(given, size=min(len(given), int(rng.integers(2, 4))), replace=False))
            s = rng.uniform(0.5, 2.0)
            for c in cols:
                logits += rng.normal(scale=s, size=(cards[c], K))[codes[c]]
        elif u < 0.80 and len(numeric) >= 2:
            kind = "C"
            pick = rng.choice(len(numeric), size=min(len(numeric), int(rng.integers(2, 5))), replace=False)
            t = sum(numeric[i][1][codes[numeric[i][0]]] for i in pick)
            cuts = np.unique(np.quantile(t, rng.uniform(0.2, 0.8, size=min(3, K - 1))))
            bins = np.searchsorted(cuts, t, side="right")
            logits += rng.uniform(2, 5) * np.eye(K)[rng.integers(K, size=len(cuts) + 1)[bins]]
        else:
            kind = "D"
        q = _softmax(logits)

        gate = rec.get("gate")
        if gate:
            a = rng.uniform(0.90, 0.99) if soft else 1.0
            popts = options_for(schema, gate["parent"])
            obs = np.isin(np.asarray(popts, dtype=object), np.asarray(gate["observed_if"], dtype=object))
            p_open = dist[gate["parent"]][:, obs].sum(axis=1)
            p_real = p_open * a + (1 - p_open) * (1 - a)
            dist[name] = np.concatenate([p_real[:, None] * q, (1 - p_real)[:, None]], axis=1)
            parent_vals = np.asarray(frame[gate["parent"]], dtype=object)
            opened = np.isin(parent_vals, np.asarray(gate["observed_if"], dtype=object))
            asked = np.where(opened, rng.random(n) < a, rng.random(n) < 1 - a)
        else:
            dist[name] = q
            asked = np.ones(n, dtype=bool)
        chosen = (q.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        vals = np.asarray(values, dtype=object)[chosen]
        if gate:
            vals = np.where(asked, vals, schema["gated_value"])
        frame[name] = vals
        kinds[name] = kind + ("g" if gate else "")
    return frame.astype(object), dist, kinds


def panel_frame(schema, config, seed, frac=0.4, copy_p=0.7, flip=0.05, lookalike=0.0, respondent=0.0):
    """The official sandbox, turned into a two-round panel: a share of rows are
    second-round re-interviews of other households. Each copies its household's
    GIVEN block (round set to 2, each column changed with probability `flip`)
    and carries each answer over with probability `copy_p`, otherwise keeping a
    fresh answer of its own. With `lookalike` > 0, that share of first-round
    households first takes the GIVEN block of another first-round household
    (keeping its own answers), so identical same-round profiles are common, as
    they are in real data. With `respondent` > 0, that share of re-interviews
    is answered by a different household member, whose own columns (relation,
    age, education, marital status) differ. There is no exact oracle here; it
    measures whether a method can use near-identical donors when they exist."""
    base = make_sandbox(schema, config, seed=seed)
    rng = np.random.default_rng([seed, 9191])
    recs = schema["items"]
    items = generated_items(schema)
    given = [k for k in items if recs[k]["class"] == "GIVEN"]
    pred = [k for k in items if recs[k]["class"] == "PREDICT"]
    n = len(base)
    second = rng.choice(n, size=int(frac * n), replace=False)
    pool = np.setdiff1d(np.arange(n), second)
    src = rng.choice(pool, size=len(second), replace=True)
    out = base.copy()
    if lookalike > 0:   # its own generator, so the default world is unchanged
        rng2 = np.random.default_rng([seed, 7373])
        la = pool[rng2.random(len(pool)) < lookalike]
        la_src = rng2.choice(pool, size=len(la), replace=True)
        for g in given:
            col = out[g].to_numpy(dtype=object).copy()
            col[la] = col[la_src]
            out[g] = col
    for g in given:
        col = out[g].to_numpy(dtype=object).copy()
        col[second] = col[src]
        opts = np.asarray(options_for(schema, g), dtype=object)
        ch = rng.random(len(second)) < flip
        col[second[ch]] = opts[rng.integers(len(opts), size=int(ch.sum()))]
        out[g] = col
    if "wave" in given:
        w = np.full(n, "1", dtype=object)
        w[second] = "2"
        out["wave"] = w
    if respondent > 0:  # a different member answers: their own columns change together
        rng3 = np.random.default_rng([seed, 5151])
        other = second[rng3.random(len(second)) < respondent]
        donor = rng3.choice(pool, size=len(other), replace=True)
        for g in [g for g in RESPONDENT_COLUMNS if g in given]:
            col = out[g].to_numpy(dtype=object).copy()
            col[other] = col[donor]
            out[g] = col
    for t in pred:
        col = out[t].to_numpy(dtype=object).copy()
        cp = rng.random(len(second)) < copy_p
        col[second[cp]] = col[src[cp]]
        out[t] = col
    return out.astype(object), None, {t: "P" for t in pred}


# UNHCR's columns that describe the person answering rather than the household
RESPONDENT_COLUMNS = ("s2_1_hh_relationship_lbl", "s2_1b_age", "s2_1c_education_level_lbl",
                      "s2_1d_marital_status_lbl")


def main():
    sub, inst = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.chdir(PUBLIC)
    config = load_config("config.yml")
    schema = load_schema("data/%s.json" % inst, config)
    if os.environ.get("STRESS_MODE") == "panel":
        frame_all, dist, kinds = panel_frame(schema, config, seed,
                                             copy_p=float(os.environ.get("PANEL_COPY", "0.7")),
                                             flip=float(os.environ.get("PANEL_FLIP", "0.05")),
                                             lookalike=float(os.environ.get("PANEL_LOOK", "0")),
                                             respondent=float(os.environ.get("PANEL_RESP", "0")))
        assert sub != "oracle", "the panel world has no exact oracle"
    else:
        frame_all, dist, kinds = stress_frame(schema, config, seed, soft=(inst == "unhcr"))
    respondents = assign_roles(schema, frame_all)
    respondents = respondents[["respondent_id", ROLE_COLUMN] + generated_items(schema)]
    frame, cells, truth = S.sample_rows(schema, respondents, 2)
    floor = config["scoring"]["floor"]
    ref = S.marginal_reference(schema, frame, cells, truth, floor)
    n_vis = schema["split"]["n_train"] + schema["split"]["n_dev"]

    t_pred = 0.0
    if sub == "oracle":
        vectors = [list(dist[item][row]) for row, _, item in cells]
    else:
        spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.abspath(os.path.join(HERE, sub)), "main.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        only = os.environ.get("SB_ONLY")
        if only and hasattr(mod, "_plan"):
            keep = set(only.split(","))
            orig = mod._plan
            mod._plan = lambda inst, _o=orig: [p for p in _o(inst) if p[0] in keep]
        staged = json.loads(json.dumps({"columns": list(frame.columns),
                                        "data": [[None if pd.isna(v) else v for v in row]
                                                 for row in frame.to_numpy(dtype=object)]}))
        staged_frame = pd.DataFrame(staged["data"], columns=staged["columns"])
        staged_schema = json.loads(json.dumps(schema, ensure_ascii=False))
        t = time.time()
        vectors = mod.predict(staged_frame, staged_schema)
        t_pred = time.time() - t
    vectors = [[float(x) for x in v] for v in vectors]
    S.check(schema, vectors, cells)
    fl = S.floored(vectors, schema, cells, floor)
    res = S.score(schema, config, fl, truth, cells, ref)

    per = {}
    for (row, _, item), v, y in zip(cells, fl, truth):
        opts = options_for(schema, item)
        per.setdefault(item, []).append(np.log(v[opts.index(y)]))
    out = os.path.join(HERE, ("_panel" if os.environ.get("STRESS_MODE") == "panel" else "_stress") + "_%s%s_%s_s%d.json" % (os.path.basename(os.path.normpath(sub)), ("_" + os.environ["SB_ONLY"].replace(",", "+")) if os.environ.get("SB_ONLY") else "", inst, seed))
    with open(out, "w") as fh:
        json.dump({"skill": res["skill"], "kinds": kinds,
                   "per_item": {k: float(np.mean(v)) for k, v in per.items()}}, fh)
    from collections import Counter
    print("KINDS", dict(Counter(kinds.values())))
    mode = os.environ.get("STRESS_MODE", "stress")
    if mode == "panel":
        mode += "(copy %s flip %s look %s resp %s)" % (os.environ.get("PANEL_COPY", "0.7"), os.environ.get("PANEL_FLIP", "0.05"),
                                                       os.environ.get("PANEL_LOOK", "0"), os.environ.get("PANEL_RESP", "0"))
    if os.environ.get("SB_DONORS") == "0":
        mode += " donors-off"
    print(("RESULT %-10s %-28s " + mode + " seed %d  skill %.4f  (log %.4f, ref %.4f, se %.4f)  predict %.1fs  cells %d")
          % (inst, (os.path.basename(os.path.normpath(sub)) + " " + os.environ.get("SB_ONLY", "")).strip(), seed, res["skill"], res["log_score"], ref, res["std_error"], t_pred, res["n_cells"]))


if __name__ == "__main__":
    main()
