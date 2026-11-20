"""Bayes-optimal predictor for the sandbox, built from the generator's own parameters.

The sandbox draws one hidden trait t ~ N(0,1) per respondent and every answer
from softmax(base_j + t * load_j), with skip logic on top. Replaying the
generator's random stream recovers base_j and load_j exactly. The best possible
forecast for a held-out cell is then

    P(y_j | GIVEN block) = integral over t of p(t | GIVEN block) P(y_j | t, gates)

computed on a fine grid of t. No model can beat it in expectation, so its score
is the ceiling of the sandbox. Only the visible GIVEN block of each held-out row
is read; nothing from the answer key.
"""
import os
import sys

import numpy as np

SEED = int(os.environ.get("ORACLE_SEED", "0"))
HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(os.path.dirname(HERE), "public")
sys.path.insert(0, PUBLIC)
from make_sandbox import STREAM_DATA, _generation_order, generated_items, load_config   # noqa: E402

GRID = np.linspace(-6, 6, 801)


def _params(schema):
    config = load_config(os.path.join(PUBLIC, "config.yml"))
    s = config["sandbox"]
    n = schema["dataset"]["n_rows"]
    rng = np.random.default_rng([SEED, STREAM_DATA])
    rng.normal(size=n)                                   # the trait itself
    items = generated_items(schema)
    par = {}
    for name in _generation_order(schema, items):
        width = len(schema["items"][name]["values"])
        base = np.log(rng.dirichlet(np.full(width, s["dirichlet_alpha"])) + 1e-12)
        load = rng.normal(scale=s["trait_scale"], size=width)
        rng.random((n, 1))
        logits = base[None, :] + GRID[:, None] * load[None, :]
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        par[name] = p / p.sum(1, keepdims=True)          # [G, width]: P(answer | t)
    return par, _generation_order(schema, items)


def predict(frame, schema):
    items = schema["items"]
    gated = schema["gated_value"]
    par, order = _params(schema)
    names = [n for n, r in items.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in names if items[n]["class"] == "GIVEN"]
    targets = [n for n in names if items[n]["class"] == "PREDICT"]
    nan = frame[targets].isna().to_numpy()
    hid = np.where(nan.any(axis=1))[0]
    H = len(hid)

    # posterior over the trait from each held-out row's GIVEN answers
    logpost = np.tile(-0.5 * GRID ** 2, (H, 1))
    for g in given:
        vals = list(items[g]["values"])
        lut = {v: i for i, v in enumerate(vals)}
        col = frame[g].to_numpy(dtype=object)[hid]
        lp = np.log(np.maximum(par[g], 1e-300))          # [G, width]
        for r, v in enumerate(col):
            if v in lut:                                  # a skipped item says nothing about t
                logpost[r] += lp[:, lut[v]]
    logpost -= logpost.max(1, keepdims=True)
    post = np.exp(logpost)
    post /= post.sum(1, keepdims=True)                    # [H, G]

    # P(asked | t) down the gate chains, per held-out row
    asked, full = {}, {}
    for name in order:
        if items[name]["class"] not in ("GIVEN", "PREDICT"):
            continue
        gate = items[name].get("gate")
        if not gate:
            a = np.ones((H, len(GRID)))
        elif items[gate["parent"]]["class"] == "GIVEN":
            pv = frame[gate["parent"]].to_numpy(dtype=object)[hid]
            a = np.isin(pv, gate["observed_if"]).astype(float)[:, None] * np.ones((1, len(GRID)))
        else:
            pvals = list(items[gate["parent"]]["values"])
            idx = [pvals.index(v) for v in gate["observed_if"] if v in pvals]
            a = asked[gate["parent"]] * par[gate["parent"]][:, idx].sum(1)[None, :]
        asked[name] = a
        if items[name]["class"] == "PREDICT":
            # P(y | x) = sum_t post(t) [a(t) P(v | t), 1 - a(t)]
            real = np.einsum("hg,hg,gk->hk", post, a, par[name])
            if gate:
                sent = 1.0 - (post * a).sum(1, keepdims=True)
                full[name] = np.concatenate([real, np.maximum(sent, 0)], axis=1)
            else:
                full[name] = real
    out = []
    for i, r in enumerate(hid):
        for j, t in enumerate(targets):
            if nan[r, j]:
                v = full[t][i]
                out.append(v / v.sum())
    return out
