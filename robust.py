"""Adversarial inputs: every one must come back as legal output, never a crash.

    python robust.py SUBMISSION_DIR

Each case perturbs a real-shaped frame in a way the hosted data might and the
sandbox never does, then runs the grader's own check() on the result.
"""
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(HERE, "public")
sys.path.insert(0, PUBLIC)
import score as S                                                    # noqa: E402
from make_sandbox import load_config, load_schema, write_sandbox     # noqa: E402

os.environ.setdefault("SB_TOTAL_BUDGET", "240")


def load_main(sub):
    spec = importlib.util.spec_from_file_location("main_%d" % time.time_ns(), os.path.join(HERE, sub, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def staged(frame):
    return pd.DataFrame([[None if pd.isna(v) else v for v in row] for row in frame.to_numpy(dtype=object)],
                        columns=list(frame.columns))


def cells_for(schema, frame):
    items = S.generated_items(schema)
    scored = set(S.scored_items(schema))
    vals = frame[items].to_numpy(dtype=object)
    return [(r, None, it) for r in range(len(frame)) for c, it in enumerate(items)
            if it in scored and pd.isna(vals[r, c])]


def run_case(sub, name, schema, frame):
    mod = load_main(sub)
    t = time.time()
    try:
        vecs = mod.predict(staged(frame), json.loads(json.dumps(schema)))
        vecs = [[float(x) for x in v] for v in vecs]
        S.check(schema, vecs, cells_for(schema, frame))
        print("PASS %-48s %5.1fs  %d vectors" % (name, time.time() - t, len(vecs)), flush=True)
        return True
    except Exception:
        print("FAIL %-48s\n%s" % (name, traceback.format_exc()[-1500:]), flush=True)
        return False


def main():
    sub = sys.argv[1]
    os.chdir(PUBLIC)
    config = load_config("config.yml")
    ok = True

    def base(inst, phase=2, seed=5):
        schema = load_schema("data/%s.json" % inst, config)
        box = os.path.join(HERE, "_box", "%s_s%d" % (inst, seed))
        if not os.path.exists(os.path.join(box, "respondents.parquet")):
            write_sandbox(schema, config, box, seed=seed)
        frame, _, _ = S.sample_rows(schema, S.load_frames(box, schema), phase)
        return schema, frame

    schema, frame = base("sample", 2)
    ok &= run_case(sub, "sample schema, phase 2", schema, frame)
    schema, frame = base("sample", 1)
    ok &= run_case(sub, "sample schema, phase 1", schema, frame)

    schema, frame = base("unicef", 2)
    given = [n for n, r in schema["items"].items() if r["class"] == "GIVEN"]
    targets = [n for n, r in schema["items"].items() if r["class"] == "PREDICT"]
    hid = frame[targets].isna().all(axis=1).to_numpy()

    f = frame.copy()
    f.loc[np.where(hid)[0][:50], given[0]] = "Atlantis"            # a level the schema never lists
    ok &= run_case(sub, "unseen GIVEN level in held-out rows", schema, f)

    f = frame.copy()
    f.loc[np.where(hid)[0][:50], given[1]] = None                   # a missing GIVEN value
    f.loc[np.where(~hid)[0][:50], given[2]] = None
    ok &= run_case(sub, "missing GIVEN values", schema, f)

    f = frame.copy()
    vis = np.where(~hid)[0]
    first = schema["items"][targets[0]]["values"][0]
    f.loc[vis, targets[0]] = first                                   # an item everybody answers alike
    ok &= run_case(sub, "constant item", schema, f)

    gated = [t for t in targets if schema["items"][t].get("gate")]
    f = frame.copy()
    f.loc[vis, gated[0]] = schema["gated_value"]                     # an item nobody visible was asked
    ok &= run_case(sub, "item gated for every visible row", schema, f)

    f = frame.copy()
    for g in given[:2]:                                              # numbers where strings are expected
        f[g] = f[g].map(lambda v: int(v) if isinstance(v, str) and v.isdigit() else v)
    ok &= run_case(sub, "numeric types in GIVEN", schema, f)

    f = frame.loc[np.concatenate([vis[:40], np.where(hid)[0][:20]])].reset_index(drop=True)
    ok &= run_case(sub, "only 40 visible rows", schema, f)

    f = frame.loc[vis].reset_index(drop=True)
    ok &= run_case(sub, "no held-out rows at all", schema, f)

    f = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    ok &= run_case(sub, "visible and held-out rows interleaved", schema, f)

    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
