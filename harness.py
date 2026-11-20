"""Run a submission against a sandbox exactly the way the grader scores it.

    python harness.py SUBMISSION_DIR INSTRUMENT [PHASE] [SEED]

Uses the grader's own sample_rows / check / floored / marginal_reference /
score, in-process, so masking, validation, the floor and the skill formula are
identical to the hosted run. The sandbox is regenerated per seed so a model is
never checked on the data it was debugged on.
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(HERE, "public")
sys.path.insert(0, PUBLIC)

import score as S                                      # noqa: E402
from make_sandbox import load_config, load_schema, write_sandbox   # noqa: E402


def main():
    sub, inst = sys.argv[1], sys.argv[2]
    phase = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.chdir(PUBLIC)
    config = load_config("config.yml")
    schema = load_schema("data/%s.json" % inst, config)
    box = os.path.join(HERE, "_box", "%s_s%d" % (inst, seed))
    if not os.path.exists(os.path.join(box, "respondents.parquet")):
        write_sandbox(schema, config, box, seed=seed)
    respondents = S.load_frames(box, schema)
    frame, cells, truth = S.sample_rows(schema, respondents, phase)
    floor = config["scoring"]["floor"]
    ref = S.marginal_reference(schema, frame, cells, truth, floor)

    spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.abspath(os.path.join(HERE, sub)), "main.py"))
    mod = importlib.util.module_from_spec(spec)
    t = time.time()
    spec.loader.exec_module(mod)
    only = os.environ.get("SB_ONLY")
    if only and hasattr(mod, "_plan"):
        keep = set(only.split(","))
        orig = mod._plan
        mod._plan = lambda inst, _o=orig: [p for p in _o(inst) if p[0] in keep]
    t_imp = time.time() - t
    # the hosted driver hands predict() a frame rebuilt from JSON: strings and None
    import json
    import pandas as pd
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
    print("RESULT %-10s %-28s phase %d seed %d  skill %.4f  (log %.4f, ref %.4f, se %.4f)  import %.1fs predict %.1fs  cells %d"
          % (inst, (os.path.basename(os.path.normpath(sub)) + " " + os.environ.get("SB_ONLY", "")).strip(), phase, seed, res["skill"], res["log_score"], ref, res["std_error"], t_imp, t_pred, res["n_cells"]))


if __name__ == "__main__":
    main()
