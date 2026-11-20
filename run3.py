"""All three instruments in one process, as the hosted container runs them."""
import json, os, sys, time, importlib.util
ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")
sys.path.insert(0, PUBLIC)
import pandas as pd
import score as S
from make_sandbox import load_config, load_schema, write_sandbox

def main(sub, order, seed=0, phase=2):
    os.chdir(PUBLIC)
    config = load_config("config.yml")
    floor = config["scoring"]["floor"]
    jobs = []
    for inst in order:
        schema = load_schema("data/%s.json" % inst, config)
        box = os.path.join(ROOT, "_box", "%s_s%d" % (inst, seed))
        if not os.path.exists(os.path.join(box, "respondents.parquet")):
            write_sandbox(schema, config, box, seed=seed)
        resp = S.load_frames(box, schema)
        frame, cells, truth = S.sample_rows(schema, resp, phase)
        ref = S.marginal_reference(schema, frame, cells, truth, floor)
        staged = pd.DataFrame([[None if pd.isna(v) else v for v in row] for row in frame.to_numpy(dtype=object)],
                              columns=list(frame.columns))
        jobs.append((inst, schema, staged, cells, truth, ref))
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("main", os.path.join(ROOT, sub, "main.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    skills = {}
    for inst, schema, staged, cells, truth, ref in jobs:
        t = time.time()
        vecs = mod.predict(staged, json.loads(json.dumps(schema)))
        vecs = [[float(x) for x in v] for v in vecs]
        S.check(schema, vecs, cells)
        res = S.score(schema, config, S.floored(vecs, schema, cells, floor), truth, cells, ref)
        skills[inst] = res["skill"]
        print("RESULT %-10s skill %.4f  se %.4f  %.1fs" % (inst, res["skill"], res["std_error"], time.time() - t), flush=True)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print("PEAK RAM %.2f GB (grader limit 16 GB)" % peak, flush=True)
    print("TOTAL wall %.1fs  mean skill %.4f  %s" % (time.time() - t0, sum(skills.values()) / len(skills), skills), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
