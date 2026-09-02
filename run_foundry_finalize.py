# Copyright Sierra
"""Self-healing consistency sweep for one model's foundry_eval artifacts.

Strips transient-error attempt entries, then tops up reroll / reflexion /
both intervention runs until every failing baseline task has full N=5
coverage with no error entries (or the round budget runs out).
"""
import glob, json, os, shutil, subprocess, sys, time

MODEL, CONC = sys.argv[1], sys.argv[2]
ROOT = f"results/foundry_eval/{MODEL}"
B = glob.glob(f"{ROOT}/react-{MODEL}-0.0_range_0--1_*")[0]
PY = ".venv/bin/python"
COMMON = ["--model", MODEL, "--model-provider", "azure_ai", "--user-model", "gpt-4o-mini",
          "--user-model-provider", "azure", "--env", "retail"]
INTERVENORS = [("gpt-4o-mini", "0.2"), ("gpt-5-mini", "1.0")]
N = "5"


def log(msg):
    print(f"FINALIZE[{MODEL}] {msg} {time.strftime('%H:%M:%S')}", flush=True)


def is_err(r):
    return isinstance(r.get("info"), dict) and "error" in r["info"]


def load(p):
    with open(p) as f:
        return json.load(f)


def dump(p, d):
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def latest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None


def run(cmd, logfile):
    with open(logfile, "a") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode


def strip_attempt_errors(path):
    d = load(path)
    kept = [r for r in d if not is_err(r)]
    dump(path, kept)
    return len(d) - len(kept)


def strip_task_errors(path):
    d = load(path)
    bad = {r["task_id"] for r in d if is_err(r)}
    kept = [r for r in d if r["task_id"] not in bad]
    dump(path, kept)
    return bad


baseline = load(f"{B}/transcript.json")
ALL = {r["task_id"] for r in baseline}
FAILS = {r["task_id"] for r in baseline if r["reward"] != 1.0}


def reroll_complete(path):
    d = load(path)
    per = {}
    for r in d:
        if r["task_id"] in FAILS and str(r.get("intervened_message", "")).startswith("(no text inserted"):
            per[r["task_id"]] = per.get(r["task_id"], 0) + 1
    no_point = {r["task_id"] for r in d if "no intervention point" in str(r.get("intervened_message", ""))}
    short = [t for t in FAILS if t not in no_point and per.get(t, 0) < int(N)]
    return not short, len(short)


def reflexion_complete(path):
    d = load(path)
    per, solved = {}, set()
    for r in d:
        if r["task_id"] in FAILS:
            per[r["task_id"]] = per.get(r["task_id"], 0) + 1
            if r["reward"] == 1.0:
                solved.add(r["task_id"])
    short = [t for t in FAILS if t not in solved and per.get(t, 0) < int(N)]
    return not short, len(short)


for rnd in range(1, 6):
    log(f"round {rnd} START")
    all_done = True

    # --- partial reroll ---
    p = latest(f"{B}/partial-reroll-by_*/partial-reroll-transcripts.json")
    if p:
        n = strip_attempt_errors(p)
        ok, short = reroll_complete(p)
        log(f"reroll: stripped {n} error entries, {short} failing tasks short of N")
    else:
        ok = False
        log("reroll: no folder yet")
    if not ok:
        all_done = False
        rc = run([PY, "run_partial_reroll.py", "--baseline_path", B, *COMMON,
                  "--intervention_model", "gpt-4o-mini", "--intervention-model-provider", "azure",
                  "--temperature", "1.0", "--best_of_N", N, "--max-concurrency", CONC], f"{ROOT}/finalize_reroll.log")
        log(f"reroll: run rc={rc}")

    # --- reflexion ---
    p = latest(f"{B}/reflexion-by_*/reflexion-transcripts.json")
    if p:
        n = strip_attempt_errors(p)
        ok, short = reflexion_complete(p)
        log(f"reflexion: stripped {n} error entries, {short} failing tasks short")
    else:
        ok = False
        log("reflexion: no folder yet")
    if not ok:
        all_done = False
        rc = run([PY, "run_reflexion.py", "--baseline_path", B, *COMMON,
                  "--intervention_model", "gpt-4o-mini", "--intervention-model-provider", "azure",
                  "--temperature", "1.0", "--best_of_N", N, "--max-concurrency", CONC], f"{ROOT}/finalize_reflexion.log")
        log(f"reflexion: run rc={rc}")

    # --- interventions ---
    for iv, ivtemp in INTERVENORS:
        target = latest(f"{B}/intervened-by_{iv}_*/intervened-transcripts.json")
        cmd = [PY, "run.py", "--run_intervention", "--baseline_path", B, *COMMON,
               "--agent-strategy", "react-intervened", "--intervention_model", iv,
               "--intervention-model-provider", "azure", "--intervenor-temperature", ivtemp,
               "--temperature", "0.0", "--best_of_N", N, "--max-concurrency", CONC]
        logfile = f"{ROOT}/finalize_interv_{iv}.log"
        if target is None:
            all_done = False
            log(f"interv-{iv}: no folder, running full")
            rc = run(cmd, logfile)
            log(f"interv-{iv}: full run rc={rc}")
            continue
        bad = strip_task_errors(target)
        covered = {r["task_id"] for r in load(target)}
        missing = sorted(ALL - covered)
        log(f"interv-{iv}: dropped {len(bad)} errored tasks, {len(missing)} tasks missing")
        if not missing:
            continue
        all_done = False
        before = set(glob.glob(f"{B}/intervened-by_{iv}_*"))
        rc = run(cmd + ["--task-ids", *map(str, missing)], logfile)
        log(f"interv-{iv}: top-up run rc={rc}")
        new_dirs = set(glob.glob(f"{B}/intervened-by_{iv}_*")) - before
        tdir = os.path.dirname(target)
        for nd in new_dirs:
            npath = f"{nd}/intervened-transcripts.json"
            if os.path.exists(npath):
                base = load(target)
                seen = {r["task_id"] for r in base}
                added = [r for r in load(npath) if r["task_id"] not in seen]
                dump(target, base + added)
                log(f"interv-{iv}: merged {len(added)} entries from {os.path.basename(nd)}")
                hp, ht = f"{nd}/agent_conversation_history.json", f"{tdir}/agent_conversation_history.json"
                if os.path.exists(hp) and os.path.exists(ht):
                    h = load(ht)
                    hs = {e.get("task_id") for e in h}
                    dump(ht, h + [e for e in load(hp) if e.get("task_id") not in hs])
            shutil.rmtree(nd, ignore_errors=True)

    if all_done:
        log("ALL COMPLETE")
        sys.exit(0)

log("round budget exhausted; see per-stage logs")
sys.exit(1)
