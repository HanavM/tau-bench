# Copyright Sierra

"""Reflexion control, run against an already-completed baseline.

Mirrors the resumable, checkpointed structure of tau_bench/placeholder.py,
but instead of grafting a fixed/placeholder message onto the failed
trajectory prefix, it runs a sequence of independent full rollouts:

  attempt 0 = the existing baseline rollout (already in transcript.json)
  attempt k (k=1..best_of_N) = a brand-new env.reset() rollout, seeded with
    a natural-language reflection on why attempt k-1 failed.

Stops early (per task) the first time an attempt succeeds, since reflexion
is inherently sequential (each attempt depends on the last), unlike
intervention/placeholder's independent Bo-N branches.
"""

import os
import glob
import json
import random
import traceback
import multiprocessing
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.run import agent_factory


def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def summarize_best_of_n(results: List[EnvRunResult], n: int) -> None:
    """Best-of-(n+1) success rate: a task counts as solved if the baseline
    (attempt 0) OR any of its up-to-n reflexion attempts solved it.
    """
    best_per_task: Dict[int, float] = {}
    for r in results:
        if r.task_id not in best_per_task or r.reward > best_per_task[r.task_id]:
            best_per_task[r.task_id] = r.reward

    num_tasks = len(best_per_task)
    num_solved = sum(1 for v in best_per_task.values() if is_successful(v))
    avg_best_reward = sum(best_per_task.values()) / num_tasks if num_tasks else 0.0

    solved_pct = f"{num_solved / num_tasks:.2%}" if num_tasks else "n/a"
    print(f"\n===== Reflexion Best-of-{n + 1} performance (baseline + up to {n} reflexion attempts) =====")
    print(f"🏆 Average reward (best per task): {avg_best_reward}")
    print(f"✅ Tasks solved by baseline or at least one reflexion attempt: {num_solved}/{num_tasks} ({solved_pct})")


def report_at_k(baseline_path: str, reflexion_folder: str, ks: List[int]) -> None:
    """Recompute Best-of-(k+1) for one or more values of k from an already
    completed reflexion run, WITHOUT making any new rollouts or LLM calls.

    Since reflexion attempts are strictly sequential and each is tagged with
    its attempt_number, Bo1/Bo2/.../Bo5 (etc.) can all be derived from a
    single Bo5 (or larger) run by just ignoring the later attempts - there's
    no need to rerun with a smaller --best_of_N.
    """
    with open(os.path.join(baseline_path, "transcript.json"), "r", encoding="utf-8") as f:
        examples = json.load(f)
    baseline_entries = [
        EnvRunResult(task_id=e["task_id"], reward=e["reward"], info=e["info"], traj=e["traj"], trial=0)
        for e in examples
    ]

    ckpt_path = os.path.join(reflexion_folder, "reflexion-transcripts.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        all_results = [EnvRunResult(**r) for r in json.load(f)]

    for k in ks:
        filtered = [r for r in all_results if r.attempt_number is not None and r.attempt_number <= k]
        summarize_best_of_n(baseline_entries + filtered, k)


def find_existing_reflexion_folder(config: RunConfig) -> Optional[str]:
    prefix = f"reflexion-by_{config.intervention_model.split('/')[-1]}_"
    matches = sorted(glob.glob(os.path.join(config.baseline_path, prefix + "*")))
    return matches[-1] if matches else None


def run_reflexion(config: RunConfig) -> List[EnvRunResult]:
    """Run the reflexion control against config.baseline_path.

    config.best_of_N is the max number of NEW reflexion rollouts attempted
    per failing task (not counting the baseline attempt itself), matching
    the "N extra attempts" budget used by the intervention/placeholder
    controls. A task stops early (uses fewer than N attempts) the moment
    one of its attempts succeeds.

    Reruns top up any task that has fewer than best_of_N attempts so far
    and hasn't yet succeeded; tasks that already succeeded (at baseline or
    in a previous reflexion attempt) or that already used best_of_N
    attempts are left untouched.
    """
    assert config.baseline_path is not None, "baseline_path is required"
    assert os.path.exists(config.baseline_path), f"baseline_path does not exist: {config.baseline_path}"
    assert config.intervention_model is not None, "intervention_model is required (used as the reflection-writer model)"

    random.seed(config.seed)

    existing_folder = find_existing_reflexion_folder(config)
    if existing_folder:
        reflexion_folder = existing_folder
        print(f"Resuming existing reflexion folder: {reflexion_folder}")
    else:
        time_str = datetime.now().strftime("%m%d%H%M%S")
        reflexion_folder = os.path.join(
            config.baseline_path, f"reflexion-by_{config.intervention_model.split('/')[-1]}_{time_str}"
        )
        os.makedirs(reflexion_folder, exist_ok=True)
    ckpt_path = os.path.join(reflexion_folder, "reflexion-transcripts.json")

    existing_results: List[EnvRunResult] = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            existing_results = [EnvRunResult(**r) for r in json.load(f)]

    by_task: Dict[int, List[EnvRunResult]] = {}
    for r in existing_results:
        by_task.setdefault(r.task_id, []).append(r)

    def already_succeeded(entries: List[EnvRunResult]) -> bool:
        return any(is_successful(e.reward) for e in entries)

    def attempts_so_far(entries: List[EnvRunResult]) -> int:
        return sum(1 for e in entries if e.attempt_number is not None)

    def sorted_attempts(entries: List[EnvRunResult]) -> List[EnvRunResult]:
        return sorted((e for e in entries if e.attempt_number is not None), key=lambda e: e.attempt_number)

    with open(os.path.join(config.baseline_path, "transcript.json"), "r", encoding="utf-8") as f:
        examples = json.load(f)

    if config.task_ids:
        examples = [e for e in examples if e["task_id"] in set(config.task_ids)]

    work_items = []
    skipped_terminal = 0
    skipped_full_or_solved = 0
    skipped_baseline_errored = 0
    for e in examples:
        task_id = e["task_id"]
        if is_successful(e["reward"]):
            skipped_terminal += 1
            continue
        if "task" not in e["info"] or not e["traj"]:
            # The baseline run itself errored out for this task (e.g. an
            # exception mid-episode) - there's no real trajectory or task
            # metadata to reflect on, so this task is left as-is rather than
            # crashing the whole reflexion run.
            skipped_baseline_errored += 1
            continue
        entries = by_task.get(task_id, [])
        if already_succeeded(entries) or attempts_so_far(entries) >= config.best_of_N:
            skipped_full_or_solved += 1
            continue
        work_items.append(e)

    print(
        f"{skipped_terminal} task(s) already passed at baseline (permanently terminal), "
        f"{skipped_baseline_errored} task(s) skipped (baseline run itself errored, no trajectory to reflect on), "
        f"{skipped_full_or_solved} task(s) already solved by reflexion or used up {config.best_of_N} attempt(s), "
        f"{len(work_items)} task(s) need reflexion attempts"
    )

    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(tools_info=env.tools_info, wiki=env.wiki, config=config)
    assert hasattr(agent, "reflect") and hasattr(agent, "solve_with_reflection"), (
        f"agent-strategy={config.agent_strategy} does not support reflexion; "
        f"use --agent-strategy react-reflexion"
    )

    lock = multiprocessing.Lock()

    def _save(r: EnvRunResult) -> None:
        with lock:
            data = []
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    data = json.load(f)
            with open(ckpt_path, "w") as f:
                json.dump(data + [r.model_dump()], f, indent=2)

    def _run(baseline_example: dict) -> List[EnvRunResult]:
        idx = baseline_example["task_id"]
        existing_entries = by_task.get(idx, [])
        existing_count = attempts_so_far(existing_entries)

        isolated_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=config.task_split,
            user_provider=config.user_model_provider,
            task_index=idx,
        )

        task_instruction = baseline_example["info"]["task"]["instruction"]

        # Reconstruct prior reflections (in attempt order) and the state of
        # the most recent attempt, so a resumed run continues the same
        # sequential chain instead of restarting it.
        reflections: List[str] = []
        prev_reward = baseline_example["reward"]
        prev_info = baseline_example["info"]
        prev_traj = baseline_example["traj"]
        for prior in sorted_attempts(existing_entries):
            if prior.reflection_text:
                reflections.append(prior.reflection_text)
            prev_reward = prior.reward
            prev_info = prior.info
            prev_traj = prior.traj

        print(f"Running reflexion on task {idx} ({existing_count} existing attempt(s))")
        produced: List[EnvRunResult] = []

        try:
            for attempt in range(existing_count + 1, config.best_of_N + 1):
                reflection_text = agent.reflect(
                    wiki=isolated_env.wiki,
                    task_instruction=task_instruction,
                    outcome=prev_info.get("reward_info", prev_info),
                    trajectory=prev_traj,
                    reflection_model=config.intervention_model,
                    reflection_provider=config.intervention_model_provider,
                )
                reflections.append(reflection_text)

                print(f"task {idx}: attempt {attempt} reflection: {reflection_text[:120]!r}")

                res = agent.solve_with_reflection(
                    env=isolated_env,
                    reflections=reflections,
                    task_index=idx,
                )

                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=0,
                    reflection_text=reflection_text,
                    attempt_number=attempt,
                    success_prev=prev_reward,
                    success_after=res.reward,
                    improved=(prev_reward == 0 and res.reward != 0),
                )
                print(
                    "REFLEXION",
                    "✅" if is_successful(result.reward) else "❌",
                    f"task_id={idx} attempt={attempt}",
                )
                produced.append(result)
                _save(result)

                prev_reward, prev_info, prev_traj = res.reward, res.info, res.messages
                if is_successful(res.reward):
                    break

        except Exception as e:
            result = EnvRunResult(
                task_id=idx,
                reward=0.0,
                info={"error": str(e), "traceback": traceback.format_exc()},
                traj=[],
                trial=0,
                attempt_number=existing_count + 1,
            )
            print(f"task {idx}: error during reflexion run: {e}")
            produced.append(result)
            _save(result)

        return produced

    new_results: List[EnvRunResult] = []
    if work_items:
        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            nested = list(executor.map(_run, work_items))
            new_results = [r for sub in nested for r in sub]

    results = existing_results + new_results

    with open(ckpt_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
        print(f"\nReflexion results saved to {ckpt_path}\n")

    # summarize_best_of_n needs attempt-0 (baseline) rewards too, since a
    # task counts as solved if baseline OR any reflexion attempt succeeded.
    baseline_entries = [
        EnvRunResult(task_id=e["task_id"], reward=e["reward"], info=e["info"], traj=e["traj"], trial=0)
        for e in examples
    ]
    summarize_best_of_n(results + baseline_entries, config.best_of_N)

    return results
