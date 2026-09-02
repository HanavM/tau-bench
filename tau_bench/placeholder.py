# Copyright Sierra

"""Placeholder-intervention control.

Mirrors the real intervention pipeline in tau_bench/run.py's `run()`
(run_intervention=True branch) exactly, EXCEPT for one change: instead of
inserting the LLM-generated corrective `intervention_text` at the identified
failure point, it inserts a fixed, content-free placeholder message at that
same point.

This isolates two effects that are otherwise confounded when comparing
"baseline" vs "intervention Bo-N":
  1. The architecture cost of the intervention pipeline (grafting a
     placeholder/correction onto a fixed, already-failed trajectory prefix
     and continuing from there) instead of a genuinely fresh rollout.
  2. The actual content of the generated intervention.

`agent.run_intervention(...)` is still called for real, since it is what
identifies WHERE in the trajectory to intervene (the failure point) — we
only discard the `intervention_text` it produces and replace it with the
placeholder. This keeps the insertion-point distribution identical to the
real intervention run, so any gap between "real intervention Bo-N" and
"placeholder Bo-N" can be attributed to the corrective content itself, not to
where the nudge was inserted.
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
    """Each failing task may have up to n attempt-entries (one per placeholder
    attempt) plus terminal single-entry cases (already-passed / no
    intervention point found). Report the best-of-n success rate — the
    same "success if ANY attempt solved it" scoring used for the real
    intervention Bo-N numbers — rather than averaging over raw entries,
    which would double count multi-attempt tasks.
    """
    best_per_task: Dict[int, float] = {}
    for r in results:
        if r.task_id not in best_per_task or r.reward > best_per_task[r.task_id]:
            best_per_task[r.task_id] = r.reward

    num_tasks = len(best_per_task)
    num_solved = sum(1 for v in best_per_task.values() if is_successful(v))
    avg_best_reward = sum(best_per_task.values()) / num_tasks if num_tasks else 0.0

    solved_pct = f"{num_solved / num_tasks:.2%}" if num_tasks else "n/a"
    print(f"\n===== Placeholder Best-of-{n} performance (task counts as a success if ANY placeholder attempt solved it) =====")
    print(f"🏆 Average reward (best per task): {avg_best_reward}")
    print(f"✅ Tasks solved by at least one placeholder attempt: {num_solved}/{num_tasks} ({solved_pct})")


PLACEHOLDER_TEXT = (
    "Please continue and do your best to complete the customer's request successfully."
)


def _parse_marker_id(raw_id: Any) -> int:
    """Same id-parsing convention used in tau_bench/run.py's run() intervention loop."""
    try:
        if isinstance(raw_id, int):
            return raw_id
        idx_b = raw_id.rfind("B")
        if idx_b == -1:
            idx_b = raw_id.rfind("A")
        return int(raw_id[idx_b + 1 :])
    except Exception as e:
        print(f"error converting marker id {raw_id} to int: {e}")
        return 1


def _add_placeholder(trajectory: List[Dict[str, Any]], intervention_id: Any) -> List[Dict[str, Any]]:
    idx_intervention = _parse_marker_id(intervention_id)
    new_trajectory = trajectory[: min(idx_intervention + 1, len(trajectory) - 1)]
    new_trajectory.append(
        {
            "role": "system",
            "content": "[*INTERVENTION*: " + PLACEHOLDER_TEXT + "]",
        }
    )
    return new_trajectory


def find_existing_placeholder_folder(config: RunConfig) -> Optional[str]:
    prefix = f"placeholder-by_{config.model.split('/')[-1]}_"
    matches = sorted(glob.glob(os.path.join(config.baseline_path, prefix + "*")))
    return matches[-1] if matches else None


def run_placeholder(config: RunConfig) -> List[EnvRunResult]:
    """Run the placeholder-intervention control against config.baseline_path.

    config.best_of_N is treated as a TARGET total attempts per failing task,
    not "attempts to run this call". Reruns against the same
    baseline_path/model top up any task that has fewer than best_of_N
    attempt-entries so far (e.g. rerunning with best_of_N=5 after an
    earlier best_of_N=3 run appends 2 more attempts per task rather than
    discarding and redoing the first 3). Already-terminal tasks (baseline
    already passed, or no intervention point found) and tasks that already
    have >= best_of_N attempts are left untouched.
    """
    assert config.baseline_path is not None, "baseline_path is required"
    assert os.path.exists(config.baseline_path), f"baseline_path does not exist: {config.baseline_path}"

    # Single fixed seed for the whole run — every task and every attempt
    # uses this same seed, not a different one per attempt/task.
    random.seed(config.seed)

    existing_folder = find_existing_placeholder_folder(config)
    if existing_folder:
        placeholder_folder = existing_folder
        print(f"Resuming existing placeholder folder: {placeholder_folder}")
    else:
        time_str = datetime.now().strftime("%m%d%H%M%S")
        placeholder_folder = os.path.join(
            config.baseline_path, f"placeholder-by_{config.model.split('/')[-1]}_{time_str}"
        )
        os.makedirs(placeholder_folder, exist_ok=True)
    ckpt_path = os.path.join(placeholder_folder, "placeholder-transcripts.json")

    existing_results: List[EnvRunResult] = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            existing_results = [EnvRunResult(**r) for r in json.load(f)]

    by_task: Dict[int, List[EnvRunResult]] = {}
    for r in existing_results:
        by_task.setdefault(r.task_id, []).append(r)

    # Only "already passed at baseline" is a permanent terminal state (the
    # baseline reward can't change). "No intervention point found" is NOT
    # treated as permanently terminal — that call can fail transiently
    # (LLM stochasticity, or a smaller top-up N behaving differently than a
    # larger single-shot N), so a task that already has real attempts
    # should still be eligible for more attempts on a future top-up rather
    # than being silently frozen. Real attempts are counted as any entry
    # that isn't a "-1" terminal/no-point-found marker (error-fallback
    # entries still count as a used, failed attempt).
    terminal_task_ids = {
        task_id for task_id, entries in by_task.items()
        if any(e.intervened_message == "no placeholder needed. already passed" for e in entries)
    }
    attempt_counts = {
        task_id: sum(1 for e in entries if e.intervened_index != "-1")
        for task_id, entries in by_task.items()
        if task_id not in terminal_task_ids
    }

    with open(os.path.join(config.baseline_path, "transcript.json"), "r", encoding="utf-8") as f:
        examples = json.load(f)

    if config.task_ids:
        examples = [e for e in examples if e["task_id"] in set(config.task_ids)]

    # (example, existing_count, needed) for every task that still needs
    # more attempts to reach the best_of_N target.
    work_items = []
    skipped_terminal = 0
    skipped_full = 0
    skipped_baseline_errored = 0
    for e in examples:
        task_id = e["task_id"]
        if task_id in terminal_task_ids:
            skipped_terminal += 1
            continue
        if "task" not in e["info"] or not e["traj"]:
            # The baseline run itself errored out for this task - there's no
            # real trajectory to find a failure point in.
            skipped_baseline_errored += 1
            continue
        existing_count = attempt_counts.get(task_id, 0)
        needed = max(0, config.best_of_N - existing_count)
        if needed == 0:
            skipped_full += 1
            continue
        work_items.append((e, existing_count, needed))

    print(
        f"{skipped_baseline_errored} task(s) skipped (baseline run itself errored, no trajectory), "
        f"{skipped_terminal} task(s) already passed at baseline (permanently terminal), "
        f"{skipped_full} task(s) already have >= {config.best_of_N} real attempts, "
        f"{len(work_items)} task(s) need {'more ' if attempt_counts else ''}attempts"
    )

    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(tools_info=env.tools_info, wiki=env.wiki, config=config)
    assert hasattr(agent, "run_intervention") and hasattr(agent, "solve_with_intervention"), (
        f"agent-strategy={config.agent_strategy} does not support interventions; "
        f"use --agent-strategy react-intervened"
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

    def _run(work_item) -> List[EnvRunResult]:
        baseline_example, existing_count, needed = work_item
        idx = baseline_example["task_id"]
        isolated_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=config.task_split,
            user_provider=config.user_model_provider,
            task_index=idx,
        )
        result = EnvRunResult(
            task_id=idx,
            reward=baseline_example["reward"],
            info=baseline_example["info"],
            traj=baseline_example["traj"],
            trial=0,
        )

        print(f"Running placeholder on task {idx} ({existing_count} existing, {needed} more needed)")
        produced: List[EnvRunResult] = []

        try:
            if result.reward == 1.0:
                r = EnvRunResult(
                    intervened_message="no placeholder needed. already passed",
                    intervened_index=str(-1),
                    improved=0,
                    task_id=idx,
                    reward=result.reward,
                    info=result.info,
                    traj=result.traj,
                    trial=0,
                )
                print(f"task {idx}: already passed, no placeholder needed")
                produced.append(r)
                _save(r)
            else:
                answer_list, _conv = agent.run_intervention(
                    env=isolated_env,
                    task_index=idx,
                    result=result,
                    N=needed,
                )

                if answer_list is None or answer_list[0] is False:
                    r = EnvRunResult(
                        intervened_message="no intervention point found (placeholder)",
                        failure_brief="no intervention point found (placeholder)",
                        failure_index=str(-1),
                        intervened_index=str(-1),
                        improved=0,
                        task_id=idx,
                        reward=result.reward,
                        success_prev=result.reward,
                        success_after=None,
                        info=result.info,
                        traj=result.traj,
                        trial=0,
                    )
                    produced.append(r)
                    _save(r)
                else:
                    for offset, candidate in enumerate(answer_list):
                        attempt_i = existing_count + offset
                        intervention_id = candidate.get("id", 1)
                        failure_id = candidate.get("failure_id", -1)
                        failure_brief = candidate.get("failure_brief", "")

                        print(
                            f"task {idx}: placeholder attempt {attempt_i} at marker {intervention_id} "
                            f"(real failure_brief discarded: {failure_brief[:80]!r})"
                        )

                        try:
                            new_trajectory = _add_placeholder(result.traj, intervention_id)
                            res_placeholder = agent.solve_with_intervention(
                                env=isolated_env,
                                task_index=idx,
                                messages=new_trajectory,
                            )
                            candidate_result = EnvRunResult(
                                failure_brief=failure_brief,
                                failure_index=str(_parse_marker_id(failure_id)),
                                intervened_message=PLACEHOLDER_TEXT,
                                intervened_first_or_last=str(attempt_i),
                                intervened_index=str(_parse_marker_id(intervention_id)),
                                improved=(result.reward == 0 and res_placeholder.reward != 0),
                                success_prev=result.reward,
                                success_after=res_placeholder.reward,
                                task_id=idx,
                                reward=res_placeholder.reward,
                                info=res_placeholder.info,
                                traj=res_placeholder.messages,
                                trial=0,
                            )
                        except Exception as e:
                            print(f"task {idx}: attempt {attempt_i} errored, skipping just this attempt: {e}")
                            candidate_result = EnvRunResult(
                                failure_brief=failure_brief,
                                failure_index=str(_parse_marker_id(failure_id)),
                                intervened_first_or_last=str(attempt_i),
                                intervened_index=str(_parse_marker_id(intervention_id)),
                                task_id=idx,
                                reward=0.0,
                                info={"error": str(e), "traceback": traceback.format_exc()},
                                traj=[],
                                trial=0,
                            )

                        print(
                            "PLACEHOLDER",
                            "✅" if candidate_result.reward == 1 else "❌",
                            f"task_id={idx} attempt={attempt_i}",
                        )
                        produced.append(candidate_result)
                        _save(candidate_result)

        except Exception as e:
            r = EnvRunResult(
                task_id=idx,
                reward=0.0,
                info={"error": str(e), "traceback": traceback.format_exc()},
                traj=[],
                trial=0,
            )
            print(f"task {idx}: error during placeholder run: {e}")
            produced.append(r)
            _save(r)

        return produced

    new_results: List[EnvRunResult] = []
    if work_items:
        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            nested = list(executor.map(_run, work_items))
            new_results = [r for sub in nested for r in sub]

    results = existing_results + new_results

    with open(ckpt_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
        print(f"\nPlaceholder results saved to {ckpt_path}\n")

    summarize_best_of_n(results, config.best_of_N)

    return results
