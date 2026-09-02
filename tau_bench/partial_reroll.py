# Copyright Sierra

"""Partial-rerolling control ("reviewer 3" baseline).

In reinforcement learning, running an agent/policy forward to some end state
is called a rollout (or playout). "Rerolling" here means: take the SAME
already-failed trajectory prefix used by the real intervention run (cut at
the exact same point, identified by the same agent.run_intervention() call),
but instead of grafting the LLM-generated `intervention_text` (or, in
placeholder.py, a fixed content-free placeholder) onto that cut point,
insert NOTHING at all and just let the agent continue generating from
there, relying purely on sampling stochasticity (temperature) to produce a
different continuation. Repeated best_of_N times per failing task, exactly
like the placeholder and real-intervention controls.

This isolates a THIRD effect, cleanly separating it from the placeholder
control's "architecture cost of grafting + continuing from a fixed prefix"
effect (see placeholder.py's docstring):
  - intervention Bo-N: fixed prefix + real corrective text + continue
  - placeholder Bo-N: fixed prefix + content-free filler text + continue
  - partial-reroll Bo-N: fixed prefix + NOTHING inserted + continue (pure
    re-sampling of "what would the agent have done differently here, by
    chance, if it just rolled the dice again")

If partial-reroll Bo-N looks a lot like placeholder Bo-N, that's expected
(neither has a "text with content" advantage). If intervention Bo-N clears
both by a wide margin, that's evidence the intervention TEXT itself is
carrying the improvement, not just re-sampling from the failure point.
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
    """Same "success if ANY attempt solved it" scoring used for the real
    intervention and placeholder Bo-N numbers, so all three are directly
    comparable.
    """
    best_per_task: Dict[int, float] = {}
    for r in results:
        if r.task_id not in best_per_task or r.reward > best_per_task[r.task_id]:
            best_per_task[r.task_id] = r.reward

    num_tasks = len(best_per_task)
    num_solved = sum(1 for v in best_per_task.values() if is_successful(v))
    avg_best_reward = sum(best_per_task.values()) / num_tasks if num_tasks else 0.0

    solved_pct = f"{num_solved / num_tasks:.2%}" if num_tasks else "n/a"
    print(f"\n===== Partial-Reroll Best-of-{n} performance (task counts as a success if ANY reroll attempt solved it) =====")
    print(f"🏆 Average reward (best per task): {avg_best_reward}")
    print(f"✅ Tasks solved by at least one reroll attempt: {num_solved}/{num_tasks} ({solved_pct})")


NO_TEXT_INSERTED_MARKER = "(no text inserted - partial reroll)"


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


def _truncate_at_reroll_point(trajectory: List[Dict[str, Any]], intervention_id: Any) -> List[Dict[str, Any]]:
    """Cut the trajectory at the exact same point intervention/placeholder
    would insert their message at, but append nothing - the agent's next
    generation call picks up right where the failed trajectory prefix ends.
    """
    idx_intervention = _parse_marker_id(intervention_id)
    return trajectory[: min(idx_intervention + 1, len(trajectory) - 1)]


def find_existing_reroll_folder(config: RunConfig) -> Optional[str]:
    prefix = f"partial-reroll-by_{config.model.split('/')[-1]}_"
    matches = sorted(glob.glob(os.path.join(config.baseline_path, prefix + "*")))
    return matches[-1] if matches else None


def run_partial_reroll(config: RunConfig) -> List[EnvRunResult]:
    """Run the partial-rerolling control against config.baseline_path.

    Same resumable/top-up semantics as run_placeholder: config.best_of_N is
    a target total attempts per failing task, reruns top up any task with
    fewer than best_of_N attempt-entries so far, and already-terminal tasks
    (baseline already passed, or no intervention point found) are left
    untouched.
    """
    assert config.baseline_path is not None, "baseline_path is required"
    assert os.path.exists(config.baseline_path), f"baseline_path does not exist: {config.baseline_path}"

    random.seed(config.seed)

    existing_folder = find_existing_reroll_folder(config)
    if existing_folder:
        reroll_folder = existing_folder
        print(f"Resuming existing partial-reroll folder: {reroll_folder}")
    else:
        time_str = datetime.now().strftime("%m%d%H%M%S")
        reroll_folder = os.path.join(
            config.baseline_path, f"partial-reroll-by_{config.model.split('/')[-1]}_{time_str}"
        )
        os.makedirs(reroll_folder, exist_ok=True)
    ckpt_path = os.path.join(reroll_folder, "partial-reroll-transcripts.json")

    existing_results: List[EnvRunResult] = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            existing_results = [EnvRunResult(**r) for r in json.load(f)]

    by_task: Dict[int, List[EnvRunResult]] = {}
    for r in existing_results:
        by_task.setdefault(r.task_id, []).append(r)

    terminal_task_ids = {
        task_id for task_id, entries in by_task.items()
        if any(e.intervened_message == "no reroll needed. already passed" for e in entries)
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

        print(f"Running partial reroll on task {idx} ({existing_count} existing, {needed} more needed)")
        produced: List[EnvRunResult] = []

        try:
            if result.reward == 1.0:
                r = EnvRunResult(
                    intervened_message="no reroll needed. already passed",
                    intervened_index=str(-1),
                    improved=0,
                    task_id=idx,
                    reward=result.reward,
                    info=result.info,
                    traj=result.traj,
                    trial=0,
                )
                print(f"task {idx}: already passed, no reroll needed")
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
                        intervened_message="no intervention point found (partial reroll)",
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

                        print(f"task {idx}: reroll attempt {attempt_i} from marker {intervention_id} (no text inserted)")

                        try:
                            new_trajectory = _truncate_at_reroll_point(result.traj, intervention_id)
                            res_reroll = agent.solve_with_intervention(
                                env=isolated_env,
                                task_index=idx,
                                messages=new_trajectory,
                            )
                            candidate_result = EnvRunResult(
                                intervened_message=NO_TEXT_INSERTED_MARKER,
                                intervened_first_or_last=str(attempt_i),
                                intervened_index=str(_parse_marker_id(intervention_id)),
                                improved=(result.reward == 0 and res_reroll.reward != 0),
                                success_prev=result.reward,
                                success_after=res_reroll.reward,
                                task_id=idx,
                                reward=res_reroll.reward,
                                info=res_reroll.info,
                                traj=res_reroll.messages,
                                trial=0,
                            )
                        except Exception as e:
                            print(f"task {idx}: attempt {attempt_i} errored, skipping just this attempt: {e}")
                            candidate_result = EnvRunResult(
                                intervened_first_or_last=str(attempt_i),
                                intervened_index=str(_parse_marker_id(intervention_id)),
                                task_id=idx,
                                reward=0.0,
                                info={"error": str(e), "traceback": traceback.format_exc()},
                                traj=[],
                                trial=0,
                            )

                        print(
                            "PARTIAL-REROLL",
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
            print(f"task {idx}: error during partial reroll run: {e}")
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
        print(f"\nPartial-reroll results saved to {ckpt_path}\n")

    summarize_best_of_n(results, config.best_of_N)

    return results
