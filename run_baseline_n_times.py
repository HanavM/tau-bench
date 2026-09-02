# Copyright Sierra

import os
import json
import argparse
from tau_bench.types import RunConfig, EnvRunResult
from tau_bench.run import run_baseline_n_times, display_metrics
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run an already-completed baseline evaluation additional times "
            "(no intervention), each with a different seed, so it can be "
            "compared against an intervention run. --num-trials is the total "
            "desired number of baseline runs (N); N-1 new runs are performed "
            "since --input_path already holds one completed run at --seed."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the already-completed baseline run folder (run with --seed).",
    )
    parser.add_argument("--num-trials", type=int, default=5, help="Total number of baseline runs desired (N), including the one already done")
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot", "react-intervened", "react-reflexion"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="The seed the --input_path baseline run was (or will be) run with; this is also the seed the intervention run uses. New runs get seed, seed+1, seed+2, ...",
    )
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    parser.add_argument("--best_of_N", type=int, default=3)

    args = parser.parse_args()
    print(args)

    config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        best_of_N=args.best_of_N,
        run_intervention=False,
    )
    return config, args.input_path


def main():
    config, input_path = parse_args()
    all_results = run_baseline_n_times(config, input_path)

    base_seed = config.seed
    with open(os.path.join(input_path, "transcript.json"), "r") as f:
        baseline_results = [EnvRunResult(**r) for r in json.load(f)]

    print(f"\n===== Baseline performance (single run, seed={base_seed}) =====")
    display_metrics([r.model_copy(update={"trial": 0}) for r in baseline_results])

    # Relabel trial = seed index (0=baseline seed, 1..N-1=new seeds) purely for
    # this aggregate computation; does not touch the per-seed result files.
    combined: list[EnvRunResult] = [r.model_copy(update={"trial": 0}) for r in baseline_results]
    for trial_idx, results in enumerate(all_results, start=1):
        combined.extend(r.model_copy(update={"trial": trial_idx}) for r in results)

    print(f"\n===== Aggregate performance across {config.num_trials} run(s) (seeds {base_seed}..{base_seed + config.num_trials - 1}) =====")
    display_metrics(combined)

    # Best-of-N (control for the intervention run's "success if any of N
    # interventions worked" scoring): a task counts as a success here if ANY
    # of the N no-intervention seed runs solved it.
    best_per_task: dict[int, float] = {}
    for r in combined:
        if r.task_id not in best_per_task or r.reward > best_per_task[r.task_id]:
            best_per_task[r.task_id] = r.reward

    num_tasks = len(best_per_task)
    num_solved_by_any_seed = sum(1 for v in best_per_task.values() if is_successful(v))
    avg_best_reward = sum(best_per_task.values()) / num_tasks

    print(f"\n===== Best-of-{config.num_trials} performance (task counts as a success if ANY of the {config.num_trials} seeds solved it) =====")
    print(f"🏆 Average reward (best per task): {avg_best_reward}")
    print(f"✅ Tasks solved by at least one seed: {num_solved_by_any_seed}/{num_tasks} ({num_solved_by_any_seed / num_tasks:.2%})")


if __name__ == "__main__":
    main()
