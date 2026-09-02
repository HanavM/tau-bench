# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.placeholder import run_placeholder
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Placeholder-intervention control: reuses the real intervention "
            "pipeline's failure-point identification (agent.run_intervention) "
            "but replaces the generated corrective text with a fixed, "
            "content-free placeholder before continuing. Isolates whether "
            "the intervention's *content* is doing anything, vs. just the "
            "architecture of resuming from a fixed failed-trajectory prefix."
        )
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to the completed baseline run folder (must contain transcript.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
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
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="react-intervened",
        choices=["react-intervened"],
        help="Only react-intervened supports run_intervention/solve_with_intervention",
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
    )
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) only run placeholder on these task IDs")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument(
        "--best_of_N",
        type=int,
        default=3,
        help="Number of placeholder attempts per failing task (should match the real intervention run's N)",
    )

    args = parser.parse_args()
    print(args)

    config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        user_strategy=args.user_strategy,
        best_of_N=args.best_of_N,
        run_intervention=False,
        baseline_path=args.baseline_path,
    )
    return config


def main():
    config = parse_args()
    run_placeholder(config)


if __name__ == "__main__":
    main()
