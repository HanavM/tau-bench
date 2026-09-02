# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.reflexion import run_reflexion
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reflexion control: given an already-completed baseline run, for each "
            "failing task runs up to --best_of_N sequential from-scratch rollouts, "
            "each seeded with a self-written natural-language reflection on why the "
            "previous attempt failed (https://arxiv.org/abs/2303.11366). Stops early "
            "per task the moment an attempt succeeds. Compare its Best-of-(N+1) "
            "number against the real intervention Bo-N and the placeholder Bo-N runs "
            "to see how this different kind of extra-attempt budget compares."
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
        "--intervention_model",
        type=str,
        default="gpt-4o",
        help="The model used to write reflections (kept separate from the agent model, same convention as the intervention pipeline)",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--intervention-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the reflection-writer model (defaults to the agent's provider)",
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
        default="react-reflexion",
        choices=["react-reflexion"],
        help="Only react-reflexion supports reflect/solve_with_reflection",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The sampling temperature for the action model (reflexion relies on sampling diversity across attempts, so a non-zero temperature is recommended)",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
    )
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) only run reflexion on these task IDs")
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
        help="Max number of NEW reflexion attempts per failing task (not counting the baseline attempt); a task stops early once it succeeds",
    )

    args = parser.parse_args()
    print(args)

    config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        intervention_model=args.intervention_model,
        intervention_model_provider=args.intervention_model_provider,
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
    run_reflexion(config)


if __name__ == "__main__":
    main()
