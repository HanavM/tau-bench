# Copyright Sierra

"""Report Best-of-(k+1) reflexion performance for one or more k, reusing an
already-completed reflexion run's saved transcripts. Makes NO new rollouts
or LLM calls - if you already ran --best_of_N 5, you can see Bo1..Bo5 (or
any subset) for free by pointing this at that same folder.
"""

import argparse
from tau_bench.reflexion import report_at_k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to the baseline run folder (must contain transcript.json).",
    )
    parser.add_argument(
        "--reflexion_folder",
        type=str,
        required=True,
        help="Path to the completed reflexion-by_<model>_<timestamp> folder (must contain reflexion-transcripts.json).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        required=True,
        help="One or more k values (max reflexion attempts to include). e.g. --k 1 2 3 4 5",
    )
    args = parser.parse_args()

    report_at_k(args.baseline_path, args.reflexion_folder, sorted(set(args.k)))


if __name__ == "__main__":
    main()
