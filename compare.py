"""Compare the policies on cost, on common random numbers: every policy sees
the same failures and the same repair times, so the differences are paired.

    python compare.py                  # the hand-written policies (about 10 s)
    python compare.py --trajectories 2048

A trained policy (see train.py) is included automatically when agents/trained
exists.
"""
import argparse
import os

import dynaplex

from mdp import EmptiestFirst, FirstComeFirstServed, MostExposedFirst
from network import default_mdp

TRAINED_AGENT = os.path.join("agents", "trained")


def policies(mdp) -> dict:
    result = {
        "FirstComeFirstServed": FirstComeFirstServed(mdp),
        "Random": dynaplex.RandomPolicy(mdp),
        "FirstComeFirstServed(reserve=1)": FirstComeFirstServed(mdp, reserve=1),
        "EmptiestFirst(reserve=1)": EmptiestFirst(mdp, reserve=1),
        "MostExposedFirst(reserve=0)": MostExposedFirst(mdp, reserve=0),
        "MostExposedFirst(reserve=1)": MostExposedFirst(mdp, reserve=1),
    }
    if os.path.isdir(TRAINED_AGENT):
        result["Trained"] = dynaplex.NNAgent.load(TRAINED_AGENT)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1500, help="periods before costs count")
    parser.add_argument("--horizon", type=int, default=9000, help="periods that count (1500 days)")
    args = parser.parse_args()

    mdp = default_mdp()
    comparer = dynaplex.PolicyComparer(
        mdp, number_of_trajectories=args.trajectories, warmup_time=args.warmup,
        horizon=args.horizon, seed=0, checks=False)
    print(f"cost per period (4 hours), {args.trajectories} runs of {args.horizon} periods each:\n")
    print(comparer.compare(policies(mdp)))


if __name__ == "__main__":
    main()
