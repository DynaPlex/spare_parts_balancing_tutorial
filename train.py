"""Train a neural-network policy with Deep Controlled Learning (DCL), one
generation: from every state a hand-written base policy would visit, try each
allowed action, roll the base policy out from there, and let the network learn
which action led to the lowest cost. Then compare.

    python train.py                     # from MostExposedFirst, a few minutes on a laptop
    python train.py --base fcfs         # from the textbook rule
    python train.py --samples 8000      # more samples, better (and slower)

The trained policy is written to agents/trained, where compare.py and
watch.py --policy trained pick it up. Runs resume: the samples and agents land
in dynaplex_runs/, and rerunning with the same settings reuses them.
"""
import argparse
import os
import shutil
import time

import dynaplex

from featurizer import SparePartsFeaturizer
from mdp import FirstComeFirstServed, MostExposedFirst
from network import default_mdp

BASE_POLICIES = {
    "exposed": lambda mdp: MostExposedFirst(mdp, reserve=1),
    "fcfs": lambda mdp: FirstComeFirstServed(mdp),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", choices=BASE_POLICIES, default="exposed", help="the policy to improve on")
    parser.add_argument("--samples", type=int, default=4000, help="labelled decision states (n)")
    parser.add_argument("--rollouts", type=int, default=64, help="rollouts per candidate action (m)")
    parser.add_argument("--horizon", type=int, default=360, help="rollout length in periods (60 days)")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--out", default=os.path.join("agents", "trained"))
    args = parser.parse_args()

    mdp = default_mdp()
    base = BASE_POLICIES[args.base](mdp)
    dcl = dynaplex.DCL(
        mdp, base, features=SparePartsFeaturizer,
        n=args.samples, m=args.rollouts, h=args.horizon,
        network=dynaplex.MLP(hidden=(64, 64)),
        train=dict(loss="ce", epochs=60, batch_size=64, lr=1e-3, patience=10, val_fraction=0.1),
    )
    started = time.time()
    agents = dcl.run(generations=args.generations)
    agent = agents[-1]
    shutil.rmtree(args.out, ignore_errors=True)     # a rerun replaces the previous policy
    agent.save(args.out)
    print(f"\ntrained in {time.time() - started:.0f} s; policy saved to {args.out}\n")

    comparer = dynaplex.PolicyComparer(mdp, number_of_trajectories=512, warmup_time=1500,
                                       horizon=9000, seed=0, checks=False)
    print(comparer.compare({
        "FirstComeFirstServed": FirstComeFirstServed(mdp),
        "MostExposedFirst(reserve=1)": MostExposedFirst(mdp, reserve=1),
        **{f"Trained (generation {a.info['generation']})": a for a in agents},
    }))


if __name__ == "__main__":
    main()
