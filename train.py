"""Train a neural-network policy with Deep Controlled Learning (DCL), one
generation: from every state a hand-written base policy would visit, try each
allowed action, roll the base policy out from there, and let the network learn
which action led to the lowest cost. Then compare.

    python train.py                     # from the textbook rule; about a minute on a laptop
    python train.py --samples 16000     # more samples, better (and slower)

Two things make the labels usable here: paired rollouts (the model draws its
random numbers so that every candidate action sees the same failures) and
soft labels (near-ties between actions become near-equal targets instead of
coin flips).

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
from mdp import FirstComeFirstServed
from network import default_mdp



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=8000, help="labelled decision states (n)")
    parser.add_argument("--rollouts", type=int, default=256, help="rollouts per candidate action (m)")
    parser.add_argument("--horizon", type=int, default=540, help="rollout length in periods (90 days)")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--loss", choices=["soft_ce", "ce", "count_ce"], default="soft_ce",
                        help="soft_ce = near-ties get near-equal targets; ce = the winning action only")
    parser.add_argument("--out", default=os.path.join("agents", "trained"))
    args = parser.parse_args()

    mdp = default_mdp()
    base = FirstComeFirstServed(mdp)
    dcl = dynaplex.DCL(
        mdp, base, features=SparePartsFeaturizer,
        n=args.samples, m=args.rollouts, h=args.horizon,
        network=dynaplex.MLP(hidden=(64, 64)),
        train=dict(loss=args.loss, epochs=60, batch_size=64, lr=1e-3, patience=10, val_fraction=0.1),
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
        **{f"Trained (generation {a.info['generation']})": a for a in agents},
    }))


if __name__ == "__main__":
    main()
