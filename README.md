# Spare parts balancing

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example, written for a
hands-on tutorial on simulation models, policies, and deep reinforcement
learning (DRL). The model is not company-specific, but it has characteristics
that match real-world challenges: try to understand the business context as
well as the code. The whole model is plain Python in one file, a textbook
policy sits next to it, and the scripts let you watch a policy run, train a
neural network with DCL, and compare policies on cost.

## The model

A service organisation owns a pool of eight identical, expensive, repairable
spare parts. Two hundred systems at 33 sites all over the world contain this
part, and now and then one fails: about one failure every two weeks, more
often where there are more systems. A failed system is down until a
serviceable part is installed.

- **Stock points.** Eight sites can hold stock, one part each: Amsterdam,
  Paris, Miami, Dubai, Singapore, Kuala Lumpur, São Paulo and Shanghai.
  Amsterdam is also the repair shop. The other 25 sites hold nothing. The pool
  starts in Amsterdam, and the first decisions position it. (`network.py` is the
  table: flip `can_hold_stock` on a site, or change the pool size, and every
  script follows.)
- **Fulfilment.** A failure is served from the nearest stock point that has a
  part on hand, its own shelf first. The part flies there (one period from the
  own shelf, up to ten across the world; locations carry the code of their
  nearest airport), is installed, and the failed unit travels back to Amsterdam
  for repair. If no stock point has a part, one is borrowed.
- **Repair.** Six repair servers; a repair takes ten weeks on average. A repaired
  part is Amsterdam stock again, so the pool is closed.
- **Orders.** A stock point that ships its part orders a replacement from
  Amsterdam. Orders wait until Amsterdam decides to fill them.
- **The decision.** Whenever Amsterdam has a part on hand and orders are open,
  it chooses which order to fill, or holds the part until the next repair
  completes or the next order opens.
- **Cost.** Downtime costs 10K per hour: 40K per period (four hours) for every
  system that is down. A borrowed part costs 1.6M. The objective is the average
  cost per period.

Travel and repair times are geometric: every period, a travelling part arrives
and a busy server finishes with a fixed probability. That is not realistic; it
keeps the state free of clocks and the code short.

## Quickstart

Requires **Python 3.11–3.14** on Windows, macOS or Linux.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # the model's own checks (a few seconds)
.venv/bin/python watch.py           # a window: watch the textbook rule run
.venv/bin/python compare.py         # the policies on cost, paired
.venv/bin/python train.py           # train a policy with DCL (about a minute), then compare
.venv/bin/python watch.py --policy trained
```

(Windows: `python -m venv .venv`, then `.venv\Scripts\pip` and
`.venv\Scripts\python` in place of `.venv/bin/...`.)

## The files

| File | What it is |
|---|---|
| `mdp.py` | **The model**: parts, stock points, the repair shop, one period of time, the allocation decision, and the textbook policy. Start here. |
| `network.py` | The map: one table of sites (invented network, real cities) with the installed base and which sites hold stock, travel times, and `default_mdp()`, the configuration every script uses. Ordinary Python: change any number, flip a flag. |
| `featurizer.py` | What the neural network sees: the numbers written from a state. |
| `test_mdp.py` | Readable checks of the model, on hand-built situations. |
| `watch.py` | Animated map of any policy running the network. |
| `compare.py` | All policies on the same random failures and repair times, with paired differences. |
| `train.py` | One generation of Deep Controlled Learning from the textbook rule; saves `agents/trained`. |

## The policies

- **FirstComeFirstServed**: fill the oldest open order. The textbook rule.
- **Trained**: whatever `train.py` learns from the textbook rule's rollouts.
