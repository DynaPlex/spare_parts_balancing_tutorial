# Spare parts balancing

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example, written for a
hands-on tutorial on simulation models, policies, and deep reinforcement
learning (DRL). In DRL, the computer learns good policies for the decisions
in a dynamic simulation model. In this tutorial, the model represents the
control of a spare parts network in response to dynamic, unpredictable
demand. The model is not company-specific, but it has characteristics that
match real-world challenges: try to understand the business context as well
as the code.

The whole model is plain Python in one file, a textbook policy sits next to
it, and the scripts let you watch a policy run, train a neural network with
DCL, and compare policies on cost. The model is still relatively basic, and
several real-world elements are not incorporated. After working through this
code, you may adapt it to include elements such as prognostics, lateral
transshipments, multiple repair shops, or multiple components. Alternatively,
you can start a whole new model that represents your own business.

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
.venv/bin/python train.py           # train a policy with DCL (a minute or two), then compare
.venv/bin/python watch.py --policy trained
```

(Windows: `python -m venv .venv`, then `.venv\Scripts\pip` and
`.venv\Scripts\python` in place of `.venv/bin/...`.)

## The files

| File | What it is |
|---|---|
| `mdp.py` | **The model**: parts, stock points, the repair shop, one period of time, the allocation decision, the textbook policy, and an empty `MyPolicy` for your own rule. Start here. |
| `network.py` | The map: one table of sites (invented network, real cities) with the installed base and which sites hold stock, travel times, and `default_mdp()`, the configuration every script uses. Ordinary Python: change any number, flip a flag. |
| `featurizer.py` | What the neural network sees: the numbers written from a state. |
| `test_mdp.py` | Readable checks of the model, on hand-built situations. |
| `watch.py` | Animated map of any policy running the network; `--step` to click through it day by day. |
| `world_map.json` | The countries `watch.py` draws behind the network: outlines from [Natural Earth](https://www.naturalearthdata.com) (1:110m, public domain). |
| `compare.py` | All policies on the same random failures and repair times, with paired differences. |
| `train.py` | One generation of Deep Controlled Learning from the textbook rule; saves `agents/trained`. |

## The policies

- **FirstComeFirstServed**: fill the oldest open order. The textbook rule.
- **Trained**: whatever `train.py` learns from the textbook rule's rollouts.
- **MyPolicy**: yours to write, at the bottom of `mdp.py`. Until you do, it
  stops with "not written yet". `python compare.py --mine` puts it next to the
  others on cost, and `python watch.py --policy mine` shows it on the map.

## Watching step by step

`python watch.py --step` shows the same map without the animation: nothing
moves by itself. Press the **right arrow** (or the space bar) for the next day,
hold it down to run on, and close the window to stop. With
`--periods-per-frame 1` every press is one four-hour period.

The line at the top shows the last decision Amsterdam took: that, step by
step, is how to find out what a trained policy does
(`python watch.py --policy trained --step`).

## After you change the model: delete `dynaplex_runs/`

`train.py` keeps its samples and trained networks in `dynaplex_runs/`, so that an
interrupted run can resume. DynaPlex 1.14 recognises an earlier run by its
settings and numbers, **not by the code**. If you change `mdp.py`,
`featurizer.py` or the policy that training starts from, and train again with
the same settings, it prints `agent_gen1 exists — skipping`, finishes in a
second, and gives you the policy that was trained on the old model.

So after any such change, delete the folder and train again:

```bash
rm -rf dynaplex_runs            # Windows: rmdir /s /q dynaplex_runs
.venv/bin/python train.py
```

Changing a number in `network.py` (the pool size, a cost, which sites hold
stock) is recognised, and needs no deleting.

## Questions, problems, ideas

Post an issue or a question at
[github.com/DynaPlex/DynaPlex/issues](https://github.com/DynaPlex/DynaPlex/issues).
The documentation, with tutorials and more examples, is at
[dynaplex.github.io/DynaPlex](https://dynaplex.github.io/DynaPlex/).
