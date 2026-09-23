# Spare parts balancing

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example, written for a
hands-on tutorial on deep reinforcement learning (DRL). A service organisation
owns a handful of expensive, repairable spare parts, spread over stock points
around the world. Parts fail, systems are down until a replacement arrives, the
failed units go back to the repair
shop in Amsterdam, and every repaired part raises the same question: **which
station gets it, or should it wait in Amsterdam?**

The example is about **modelling**: the whole model is plain Python in one
file, the hand-written policies sit next to it, and you can watch any policy
run the network, train a neural network to do the job, and compare on cost.

## The story

The part is expensive, so the pool owns exactly one per station: eight parts
for eight stock points, and about five of them are in the repair shop at any
time. Whenever a station ships its part to a system that is down it orders a
replacement, and the orders queue up in Amsterdam. Every time a repair
completes, the shop asks: which open order do we fill first? The textbook
answer is first come, first served. But a part sent to Singapore is out of
service for a day and a half, Miami's region has no other station nearby, and
a shop with five parts in repair will release the next one soon. There is room
to be smarter, and
that is the room a trained policy has to find.

What it costs: one per period (four hours) for every system that is down,
waiting for a part, and a lump sum when no station has a part at all and one has to be
borrowed elsewhere.

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
| `mdp.py` | **The model**: parts, stations, the repair shop, one period of time, the allocation decision, and the textbook policy. Start here. |
| `network.py` | The map (invented network, real cities), travel times, and `default_mdp()`, the configuration every script uses. Ordinary Python: change any number. |
| `featurizer.py` | What the neural network sees: the numbers written from a state. |
| `test_mdp.py` | Readable checks of the model, on hand-built situations. |
| `watch.py` | Animated map of any policy running the network. |
| `compare.py` | All policies on the same random failures and repair times, with paired differences. |
| `train.py` | One generation of Deep Controlled Learning from the textbook rule; saves `agents/trained`. |

## The policies

- **FirstComeFirstServed**: fill the oldest order. The textbook rule, and the only hand-written policy here.
- **Trained**: whatever `train.py` learns from the textbook rule's rollouts. Considerably cheaper.

The challenge: watch the trained policy, work out what it does differently,
and write that down as a rule of your own next to `FirstComeFirstServed`.
How close can an explainable rule get? Other things to try: change the
features, change the network size, give Amsterdam two parts, make the repair
shop slower, and see what each does.
