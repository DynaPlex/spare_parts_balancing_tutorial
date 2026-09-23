"""What the neural network sees: a fixed-length vector of numbers describing
the state, written by `write_features`. Everything here is plain arithmetic
over the state; DynaPlex compiles it, like the model.

Guidance for the tutorial: the network can only learn from what is written
here. The features below cover the three things a good allocation depends on:
where the stock is (and is about to be), what is on order, and how busy the
repair shop is. The map itself (which station is far from which) is not a
feature: it never changes, so the network can learn it from the stock levels
by position. Add to them, drop some, retrain, and see what
changes.
"""
from dataclasses import dataclass
from typing import Final

from dynaplex.modelling import Featurizer, GlobalStateWriter, featurizer

from mdp import AMS, PartStatus, SparePartsMDP, State


@featurizer
@dataclass(slots=True)
class SparePartsFeaturizer(Featurizer):
    mdp: Final[SparePartsMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        # Per stock point: what it has, what is coming, what it is waiting for.
        for k in range(self.mdp.n_stock_points):
            point = state.stock_points[k]
            self.v.append(float(point.on_hand))
            self.v.append(float(point.inbound))
            self.v.append(float(len(point.open_orders)))
            # (The age of the oldest order, state.period - point.open_orders[0], would be
            # a natural feature too; DynaPlex 1.14.0 wrongly flags that read as a state
            # modification inside a featurizer. Lengths and is_empty() are fine.)

        # The repair shop: parts being repaired come out at a fixed rate each.
        self.v.append(float(state.queued))
        self.v.append(float(state.busy_servers))

        # The rest of the pool: on its way back, or on its way to a system that is down.
        returning = 0
        for part in state.parts:
            if part.status == PartStatus.RETURNING:
                returning += 1
        self.v.append(float(returning))
        self.v.append(float(state.systems_down))
        self.v.append(float(state.stock_points[AMS].on_hand))
