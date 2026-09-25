"""Spare parts balancing: a closed pool of repairable parts, one repair shop,
stock points around the world, and the question where each repaired part
should go — or whether it should go anywhere at all.

The setting. A service organisation owns a fixed number of interchangeable
parts. Systems all over the world contain this part, and it can fail. A failure
takes the system down until a serviceable part arrives, so it is served at once
from the nearest stock point that has one. The failed unit
travels back to the repair shop in Amsterdam (AMS), is repaired, and becomes
AMS stock again: the loop is closed.

Locations are numbered; 0 .. n_stock_points-1 are the stock points, and stock
point 0 is AMS, which is also the repair shop. The remaining locations are
sites without stock. `network.py` holds the names and the map.

Time. One period is a few hours (`network.py` says how many). Every period:

  1. Travel. Every travelling part arrives with probability 1 / (mean travel
     time of its leg), so travel times are geometric and the state needs no
     clocks for them. Every leg takes at least one period, also from a site's
     own shelf: installing takes time too.
       - a part sent to a regional stock point joins its stock;
       - a part that reaches a system that is down is installed, and the failed
         unit it replaces starts its way back to AMS;
       - a failed unit that reaches AMS enters the repair shop.
  2. Repair. The shop has `repair_servers` parallel servers; failed units
     wait for a free one. The parts are all alike, so it does not matter which
     waiting unit goes first, and the queue is just a count. Every busy server has, every period, the same fixed
     probability 1 / `repair_mean` of finishing its repair, so repair times are
     geometric, like travel times, and the state needs no clocks for them
     either. This is not realistic: a repair that has been going on for weeks
     is no closer to done than one that started today. We do it only to keep
     the code simple and readable.
  3. Demand. With large probability nothing happens; otherwise one part fails,
     at a location drawn from `demand_prob`. The nearest stock point (smallest
     mean travel time; its own shelf first) with a part on hand ships one. If no stock point has a
     part on hand, the organisation borrows one elsewhere: a LOAN, at `loan_cost`,
     which does not touch our pool.
  4. Cost. `downtime_cost` per period for every system that is down, waiting
     for its part. In the tutorial's story downtime costs 10K per hour, so 40K
     per four-hour period, and a loan 1.6M (`network.py` sets both).

Base stock. Regional stock point k has a fixed base-stock level: whenever it
ships a part to a customer it orders one from AMS, and records the period of
the order. Hence, always,

    on_hand + inbound + len(open_orders) == base_stock        (k >= 1)

AMS has no level of its own: it holds whatever is not elsewhere. The pool
(`pool_size` parts) may be smaller than the sum of the base-stock levels;
then some orders stay open until a repaired part is sent their way.

The decision is the ALLOCATION of AMS stock to those orders. It comes up when
AMS has a part on hand and at least one order is open. Action k >= 1 sends a
part to stock point k (valid if k has an open order; its oldest order is then
filled). Action 0 holds: nothing is sent, and the question is not asked again until
something changes: a repair completes, or a new order opens somewhere. Holding is a real option: a part on
its way to Singapore serves nobody for a day, and is a poor answer to the next
failure in Miami.

A part is always in exactly one `PartStatus`; `parts` is the truth, and the
counts on the stock points mirror it so that policies and features need not
scan. (With hundreds of parts a heap of repair completions would beat the scan
in step 2; with a few dozen, one list and one loop is easier to read.)

Infinite horizon; the objective is the average cost per period.
"""
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

import dynaplex
from dynaplex import const_dataclass, mdp, policy
from dynaplex.modelling import (
    AliasSampler,
    ConstArray2D,
    ConstList,
    DiscreteDist,
    FifoQueue,
    HorizonType,
    StateCategory,
    TrajectoryContext,
    Validity,
)

AMS = 0


class PartStatus(Enum):
    STOCK = auto()          # serviceable, on hand at stock point `origin`
    OUTBOUND = auto()       # serviceable, travelling from AMS to stock point `dest`
    TO_CUSTOMER = auto()    # serviceable, travelling from `origin` to the down system at `dest`
    RETURNING = auto()      # failed, travelling from `origin` back to AMS
    REPAIR_QUEUE = auto()   # failed, at AMS, waiting for a free server
    IN_REPAIR = auto()      # being repaired at AMS


@dataclass(slots=True)
class Part:
    status: PartStatus
    origin: int             # location where it is, or that it left
    dest: int               # location it travels to (travelling statuses only)


@dataclass(slots=True)
class StockPoint:
    on_hand: int
    inbound: int            # parts OUTBOUND to this stock point
    open_orders: FifoQueue  # period in which each unfilled order was placed, oldest first


@dataclass(slots=True)
class State:
    parts: list[Part]
    stock_points: list[StockPoint]
    queued: int                 # REPAIR_QUEUE parts: failed units waiting for a server
    busy_servers: int
    systems_down: int           # TO_CUSTOMER parts: systems waiting for one
    orders_open: int            # unfilled orders, over all stock points
    period: int
    holding: bool               # the last decision was to hold; cleared when something changes
    category: StateCategory


@mdp
@const_dataclass(init=False, slots=True)
class SparePartsMDP:
    """See the module docstring."""

    n_stock_points: int
    n_locations: int
    n_parts: int
    mean_travel: ConstArray2D[np.float64]   # [from, to]: mean travel time in periods, at least 1
    travel_prob: ConstArray2D[np.float64]   # [from, to]: per-period arrival probability
    demand_prob: ConstList[float]           # per location, per period
    base_stock: ConstList[int]              # per stock point; [0] (AMS) is 0
    repair_servers: int
    repair_prob: float                      # per busy server, per period: chance the repair is done
    downtime_cost: float
    loan_cost: float

    # derived, for sampling:
    event_sampler: AliasSampler     # 0 = no demand this period, i + 1 = demand at location i

    num_actions: int
    horizon_type: HorizonType

    def __init__(self, n_stock_points: int, mean_travel_time: np.ndarray,
                 demand_prob: list[float], pool_size: int, base_stock: list[int],
                 repair_mean: float, repair_servers: int,
                 downtime_cost: float = 40_000.0, loan_cost: float = 1_600_000.0):
        # some validations:
        n_locations = len(demand_prob)
        if mean_travel_time.shape != (n_locations, n_locations):
            raise ValueError("mean_travel_time must be n_locations x n_locations")
        if np.any(mean_travel_time < 1.0):
            raise ValueError("a shipment takes at least one period, also from the site's own shelf")
        if len(base_stock) != n_stock_points or min(base_stock) < 0:
            raise ValueError("base_stock needs one non-negative entry per stock point")
        if base_stock[AMS] != 0:
            raise ValueError("base_stock[0] must be 0: AMS holds whatever is not elsewhere")
        if pool_size < 1:
            raise ValueError("the pool needs at least one part")
        if repair_mean < 1.0:
            raise ValueError("a repair takes at least one period on average")
        total_demand_prob = sum(demand_prob)
        if min(demand_prob) < 0.0 or total_demand_prob >= 1.0:
            raise ValueError("demand_prob: probabilities per period, summing to less than 1")
        # The repair shop must keep up with the failures, or its queue will swallow the pool. 
        load = total_demand_prob * repair_mean / repair_servers
        if load >= 1.0:
            raise ValueError(
                f"unstable repair shop: {total_demand_prob:.4f} failures per period x "
                f"{repair_mean:.1f} periods per repair needs more than "
                f"{repair_servers} servers (load {load:.2f}, must be below 1)")

        self.n_stock_points = n_stock_points
        self.n_locations = n_locations
        self.n_parts = pool_size
        self.mean_travel = mean_travel_time.astype(np.float64)
        self.travel_prob = 1.0 / mean_travel_time
        self.demand_prob = list(demand_prob)
        self.base_stock = list(base_stock)
        self.repair_servers = repair_servers
        self.repair_prob = 1.0 / repair_mean
        self.downtime_cost = downtime_cost
        self.loan_cost = loan_cost
        self.event_sampler = DiscreteDist.custom(
            [1.0 - total_demand_prob] + list(demand_prob)).alias_sampler()
        self.num_actions = n_stock_points
        self.horizon_type = HorizonType.INFINITE

    # ---- helpers ----------------------------------------------------------

    def _nearest_with_stock(self, state: State, location: int) -> int:
        """The stock point with a part on hand that reaches `location` fastest
        (ties: the site's own shelf); -1 if none."""
        best = -1
        best_time = 0.0
        for k in range(self.n_stock_points):
            if state.stock_points[k].on_hand > 0:
                time = self.mean_travel[k, location]
                if best < 0 or time < best_time or (time == best_time and k == location):
                    best = k
                    best_time = time
        return best

    def _take_from_stock(self, state: State, k: int) -> int:
        """Index of a part on hand at stock point `k`; the counts are updated."""
        for index in range(self.n_parts):
            if state.parts[index].status == PartStatus.STOCK and state.parts[index].origin == k:
                state.stock_points[k].on_hand -= 1
                return index
        dynaplex.fail("no part on hand at this stock point: the counts and the parts disagree")

    def _set_category(self, state: State) -> None:
        """An allocation decision is due when AMS has a part, somebody has an
        order open, and we are not inside a hold."""
        decision_due = state.stock_points[AMS].on_hand > 0 and state.orders_open > 0 and not state.holding
        state.category = StateCategory.AWAIT_ACTION if decision_due else StateCategory.AWAIT_EVENT

    # ---- MDP contract -----------------------------------------------------

    def get_initial_state(self, context: TrajectoryContext) -> State:
        """The whole pool on the shelf in AMS and every regional stock point
        waiting for its base stock: the first decisions position the pool.
        (A pool smaller than the levels leaves some of those orders open.)"""
        parts: list[Part] = []
        for _ in range(self.n_parts):
            parts.append(Part(status=PartStatus.STOCK, origin=AMS, dest=AMS))
        stock_points: list[StockPoint] = []
        stock_points.append(StockPoint(on_hand=self.n_parts, inbound=0, open_orders=FifoQueue()))
        orders_open = 0
        for k in range(1, self.n_stock_points):
            orders = FifoQueue()
            for _ in range(self.base_stock[k]):
                orders.push_back(0)
            stock_points.append(StockPoint(on_hand=0, inbound=0, open_orders=orders))
            orders_open += self.base_stock[k]
        state = State(parts=parts, stock_points=stock_points, queued=0, busy_servers=0, systems_down=0,
                      orders_open=orders_open, period=0, holding=False, category=StateCategory.AWAIT_EVENT)
        self._set_category(state)
        return state

    def modify_state_with_action(self, state: State, context: TrajectoryContext,
                                 action: int) -> None:
        if action == 0:
            state.holding = True
        else:
            part = state.parts[self._take_from_stock(state, AMS)]
            part.status = PartStatus.OUTBOUND
            part.dest = action
            destination = state.stock_points[action]
            destination.inbound += 1
            destination.open_orders.pop_front()
            state.orders_open -= 1
        self._set_category(state)

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        state.period += 1

        # 1 + 2. travel, and the repairs in progress: one pass over the parts.
        # Every part draws one random number every period, whether it needs one or
        # not, so that a period always consumes the same number of draws whatever
        # the state: rollouts from the same seed then see the same failures and the
        # same travel and repair luck regardless of the decision taken (common
        # random numbers), and comparisons between decisions are far less noisy.
        for index in range(self.n_parts):
            part = state.parts[index]
            u = context.rng.random()
            if part.status == PartStatus.OUTBOUND:
                if u < self.travel_prob[part.origin, part.dest]:
                    part.status = PartStatus.STOCK
                    part.origin = part.dest
                    state.stock_points[part.dest].inbound -= 1
                    state.stock_points[part.dest].on_hand += 1
            elif part.status == PartStatus.TO_CUSTOMER:
                if u < self.travel_prob[part.origin, part.dest]:
                    # installed; from here on this slot is the failed unit it replaced
                    state.systems_down -= 1
                    part.status = PartStatus.RETURNING
                    part.origin = part.dest
                    part.dest = AMS
            elif part.status == PartStatus.RETURNING:
                if u < self.travel_prob[part.origin, part.dest]:
                    part.status = PartStatus.REPAIR_QUEUE
                    part.origin = AMS
                    state.queued += 1
            elif part.status == PartStatus.IN_REPAIR:
                # every busy server finishes with the same probability, every period
                if u < self.repair_prob:
                    part.status = PartStatus.STOCK
                    state.stock_points[AMS].on_hand += 1
                    state.busy_servers -= 1
                    state.holding = False               # new supply ends a hold

        # 2. the shop: free servers take waiting units (any of them: the parts are alike)
        if state.queued > 0 and state.busy_servers < self.repair_servers:
            for part in state.parts:
                if part.status == PartStatus.REPAIR_QUEUE and state.busy_servers < self.repair_servers:
                    part.status = PartStatus.IN_REPAIR
                    state.queued -= 1
                    state.busy_servers += 1

        # 3. demand
        event = self.event_sampler.sample(context.rng)
        if event > 0:
            location = event - 1
            k = self._nearest_with_stock(state, location)
            if k < 0:
                context.cumulative_cost += self.loan_cost
            else:
                shipped = state.parts[self._take_from_stock(state, k)]
                shipped.status = PartStatus.TO_CUSTOMER
                shipped.dest = location
                state.systems_down += 1
                if k != AMS:
                    state.stock_points[k].open_orders.push_back(state.period)
                    state.orders_open += 1
                    state.holding = False               # a new order ends a hold

        # 4. cost
        context.cumulative_cost += self.downtime_cost * state.systems_down
        context.time_elapsed += 1
        self._set_category(state)

    def write_action_validity(self, state: State, valid: Validity) -> None:
        valid.set(0, True)
        for k in range(1, self.n_stock_points):
            valid.set(k, not state.stock_points[k].open_orders.is_empty())


# ---- the hand-written allocation policy -------------------------------------


# @policy
@const_dataclass(slots=True)
class FirstComeFirstServed:
    """The textbook rule: fill the oldest open order. Never holds."""

    mdp: SparePartsMDP

    def get_action(self, state: State) -> int:
        best = 0
        oldest = state.period + 1
        for k in range(1, self.mdp.n_stock_points):
            point = state.stock_points[k]
            if not point.open_orders.is_empty() and point.open_orders[0] < oldest:
                best = k
                oldest = point.open_orders[0]
        return best
