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
  2. Repair. The shop has `repair_servers` parallel servers and a first-come
     first-served queue. A repair time is drawn from `repair_time` — any
     distribution — when the repair STARTS, and the part records the period it
     will be finished. So the completion dates of the repairs in progress are
     in the state: a policy may use the shop's outlook. (Drawing the time when
     a part JOINS the queue would extend that outlook to the queue.)
  3. Demand. With large probability nothing happens; otherwise one part fails,
     at a location drawn from `demand_prob`. The nearest stock point (smallest
     mean travel time; its own shelf first) with a part on hand ships one. If no stock point has a
     part on hand, the organisation borrows one elsewhere: a LOAN, at `loan_cost`,
     which does not touch our pool.
  4. Cost. `downtime_cost` per period for every system that is down, waiting
     for its part.

Base stock. Regional stock point k has a fixed base-stock level: whenever it
ships a part to a customer it orders one from AMS, and records the period of
the order. Hence, always,

    on_hand + inbound + len(open_orders) == base_stock        (k >= 1)

AMS has whatever is not elsewhere.

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
    REPAIR_QUEUE = auto()   # failed, at AMS, waiting for a free repair server
    IN_REPAIR = auto()      # being repaired; serviceable at `repair_done_at`


@dataclass(slots=True)
class Part:
    status: PartStatus
    origin: int             # location where it is, or that it left
    dest: int               # location it travels to (travelling statuses only)
    repair_done_at: int     # IN_REPAIR only


@dataclass(slots=True)
class StockPoint:
    on_hand: int
    inbound: int            # parts OUTBOUND to this stock point
    open_orders: FifoQueue  # period in which each unfilled order was placed, oldest first


@dataclass(slots=True)
class State:
    parts: list[Part]
    stock_points: list[StockPoint]
    repair_queue: FifoQueue     # indices into `parts`, first come first served
    busy_servers: int
    systems_down: int           # TO_CUSTOMER parts: systems waiting for one
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
    base_stock: ConstList[int]              # [0] is the initial AMS stock
    repair_servers: int
    downtime_cost: float
    loan_cost: float

    # derived, for sampling:
    event_sampler: AliasSampler     # 0 = no demand this period, i + 1 = demand at location i
    repair_sampler: AliasSampler

    num_actions: int
    horizon_type: HorizonType

    def __init__(self, n_stock_points: int, mean_travel_time: np.ndarray,
                 demand_prob: list[float], base_stock: list[int],
                 repair_time: DiscreteDist, repair_servers: int,
                 downtime_cost: float = 1.0, loan_cost: float = 40.0):
        # some validations:
        n_locations = len(demand_prob)
        if mean_travel_time.shape != (n_locations, n_locations):
            raise ValueError("mean_travel_time must be n_locations x n_locations")
        if np.any(mean_travel_time < 1.0):
            raise ValueError("a shipment takes at least one period, also from the site's own shelf")
        if len(base_stock) != n_stock_points or min(base_stock) < 0:
            raise ValueError("base_stock needs one non-negative entry per stock point")
        if repair_time.min() < 1:
            raise ValueError("a repair takes at least one period")
        total_demand_prob = sum(demand_prob)
        if min(demand_prob) < 0.0 or total_demand_prob >= 1.0:
            raise ValueError("demand_prob: probabilities per period, summing to less than 1")
        # The repair shop must keep up with the failures, or its queue will swallow the pool. 
        load = total_demand_prob * repair_time.expectation() / repair_servers
        if load >= 1.0:
            raise ValueError(
                f"unstable repair shop: {total_demand_prob:.4f} failures per period x "
                f"{repair_time.expectation():.1f} periods per repair needs more than "
                f"{repair_servers} servers (load {load:.2f}, must be below 1)")

        self.n_stock_points = n_stock_points
        self.n_locations = n_locations
        self.n_parts = sum(base_stock)
        self.mean_travel = mean_travel_time.astype(np.float64)
        self.travel_prob = 1.0 / mean_travel_time
        self.demand_prob = list(demand_prob)
        self.base_stock = list(base_stock)
        self.repair_servers = repair_servers
        self.downtime_cost = downtime_cost
        self.loan_cost = loan_cost
        self.event_sampler = DiscreteDist.custom(
            [1.0 - total_demand_prob] + list(demand_prob)).alias_sampler()
        self.repair_sampler = repair_time.alias_sampler()
        self.num_actions = n_stock_points
        self.horizon_type = HorizonType.INFINITE

    # ---- helpers ----------------------------------------------------------

    def _enter_repair_shop(self, state: State, context: TrajectoryContext, index: int) -> None:
        part = state.parts[index]
        part.origin = AMS
        if state.busy_servers < self.repair_servers:
            state.busy_servers += 1
            part.status = PartStatus.IN_REPAIR
            part.repair_done_at = state.period + self.repair_sampler.sample(context.rng)
        else:
            part.status = PartStatus.REPAIR_QUEUE
            state.repair_queue.push_back(index)

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
        state.category = StateCategory.AWAIT_EVENT
        if state.stock_points[AMS].on_hand > 0 and not state.holding:
            for k in range(1, self.n_stock_points):
                if not state.stock_points[k].open_orders.is_empty():
                    state.category = StateCategory.AWAIT_ACTION

    def exposure(self, state: State, k: int) -> float:
        """How much stock point `k` is missed right now: the expected extra travel
        per period (demand probability x extra periods, summed over the world)
        that failures suffer because `k` has nothing on hand or on its way, and
        are served from the nearest stock point that does. 0 if `k` is covered.
        A part that nobody can supply counts as a loan-length wait."""
        own = state.stock_points[k]
        if own.on_hand + own.inbound > 0:
            return 0.0
        total = 0.0
        for loc in range(self.n_locations):
            covered = -1.0
            for j in range(self.n_stock_points):
                point = state.stock_points[j]
                if point.on_hand + point.inbound > 0 and (
                        covered < 0.0 or self.mean_travel[j, loc] < covered):
                    covered = self.mean_travel[j, loc]
            if covered < 0.0:
                covered = self.loan_cost / self.downtime_cost
            extra = covered - self.mean_travel[k, loc]
            if extra > 0.0:
                total += self.demand_prob[loc] * extra
        return total

    # ---- MDP contract -----------------------------------------------------

    def get_initial_state(self, context: TrajectoryContext) -> State:
        """Every stock point at its base stock, nothing moving, nothing broken."""
        parts: list[Part] = []
        stock_points: list[StockPoint] = []
        for k in range(self.n_stock_points):
            stock_points.append(StockPoint(
                on_hand=self.base_stock[k], inbound=0, open_orders=FifoQueue()))
            for _ in range(self.base_stock[k]):
                parts.append(Part(status=PartStatus.STOCK, origin=k, dest=k, repair_done_at=0))
        return State(parts=parts, stock_points=stock_points, repair_queue=FifoQueue(),
                     busy_servers=0, systems_down=0, period=0, holding=False,
                     category=StateCategory.AWAIT_EVENT)

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
        self._set_category(state)

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        state.period += 1

        # 1 + 2. travel and repair: one pass over the parts
        for index in range(self.n_parts):
            part = state.parts[index]
            if part.status == PartStatus.OUTBOUND:
                if context.rng.random() < self.travel_prob[part.origin, part.dest]:
                    part.status = PartStatus.STOCK
                    part.origin = part.dest
                    state.stock_points[part.dest].inbound -= 1
                    state.stock_points[part.dest].on_hand += 1
            elif part.status == PartStatus.TO_CUSTOMER:
                if context.rng.random() < self.travel_prob[part.origin, part.dest]:
                    # installed; from here on this slot is the failed unit it replaced
                    state.systems_down -= 1
                    part.status = PartStatus.RETURNING
                    part.origin = part.dest
                    part.dest = AMS
            elif part.status == PartStatus.RETURNING:
                if context.rng.random() < self.travel_prob[part.origin, part.dest]:
                    self._enter_repair_shop(state, context, index)
            elif part.status == PartStatus.IN_REPAIR:
                if part.repair_done_at <= state.period:
                    part.status = PartStatus.STOCK
                    state.stock_points[AMS].on_hand += 1
                    state.busy_servers -= 1
                    state.holding = False               # new supply ends a hold
                    if not state.repair_queue.is_empty():
                        self._enter_repair_shop(state, context, state.repair_queue.pop_front())

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
                    state.holding = False               # a new order ends a hold

        # 4. cost
        context.cumulative_cost += self.downtime_cost * state.systems_down
        context.time_elapsed += 1
        self._set_category(state)

    def write_action_validity(self, state: State, valid: Validity) -> None:
        valid.set(0, True)
        for k in range(1, self.n_stock_points):
            valid.set(k, not state.stock_points[k].open_orders.is_empty())


# ---- hand-written allocation policies ---------------------------------------


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


# @policy
@const_dataclass(slots=True)
class EmptiestFirst:
    """Send to the stock point with the least stock on hand or on its way
    (ties: the older order) — but keep `reserve` parts in AMS, which can reach
    any site in the world."""

    mdp: SparePartsMDP
    reserve: int = 1

    def get_action(self, state: State) -> int:
        if state.stock_points[AMS].on_hand <= self.reserve:
            return 0
        best = 0
        least = self.mdp.n_parts + 1
        oldest = state.period + 1
        for k in range(1, self.mdp.n_stock_points):
            point = state.stock_points[k]
            if not point.open_orders.is_empty():
                position = point.on_hand + point.inbound
                order = point.open_orders[0]
                if position < least or (position == least and order < oldest):
                    best = k
                    least = position
                    oldest = order
        return best


# @policy
@const_dataclass(slots=True)
class MostExposedFirst:
    """Cover the world: fill the open order of the stock point whose region
    suffers most from its absence (`SparePartsMDP.exposure`) — but keep
    `reserve` parts in AMS. A one-line idea that beats first-come first-served
    by about a tenth; the challenge for a trained policy is to beat this."""

    mdp: SparePartsMDP
    reserve: int = 1

    def get_action(self, state: State) -> int:
        if state.stock_points[AMS].on_hand <= self.reserve:
            return 0
        best = 0
        most = -1.0
        for k in range(1, self.mdp.n_stock_points):
            if not state.stock_points[k].open_orders.is_empty():
                exposure = self.mdp.exposure(state, k)
                if exposure > most:
                    best = k
                    most = exposure
        return best
