"""Readable checks of the model: the map, the numbering of the locations, the
initial state, the accounting identity, fulfilment from the nearest shelf,
holding, and the textbook policy on hand-built situations. Run with
`python -m pytest`."""
import numpy as np
import pytest

import dynaplex
from dynaplex.modelling import StateCategory, new_context

from featurizer import SparePartsFeaturizer
from mdp import AMS, FirstComeFirstServed, PartStatus, SparePartsMDP
from network import LOCATIONS, REPAIR_SHOP, STOCK_POINTS, TOTAL_SYSTEMS, default_mdp, mean_travel_periods

CODE = {loc.code: i for i, loc in enumerate(LOCATIONS)}


def simulate(mdp, policy, periods: int, seed: int = 1):
    """Run the model in plain Python and hand every state to `check`-style callers."""
    context = new_context(mdp, seed)
    state = mdp.get_initial_state(context)
    for _ in range(periods):
        while state.category == StateCategory.AWAIT_ACTION:
            mdp.modify_state_with_action(state, context, policy.get_action(state))
        mdp.modify_state_with_event(state, context)
        yield state


# ---- the map --------------------------------------------------------------

def test_travel_times_are_whole_periods_between_one_and_ten():
    m = mean_travel_periods()
    assert np.all(np.diag(m) == 1)                # the site's own shelf: installing takes a period
    assert m.min() == 1 and m.max() == 10
    assert np.array_equal(m, m.T)
    assert m[CODE["AMS"], CODE["CDG"]] == 1       # next door
    assert m[CODE["AMS"], CODE["JFK"]] == 4       # across the Atlantic
    assert m[CODE["AMS"], CODE["SIN"]] == 7       # far east
    assert m[CODE["AMS"], CODE["SYD"]] == 10      # the other side of the world
    assert m[CODE["SIN"], CODE["KUL"]] == 1       # neighbours


def test_the_repair_shop_is_location_zero_and_the_stock_points_come_first():
    assert LOCATIONS[AMS].code == REPAIR_SHOP and LOCATIONS[AMS].can_hold_stock
    n = len(STOCK_POINTS)
    assert all(loc.can_hold_stock for loc in LOCATIONS[:n])
    assert not any(loc.can_hold_stock for loc in LOCATIONS[n:])
    assert len(LOCATIONS) == len({loc.code for loc in LOCATIONS})     # codes are unique


def test_demand_is_proportional_to_the_installed_base():
    mdp = default_mdp(demands_per_week=0.5)
    per_period = 0.5 / (7 * 6)
    for i, loc in enumerate(LOCATIONS):
        assert mdp.demand_prob[i] == pytest.approx(per_period * loc.systems / TOTAL_SYSTEMS)
    assert sum(mdp.demand_prob) == pytest.approx(per_period)


# ---- the initial state ------------------------------------------------------

def test_initial_state_has_the_whole_pool_in_amsterdam_and_every_order_open():
    mdp = default_mdp()
    state = mdp.get_initial_state(new_context(mdp))
    assert mdp.n_parts == len(STOCK_POINTS) == 8
    assert [point.on_hand for point in state.stock_points] == [8] + [0] * 7
    assert [len(point.open_orders) for point in state.stock_points] == [0] + [1] * 7
    assert state.orders_open == 7 and all(part.status == PartStatus.STOCK for part in state.parts)
    assert state.category == StateCategory.AWAIT_ACTION       # position the pool


def test_a_pool_smaller_than_the_number_of_stock_points_leaves_orders_open():
    mdp = default_mdp(pool_size=5)
    state = mdp.get_initial_state(new_context(mdp))
    assert mdp.n_parts == 5 and state.stock_points[AMS].on_hand == 5 and state.orders_open == 7
    fcfs = FirstComeFirstServed(mdp)
    for state in simulate(mdp, fcfs, periods=2000):
        assert state.orders_open >= 2                                  # never enough parts for every shelf
        assert len(state.parts) == 5


def positioned(mdp, context):
    """The initial state after the pool has been positioned and every part has
    arrived: part k on the shelf of stock point k, the rest in AMS."""
    state = mdp.get_initial_state(context)
    for k in range(1, mdp.n_stock_points):
        for _ in range(mdp.base_stock[k]):
            state.stock_points[AMS].on_hand -= 1
            state.stock_points[k].on_hand += 1
            state.stock_points[k].open_orders.pop_front()
            state.orders_open -= 1
            state.parts[k].origin = k
            state.parts[k].dest = k
    mdp._set_category(state)
    return state


def test_unstable_repair_shop_is_refused():
    with pytest.raises(ValueError, match="unstable"):
        default_mdp(demands_per_week=20.0, repair_servers=1)


# ---- the accounting identity ------------------------------------------------

def test_counts_mirror_the_parts_throughout_a_long_run():
    mdp = default_mdp()
    for state in simulate(mdp, FirstComeFirstServed(mdp), periods=3000):
        by_status = {status: 0 for status in PartStatus}
        for part in state.parts:
            by_status[part.status] += 1
        for k, point in enumerate(state.stock_points):
            on_hand = sum(1 for p in state.parts if p.status == PartStatus.STOCK and p.origin == k)
            inbound = sum(1 for p in state.parts if p.status == PartStatus.OUTBOUND and p.dest == k)
            assert point.on_hand == on_hand and point.inbound == inbound
            if k != AMS:
                assert point.on_hand + point.inbound + len(point.open_orders) == mdp.base_stock[k]
        assert state.systems_down == by_status[PartStatus.TO_CUSTOMER]
        assert state.orders_open == sum(len(point.open_orders) for point in state.stock_points)
        assert state.busy_servers == by_status[PartStatus.IN_REPAIR] <= mdp.repair_servers
        assert state.queued == by_status[PartStatus.REPAIR_QUEUE]
        if by_status[PartStatus.REPAIR_QUEUE] > 0:
            assert state.busy_servers == mdp.repair_servers


# ---- a failure is served from the nearest shelf ----------------------------

def tiny_mdp(demand_at: str, stock_at: list[str]) -> SparePartsMDP:
    """A three-location world (AMS, CDG, MIA) where every failure is at `demand_at`."""
    codes = ["AMS", "CDG", "MIA"]
    travel = mean_travel_periods()[np.ix_([CODE[c] for c in codes], [CODE[c] for c in codes])]
    return SparePartsMDP(
        n_stock_points=3, mean_travel_time=travel,
        demand_prob=[0.1 if c == demand_at else 0.0 for c in codes],
        pool_size=len(stock_at),
        base_stock=[0] + [1 if c in stock_at else 0 for c in codes[1:]],
        repair_mean=10.0, repair_servers=2)


def first_demand(mdp):
    context = new_context(mdp, seed=3)
    state = positioned(mdp, context)
    while all(part.status == PartStatus.STOCK for part in state.parts):
        mdp.modify_state_with_event(state, context)
    return state


def test_failure_at_a_stocked_site_is_served_from_its_own_shelf():
    mdp = tiny_mdp(demand_at="CDG", stock_at=["AMS", "CDG"])
    state = first_demand(mdp)
    assert state.systems_down == 1
    shipped = [p for p in state.parts if p.status == PartStatus.TO_CUSTOMER]
    assert len(shipped) == 1 and shipped[0].origin == CODE["CDG"] == shipped[0].dest
    orders = state.stock_points[CODE["CDG"]].open_orders
    assert len(orders) == 1 and orders[0] == state.period
    assert state.category == StateCategory.AWAIT_ACTION      # AMS has a part, CDG has an order


def test_installed_part_comes_back_to_the_shop_as_a_failed_unit():
    mdp = tiny_mdp(demand_at="AMS", stock_at=["AMS"])
    context = new_context(mdp, seed=3)
    state = mdp.get_initial_state(context)
    while state.parts[0].status == PartStatus.STOCK:
        mdp.modify_state_with_event(state, context)
    assert state.parts[0].status == PartStatus.TO_CUSTOMER   # from the AMS shelf to the AMS system
    mdp.modify_state_with_event(state, context)               # mean travel 1: installed next period
    assert state.parts[0].status == PartStatus.RETURNING and state.parts[0].origin == AMS
    mdp.modify_state_with_event(state, context)               # and back in the shop the period after
    assert state.parts[0].status == PartStatus.IN_REPAIR and state.busy_servers == 1


def test_failure_at_an_unstocked_site_waits_for_the_nearest_part():
    mdp = tiny_mdp(demand_at="MIA", stock_at=["AMS", "CDG"])
    state = first_demand(mdp)
    assert state.systems_down == 1
    travelling = [p for p in state.parts if p.status == PartStatus.TO_CUSTOMER]
    assert len(travelling) == 1 and travelling[0].dest == CODE["MIA"]
    assert travelling[0].origin in (AMS, CODE["CDG"])         # both are 5 periods from Miami


# ---- holding ----------------------------------------------------------------

def test_holding_postpones_the_question_until_something_changes():
    mdp = default_mdp()
    context = new_context(mdp, seed=5)
    state = with_open_orders(mdp, ["MIA"])
    assert state.category == StateCategory.AWAIT_ACTION
    mdp.modify_state_with_action(state, context, 0)
    assert state.holding and state.category == StateCategory.AWAIT_EVENT
    orders_before = state.orders_open
    ams_before = state.stock_points[AMS].on_hand
    while state.category == StateCategory.AWAIT_EVENT:
        mdp.modify_state_with_event(state, context)
    # The question is back only because an order opened or a repair completed.
    assert not state.holding
    assert state.orders_open > orders_before or state.stock_points[AMS].on_hand > ams_before


# ---- the hand-written policies ----------------------------------------------

def with_open_orders(mdp, codes: list[str]):
    """The positioned pool, except that the stock points in `codes` have shipped
    their part to a customer (in this order) and ordered a replacement."""
    state = positioned(mdp, new_context(mdp))
    for age, code in enumerate(codes):
        k = CODE[code]
        state.stock_points[k].on_hand = 0
        state.stock_points[k].open_orders.push_back(age)
        state.orders_open += 1
        state.parts[k].status = PartStatus.TO_CUSTOMER
        state.parts[k].dest = k
        state.systems_down += 1
    state.period = len(codes)
    mdp._set_category(state)
    return state


def test_first_come_first_served_fills_the_oldest_order():
    mdp = default_mdp()
    state = with_open_orders(mdp, ["MIA", "CDG"])
    assert FirstComeFirstServed(mdp).get_action(state) == CODE["MIA"]
    state = with_open_orders(mdp, ["CDG", "MIA"])
    assert FirstComeFirstServed(mdp).get_action(state) == CODE["CDG"]


# ---- DynaPlex's own checks ---------------------------------------------------

def test_model_passes_the_dynaplex_checks_with_the_featurizer():
    mdp = default_mdp()
    report = dynaplex.check_mdp(mdp, FirstComeFirstServed(mdp), features=SparePartsFeaturizer,
                                seeds=8, periods=2000, relax_program_flow=True)
    assert report.decisions > 0 and report.feature_rows > 0
