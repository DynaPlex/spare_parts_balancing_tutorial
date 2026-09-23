"""Readable checks of the model: the map, the initial state, the accounting
identity, fulfilment from the nearest shelf, holding, and the policies
on hand-built situations. Run with `python -m pytest`."""
import numpy as np
import pytest

import dynaplex
from dynaplex.modelling import DiscreteDist, StateCategory, new_context, probe_state

from featurizer import SparePartsFeaturizer
from mdp import AMS, EmptiestFirst, FirstComeFirstServed, MostExposedFirst, PartStatus, SparePartsMDP
from network import LOCATIONS, STOCK_POINTS, default_mdp, mean_travel_periods

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


# ---- the initial state ------------------------------------------------------

def test_initial_state_has_one_part_on_every_shelf():
    mdp = default_mdp()
    state = mdp.get_initial_state(new_context(mdp))
    assert mdp.n_parts == len(STOCK_POINTS) == 8
    assert [point.on_hand for point in state.stock_points] == [1] * 8
    assert all(part.status == PartStatus.STOCK for part in state.parts)
    assert state.category == StateCategory.AWAIT_EVENT


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
        assert len(state.repair_queue) == by_status[PartStatus.REPAIR_QUEUE]
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
        base_stock=[1 if c in stock_at else 0 for c in codes],
        repair_time=DiscreteDist.constant(10), repair_servers=2)


def first_demand(mdp):
    context = new_context(mdp, seed=3)
    state = mdp.get_initial_state(context)
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
    state = probe_state(mdp, seed=5)
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
    """The initial state, except that the stock points in `codes` have shipped
    their part to a customer (in this order) and ordered a replacement."""
    state = mdp.get_initial_state(new_context(mdp))
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


def test_most_exposed_first_prefers_the_region_without_a_backup():
    mdp = default_mdp()
    state = with_open_orders(mdp, ["CDG", "MIA"])
    # Paris is next door to Amsterdam; Miami's region would be served from
    # Amsterdam across the ocean. Miami is the bigger loss, whatever the order age.
    assert mdp.exposure(state, CODE["MIA"]) > mdp.exposure(state, CODE["CDG"]) > 0.0
    assert MostExposedFirst(mdp, reserve=0).get_action(state) == CODE["MIA"]
    assert mdp.exposure(state, CODE["DXB"]) == 0.0           # Dubai still has its part


def test_reserve_keeps_the_last_part_in_amsterdam():
    mdp = default_mdp()
    state = with_open_orders(mdp, ["MIA"])
    assert state.stock_points[AMS].on_hand == 1
    assert MostExposedFirst(mdp, reserve=1).get_action(state) == 0
    assert EmptiestFirst(mdp, reserve=1).get_action(state) == 0
    assert MostExposedFirst(mdp, reserve=0).get_action(state) == CODE["MIA"]


# ---- DynaPlex's own checks ---------------------------------------------------

def test_model_passes_the_dynaplex_checks_with_the_featurizer():
    mdp = default_mdp()
    report = dynaplex.check_mdp(mdp, MostExposedFirst(mdp), features=SparePartsFeaturizer,
                                seeds=8, periods=2000, relax_program_flow=True)
    assert report.decisions > 0 and report.feature_rows > 0
