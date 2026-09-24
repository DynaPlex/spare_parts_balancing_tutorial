"""Watch a policy run the network: a map with the parts moving, the systems
that are down, and the shop repairing, one frame per period.

    python watch.py                        # the textbook rule
    python watch.py --policy trained       # after train.py
    python watch.py --fps 20 --periods 20000

The model runs in plain Python here (no compilation): the same mdp.py, called
directly, so you can put a print() or a breakpoint anywhere in it and watch.

Colours: green = serviceable, blue = on its way to a system that is down (the
red cross), red = failed and returning, orange = in the repair shop.
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dynaplex.modelling import StateCategory, new_context

from mdp import AMS, FirstComeFirstServed, PartStatus
from network import HOURS_PER_PERIOD, LOCATIONS, STOCK_POINTS, default_mdp

PERIODS_PER_DAY = 24 // HOURS_PER_PERIOD
COLOUR = {
    PartStatus.STOCK: "tab:green", PartStatus.OUTBOUND: "tab:green",
    PartStatus.TO_CUSTOMER: "tab:blue", PartStatus.RETURNING: "tab:red",
    PartStatus.REPAIR_QUEUE: "tab:orange", PartStatus.IN_REPAIR: "tab:orange",
}
TRAVELLING = (PartStatus.OUTBOUND, PartStatus.TO_CUSTOMER, PartStatus.RETURNING)
# Label offsets (points) where cities crowd each other; everything else goes up-right.
LABEL_OFFSET = {"CDG": (-24, -12), "LHR": (-26, 2), "FRA": (6, -10), "KUL": (-26, 4), "SIN": (6, -12),
                "PVG": (6, 4), "HKG": (6, -10), "BOM": (-26, -10), "DEL": (6, 4)}
STOCK_LABEL_OFFSET = {"AMS": (10, 6), "CDG": (-90, -14), "KUL": (-115, 4), "SIN": (8, -14)}


def make_policy(name: str, mdp, context):
    """A function state -> action, whichever kind of policy is asked for."""
    if name == "fcfs":
        return FirstComeFirstServed(mdp).get_action
    if name == "random":
        import dynaplex
        random_policy = dynaplex.RandomPolicy(mdp)
        return lambda state: random_policy.get_action(state, context)
    if name == "trained":
        import dynaplex
        path = os.path.join("agents", "trained")
        if not os.path.isdir(path):
            raise SystemExit(f"{path} does not exist yet: run train.py first")
        agent = dynaplex.NNAgent.load(path)
        return lambda state: int(agent([state], mdp=mdp)[0])
    raise SystemExit(f"unknown policy {name!r}")


class Run:
    """The simulation, advanced one period per frame, plus what the picture
    needs and the model does not keep: how long each part has been travelling."""

    def __init__(self, mdp, policy_name: str, seed: int):
        self.mdp = mdp
        self.context = new_context(mdp, seed)
        self.state = mdp.get_initial_state(self.context)
        self.policy = make_policy(policy_name, mdp, self.context)
        self.legs: dict[int, tuple | None] = {}  # part index -> (status, origin, dest) while travelling
        self.departed: dict[int, int] = {}      # part index -> period its current leg started
        self.last_decision = ""

    def step(self) -> None:
        state, mdp = self.state, self.mdp
        while state.category == StateCategory.AWAIT_ACTION:
            action = self.policy(state)
            mdp.modify_state_with_action(state, self.context, action)
            self.last_decision = (f"day {state.period / PERIODS_PER_DAY:.1f}: "
                                  + ("hold" if action == 0 else f"AMS → {LOCATIONS[action].code}"))
        mdp.modify_state_with_event(state, self.context)
        for index, part in enumerate(state.parts):
            leg = (part.status, part.origin, part.dest) if part.status in TRAVELLING else None
            if self.legs.get(index) != leg:
                self.legs[index] = leg
                self.departed[index] = state.period - 1

    def position(self, index: int, part):
        """(lon, lat) of a part: at its location, or along its leg, as far as
        its mean travel time says it should be (never quite arriving)."""
        if part.status not in TRAVELLING:
            loc = LOCATIONS[part.origin]
            return loc.lon, loc.lat
        a, b = LOCATIONS[part.origin], LOCATIONS[part.dest]
        elapsed = self.state.period - self.departed.get(index, self.state.period)
        fraction = min(0.92, elapsed / max(1.0, self.mdp.mean_travel[part.origin, part.dest]))
        return a.lon + fraction * (b.lon - a.lon), a.lat + fraction * (b.lat - a.lat)


def draw(ax, run: Run, policy_name: str) -> None:
    state = run.state
    ax.clear()
    ax.set_xlim(-130, 160)
    ax.set_ylim(-45, 65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f4f4f8")

    for loc in LOCATIONS:
        ax.plot(loc.lon, loc.lat, "o", color="#b0b0b8", markersize=3)
        ax.annotate(loc.code, (loc.lon, loc.lat), textcoords="offset points",
                    xytext=LABEL_OFFSET.get(loc.code, (4, 4)), fontsize=7, color="#707078")
    for k, loc in enumerate(STOCK_POINTS):
        point = state.stock_points[k]
        label = f"{loc.code}: {point.on_hand} on hand"
        if point.inbound:
            label += f", {point.inbound} coming"
        if not point.open_orders.is_empty():
            label += f", {len(point.open_orders)} ordered"
        if k == AMS:
            label += f"\nshop: {state.busy_servers} in repair, {state.queued} queued"
        ax.plot(loc.lon, loc.lat, "s", color="#404048", markersize=7, markerfacecolor="none")
        ax.annotate(label, (loc.lon, loc.lat), textcoords="offset points",
                    xytext=STOCK_LABEL_OFFSET.get(loc.code, (6, -12)), fontsize=7, color="#202028",
                    fontweight="bold")

    stacked: dict[tuple[float, float], int] = {}
    for index, part in enumerate(state.parts):
        lon, lat = run.position(index, part)
        n = stacked.get((lon, lat), 0)
        stacked[(lon, lat)] = n + 1
        ax.plot(lon + 2.5 * n, lat + 2.5, "o", color=COLOUR[part.status], markersize=7,
                markeredgecolor="white")
        if part.status == PartStatus.TO_CUSTOMER:
            site = LOCATIONS[part.dest]
            ax.plot(site.lon, site.lat, "x", color="tab:red", markersize=11, markeredgewidth=2.5)

    cost = run.context.cumulative_cost
    day = state.period / PERIODS_PER_DAY
    per_period = cost / state.period if state.period else 0.0
    ax.set_title(f"{policy_name}   |   day {day:.1f}   |   {state.systems_down} systems down   |   "
                 f"cost {cost / 1e6:.1f}M ({per_period:,.0f} per period)   |   {run.last_decision}",
                 fontsize=9, loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["fcfs", "random", "trained"], default="fcfs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--periods", type=int, default=10000, help="periods to show (1667 days)")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    run = Run(default_mdp(), args.policy, args.seed)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

    def frame(_):
        run.step()
        draw(ax, run, args.policy)

    animation = FuncAnimation(fig, frame, frames=args.periods, interval=1000 / args.fps, repeat=False,
                              cache_frame_data=False)
    plt.show()
    del animation


if __name__ == "__main__":
    main()
