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


class Picture:
    """The map is drawn once. Every frame only moves the parts, the crosses,
    the stock-point labels and the status line; matplotlib redraws just those
    over the cached map (blitting), which is what keeps the animation smooth."""

    def __init__(self, ax, run: Run, policy_name: str):
        self.run = run
        self.policy_name = policy_name
        ax.set_xlim(-130, 160)
        ax.set_ylim(-45, 72)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f4f4f8")
        for loc in LOCATIONS:
            ax.plot(loc.lon, loc.lat, "o", color="#b0b0b8", markersize=3)
            ax.annotate(loc.code, (loc.lon, loc.lat), textcoords="offset points",
                        xytext=LABEL_OFFSET.get(loc.code, (4, 4)), fontsize=7, color="#707078")
        for loc in STOCK_POINTS:
            ax.plot(loc.lon, loc.lat, "s", color="#404048", markersize=7, markerfacecolor="none")

        # The animated artists, one per thing that changes.
        self.crosses = ax.plot([], [], "x", color="tab:red", markersize=11, markeredgewidth=2.5,
                               animated=True)[0]
        self.dots = {colour: ax.plot([], [], "o", color=colour, markersize=7, markeredgecolor="white",
                                     animated=True)[0] for colour in set(COLOUR.values())}
        self.labels = [ax.annotate("", (loc.lon, loc.lat), textcoords="offset points",
                                   xytext=STOCK_LABEL_OFFSET.get(loc.code, (6, -12)), fontsize=7,
                                   color="#202028", fontweight="bold", animated=True)
                       for loc in STOCK_POINTS]
        self.status = ax.text(0.005, 0.99, "", transform=ax.transAxes, va="top", fontsize=9, animated=True)

    def artists(self) -> list:
        return [self.crosses, *self.dots.values(), *self.labels, self.status]

    def update(self) -> list:
        run, state = self.run, self.run.state
        xs: dict[str, list[float]] = {colour: [] for colour in self.dots}
        ys: dict[str, list[float]] = {colour: [] for colour in self.dots}
        cross_x, cross_y = [], []
        stacked: dict[tuple[float, float], int] = {}
        for index, part in enumerate(state.parts):
            lon, lat = run.position(index, part)
            n = stacked.get((lon, lat), 0)
            stacked[(lon, lat)] = n + 1
            xs[COLOUR[part.status]].append(lon + 2.5 * n)
            ys[COLOUR[part.status]].append(lat + 2.5)
            if part.status == PartStatus.TO_CUSTOMER:
                site = LOCATIONS[part.dest]
                cross_x.append(site.lon)
                cross_y.append(site.lat)
        for colour, dots in self.dots.items():
            dots.set_data(xs[colour], ys[colour])
        self.crosses.set_data(cross_x, cross_y)

        for k, (loc, label) in enumerate(zip(STOCK_POINTS, self.labels)):
            point = state.stock_points[k]
            text = f"{loc.code}: {point.on_hand} on hand"
            if point.inbound:
                text += f", {point.inbound} coming"
            if not point.open_orders.is_empty():
                text += f", {len(point.open_orders)} ordered"
            if k == AMS:
                text += f"\nshop: {state.busy_servers} in repair, {state.queued} queued"
            label.set_text(text)

        cost = run.context.cumulative_cost
        day = state.period / PERIODS_PER_DAY
        per_period = cost / state.period if state.period else 0.0
        self.status.set_text(f"{self.policy_name}   |   day {day:.1f}   |   {state.systems_down} systems down"
                             f"   |   cost {cost / 1e6:.1f}M ({per_period:,.0f} per period)"
                             f"   |   {run.last_decision}")
        return self.artists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["fcfs", "random", "trained"], default="fcfs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--periods", type=int, default=10000, help="periods to show (1667 days)")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    run = Run(default_mdp(), args.policy, args.seed)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    picture = Picture(ax, run, args.policy)

    def frame(_):
        run.step()
        return picture.update()

    animation = FuncAnimation(fig, frame, init_func=picture.update, frames=args.periods,
                              interval=1000 / args.fps, repeat=False, blit=True, cache_frame_data=False)
    plt.show()
    del animation


if __name__ == "__main__":
    main()
