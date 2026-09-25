"""The geography and the default numbers: where the sites are, which of them
hold stock, how long a shipment takes, and `default_mdp()`, the configuration
every script uses.

This file is ordinary Python (strings, dictionaries, anything goes). The model
in `mdp.py` only ever sees the numbers computed here: a matrix of mean travel
times, a demand probability per location, and which locations hold stock. The
network is invented; the cities are real so that the map is recognizable.
Parts travel by air freight, which is why every location carries the code of
its nearest airport.

To experiment: flip `can_hold_stock` on a location, change `pool_size` in
`default_mdp()`, or give a site more systems. Nothing else needs to change.
The pool may be smaller than the number of stock points: then some shelf is
always waiting for a part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mdp import SparePartsMDP

HOURS_PER_PERIOD = 4
REPAIR_SHOP = "AMS"


@dataclass(frozen=True)
class Location:
    code: str               # nearest airport
    city: str
    lat: float
    lon: float
    systems: int            # installed base: how many systems here contain the part
    # A location that can hold stock is a stock point: it keeps a part on its
    # shelf when it has one, and orders a replacement from the repair shop
    # whenever it ships one. A location that cannot is served from the nearest
    # stock point that has a part. The shop itself must be able to hold stock.
    can_hold_stock: bool = False


# Every location where a system can fail. Demand is proportional to the
# number of systems at a location.
TABLE = [
    Location("AMS", "Amsterdam", 52.31, 4.76, 10, can_hold_stock=True),
    Location("CDG", "Paris", 49.01, 2.55, 9, can_hold_stock=True),
    Location("MIA", "Miami", 25.79, -80.29, 9, can_hold_stock=True),
    Location("DXB", "Dubai", 25.25, 55.36, 9, can_hold_stock=True),
    Location("SIN", "Singapore", 1.36, 103.99, 10, can_hold_stock=True),
    Location("KUL", "Kuala Lumpur", 2.75, 101.71, 5, can_hold_stock=True),
    Location("GRU", "Sao Paulo", -23.43, -46.47, 5, can_hold_stock=True),
    Location("PVG", "Shanghai", 31.14, 121.81, 12, can_hold_stock=True),
    Location("JFK", "New York", 40.64, -73.78, 18),
    Location("LAX", "Los Angeles", 33.94, -118.41, 12),
    Location("ORD", "Chicago", 41.98, -87.90, 9),
    Location("YYZ", "Toronto", 43.68, -79.63, 4),
    Location("MEX", "Mexico City", 19.44, -99.07, 4),
    Location("BOG", "Bogota", 4.70, -74.15, 2),
    Location("LIM", "Lima", -12.02, -77.11, 2),
    Location("EZE", "Buenos Aires", -34.82, -58.54, 3),
    Location("LHR", "London", 51.47, -0.45, 12),
    Location("FRA", "Frankfurt", 50.04, 8.56, 6),
    Location("MAD", "Madrid", 40.49, -3.57, 3),
    Location("IST", "Istanbul", 41.26, 28.74, 4),
    Location("CAI", "Cairo", 30.12, 31.41, 2),
    Location("LOS", "Lagos", 6.58, 3.32, 2),
    Location("NBO", "Nairobi", -1.32, 36.93, 1),
    Location("JNB", "Johannesburg", -26.14, 28.25, 2),
    Location("DEL", "Delhi", 28.56, 77.10, 5),
    Location("BOM", "Mumbai", 19.09, 72.87, 3),
    Location("BKK", "Bangkok", 13.69, 100.75, 5),
    Location("CGK", "Jakarta", -6.13, 106.66, 3),
    Location("HKG", "Hong Kong", 22.31, 113.91, 8),
    Location("PEK", "Beijing", 40.08, 116.58, 7),
    Location("NRT", "Tokyo", 35.77, 140.39, 8),
    Location("ICN", "Seoul", 37.46, 126.44, 4),
    Location("SYD", "Sydney", -33.95, 151.18, 2),
]

# The model numbers the locations: the repair shop is 0, the other stock points
# come next, then the sites without stock. The table above can be in any order.
if not any(loc.code == REPAIR_SHOP and loc.can_hold_stock for loc in TABLE):
    raise ValueError(f"the repair shop {REPAIR_SHOP} must be in the table and hold stock: "
                     "repaired parts land on its shelf")
LOCATIONS = sorted(TABLE, key=lambda loc: (loc.code != REPAIR_SHOP, not loc.can_hold_stock))
STOCK_POINTS = [loc for loc in LOCATIONS if loc.can_hold_stock]
TOTAL_SYSTEMS = sum(loc.systems for loc in LOCATIONS)


def distance_km(a: Location, b: Location) -> float:
    """Great-circle distance (haversine)."""
    lat_a, lat_b = math.radians(a.lat), math.radians(b.lat)
    d_lat = lat_b - lat_a
    d_lon = math.radians(b.lon - a.lon)
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(d_lon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def mean_travel_periods(handling_hours: float = 3.0, speed_kmh: float = 450.0,
                        max_periods: int = 10) -> np.ndarray:
    """Mean door-to-door time between every pair of locations, in whole
    periods: a fixed handling time (paperwork, pick-up) plus the distance at an
    effective speed, rounded, at least one period and at most `max_periods`.
    That speed is about half the cruise speed of air freight — a shipment waits
    for the next departure, and the far side of the world needs a transfer. A
    part on the site's own shelf still needs the handling time: the diagonal is
    one period, installing included."""
    n = len(LOCATIONS)
    mean = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(LOCATIONS):
        for j, b in enumerate(LOCATIONS):
            hours = handling_hours + distance_km(a, b) / speed_kmh
            mean[i, j] = min(max_periods, max(1, round(hours / HOURS_PER_PERIOD)))
    return mean


def default_mdp(
    demands_per_week: float = 0.5,
    repair_mean_days: float = 70.0,
    repair_servers: int = 6,
    pool_size: int = 8,
    downtime_cost_per_hour: float = 10_000.0,
    loan_cost: float = 1_600_000.0,     # about a week of downtime
    handling_hours: float = 3.0,
    speed_kmh: float = 450.0,
    max_travel_periods: int = 10,
) -> SparePartsMDP:
    """The tutorial's configuration. Every number is an argument: change how
    fast-moving the part is (`demands_per_week`, over all systems together),
    how good the repair shop is, or how many parts the pool owns (`pool_size`;
    it may be smaller than the number of stock points, then some shelves stay
    empty), and see what the policies make of it. Money has no currency here:
    a system that is down costs 10K per hour, a borrowed part 1.6M."""
    periods_per_day = 24 // HOURS_PER_PERIOD
    demand_prob = demands_per_week / (7 * periods_per_day)
    return SparePartsMDP(
        n_stock_points=len(STOCK_POINTS),
        mean_travel_time=mean_travel_periods(handling_hours, speed_kmh, max_travel_periods),
        demand_prob=[demand_prob * loc.systems / TOTAL_SYSTEMS for loc in LOCATIONS],
        pool_size=pool_size,
        base_stock=[0] + [1] * (len(STOCK_POINTS) - 1),
        repair_mean=repair_mean_days * periods_per_day,
        repair_servers=repair_servers,
        downtime_cost=downtime_cost_per_hour * HOURS_PER_PERIOD,
        loan_cost=loan_cost,
    )
