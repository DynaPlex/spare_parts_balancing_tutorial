"""The geography and the default numbers: where the stock points and the
sites are, how long a shipment takes, and `default_mdp()`, the configuration
every script uses.

This file is ordinary Python (strings, dictionaries, anything goes). The model
in `mdp.py` only ever sees the numbers computed here: a matrix of mean travel
times and a demand probability per location. The network is invented; the
cities are real so that the map is recognizable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mdp import SparePartsMDP

HOURS_PER_PERIOD = 4


@dataclass(frozen=True)
class Location:
    code: str
    city: str
    lat: float
    lon: float
    demand_weight: float    # relative share of the world's demand


# Stock points first, the repair shop first of all: location index 0 is AMS.
STOCK_POINTS = [
    Location("AMS", "Amsterdam", 52.31, 4.76, 3),
    Location("CDG", "Paris", 49.01, 2.55, 3),
    Location("MIA", "Miami", 25.79, -80.29, 2),
    Location("DXB", "Dubai", 25.25, 55.36, 2),
    Location("SIN", "Singapore", 1.36, 103.99, 2),
    Location("KUL", "Kuala Lumpur", 2.75, 101.71, 1),
    Location("GRU", "Sao Paulo", -23.43, -46.47, 1),
    Location("PVG", "Shanghai", 31.14, 121.81, 2),
]

# Sites where a part can fail, but that hold no stock.
SITES = [
    Location("JFK", "New York", 40.64, -73.78, 3),
    Location("LAX", "Los Angeles", 33.94, -118.41, 2),
    Location("ORD", "Chicago", 41.98, -87.90, 2),
    Location("YYZ", "Toronto", 43.68, -79.63, 1),
    Location("MEX", "Mexico City", 19.44, -99.07, 1),
    Location("BOG", "Bogota", 4.70, -74.15, 1),
    Location("LIM", "Lima", -12.02, -77.11, 1),
    Location("EZE", "Buenos Aires", -34.82, -58.54, 1),
    Location("LHR", "London", 51.47, -0.45, 3),
    Location("FRA", "Frankfurt", 50.04, 8.56, 2),
    Location("MAD", "Madrid", 40.49, -3.57, 1),
    Location("IST", "Istanbul", 41.26, 28.74, 2),
    Location("CAI", "Cairo", 30.12, 31.41, 1),
    Location("LOS", "Lagos", 6.58, 3.32, 1),
    Location("NBO", "Nairobi", -1.32, 36.93, 1),
    Location("JNB", "Johannesburg", -26.14, 28.25, 1),
    Location("DEL", "Delhi", 28.56, 77.10, 2),
    Location("BOM", "Mumbai", 19.09, 72.87, 1),
    Location("BKK", "Bangkok", 13.69, 100.75, 2),
    Location("CGK", "Jakarta", -6.13, 106.66, 1),
    Location("HKG", "Hong Kong", 22.31, 113.91, 2),
    Location("PEK", "Beijing", 40.08, 116.58, 2),
    Location("NRT", "Tokyo", 35.77, 140.39, 2),
    Location("ICN", "Seoul", 37.46, 126.44, 1),
    Location("SYD", "Sydney", -33.95, 151.18, 1),
]

LOCATIONS = STOCK_POINTS + SITES


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
    regional_base_stock: int = 1,
    ams_stock: int = 1,
    loan_cost: float = 40.0,
    handling_hours: float = 3.0,
    speed_kmh: float = 450.0,
    max_travel_periods: int = 10,
) -> SparePartsMDP:
    """The tutorial's configuration. Every number is an argument: change how
    fast-moving the part is (`demands_per_week`), how good the repair shop is,
    or how many parts the pool owns, and see what the policies make of it."""
    periods_per_day = 24 // HOURS_PER_PERIOD
    demand_prob = demands_per_week / (7 * periods_per_day)
    total_weight = sum(loc.demand_weight for loc in LOCATIONS)
    return SparePartsMDP(
        n_stock_points=len(STOCK_POINTS),
        mean_travel_time=mean_travel_periods(handling_hours, speed_kmh, max_travel_periods),
        demand_prob=[demand_prob * loc.demand_weight / total_weight for loc in LOCATIONS],
        base_stock=[ams_stock] + [regional_base_stock] * (len(STOCK_POINTS) - 1),
        repair_mean=repair_mean_days * periods_per_day,
        repair_servers=repair_servers,
        loan_cost=loan_cost,
    )
