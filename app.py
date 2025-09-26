"""STEP-3 application stubs consuming PROCESS artifacts only."""

import math
from collections import defaultdict

from pipeline import haversine_km


def build_item_coordinates(proc):
    """Collect item coordinates from rich review units."""
    coords = defaultdict(dict)
    for unit in proc.get('RICH_REVIEWS', []):
        city = unit.get('city')
        item_id = unit.get('item_id')
        loc = unit.get('coords')
        if city and item_id and loc and loc[0] is not None and loc[1] is not None:
            coords[city][item_id] = loc
    return coords


def compute_utility(proc, query=None):
    """Compute utility scores for each user-item pair given PROC artifacts."""
    user_geo = proc.get('USER_GEO', {})
    item_perf = proc.get('ITEM_ASPECT_PERF', {})
    coords = build_item_coordinates(proc)
    utilities = defaultdict(dict)
    for city, users in user_geo.items():
        city_perf = item_perf.get(city, {})
        city_coords = coords.get(city, {})
        for uid, geo in users.items():
            utilities[city][uid] = {}
            center = geo.get('center')
            alpha = geo.get('alpha', 0.0)
            for item_id, aspects in city_perf.items():
                perf = 0.0
                if aspects:
                    perf = sum(aspects.values()) / len(aspects)
                if perf <= 0.0:
                    utilities[city][uid][item_id] = 0.0
                    continue
                if not center or item_id not in city_coords:
                    distance = 0.0
                else:
                    loc = city_coords[item_id]
                    distance = haversine_km(center[0], center[1], loc[0], loc[1])
                utilities[city][uid][item_id] = perf * math.exp(-alpha * distance)
    return utilities


def apps_run(args, proc):
    """Example STEP-3 runner returning utility grids."""
    utilities = compute_utility(proc)
    summary = {
        'cities': len(utilities),
        'users': sum(len(users) for users in utilities.values()),
        'items': sum(len(items) for users in utilities.values() for items in users.values()),
    }
    print(f"[apps] utility summary: {summary}")
    return utilities
