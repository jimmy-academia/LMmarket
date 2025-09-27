# data/process_yelp.py
"""Compute per-user location priors using an exponential distance decay model."""

import math


EARTH_RADIUS_KM = 6371.0088
DEFAULT_MIN_REVIEWS = 5
EPSILON = 1e-9
MAX_STEPS = 128


def _haversine_km(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_lat = math.sin(dlat * 0.5)
    sin_lon = math.sin(dlon * 0.5)
    c = sin_lat * sin_lat + math.cos(lat1) * math.cos(lat2) * sin_lon * sin_lon
    arc = 2 * math.atan2(math.sqrt(c), math.sqrt(max(0.0, 1 - c)))
    return EARTH_RADIUS_KM * arc


def _geometric_median(points):
    if not points:
        return None
    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    for _ in range(MAX_STEPS):
        num_lat = 0.0
        num_lon = 0.0
        denom = 0.0
        coincident = False
        for p in points:
            d = math.hypot(p[0] - lat, p[1] - lon)
            if d <= EPSILON:
                coincident = True
                continue
            w = 1.0 / d
            num_lat += p[0] * w
            num_lon += p[1] * w
            denom += w
        if coincident and denom == 0.0:
            return (lat, lon)
        if denom == 0.0:
            break
        new_lat = num_lat / denom
        new_lon = num_lon / denom
        shift = math.hypot(new_lat - lat, new_lon - lon)
        lat, lon = new_lat, new_lon
        if shift <= EPSILON:
            break
    return (lat, lon)


def _iter_user_points(user_ids, review_map, coord_map):
    for rid in user_ids:
        item_id = review_map.get(rid)
        if not item_id:
            continue
        coord = coord_map.get(item_id)
        if coord is None:
            continue
        yield coord


def process_yelp_data(args, DATA):
    min_reviews = getattr(args, "min_user_location_reviews", DEFAULT_MIN_REVIEWS)
    user_loc = {}
    for city, payload in DATA.items():
        reviews = payload.get("REVIEWS")
        if not reviews:
            reviews = []
        users = payload.get("USERS")
        if not users:
            users = {}
        info = payload.get("INFO")
        if not info:
            info = {}
        review_map = {}
        for entry in reviews:
            rid = entry.get("review_id")
            item_id = entry.get("item_id")
            if not item_id:
                item_id = entry.get("business_id")
            if rid and item_id:
                review_map[rid] = item_id
        coord_map = {}
        for item_id, meta in info.items():
            coords = meta.get("coords") if meta else None
            if not coords or len(coords) != 2:
                continue
            lat, lon = coords
            if lat is None or lon is None:
                continue
            coord_map[item_id] = (lat, lon)
        for uid, user_ids in users.items():
            points = list(_iter_user_points(user_ids, review_map, coord_map))
            if len(points) < min_reviews:
                continue
            center = _geometric_median(points)
            if center is None:
                continue
            distances = []
            for p in points:
                d = _haversine_km(center, p)
                if d > 0:
                    distances.append(d)
            if not distances:
                continue
            mean_distance = sum(distances) / len(distances)
            if mean_distance <= 0:
                continue
            user_loc.setdefault(city, {})[uid] = {
                "center_lat": center[0],
                "center_lon": center[1],
                "lambda_per_km": 1.0 / mean_distance,
            }
    return user_loc