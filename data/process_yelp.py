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
    assert points
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


def process_yelp_data(args, DATA):
    if hasattr(args, "min_user_location_reviews"):
        min_reviews = args.min_user_location_reviews
    else:
        min_reviews = DEFAULT_MIN_REVIEWS
    assert min_reviews > 0

    reviews = DATA["reviews"]
    items = DATA["items"]
    users = DATA["users"]

    review_to_item = {}
    for rid, record in reviews.items():
        item_id = record["item_id"]
        if item_id in items:
            review_to_item[rid] = item_id

    item_to_coord = {}
    for item_id, meta in items.items():
        coord = meta.get("coords")
        if coord:
            item_to_coord[item_id] = coord

    result = {}
    for uid, info in users.items():
        review_ids = info.get("review_ids") or []
        points = []
        for rid in review_ids:
            item_id = review_to_item.get(rid)
            if not item_id:
                continue
            coord = item_to_coord.get(item_id)
            if not coord:
                continue
            points.append(coord)
        if len(points) < min_reviews:
            continue
        center = _geometric_median(points)
        distances = [max(_haversine_km(center, p), EPSILON) for p in points]
        mean_distance = sum(distances) / len(distances)
        if mean_distance <= 0.0:
            continue
        result[uid] = {
            "center_lat": center[0],
            "center_lon": center[1],
            "lambda_per_km": 1.0 / mean_distance,
        }

    return result
