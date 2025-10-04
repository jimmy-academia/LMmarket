import math

EARTH_RADIUS_KM = 6371.0088
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
