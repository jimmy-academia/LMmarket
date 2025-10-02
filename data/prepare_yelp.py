import json
from tqdm import tqdm
from collections import defaultdict, Counter

'''
Config setting and helper function
'''

MIN_REVIEWS_PER_RESTAURANT = 50
MIN_RESTAURANTS_PER_CITY   = 500
MIN_TEXT_LEN               = 200
MIN_USER_REVIEWS           = 50

def iter_line(filepath, total=None):
    if total is None:
        with open(filepath, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, ncols=90):
            yield line

"""
Main interface
"""

def prepare_yelp_data(dset_root):
    dset_root = dset_root / 'yelp'
    business_file = dset_root/"yelp_academic_dataset_business.json" # 150346
    review_file = dset_root/"yelp_academic_dataset_review.json" # 6990280
    tip_file = dset_root/"yelp_academic_dataset_tip.json" # 908915
    user_file = dset_root/"yelp_academic_dataset_user.json" # 1987897

    restaurants = load_restaurants(business_file)
    city_restaurants = filter_cities(restaurants)

    biz_to_city = {bid: city for city, biz_map in city_restaurants.items() for bid in biz_map.keys()}

    entry_by_city, users, user_city_counts, _ = load_entries(review_file, tip_file, biz_to_city)
    user_by_city = filter_user(users, user_city_counts)
    info_by_city = build_info_by_city(city_restaurants)

    return build_city_payloads(entry_by_city, user_by_city, info_by_city)

    # return {"USERS": user_by_city, "ITEMS": item_by_city, "REVIEWS": entry_by_city, "INFO": info_by_city}

"""
Module functions
-----------------------------------
"""


def load_restaurants(business_file):
    """Return {biz_id: biz_info} for restaurants with >MIN_REVIEWS_PER_RESTAURANT reviews."""
    restaurants = {}
    with open(business_file, encoding="utf-8") as f:
        for line in f:
            biz = json.loads(line)
            categories = biz.get("categories")
            if not categories or "restaurant" not in categories.lower():
                continue
            address = biz.get("address")
            if not address:
                continue
            review_count = biz.get("review_count", 0)
            if review_count <= MIN_REVIEWS_PER_RESTAURANT:
                continue
            latitude = biz.get("latitude")
            longitude = biz.get("longitude")
            if latitude is None or longitude is None:
                continue
            restaurants[biz["business_id"]] = biz
    return restaurants

def filter_cities(restaurants):
    """Return {city: {biz_id: info}} only for cities with >MIN_RESTAURANTS_PER_CITY restaurants."""
    city_map = defaultdict(dict)
    counts = Counter()
    for biz_id, biz in restaurants.items():
        city = biz.get("city", "").strip().lower()
        counts[city] += 1
        city_map[city][biz_id] = biz
    counts.pop("", None)
    big_cities = {c for c, cnt in counts.items() if cnt > MIN_RESTAURANTS_PER_CITY}
    return {c: city_map[c] for c in big_cities}

def load_entries(review_file, tip_file, biz_to_city):
    entry_by_city = defaultdict(list)
    users = defaultdict(list)
    item_by_city = defaultdict(lambda: defaultdict(list))
    user_city_counts = defaultdict(Counter)
    tip_id = 0

    for filepath, kind, total in [(review_file, "review", 6_990_280), (tip_file, "tip", 908_915)]:
        for line in iter_line(filepath, total=total):
            obj = json.loads(line)
            bid = obj["business_id"]
            if bid in biz_to_city:
                text = obj.get("text", "")
                if len(text) <= MIN_TEXT_LEN:
                    continue
                uid = obj["user_id"]
                city = biz_to_city[bid]
                if kind == "review":
                    rid = obj["review_id"]
                else:
                    tip_id += 1
                    rid = f"tip_{tip_id}"

                row = {
                    "review_id": rid,
                    "user_id": uid,
                    "item_id": bid,
                    "text": text,
                    "kind": kind,
                }
                entry_by_city[city].append(row)
                users[uid].append(rid)
                item_by_city[city][bid].append(rid)
                user_city_counts[uid][city] += 1

    return entry_by_city, users, user_city_counts, item_by_city

def filter_user(users, user_city_counts):
    users_by_city = defaultdict(dict)
    for uid, rids in users.items():
        if len(rids) > MIN_USER_REVIEWS:
            city = min(user_city_counts[uid].items(), key=lambda kv: (-kv[1], kv[0]))[0]
            users_by_city[city][uid] = rids
    return users_by_city

def _biz_info(b):
    """Pick small, useful fields from business.json."""
    # categories -> list (lowercased, stripped)
    cats = [c.strip() for c in (b.get("categories") or "").split(",") if c.strip()]
    return {
        "name": b.get("name"),
        "address": b.get("address"),
        "city": (b.get("city") or "").strip(),
        "state": b.get("state"),
        "postal_code": b.get("postal_code"),
        "coords": [b.get("latitude"), b.get("longitude")],
        "stars": b.get("stars"),
        "review_count": b.get("review_count"),
        "is_open": b.get("is_open"),
        "categories": cats,
        # keep raw attributes/hours as-is (strings/dicts in Yelp dump)
        "attributes": b.get("attributes"),
        "hours": b.get("hours"),
    }

def build_info_by_city(city_restaurants):
    """Return {city: {biz_id: small_info_dict}}"""
    info_by_city = defaultdict(dict)
    for city, biz_map in city_restaurants.items():
        for bid, biz in biz_map.items():
            info_by_city[city][bid] = _biz_info(biz)
    return dict(info_by_city)


def build_city_payloads(entry_by_city, user_by_city, info_by_city):
    data = {}
    for city, entries in entry_by_city.items():
        city_users = user_by_city[city]
        assert isinstance(city_users, dict)
        assert city in info_by_city
        city_info = info_by_city[city]
        reviews = []
        review_ids = []
        items = {}
        for entry in entries:
            rid = entry["review_id"]
            uid = entry["user_id"]
            item_id = entry["item_id"]
            text = entry["text"].strip()
            assert text
            record = {
                "review_id": rid,
                "user_id": uid,
                "item_id": item_id,
                "text": text,
                "kind": entry["kind"],
            }
            reviews.append(record)
            review_ids.append(rid)
            items.setdefault(item_id, []).append(rid)

        review_id_set = set(review_ids)
        users = {}
        for uid, rids in city_users.items():
            cleaned = [rid for rid in rids if rid in review_id_set]
            assert cleaned
            users[uid] = cleaned

        info = {}
        for item_id in items.keys():
            meta = dict(city_info[item_id])
            coords = meta["coords"]
            lat = float(coords[0])
            lon = float(coords[1])
            meta["coords"] = (lat, lon)
            info[item_id] = meta

        data[city] = {
            "USERS": users,
            "ITEMS": items,
            "REVIEWS": reviews,
            "INFO": info,
        }
    assert_city_payloads(data)
    return data


def assert_city_payloads(data):
    assert isinstance(data, dict)
    for city, payload in data.items():
        assert isinstance(city, str) and city
        assert isinstance(payload, dict)
        reviews = payload["REVIEWS"]
        users = payload["USERS"]
        items = payload["ITEMS"]
        info = payload["INFO"]
        assert isinstance(reviews, list) and reviews
        assert isinstance(users, dict)
        assert isinstance(items, dict) and items
        assert isinstance(info, dict) and info
        review_ids = set()
        for review in reviews:
            assert isinstance(review, dict)
            rid = review["review_id"]
            uid = review["user_id"]
            item_id = review["item_id"]
            text = review["text"]
            kind = review["kind"]
            assert isinstance(rid, str) and rid
            assert isinstance(uid, str) and uid
            assert isinstance(item_id, str) and item_id
            assert isinstance(text, str) and text
            assert isinstance(kind, str) and kind
            review_ids.add(rid)
            assert rid in items[item_id]
        for uid, rids in users.items():
            assert isinstance(uid, str) and uid
            assert isinstance(rids, list) and rids
            for rid in rids:
                assert rid in review_ids
        for item_id, linked in items.items():
            assert isinstance(item_id, str) and item_id
            assert isinstance(linked, list) and linked
            for rid in linked:
                assert rid in review_ids
            meta = info[item_id]
            assert isinstance(meta, dict)
            coords = meta["coords"]
            assert isinstance(coords, tuple) and len(coords) == 2
            lat, lon = coords
            assert isinstance(lat, float)
            assert isinstance(lon, float)
    

    

