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

    payload = build_global_payloads(entry_by_city, user_by_city, info_by_city)
    payload["schema"] = "yelp.v2.pooled"
    return payload

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
                for key in ("stars", "useful", "funny", "cool", "date"):
                    value = obj.get(key)
                    if value is not None:
                        row[key] = value
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


def build_global_payloads(entry_by_city, user_by_city, info_by_city):
    users = {}
    items = {}
    reviews = {}

    for city, city_info in info_by_city.items():
        for item_id, meta in city_info.items():
            coords = meta.get("coords")
            lat = float(coords[0])
            lon = float(coords[1])
            raw = {}
            for key, value in meta.items():
                if key in ("coords", "name", "categories"):
                    continue
                raw[key] = value
            item_entry = {
                "review_ids": [],
                "city": city,
                "coords": (lat, lon),
                "name": meta.get("name"),
                "categories": meta.get("categories"),
            }
            if raw:
                item_entry["raw"] = raw
            items[item_id] = item_entry

    for city, entries in entry_by_city.items():
        city_users = user_by_city.get(city, {})
        for entry in entries:
            uid = entry["user_id"]
            if uid not in city_users:
                continue
            rid = entry["review_id"]
            item_id = entry["item_id"]
            if item_id not in items:
                continue
            text = entry["text"].strip()
            if not text:
                continue
            item_city = items[item_id]["city"]
            record = {
                "user_id": uid,
                "item_id": item_id,
                "city": item_city,
                "text": text,
                "kind": entry["kind"],
            }
            for key in ("stars", "useful", "funny", "cool", "date", "ts"):
                if key in entry:
                    target = "ts" if key in ("date", "ts") else key
                    record[target] = entry[key]
            reviews[rid] = record
            items[item_id]["review_ids"].append(rid)

    review_city = {rid: record["city"] for rid, record in reviews.items()}
    for city, city_users in user_by_city.items():
        for uid, rids in city_users.items():
            valid = []
            seen = set()
            for rid in rids:
                if rid in reviews and rid not in seen:
                    valid.append(rid)
                    seen.add(rid)
            if not valid:
                continue
            info = users.get(uid)
            if not info:
                info = {"review_ids": [], "city_hist": {}}
                users[uid] = info
            info["review_ids"].extend(valid)

    for uid, info in users.items():
        seen = set()
        cleaned = []
        for rid in info["review_ids"]:
            if rid in review_city and rid not in seen:
                cleaned.append(rid)
                seen.add(rid)
        info["review_ids"] = cleaned
        hist = {}
        for rid in cleaned:
            city = review_city[rid]
            hist[city] = hist.get(city, 0) + 1
        info["city_hist"] = hist

    _assert_global_payloads(users, items, reviews)
    return {"users": users, "items": items, "reviews": reviews}


def _assert_global_payloads(users, items, reviews):
    assert isinstance(users, dict) and isinstance(items, dict) and isinstance(reviews, dict)
    for rid, record in reviews.items():
        assert record["item_id"] in items
        assert record["user_id"] in users
        assert record["city"] == items[record["item_id"]]["city"]
    for item_id, meta in items.items():
        for rid in meta["review_ids"]:
            assert rid in reviews
    for uid, info in users.items():
        for rid in info["review_ids"]:
            assert rid in reviews


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_root", required=True)
    args = parser.parse_args()
    payload = prepare_yelp_data(Path(args.dset_root))
    users = payload.get("users", {})
    items = payload.get("items", {})
    reviews = payload.get("reviews", {})
    print(len(users), len(items), len(reviews))
    sample_user = next(iter(users.items()), None)
    sample_item = next(iter(items.items()), None)
    sample_review = next(iter(reviews.items()), None)
    if sample_user:
        print("user", sample_user[0], sample_user[1])
    if sample_item:
        print("item", sample_item[0], sample_item[1])
    if sample_review:
        print("review", sample_review[0], sample_review[1])
