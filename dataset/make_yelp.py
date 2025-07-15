# python -m dataset.make_yelp

import json
from collections import defaultdict, Counter

from .create_profiles import build_user_profiles, build_item_profiles

from utils import load_make, iter_line

def load_yelp_data():
    '''
    retrieve the information from the json files
    '''
    dsetroot = get_dset_root()
    dsetroot = dsetroot / 'yelp'
    business_file = dsetroot/"yelp_academic_dataset_business.json" # 150346
    review_file = dsetroot/"yelp_academic_dataset_review.json" # 6990280
    tip_file = dsetroot/"yelp_academic_dataset_tip.json" # 908915
    # ignore social network for now
    # user_file = dsetroot/"yelp_academic_dataset_user.json" # 1987897

    # --- Part 1: Process business_file to build the CITYR skeleton ---
    all_restaurant_ids = set()
    city_restaurant_counts = Counter()
    city_restaurants_filtered = defaultdict(dict)
    biz_id_to_biz = dict()

    for line in iter_line(business_file, 150346):
        biz = json.loads(line)
        if biz.get("categories") and "restaurant" in biz["categories"].lower():
            biz_id = biz["business_id"]
            city = biz.get("city", "X").strip()
            review_count = biz.get("review_count", 0)

            all_restaurant_ids.add(biz_id)
            city_restaurant_counts[city] += 1
            biz_id_to_biz[biz_id] = biz

            if review_count > 50:
                # Store the entire business JSON object
                for key in ['address', 'city', 'state', 'postal_code', 'latitude', 'longitude']:
                    if key in biz:
                        del biz[key]
                city_restaurants_filtered[city][biz_id] = biz

    city_restaurant_counts.pop("X", None)
    # filter city with > 500 restaurant each with > 50 reviews
    cities = {city for city, count in city_restaurant_counts.items() if count > 500}

    # --- Part 2: Build the final CITYR structure and helper map ---
    CITYR_dict = defaultdict(dict)
    biz_to_city_map = {}

    for city, businesses in city_restaurants_filtered.items():
        if city in cities:
            for biz_id, biz_info in businesses.items():
                # For each target business, create its final structure
                CITYR_dict[city][biz_id] = {
                    "info": biz_info,
                    "interactions": [] # Ready to hold reviews and tips
                }
                biz_to_city_map[biz_id] = city

    # --- Part 3: Process interaction files (reviews AND tips) ---
    user_interactions = defaultdict(list)

    # Process reviews
    for line in iter_line(review_file, 6990280):
        rev = json.loads(line)
        biz_id = rev["business_id"]
        if len(rev['text']) > 200 and biz_id in all_restaurant_ids:
            revtext = rev['text'] + f'| stars: {rev["stars"]}; useful: {rev["useful"]}; funny: {rev["funny"]}; cool: {rev["cool"]}'
            
            user_interactions[rev["user_id"]].append(revtext)
            
            city = biz_to_city_map.get(biz_id)
            if city:
                CITYR_dict[city][biz_id]["interactions"].append(revtext)

    # Process tips
    for line in iter_line(tip_file, 908915):
        tip = json.loads(line)
        biz_id = tip["business_id"]
        # Add a type identifier
        if len(rev['text']) > 200 and biz_id in all_restaurant_ids:
            user_interactions[tip["user_id"]].append(tip['text'])
            
        city = biz_to_city_map.get(biz_id)
        if city:
            CITYR_dict[city][biz_id]["interactions"].append(tip['text'])

    # --- Part 4: Final user filtering based on combined interactions ---
    # select users with > 50 reviews (of > 200 characters; filtered above)
    USERS = [
        {"user_id": uid, "interactions": interactions}
        for uid, interactions in user_interactions.items()
        if len(interactions) > 50
    ]

    ITEMS = []
    for city, businesses in CITYR_dict.items():
        for biz_id, data in businesses.items():
            ITEMS.append({
                "business_id": biz_id,
                "city": city,
                **data  # includes info and interactions
            })

    return {"USERS": USERS, "ITEMS": ITEMS}


def main():
    # collect data (no LLM)
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    data_path = cache_dir / "yelp_data.json"
    data = load_make(data_path, load_yelp_data)

    # create user profile (with LLM)
    USERS = data["USERS"]
    ITEMS = data["ITEMS"]
    domain_name = 'Yelp restaurants'
    user_profile_path = cache_dir / "user_profile.json"
    item_profile_path = cache_dir / "item_profile.json"
    build_yelp_user_profile = lambda: build_user_profiles(USERS, domain_name)
    build_yelp_item_profile = lambda: build_item_profiles(ITEMS, domain_name)

    user_profile = load_make(user_profile_path, build_yelp_user_profile)
    item_profile = load_make(item_profile_path, build_yelp_item_profile)

    # finalize combination and ground truth setting


if __name__ == '__main__':
    main()