# python -m dataset.make_yelp

import json
from collections import defaultdict, Counter
from utils import get_dset_root, dumpj, iter_line
from debug import check

def filter_user_item_data():
    dsetroot = get_dset_root()
    dsetroot = dsetroot / 'yelp'
    business_file = dsetroot/"yelp_academic_dataset_business.json" # 150346
    review_file = dsetroot/"yelp_academic_dataset_review.json" # 6990280
    tip_file = dsetroot/"yelp_academic_dataset_tip.json" # 908915
    # user_file = dsetroot/"yelp_academic_dataset_user.json" # 1987897

    # --- Part 1: Process business_file to build the CITYR skeleton ---
    all_restaurant_ids = set()
    city_restaurant_counts = Counter()
    _restaurants_filtered = defaultdict(dict)

    for line in iter_line(business_file, 150346):
        biz = json.loads(line)
        if biz.get("categories") and "restaurant" in biz["categories"].lower():
            biz_id = biz["business_id"]
            city = biz.get("city", "X").strip()
            review_count = biz.get("review_count", 0)

            all_restaurant_ids.add(biz_id)
            city_restaurant_counts[city] += 1

            if review_count > 50:
                # Store the entire business JSON object
                for key in ['address', 'city', 'state', 'postal_code', 'latitude', 'longitude']:
                    if key in biz:
                        del biz[key]
                _restaurants_filtered[city][biz_id] = biz

    city_restaurant_counts.pop("X", None)
    cities = {city for city, count in city_restaurant_counts.items() if count > 500}

    # --- Part 2: Build the final CITYR structure and helper map ---
    CITYR = defaultdict(dict)
    biz_to_city_map = {}

    for city, businesses in _restaurants_filtered.items():
        if city in cities:
            for biz_id, biz_info in businesses.items():
                # For each target business, create its final structure
                CITYR[city][biz_id] = {
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
        revtext = rev['text'] + f'| stars: {rev["stars"]}; useful: {rev["useful"]}; funny: {rev["funny"]}; cool: {rev["cool"]}'
        if biz_id in all_restaurant_ids:
            user_interactions[rev["user_id"]].append(revtext)
        
        city = biz_to_city_map.get(biz_id)
        if city:
            CITYR[city][biz_id]["interactions"].append(revtext)

    # Process tips
    for line in iter_line(tip_file, 908915):
        tip = json.loads(line)
        biz_id = tip["business_id"]
        # Add a type identifier
        if biz_id in all_restaurant_ids:
            user_interactions[tip["user_id"]].append(tip['text'])
            
        city = biz_to_city_map.get(biz_id)
        if city:
            CITYR[city][biz_id]["interactions"].append(tip['text'])

    # --- Part 4: Final user filtering based on combined interactions ---
    USERS = {
        uid: interactions for uid, interactions in user_interactions.items()
        if len(interactions) > 50
    }

    return USERS, CITYR

def main():

    USERS, CITYR = filter_user_item_data()
    check()




if __name__ == '__main__':
    main()