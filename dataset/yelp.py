import json
from collections import defaultdict, Counter

from utils import get_dset_root, load_make, iter_line

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

    # --- Step 1: Identify valid restaurants ---
    all_restaurant_ids = set()
    city_restaurant_counts = Counter()
    city_restaurants_filtered = defaultdict(dict)
    biz_id_to_biz = dict()

    for line in iter_line(business_file, 150_346):
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
                    biz.pop(key, None)
                city_restaurants_filtered[city][biz_id] = biz

    city_restaurant_counts.pop("X", None)
    # filter city with > 500 restaurant each with > 50 reviews
    cities = {city for city, count in city_restaurant_counts.items() if count > 500}

    # --- Step 2: Build CITYR structure ---
    CITYR_dict = defaultdict(dict)
    biz_to_city_map = {}

    for city, businesses in city_restaurants_filtered.items():
        if city in cities:
            for biz_id, biz_info in businesses.items():
                # For each target business, create its final structure
                CITYR_dict[city][biz_id] = {
                    "info": biz_info,
                    "review_ids": [] # Ready to hold reviews and tips
                }
                biz_to_city_map[biz_id] = city

    # --- Step 3: Process reviews and tips ---
    user_reviews = defaultdict(list)  # user_id -> [review_id]
    REVIEWS = []

    # Process reviews
    for line in iter_line(review_file, 6_990_280):
        rev = json.loads(line)
        biz_id = rev["business_id"]
        rev_id = rev["review_id"]
        user_id = rev["user_id"]
        revtext = f'|{rev["stars"]} stars| ' + rev['text'] 
        # useful: {rev["useful"]}; funny: {rev["funny"]}; cool: {rev["cool"]}
        
        if len(rev['text']) > 200 and biz_id in all_restaurant_ids:
            user_reviews[user_id].append(rev_id)
            REVIEWS.append({
                "review_id": rev_id,
                "user_id": user_id,
                "business_id": biz_id,
                "text": revtext,
            })
            
            city = biz_to_city_map.get(biz_id)
            if city:
                CITYR_dict[city][biz_id]["review_ids"].append(rev_id)

    # Process tips
    tip_counter = 0
    for line in iter_line(tip_file, 908_915):
        tip = json.loads(line)
        biz_id = tip["business_id"]
        user_id = tip["user_id"]
        if len(tip['text']) > 200 and biz_id in all_restaurant_ids:
            tip_counter += 1
            tip_id = f"tip_{tip_counter}"

            user_reviews[user_id].append(tip_id)
            REVIEWS.append({
                "review_id": tip_id,
                "user_id": user_id,
                "business_id": biz_id,
                "text": tip['text'],
            })

            city = biz_to_city_map.get(biz_id)
            if city:
                CITYR_dict[city][biz_id]["review_ids"].append(tip_id)


    # --- Part 4: Final user filtering based on combined review_ids ---
    # select users with > 50 reviews (of > 200 characters; filtered above)
    USERS = [
        {"user_id": uid, "review_ids": review_ids}
        for uid, review_ids in user_reviews.items()
        if len(review_ids) > 50
    ]

    ITEMS = []
    for city, businesses in CITYR_dict.items():
        for biz_id, data in businesses.items():
            ITEMS.append({
                "business_id": biz_id,
                "city": city,
                **data  # includes info and review_ids
            })

    return {"USERS": USERS, "ITEMS": ITEMS, "REVIEWS": REVIEWS}
