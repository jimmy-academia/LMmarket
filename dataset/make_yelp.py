# python -m dataset.make_yelp

import json
from collections import defaultdict, Counter
from utils import get_dset_root, iter_line, dumpj, loadj
from llm import query_llm, safe_json_extract
from debug import check

import textwrap


def filter_user_item_data():
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
        if biz_id in all_restaurant_ids:
            user_interactions[tip["user_id"]].append(tip['text'])
            
        city = biz_to_city_map.get(biz_id)
        if city:
            CITYR_dict[city][biz_id]["interactions"].append(tip['text'])

    # --- Part 4: Final user filtering based on combined interactions ---
    # select users with > 50 reviews of > 100 characters
    USERS = [
        {"user_id": uid, "interactions": interactions}
        for uid, interactions in user_interactions.items()
        if len(interactions) > 50
    ]

    CITYR = []
    for city, businesses in CITYR_dict.items():
        city_restaurants = [
            {"business_id": biz_id, **data}
            for biz_id, data in businesses.items()
        ]
        CITYR.append(city_restaurants)

    return {"USERS": USERS, "CITYR": CITYR}

####################
### USER PROFILE ###
####################

# (later use in dynamic) ambiance_noise, ambiance_lighting, crowd_density
static_feature_descriptives = {
    "food_quality": ["Inedible or poorly prepared", "Bland or inconsistent", "Generally acceptable taste", "Tasty and well-executed", "Exceptional flavor and preparation"],
    "portion_size": ["Extremely small or unsatisfying", "Somewhat undersized", "Standard / as expected", "Generous portions", "Very large or extremely generous"],
    "service_speed": ["Extremely slow or inattentive", "Slower than expected", "Average speed", "Prompt and efficient", "Exceptionally fast and attentive"],
    "service_attitude": ["Rude or dismissive staff", "Unfriendly or cold", "Neutral or businesslike", "Friendly and courteous", "Exceptionally warm and accommodating"],
    "cleanliness": ["Dirty or unsanitary", "Below average cleanliness", "Generally clean", "Very clean and tidy", "Immaculate and spotless"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"]
}

def format_feature_prompt(user_text, feature_name, spectrum):
    scale = "\n".join([f"{i-2}. {desc}" for i, desc in enumerate(spectrum)])
    return f"""
You are evaluating a restaurant reviewer's preferences.

Below are their recent reviews:
{textwrap.shorten(user_text, width=6000)}

Now consider the feature: **{feature_name}**

Rate this user's preference or expectation on the following -2 to +2 scale:
{scale}

Return only a single float value from -2 to 2.
"""

def format_description_and_dynamic_prompt(user_text):
    return f"""
Based on the following restaurant reviews:

{textwrap.shorten(user_text, width=6000)}

Generate:
1. A short summary (1–2 sentences) of the user's restaurant preferences and habits.
2. A list of any notable preferences, special requests, or constraints — e.g. vegetarian, outdoor seating, kid-friendly, allergy-conscious, etc.

Return valid JSON:
{{
  "text_description": str,
  "dynamic_preferences": [str]
}}
"""

def build_user_profile(USERS, model="openai"):
    '''
    read user reviews (USERS) to create 
    preference vector + budget, price-sensivity (usually inverse)
    text descriptions profile summary
    record special request into dynamic feature pool;
    '''

    profiles = []

    for user in tqdm(USERS, ncols=90, desc="Building user profiles"):
        text = "\n".join(user["interactions"])  # [:20] limit to ~20 reviews
        preference_vector = []

        for feature, descriptions in static_feature_descriptives.items():
            response = query_llm(format_feature_prompt(user_text, feature_name, descriptions), model=model).strip()
            score = max(-1.0, min(1.0, float(response)/2))
            preference_vector.append(score)

        ## budget and price sensitivity
        price_sensitivity = preference_vector.pop(-1)

        extra_data = safe_json_extract(format_description_and_dynamic_prompt(user_text), model=model)
        summary = extra_data.get("text_description", "")
        dynamic_preferences = extra_data.get("dynamic_preferences", [])

        profile = {
            "user_id": user["user_id"],
            "preference_vector": preference_vector,
            "price_sensitivity": price_sensitivity,
            "text_description": summary,
            "dynamic_preferences": dynamic_preferences
        }

        profiles.append(profile)

    return profiles

####################
### ITEM PROFILE ###
####################



def build_item_profile(CITYR):
    '''
    read item reviews (CITYR) to create 
    feature vector + cost
    text descriptions profile summary
    record special features into dynamic feature pool;
    '''
    pass


def main():
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    user_item_path = cache_dir / "user_and_item_data.json"
    user_item_data = load_make(user_item_path, filter_user_item_data)

    USERS = user_item_data["USERS"]
    CITYR = user_item_data["CITYR"]

    user_profile_path = cache_dir / "user_profile.json"
    item_profile_path = cache_dir / "item_profile.json"

    user_profile = load_make(user_profile_path, lambda: build_user_profile(USERS))
    item_profile = load_make(item_profile_path, lambda: build_item_profile(CITYR))





if __name__ == '__main__':
    main()