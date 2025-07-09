# python3 -m dataset.make_amazon_beauty

import json
from collections import defaultdict, Counter
from utils import get_dset_root, iter_line, dumpj, loadj
from llm import query_llm, safe_json_extract
from debug import check

from tqdm import tqdm
import textwrap

def filter_user_item_data():
    dsetroot = get_dset_root()
    review_file = dsetroot / "Amazon" / "All_Beauty" / "All_Beauty.jsonl" #701529
    item_file = dsetroot / "Amazon" / "All_Beauty" / "meta_All_Beauty.jsonl" #112591

    all_item_ids = set()

    user_interactions = defaultdict(list)
    ITEMS_DICT = defaultdict(list)

    for line in iter_line(item_file, 112591):
        item = json.loads(line)
        item_id = item["parent_asin"]
        all_item_ids.add(item_id)
        for key in ["main_category", "images", "videos", "bought_together", "categories"]:
            if key in item:
                del item[key]
        ITEMS_DICT[item_id] = {
            "info" : item,
            "interactions" : []
        }

    for line in iter_line(review_file, 701529):
        rev = json.loads(line)
        # if rev["helpful_vote"] > 0:
        #     continue
        revtext = f"title : {rev["title"]} | text : {rev["text"]} | rating : {rev["rating"]} | helpful_vote : {rev["helpful_vote"]}"
        user_interactions[rev["user_id"]].append(revtext)
        if rev["parent_asin"] in all_item_ids:
            ITEMS_DICT[rev["parent_asin"]]["interactions"].append(revtext)

    USERS = [
        {"user_id": uid, "interactions": interactions} for uid, interactions in user_interactions.items()
        # if len(interactions) > 1
    ]
    ITEMS = [ {"user_id": uid, **interactions} for uid, interactions in ITEMS_DICT.items() if len(interactions['interactions']) > 1
    ]
    return {"USERS": USERS, "ITEMS": ITEMS}
    # 631986, 64999

####################
### USER PROFILE ###
####################

# (later use in dynamic) ambiance_noise, ambiance_lighting, crowd_density
static_feature_descriptives = {
    "quality": ["Very poor quality; product is uncomfortable, unreliable, or breaks easily", "Below average quality; some flaws, inconsistent performance or texture", "Average quality; meets basic expectations for comfort and durability", "Good quality; well-made, pleasant to use, and reliable over time", "Excellent quality; premium materials, superior feel, and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"]
}

def format_feature_prompt(user_text, feature_name, spectrum):
    scale = "\n".join([f"{i-2}. {desc}" for i, desc in enumerate(spectrum)])
    return f"""
You are evaluating a beauty goods reviewer's preferences.

Below are their recent reviews:
{textwrap.shorten(user_text, width=6000)}

Now consider the feature: **{feature_name}**

Rate this user's preference or expectation on the following -2 to +2 scale:
{scale}

Return only a single float value from -2 to 2.
"""

def format_description_and_dynamic_prompt(user_text):
    return f"""
Based on the following beauty goods reviews:

{textwrap.shorten(user_text, width=6000)}

Generate:
1. A short summary (1–2 sentences) of the user's beauty goods preferences and habits.
2. A list of any notable preferences, special requests, or constraints — e.g. Fragrance-free, Oily skin, Eco-friendly, Waterproof, etc.

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
        user_text = "\n".join(user["interactions"])  # [:20] limit to ~20 reviews
        preference_vector = []

        for feature_name, descriptions in static_feature_descriptives.items():
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

def main():
    Data = filter_user_item_data()
    USERS, ITEMS = Data["USERS"], Data["ITEMS"]
    check()

if __name__ == "__main__":
    main()
