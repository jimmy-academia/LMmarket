# python3 -m dataset.make_amazon

import json
from collections import defaultdict, Counter
from utils import get_dset_root, iter_line, dumpj, loadj
from llm import query_llm, safe_json_extract
from debug import check
import argparse

from tqdm import tqdm
import textwrap

def filter_user_item_data(catergory):
    dsetroot = get_dset_root()
    review_file = dsetroot / "Amazon" / f"{catergory}.jsonl" 
    item_file = dsetroot / "Amazon" / f"meta_{catergory}.jsonl" 

    all_item_ids = set()

    user_interactions = defaultdict(list)
    ITEMS_DICT = defaultdict(list)

    for line in iter_line(item_file):
        item = json.loads(line)
        item_id = item["parent_asin"]
        item["main_category"] = "main_category"
        all_item_ids.add(item_id)
        for key in ["main_category", "images", "videos", "bought_together", "categories"]:
            if key in item:
                del item[key]
        ITEMS_DICT[item_id] = {
            "info" : item,
            "interactions" : []
        }

    for line in iter_line(review_file):
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

def filter_all_data():
    USERS, ITEMS = [], []
    for catergory in ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"]:
        data = filter_user_item_data(catergory)
        USERS.extend(data["USERS"])
        ITEMS.extend(data["ITEMS"])
    return {"USERS": USERS, "ITEMS": ITEMS}

####################
### USER PROFILE ###
####################

# (later use in dynamic) ambiance_noise, ambiance_lighting, crowd_density
from dataset.amazon_criteria import static_feature_descriptives

def format_feature_prompt(catergory, user_text, feature_name, spectrum):
    scale = "\n".join([f"{i-2}. {desc}" for i, desc in enumerate(spectrum)])
    return f"""
You are evaluating a {catergory} goods reviewer's preferences.

Below are their recent reviews:
{textwrap.shorten(user_text, width=6000)}

Now consider the feature: **{feature_name}**

Rate this user's preference or expectation on the following -2 to +2 scale:
{scale}

Return only a single float value from -2 to 2.
"""

def format_description_and_dynamic_prompt(catergory, user_text):
    return f"""
Based on the following {catergory} goods reviews:

{textwrap.shorten(user_text, width=6000)}

Generate:
1. A short summary (1–2 sentences) of the user's {catergory} goods preferences and habits.
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

def generate_user_request_prompt(catergory, user_profile, dynamic_preferences):
    preference_vector = user_profile["preference_vector"]
    price_sensitivity = user_profile["price_sensitivity"]
    text_description = user_profile["text_description"]
    prompt = f"""
Below is a user's profile, play as this user and generate a request of {catergory} goods.
User request should be complex and detail and match user's preferences.
preference_vector : {preference_vector}
price sensitivity : {price_sensitivity}
other preferences : {dynamic_preferences}
text_description : {text_description}
preference vector  is a feature representation where each dimension is a score derived from a distinct evaluation metric.
The metrics is showed as below. It's scaled from -1 to 1.
The last metrics is for price sensitivity, it's also scaled from -1 to 1.
{static_feature_descriptives}
Your request should reflect and match the metrics well.
"""
    return query_llm(prompt=prompt)

def main(catergory):
    if catergory == "all":
        Data = filter_all_data()
    else:
        Data = filter_user_item_data(catergory)
    USERS, ITEMS = Data["USERS"], Data["ITEMS"]
    check()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catergory", "-c")
    args = parser.parse_args()
    main(args.catergory)
