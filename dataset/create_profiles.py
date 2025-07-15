from .prompts import price_sensitivity_prompt, preference_prompt, price_estimate_prompt, feature_prompt
from llm import query_llm, safe_json_extract

from tqdm import tqdm

def build_user_profile(USERS, domain, model="openai"):
    '''
    Create user profile (with LLM)
    - budget -- price-sensitivity (inverse)
    - text descriptions profile summary
    - dynamic feature pool
    '''

    profiles = []
    uid = 0
    desc = "Building user profiles for "+domain
    for user in tqdm(USERS, ncols=90, desc=desc):
        user_text = "\n".join(user["interactions"])

        prompt = price_sensitivity_prompt(user_text, domain)
        response = query_llm(prompt, model=model).strip()
        price_sensitivity = max(0, min(1.0, float(response)/4))

        prompt = preference_prompt(user_text, domain)
        extra = safe_json_extract(prompt, model=model)
        profile = {
            "uid" : uid, 
            "user_id": user["user_id"],
            "price_sensitivity": price_sensitivity,
            "text_description": extra.get("text_description", ""),
            "preferences": extra.get("preferences", [])
            "importance": extra.get("importance", [])
        }
        profiles.append(profile)
        uid += 1

    return profiles


def build_item_profile(ITEMS, domain, model="openai"):

    '''
    Create item profile (with LLM)
    - estimate set price
    - text descriptions profile summary
    - dynamic feature pool
    '''

    profiles = []
    iid = 0
    desc = "BUilding item profiles for "+domain
    for item in tqdm(ITEMS, ncols=90, desc=desc):
        item_text = str(item["info"])+ "\n".join(item["interactions"])

        prompt = price_estimate_prompt(item_text, domain)
        response = query_llm(prompt, model=model).strip()
        price = max(0, min(1.0, float(response)))

        prompt = feature_prompt(item_text, domain)
        extra = safe_json_extract(prompt, model=model)
        profile = {
            "iid" : iid, 
            "item_id": item["biz_id"],
            "price": price,
            "text_description": extra.get("text_description", ""),
            "features": extra.get("features", [])
            "importance": extra.get("importance", [])
        }
        profiles.append(profile)
        iid += 1

    return profiles

