import textwrap

max_character_width = 10000

def price_sensitivity_prompt(user_text, domain):
    spectrum = [
        "Rarely considers price when choosing",
        "Comfortable paying for better experience",
        "Balances cost with quality",
        "Prefers cheaper budget options",
        "Very sensitive to price changes"
    ]

    scale = "\n".join([f"{i}. {desc}" for i, desc in enumerate(spectrum)])

    return f"""
You are evaluating a {domain} reviewer's preferences.

Below are their recent reviews:
{textwrap.shorten(user_text, width=max_character_width)}

Now consider: the user's price_sensitivity

Rate this user's price_sensitivity on a 0 to 4 scale: {scale}

Return only a single float value from 0 to 4.
"""



def preference_prompt(user_text, domain):
    return f"""
Based on the following {domain} reviews:

{textwrap.shorten(user_text, width=max_character_width)}

Generate:
1. A short summary (2-3 sentences) of the user's preferences and habits.
2. A list of any notable preferences or constraints
3. A list of numerical values between 0 and 1 that indicate the importance of each preference in the list above.

Return valid JSON:
{{
  "text_description": str,
  "preferences": [str]
  "importance": [float]
}}
""".strip()


def price_estimate_prompt(item_text, domain):
    spectrum = [
        "Extremely cheap or budget-friendly",
        "Below average pricing",
        "Moderately priced / mid-range",
        "High-end or premium pricing",
        "Very expensive or luxury-level"
    ]
    scale = "\n".join([f"{i}. {desc}" for i, desc in enumerate(spectrum)])

    return f"""
You are analyzing reviews for a {domain} item.

Below are the recent customer reviews:
{textwrap.shorten(item_text, width=max_character_width)}

Now consider: the item's pricing_level

Rate this item's pricing_level on a 0 to 4 scale: {scale}

Return only a single float value from 0 to 4.
""".strip()


def feature_prompt(item_text, domain):
    return f"""
Based on the following {domain} reviews:

Generate:
1. A short summary (2-3 sentences) of the {domain} item.
2. A list of any notable features or constraints.
3. A list of numerical values between 0 and 1 that indicate the importance of each feature in the list above.

Return valid JSON:
{{
  "text_description": str,
  "features": [str]
  "importance": [float]
}}
""".strip()
