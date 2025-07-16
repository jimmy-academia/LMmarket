features = ["Vegetarian options", "Outdoor seating", "Friendly and attentive service", "Overpriced for portion/quality", "Creative or unique menu items", "Casual and relaxed atmosphere", "Cleanliness and hygiene", "Kid-friendly environment", "Quick food service", "Generous portion sizes", "Gluten-free options", "Lively or vibrant vibe", "Memorable signature dishes", "Good value for money", "High food quality (taste/freshness)", "Diverse menu options", "Parking availability", "Comfortable for working or studying", "Wait times during peak hours", "Good drink or cocktail selection"]


import json
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load data
with open("cache/yelp_data.json", "r") as f:
    data = json.load(f)

users = data["USERS"][:20]
items = data["ITEMS"][:20]

# Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")
feature_embeddings = model.encode(features)

# Compute similarity
def compute_scores(entities, entity_type):
    results = []
    for entity in entities:
        if entity_type == "user":
            text = " ".join(entity["interactions"])
        else:
            text = " ".join(entity["interactions"]) + " " + json.dumps(entity["info"])
        embedding = model.encode([text])[0]
        scores = cosine_similarity([embedding], feature_embeddings)[0]
        results.append(scores)
    return results

# Calculate scores
user_scores = compute_scores(users, "user")
item_scores = compute_scores(items, "item")

# Save or display results
user_df = pd.DataFrame(user_scores, columns=features, index=[u["user_id"] for u in users])
item_df = pd.DataFrame(item_scores, columns=features, index=[i["business_id"] for i in items])

user_df.to_csv("user_feature_scores.csv")
item_df.to_csv("item_feature_scores.csv")

print("Saved to user_feature_scores.csv and item_feature_scores.csv")
