import json
from collections import defaultdict, Counter
from datetime import datetime
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os

from tqdm import tqdm 

# Ensure required NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from utils import get_dset_root, dumpj

make_bar = lambda f, x: tqdm(f, ncols=88, total=x)

dsetroot = get_dset_root()
dsetroot = dsetroot/'yelp'
business_file = dsetroot/"yelp_academic_dataset_business.json"
review_file = dsetroot/"yelp_academic_dataset_review.json"
tip_file = dsetroot/"yelp_academic_dataset_business.json"
user_file = dsetroot/"yelp_academic_dataset_user.json"

# Step 1: Identify restaurant business_ids
restaurant_business_ids = set()
with open(business_file, "r", encoding="utf-8") as f:
    for line in f:
        biz = json.loads(line)
        if biz.get("categories") and "restaurant" in biz["categories"].lower():
            restaurant_business_ids.add(biz["business_id"])

# Step 2: Collect user reviews for restaurant businesses
user_reviews = defaultdict(list)
with open(review_file, "r", encoding="utf-8") as f:
    for line in f:
        rev = json.loads(line)
        if rev["business_id"] in restaurant_business_ids:
            user_reviews[rev["user_id"]].append(rev)

# Step 3: Filter users with more than 50 restaurant reviews
qualified_users = {user_id: reviews for user_id, reviews in user_reviews.items() if len(reviews) > 50}

# Step 4: Filter cities with more than 500 restaurants, retain restaurant in these cities that has more than 50 reviews.

city_restaurants_all = defaultdict(set)
city_restaurants_filtered = defaultdict(set)

with open(business_file, "r", encoding="utf-8") as f:
    for line in make_bar(f, 150346):
        biz = json.loads(line)
        if biz.get("categories") and "restaurant" in biz["categories"].lower():
            city = biz.get("city", "").strip()
            biz_id = biz["business_id"]
            review_count = biz.get("review_count", 0)

            # Group all restaurants by city
            city_restaurants_all[city].add(biz_id)

            # Filter: more than 50 reviews
            if review_count > 50:
                city_restaurants_filtered[city].add(biz_id)

# Only keep cities with more than 500 restaurants total
final_city_restaurants = {
    city: biz_ids
    for city, biz_ids in city_restaurants_filtered.items()
    if len(city_restaurants_all[city]) > 500
}

# Summary report
print(f"{'City':30s} | {'Total':>6} | {'Filtered >50 reviews':>20}")
print("-" * 60)
for city in sorted(final_city_restaurants, key=lambda x: -len(final_city_restaurants[x])):
    total = len(city_restaurants_all[city])
    filtered = len(final_city_restaurants[city])
    print(f"{city:30s} | {total:6d} | {filtered:20d}")


# Helper functions
def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

def extract_adjectives(text):
    words = word_tokenize(clean_text(text))
    filtered = [word for word in words if word not in stopwords.words('english') and word.isalpha()]
    return filtered

def estimate_tokens(text):
    return len(word_tokenize(text))

# Step 4: Process one sample user
sample_user_id, sample_reviews = next(iter(qualified_users.items()))
review_summaries = []
all_adjectives = []

for rev in sample_reviews:
    text = rev["text"]
    tokens = estimate_tokens(text)
    adjectives = extract_adjectives(text)
    all_adjectives.extend(adjectives)
    
    review_summaries.append({
        "review_id": rev["review_id"],
        "business_name": "unknown",
        "category": "Restaurants",
        "date": rev["date"].split(" ")[0],
        "stars": rev["stars"],
        "aspects": [],  # Placeholder
        "key_phrases": [],  # Placeholder
        "length_tokens": tokens
    })

# Compute overall stats
rating_distribution = Counter([int(r["stars"]) for r in sample_reviews])
dates = [datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S") for r in sample_reviews]
first_review_date = min(dates).strftime("%Y-%m-%d")
last_review_date = max(dates).strftime("%Y-%m-%d")

user_profile = {
    "overall_stats": {
        "total_reviews": len(sample_reviews),
        "average_stars": sum([r["stars"] for r in sample_reviews]) / len(sample_reviews),
        "rating_distribution": dict(rating_distribution),
        "first_review_date": first_review_date,
        "last_review_date": last_review_date
    },
    "category_stats": [
        { "category": "Restaurants", "review_count": len(sample_reviews), "average_stars": sum([r["stars"] for r in sample_reviews]) / len(sample_reviews) }
    ],
    "review_summaries": review_summaries,
    "linguistic_traits": {
        "avg_tokens": sum([r["length_tokens"] for r in review_summaries]) / len(review_summaries),
        "median_tokens": sorted([r["length_tokens"] for r in review_summaries])[len(review_summaries)//2],
        "uses_first_person": any(re.search(r'\b(i|me|my|mine|we|us|our|ours)\b', r["text"].lower()) for r in sample_reviews),
        "top_adjectives": [word for word, _ in Counter(all_adjectives).most_common(10)]
    }
}

dumpj(user_profile, 'sample_user.json')

