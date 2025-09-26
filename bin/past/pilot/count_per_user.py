import json
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# File paths
business_file = "yelp_academic_dataset_business.json"
review_file = "yelp_academic_dataset_review.json"
user_file = "yelp_academic_dataset_user.json"

# Step 1: Find all restaurant business_ids
restaurant_business_ids = set()
with open(business_file, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        biz = json.loads(line)
        if biz.get("categories") and "restaurant" in biz["categories"].lower():
            restaurant_business_ids.add(biz["business_id"])

# Step 2: Count total and restaurant-specific reviews per user
user_total_reviews = defaultdict(int)
user_restaurant_reviews = defaultdict(int)

with open(review_file, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        rev = json.loads(line)
        user_id = rev["user_id"]
        business_id = rev["business_id"]
        user_total_reviews[user_id] += 1
        if business_id in restaurant_business_ids:
            user_restaurant_reviews[user_id] += 1

# Step 3: Filter and collect user info
users_over_150_reviews = []
users_over_15_restaurant_reviews = []

with open(user_file, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        user = json.loads(line)
        uid = user["user_id"]
        name = user.get("name", "Unknown")

        if user_total_reviews[uid] > 150:
            users_over_150_reviews.append((uid, name, user_total_reviews[uid]))
        if user_restaurant_reviews[uid] > 15:
            users_over_15_restaurant_reviews.append((uid, name, user_restaurant_reviews[uid]))

# # Step 4: Print summaries
# print("\nUsers with more than 150 total reviews:")
# for uid, name, count in users_over_150_reviews:
#     print(f"{name} ({uid}) - {count} total reviews")

# print("\nUsers with more than 15 restaurant reviews:")
# for uid, name, count in users_over_15_restaurant_reviews:
#     print(f"{name} ({uid}) - {count} restaurant reviews")

# Step 5: Visualization
total_review_counts = list(user_total_reviews.values())
restaurant_review_counts = list(user_restaurant_reviews.values())

# Histogram of total reviews
plt.figure()
n_total, bins_total, _ = plt.hist(total_review_counts, bins=50, range=(0, 300), edgecolor='black')
plt.title("Distribution of Total Reviews per User")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Users")
plt.grid(True)
plt.savefig("total_reviews_distribution.jpg")

# Print total review bin counts
print("\nTotal Review Count Distribution:")
for i in range(len(n_total)):
    print(f"{int(bins_total[i]):>3}–{int(bins_total[i+1]):>3}: {int(n_total[i])} users")

# Histogram of restaurant reviews
plt.figure()
n_rest, bins_rest, _ = plt.hist(restaurant_review_counts, bins=30, range=(0, 100), edgecolor='black')
plt.title("Distribution of Restaurant Reviews per User")
plt.xlabel("Number of Restaurant Reviews")
plt.ylabel("Number of Users")
plt.grid(True)
plt.savefig("restaurant_reviews_distribution.jpg")

# Print restaurant review bin counts
print("\nRestaurant Review Count Distribution:")
for i in range(len(n_rest)):
    print(f"{int(bins_rest[i]):>3}–{int(bins_rest[i+1]):>3}: {int(n_rest[i])} users")

# plt.show()
