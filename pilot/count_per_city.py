import json
from collections import defaultdict

# Path to your dataset
file_path = "yelp_academic_dataset_business.json"

total_businesses = 0
restaurant_counts_by_city = defaultdict(int)
business_counts_by_city = defaultdict(int)
chicago_count = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        business = json.loads(line)
        total_businesses += 1

        # Check if it's a restaurant
        categories = business.get("categories", "")
        city = business.get("city", "Unknown")
        business_counts_by_city[city] += 1
        if 'chicago' in city.lower():
            chicago_count += 1
            # print(business)
        if categories and "restaurant" in categories.lower():
            restaurant_counts_by_city[city] += 1


# Print total businesses
print(f"Total businesses: {total_businesses}")

# print(chicago_count, 'Chicago')

# Print number of restaurants per city
print("Restaurant count by city:")
for city, count in sorted(restaurant_counts_by_city.items(), key=lambda x: -x[1]):
    if count > 500:
        print(f"{city}: {count}")

# for city, count in sorted(business_counts_by_city.items(), key=lambda x: -x[1]):
#     if count > 15:
#         print(f"{city}: {count}")
