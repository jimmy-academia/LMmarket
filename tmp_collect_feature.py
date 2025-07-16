# import json
# from pathlib import Path

# def collect_dynamic_features(item_file, user_file):
#     # Load item features
#     with open(item_file, 'r') as f:
#         item_data = json.load(f)

#     all_features = set()
#     for item in item_data:
#         features = item.get("features", [])
#         all_features.update(features)

#     # Load user preferences
#     with open(user_file, 'r') as f:
#         user_data = json.load(f)

#     for user in user_data:
#         preferences = user.get("preferences", [])
#         all_features.update(preferences)

#     return all_features

# # Example usage
# item_file = Path("sample/item_profile_sample.json")
# user_file = Path("sample/user_profile_sample.json")

# all_features = collect_dynamic_features(item_file, user_file)

# print(all_features)


