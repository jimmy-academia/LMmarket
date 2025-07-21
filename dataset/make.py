## python -m dataset.make --sample --dset yelp
## python -m dataset.make --sample

from yelp import load_yelp_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help="Use a small sample dataset")
    parser.add_argument('--dset', type=str, default='yelp')
    args = parser.parse_args()
    
    print(f"making dataset: {args.dset}")
    if args.sample:
        print("⚠️  Running in SAMPLE mode: using 20 users and 20 items")

    # 1. collect data (no LLM)
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    data_path = cache_dir / "yelp_data.json"
    data = load_make(data_path, load_yelp_data)
    
USERS = data["USERS"]
ITEMS = data["ITEMS"]
REVIEWS = data["REVIEWS"]
if args.sample:
    USERS = USERS[:20]
    ITEMS = ITEMS[:20]
    # REVIEWS = reviews made by the users and made to the items   

    # 2. construct feature ontology (with LLM)
    feature_graph = build_ontology_by_reviews(USERS, domain_name)

    # 3. final user profile, user request, and item profile

    # suffix = "_sample.json" if args.sample else ".json"
    # user_profile_path = cache_dir / f"user_profile{suffix}"
    # item_profile_path = cache_dir / f"item_profile{suffix}"

    
    # build_yelp_user_profile = lambda: build_user_profiles(USERS, feature_graph)
    # build_yelp_item_profile = lambda: build_item_profiles(ITEMS, feature_graph)

    # user_profile = load_make(user_profile_path, build_yelp_user_profile)
    # item_profile = load_make(item_profile_path, build_yelp_item_profile)



if __name__ == '__main__':
    main()