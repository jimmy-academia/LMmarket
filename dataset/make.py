## python -m dataset.make --sample --dset yelp
## python -m dataset.make --sample

from pathlib import Path
import argparse

from utils import load_make
from .yelp import load_yelp_data
from .build_features import build_ontology_by_reviews

def main(args):
    
    print(f"making dataset: {args.dset}")
    
    # 1. collect data (no LLM)
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    if args.dset == 'yelp':
        data_path = cache_dir / "yelp_data.json"
        data = load_make(data_path, load_yelp_data)
        print('load or created', data_path)
    else:
        input(f"{args.dset} not implemented yet!")
    
    if args.sample:
        print("⚠️  Running in SAMPLE mode: using 20 users and 20 items")
    
    USERS = data["USERS"]
    ITEMS = data["ITEMS"]
    REVIEWS = data["REVIEWS"]
    if args.sample:
        '''
        USERS = USERS[:1]
        ITEMS = ITEMS[:1]
        used_review_ids = {rid for u in USERS for rid in u["review_ids"]} | \
                          {rid for i in ITEMS for rid in i["review_ids"]}
        REVIEWS = [r for r in REVIEWS if r["review_id"] in used_review_ids]
        '''
        REVIEWS = REVIEWS[:100]  # Sample 100 reviews for testing

    # 2. construct feature ontology (with LLM)
    output = build_ontology_by_reviews(args, REVIEWS)

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