## python -m dataset.make --sample --dset yelp
## python -m dataset.make --sample

import os
from pathlib import Path

from utils import load_make
from .yelp import load_yelp_data
from .build_features import build_ontology_by_reviews
from .build_benchmark import build_benchmark, build_benchmark_tmp

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
    rid2review = {r['review_id']: r for r in REVIEWS}
    if args.sample:
        USERS = USERS[:20]
        ITEMS = ITEMS[:20]
        used_rids = {(rid, "USER", u['user_id']) for u in USERS for rid in u["review_ids"][:10]} | \
                    {(rid, "ITEM", i['business_id']) for i in ITEMS for rid in i["review_ids"][:10]}
        review_type_id_pairs = [(rid2review[rid], type_, id) for rid, type_, id in used_rids]

    # 2. construct feature ontology (with LLM)
    ontology_path = "cache/ontology_human_in_the_loop.json"
    if not os.path.exists(ontology_path):
        output = build_ontology_by_reviews(args, review_type_id_pairs)
    print("make: ontology completed")

    if not os.path.exists("cache/benchmark.json"):
        build_benchmark(USERS, ITEMS, REVIEWS, ontology_path)
    if not os.path.exists("cache/new/benchmark.json"):
        build_benchmark_tmp(args)
    print("make: benchmark completed")



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