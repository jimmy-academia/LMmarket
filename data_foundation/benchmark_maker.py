# data_foundation/benchmark_maker.py
import random
from collections import Counter
from systems.ou import OUBaseline

def build_benchmark(reviews, k=5, seed=0):
    rng = random.Random(seed)

    # 1) unique (user,item) pairs in this city
    pair_counts = Counter((r["user_id"], r["business_id"]) for r in reviews)
    unique_ids = [r["review_id"] for r in reviews if pair_counts[(r["user_id"], r["business_id"])] == 1]

    rng.shuffle(unique_ids)
    hold_ids = set(unique_ids[:k])

    # 2) collect held-out reviews and make DATA'
    test_reviews = [r for r in reviews if r["review_id"] in hold_ids]
    reviews = [r for r in reviews if r["review_id"] not in hold_ids]
    # USERS
    ou = OUBaseline(None, test_reviews)
    all_units = ou.segmentation() # annotates r["opinion_units"] in-place

    # 4) pack test samples
    tests = []
    for r in test_reviews:
        tests.append({
            "review_id": r["review_id"],
            "user_id": r["user_id"],
            "item_id": r["business_id"],
            "review_text": r["text"],
            "opinion_units": [(u["aspect"], u["excerpt"], u["sentiment"]) for u in r.get("opinion_units", [])],
        })
    from debug import check
    check()
    return reviews, tests
