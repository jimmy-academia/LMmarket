# data_foundation/benchmark_maker.py
import random
from collections import Counter
from systems.ou import OUBaseline, PROMPT_TEMPLATE

def construct_benchmark(reviews, num_test=5, seed=0):
    rng = random.Random(seed)

    # 1) unique (user,item) pairs in this city
    pair_counts = Counter((r["user_id"], r["item_id"]) for r in reviews)
    unique_ids = [r["review_id"] for r in reviews if pair_counts[(r["user_id"], r["item_id"])] == 1]

    rng.shuffle(unique_ids)
    hold_ids = set(unique_ids[:num_test])

    test_reviews = [r for r in reviews if r["review_id"] in hold_ids]
    ou = object.__new__(OUBaseline)
    ou.prompt_template = PROMPT_TEMPLATE
    ou.ou_model = "gpt-5-nano"
    ou.ou_temperature = 0
    ou.num_workers = 128
    input('check')
    units = OUBaseline.segmentation(ou, test_reviews)
    # annotates r["opinion_units"] in-place

    # 4) pack test samples
    tests = []
    for r in test_reviews:
        tests.append({
            "review_id": r["review_id"],
            "user_id": r["user_id"],
            "item_id": r["item_id"],
            "review_text": r["text"],
            "opinion_units": [(u["aspect"], u["excerpt"], u["sentiment_score"]) for u in r.get("opinion_units", [])],
        })
    return tests
