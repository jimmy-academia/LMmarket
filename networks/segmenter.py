import logging
from collections import defaultdict
from tqdm import tqdm

from wtpsplit import SaT

def segment_reviews(reviews, batch_size):
    valid_reviews = [r for r in reviews if isinstance(r, dict) and r.get("text")]
    if not valid_reviews:
        logging.error('Not valid reviews')
    step = batch_size or 32
    model = SaT("sat-12l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    total = len(valid_reviews)

    segments = []
    segment_lookup = {}
    review_segments = defaultdict(list)
    item_segments = defaultdict(list)
    item_reviews = defaultdict(list)

    for start in tqdm(range(0, total, step), ncols=88, desc="[segmenter] split"):
        batch = valid_reviews[start:start + step]
        texts = [r.get("text") for r in batch]
        splits = list(model.split(texts))
        for review, pieces in zip(batch, splits):
            rid = review.get("review_id")
            item_id = review.get("item_id")
            user_id = review.get("user_id")
            for pos, segment in enumerate(pieces):
                content = segment.strip()
                if not content:
                    continue
                seg_id = f"{rid}::{pos}" if rid else f"seg::{len(segments)}"
                record = {
                    "segment_id": seg_id,
                    "review_id": rid,
                    "item_id": item_id,
                    "user_id": user_id,
                    "position": pos,
                    "text": content,
                }
                segments.append(record)
                segment_lookup[seg_id] = record
                review_segments[rid].append(record)
                item_segments[item_id].append(seg_id)
            item_reviews[item_id].append(rid)
    return {
        "segments": segments,
        "segment_lookup": segment_lookup,
        "review_segments": review_segments,
        "item_segments": item_segments,
        "item_reviews": item_reviews
    }

