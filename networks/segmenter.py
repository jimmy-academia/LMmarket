from wtpsplit import SaT
from tqdm import tqdm


def segment_reviews(reviews, batch_size):
    segments = []
    segment_lookup = {}
    review_segments = {}
    item_segments = {}
    if isinstance(reviews, dict):
        values = reviews.values()
    else:
        values = reviews
    valid_reviews = [r for r in values if isinstance(r, dict) and r.get("text")]
    if not valid_reviews:
        return {
            "segments": segments,
            "segment_lookup": segment_lookup,
            "review_segments": review_segments,
            "item_segments": item_segments,
        }
    step = batch_size or 32
    model = SaT("sat-12l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    total = len(valid_reviews)
    for start in tqdm(range(0, total, step), ncols=88, desc="[segmenter] split"):
        batch = valid_reviews[start:start + step]
        texts = [r.get("text") for r in batch]
        splits = list(model.split(texts))
        for review, pieces in zip(batch, splits):
            rid = review.get("review_id")
            item_id = review.get("item_id")
            user_id = review.get("user_id")
            collected = []
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
                collected.append(record)
            if rid:
                review_segments[rid] = list(collected)
            if item_id:
                existing = item_segments.get(item_id)
                if existing is None:
                    existing = []
                    item_segments[item_id] = existing
                existing.extend(collected)
    return {
        "segments": segments,
        "segment_lookup": segment_lookup,
        "review_segments": review_segments,
        "item_segments": item_segments,
    }


def apply_segment_data(payload):
    data = payload if isinstance(payload, dict) else {}
    segments_value = data.get("segments")
    lookup_value = data.get("segment_lookup")
    review_value = data.get("review_segments")
    item_value = data.get("item_segments")
    segments = segments_value if isinstance(segments_value, list) else []
    segment_lookup = lookup_value if isinstance(lookup_value, dict) else {}
    review_segments = review_value if isinstance(review_value, dict) else {}
    item_segments = item_value if isinstance(item_value, dict) else {}
    return segments, segment_lookup, review_segments, item_segments
