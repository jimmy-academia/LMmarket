# data/generate_request.py

input('warning: not reviewed yet')

import argparse
import json
import random
from pathlib import Path

import numpy as np

from utils import loadp, writef
from llm import run_llm_batch

KEYWORD_TASK = "segment_to_keyword"
REQUEST_TASK = "keyword_bundle_to_request"

IMPORTANCE_LEVELS = [
    "critical",
    "high priority",
    "useful",
    "optional",
]

PERFORMANCE_LEVELS = [
    "must excel",
    "should be reliable",
    "acceptable if decent",
    "experiment-friendly",
]


def _normalize_rows(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _sample_embedding_points(embeddings, count, seed):
    rng = np.random.default_rng(seed)
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    std[std == 0] = 1.0
    noise = rng.standard_normal(size=(count, embeddings.shape[1]))
    return mean + noise * std


def _nearest_segment_indices(points, embeddings_norm):
    indices = []
    for point in points:
        norm = np.linalg.norm(point)
        if norm == 0:
            norm = 1.0
        sims = embeddings_norm @ (point / norm)
        indices.append(int(np.argmax(sims)))
    return indices


def _clean_keyword(text):
    if not text:
        return ""
    line = text.strip().splitlines()[0].strip()
    if line.startswith("- "):
        line = line[2:]
    if line.startswith("*"):
        line = line.lstrip("* ")
    if line.startswith("1."):
        line = line[2:].lstrip()
    if line.startswith("\"") and line.endswith("\""):
        line = line[1:-1]
    return line.strip()


def _build_keyword_prompts(segment_texts):
    prompts = []
    for text in segment_texts:
        prompt = (
            "Extract a concise aspect keyword (max six words) that captures the user need "
            "expressed in the following review segment. Focus on product attributes or "
            "experience factors. Respond with only the keyword.\n\n"
            f"Segment: "
            f"{text}"
        )
        prompts.append(prompt)
    return prompts


def _build_request_prompts(keyword_bundles):
    prompts = []
    for bundle in keyword_bundles:
        lines = []
        for idx, info in enumerate(bundle, 1):
            line = (
                f"{idx}. {info['keyword']} — importance: {info['importance']}; "
                f"performance: {info['performance']}"
            )
            lines.append(line)
        joined = "\n".join(lines)
        prompt = (
            "Write a natural shopping request in 2-3 sentences that a user might give to a "
            "recommendation assistant. Reference the desired aspects, keeping the tone "
            "practical and grounded. Avoid bullet lists.\n\nDesired aspects:\n"
            f"{joined}\n\nUser request:"
        )
        prompts.append(prompt)
    return prompts


def generate_user_requests(segment_path, embedding_path, output_request_path, output_meta_path, num_requests=20, keywords_per_request=3, sample_multiplier=4, seed=17, model="gpt-4.1-mini"):
    random.seed(seed)
    np.random.seed(seed)

    segment_path = Path(segment_path)
    embedding_path = Path(embedding_path)

    segments = loadp(segment_path)
    embeddings = np.load(embedding_path)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    if len(segments) != embeddings.shape[0]:
        raise ValueError("Segment and embedding counts do not match")

    points_needed = num_requests * keywords_per_request * sample_multiplier
    sampled_points = _sample_embedding_points(embeddings, points_needed, seed)
    embeddings_norm = _normalize_rows(embeddings.astype(np.float32))
    nearest = _nearest_segment_indices(sampled_points, embeddings_norm)

    required_segments = num_requests * keywords_per_request
    seen = set()
    ordered_indices = []
    for idx in nearest:
        if idx in seen:
            continue
        seen.add(idx)
        ordered_indices.append(idx)
        if len(ordered_indices) >= required_segments:
            break

    if len(ordered_indices) < required_segments:
        remaining = [i for i in range(len(segments)) if i not in seen]
        random.shuffle(remaining)
        ordered_indices.extend(remaining[:required_segments - len(ordered_indices)])

    segment_texts = []
    segment_refs = []
    for idx in ordered_indices:
        entry = segments[idx]
        text = entry.get("text")
        if not text:
            text = entry.get("segment_text")
        if not text:
            text = ""
        segment_texts.append(text.strip())
        segment_refs.append(
            {
                "segment_index": idx,
                "segment_id": entry.get("segment_id"),
                "review_id": entry.get("review_id"),
                "item_id": entry.get("item_id"),
                "text": text.strip(),
            }
        )

    keyword_prompts = _build_keyword_prompts(segment_texts)
    keyword_outputs = run_llm_batch(keyword_prompts, KEYWORD_TASK, model=model)

    keyword_records = []
    keyword_seen = set()
    for ref, raw in zip(segment_refs, keyword_outputs):
        keyword = _clean_keyword(raw)
        if not keyword:
            continue
        key = keyword.lower()
        if key in keyword_seen:
            continue
        keyword_seen.add(key)
        rec = {
            "keyword": keyword,
            "segment_index": ref["segment_index"],
            "segment_id": ref.get("segment_id"),
            "review_id": ref.get("review_id"),
            "item_id": ref.get("item_id"),
            "segment_text": ref.get("text"),
        }
        keyword_records.append(rec)

    if not keyword_records:
        raise ValueError("No keywords were generated from segments")

    pool = keyword_records[:]
    random.shuffle(pool)

    keyword_bundles = []
    cursor = 0
    for _ in range(num_requests):
        bundle = []
        for _ in range(keywords_per_request):
            if cursor >= len(pool):
                random.shuffle(pool)
                cursor = 0
            base = pool[cursor]
            cursor += 1
            entry = {
                "keyword": base["keyword"],
                "segment_index": base["segment_index"],
                "segment_id": base.get("segment_id"),
                "review_id": base.get("review_id"),
                "item_id": base.get("item_id"),
                "segment_text": base.get("segment_text"),
                "importance": random.choice(IMPORTANCE_LEVELS),
                "performance": random.choice(PERFORMANCE_LEVELS),
            }
            bundle.append(entry)
        keyword_bundles.append(bundle)

    request_prompts = _build_request_prompts(keyword_bundles)
    requests = run_llm_batch(request_prompts, REQUEST_TASK, model=model)
    requests = [r.strip() for r in requests]

    request_payload = []
    for request_text, bundle in zip(requests, keyword_bundles):
        request_payload.append(
            {
                "request": request_text,
                "aspects": bundle,
            }
        )

    request_text_block = "\n\n".join(requests)
    writef(output_request_path, request_text_block)
    writef(output_meta_path, json.dumps(request_payload, indent=2))
    return request_payload


def main():
    parser = argparse.ArgumentParser(description="Generate user requests from review segments")
    parser.add_argument("--segment_path", required=True)
    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--output_request_path", default="generated_requests.txt")
    parser.add_argument("--output_meta_path", default="generated_requests_meta.json")
    parser.add_argument("--num_requests", type=int, default=20)
    parser.add_argument("--keywords_per_request", type=int, default=3)
    parser.add_argument("--sample_multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    generate_user_requests(
        args.segment_path,
        args.embedding_path,
        args.output_request_path,
        args.output_meta_path,
        num_requests=args.num_requests,
        keywords_per_request=args.keywords_per_request,
        sample_multiplier=args.sample_multiplier,
        seed=args.seed,
        model=args.model,
    )


if __name__ == "__main__":
    main()
