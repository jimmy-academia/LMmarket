#sparse.py

from collections import defaultdict
from rank_bm25 import BM25Okapi
from .base import BaseSystem

class BM25Baseline(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cache = {}
        self.retrieve_k = getattr(args, "retrieve_k", 500)
        self.bm25_top_m = getattr(args, "bm25_top_m", 3)
        self.top_k = getattr(args, "top_k", 5)

    def _ensure_city(self, city=None):
        resolved = self.get_city_key(city)
        if not resolved:
            return None
        if resolved in self.cache:
            return self.cache[resolved]
        payload = self.get_city_data(resolved)
        if not payload:
            self.cache[resolved] = None
            return None
        prepared = self._prepare_reviews(payload)
        tokens = prepared.get("tokens")
        if not tokens:
            self.cache[resolved] = None
            return None
        bm25 = BM25Okapi(tokens)
        model = {
            "bm25": bm25,
            "review_ids": prepared.get("review_ids"),
            "review_items": prepared.get("review_items"),
            "review_texts": prepared.get("review_texts"),
        }
        self.cache[resolved] = model
        return model

    def _prepare_reviews(self, payload):
        reviews = payload.get("REVIEWS") if isinstance(payload, dict) else None
        result = {
            "tokens": [],
            "review_ids": [],
            "review_items": [],
            "review_texts": [],
        }
        if not isinstance(reviews, list):
            return result
        for entry in reviews:
            if not isinstance(entry, dict):
                continue
            review_id = entry.get("review_id")
            item_id = entry.get("item_id")
            if not item_id:
                item_id = entry.get("business_id")
            text = entry.get("text")
            if not review_id or not item_id or not text:
                continue
            tokens = self._tokenize(text)
            if not tokens:
                continue
            result["tokens"].append(tokens)
            result["review_ids"].append(review_id)
            result["review_items"].append(item_id)
            result["review_texts"].append(text)
        return result

    def _tokenize(self, text):
        if not text:
            return []
        cleaned = []
        for ch in text:
            if ch.isalnum():
                cleaned.append(ch.lower())
            else:
                cleaned.append(" ")
        tokens = [tok for tok in "".join(cleaned).split() if tok]
        return tokens

    def rank_items(self, request, city=None, top_k=None):
        if not request:
            return []
        model = self._ensure_city(city)
        if not model:
            return []
        tokens = self._tokenize(request)
        if not tokens:
            return []
        scores = model["bm25"].get_scores(tokens)
        indexed_scores = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            indexed_scores.append((index, float(score)))
        if not indexed_scores:
            return []
        indexed_scores.sort(key=lambda pair: pair[1], reverse=True)
        limit = self.retrieve_k if isinstance(self.retrieve_k, int) and self.retrieve_k > 0 else None
        if limit is not None and len(indexed_scores) > limit:
            indexed_scores = indexed_scores[:limit]
        aggregated = defaultdict(list)
        for index, score in indexed_scores:
            if index >= len(model["review_items"]):
                continue
            item_id = model["review_items"][index]
            review_id = model["review_ids"][index] if index < len(model["review_ids"]) else None
            review_text = model["review_texts"][index] if index < len(model["review_texts"]) else None
            if not item_id or not review_id:
                continue
            aggregated[item_id].append((score, review_id, review_text))
        if not aggregated:
            return []
        results = []
        for item_id, pairs in aggregated.items():
            pairs.sort(key=lambda pair: pair[0], reverse=True)
            m = self.bm25_top_m if isinstance(self.bm25_top_m, int) and self.bm25_top_m > 0 else None
            limited = pairs[:m] if m is not None else pairs
            total = 0.0
            evidence = []
            review_summaries = []
            for score, review_id, review_text in limited:
                total += float(score)
                evidence.append(review_id)
                normalized_text = self._normalize_text(review_text)
                if normalized_text:
                    review_summaries.append(normalized_text)
            short_excerpt = self._compose_excerpt(review_summaries)
            full_explanation = self._compose_explanation(review_summaries)
            results.append((item_id, total, evidence, short_excerpt, full_explanation))
        results.sort(key=lambda row: (-row[1], -len(row[2]), row[0]))
        cutoff = top_k if isinstance(top_k, int) and top_k > 0 else self.top_k
        trimmed = results[:cutoff] if cutoff and cutoff > 0 else results
        formatted = []
        for item_id, score, evidence, short_excerpt, full_explanation in trimmed:
            formatted.append({
                "item_id": item_id,
                "model_score": float(score),
                "evidence": list(evidence),
                "short_excerpt": short_excerpt,
                "full_explanation": full_explanation,
            })
        return formatted

    def recommend(self, request, city=None, top_k=None):
        return self.rank_items(request, city=city, top_k=top_k)

    def _normalize_text(self, text):
        if not text:
            return ""
        collapsed = " ".join(str(text).split())
        return collapsed.strip()

    def _compose_excerpt(self, summaries):
        if not summaries:
            return ""
        combined = " ".join(summaries)
        max_len = 160
        if len(combined) <= max_len:
            return combined
        trimmed = combined[:max_len].rstrip()
        return f"{trimmed}…"

    def _compose_explanation(self, summaries):
        if not summaries:
            return ""
        parts = []
        for idx, summary in enumerate(summaries, 1):
            parts.append(f"{idx}) {summary}")
        return " ".join(parts)
