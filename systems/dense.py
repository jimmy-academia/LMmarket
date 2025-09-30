import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseSystem


class DenseRetrieverBaseline(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.retrieve_k = getattr(args, "retrieve_k", 500)
        self.top_m = getattr(args, "dense_top_m", getattr(args, "bm25_top_m", 3))
        self.top_k = getattr(args, "top_k", self.default_top_k)
        self.encode_batch_size = getattr(args, "encode_batch_size", 64)
        self.embedder_name = getattr(args, "embedder_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.normalize = getattr(args, "normalize_embeddings", True)
        self.encoder = SentenceTransformer(self.embedder_name)
        self.city_cache = {}

    def _ensure_city(self, city=None):
        key = self.get_city_key(city)
        if not key:
            return None
        if key in self.city_cache:
            return self.city_cache[key]
        payload = self.get_city_data(key)
        if not payload:
            self.city_cache[key] = None
            return None
        reviews = payload.get("REVIEWS")
        if not isinstance(reviews, list):
            self.city_cache[key] = None
            return None
        review_ids = []
        review_items = []
        review_texts = []
        texts = []
        for entry in reviews:
            if not isinstance(entry, dict):
                continue
            rid = entry.get("review_id")
            item_id = entry.get("item_id")
            if not item_id:
                item_id = entry.get("business_id")
            text = entry.get("text")
            if not rid or not item_id or not text:
                continue
            cleaned = str(text).strip()
            if not cleaned:
                continue
            review_ids.append(rid)
            review_items.append(item_id)
            review_texts.append(cleaned)
            texts.append(cleaned)
        if not texts:
            self.city_cache[key] = None
            return None
        embeddings = self.encoder.encode(texts, batch_size=self.encode_batch_size, normalize_embeddings=self.normalize)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        embeddings = embeddings.astype("float32", copy=False)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        model = {
            "embeddings": embeddings,
            "review_ids": review_ids,
            "review_items": review_items,
            "review_texts": review_texts,
        }
        self.city_cache[key] = model
        return model

    def _encode_query(self, text):
        if not text:
            return None
        encoded = self.encoder.encode([text], batch_size=1, normalize_embeddings=self.normalize)
        if isinstance(encoded, np.ndarray):
            query = encoded[0]
        else:
            query = np.array(encoded)[0]
        query = query.astype("float32", copy=False)
        if self.normalize:
            norm = np.linalg.norm(query)
            if norm == 0:
                return None
            query = query / norm
        return query

    def recommend(self, request, city=None, top_k=None):
        if not request:
            return []
        model = self._ensure_city(city)
        if not model:
            return []
        query_vec = self._encode_query(request)
        if query_vec is None:
            return []
        embeddings = model["embeddings"]
        scores = embeddings @ query_vec
        order = np.argsort(scores)[::-1]
        limit = self.retrieve_k if isinstance(self.retrieve_k, int) and self.retrieve_k > 0 else len(order)
        aggregated = {}
        for idx in order[:limit]:
            score = float(scores[idx])
            if score <= 0:
                break
            item_id = model["review_items"][idx]
            rid = model["review_ids"][idx]
            text = model["review_texts"][idx]
            aggregated.setdefault(item_id, []).append((score, rid, text))
        if not aggregated:
            return []
        results = []
        top_m = self.top_m if isinstance(self.top_m, int) and self.top_m > 0 else None
        for item_id, entries in aggregated.items():
            entries.sort(key=lambda row: row[0], reverse=True)
            use = entries[:top_m] if top_m is not None else entries
            total = 0.0
            evidence = []
            snippets = []
            for score, rid, text in use:
                total += float(score)
                evidence.append(rid)
                summary = self._normalize_text(text)
                if summary:
                    snippets.append(summary)
            short_excerpt = self._compose_excerpt(snippets)
            full_explanation = self._compose_explanation(snippets)
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
