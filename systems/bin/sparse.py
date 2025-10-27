#sparse.py
import logging
from collections import defaultdict
from rank_bm25 import BM25Okapi
from tqdm import tqdm 

from .base import BaseSystem
from debug import check

class BM25Baseline(BaseSystem):
    def __init__(self, args, data, use_segment=False):
        super().__init__(args, data)
        self.sequence = self.segments if use_segment else self.reviews
        item_div = self.item_segment if use_segment else self.item_reviews
        self.item_length = {k: len(v) for k, v in item_div.items()}
        self.bm25 = self._ensure_bm25_model(use_segment)
        logging.info("[BM25Baseline] Initialized.")

    def recommend(self, request):
        self.request = request
        
        query_tokens = self._tokenize(request)
        scores = self.bm25.get_scores(query_tokens)
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        indexed_scores.sort(key=lambda p: p[1], reverse=True)
        indexed_scores = indexed_scores[: self.retrieve_k]

        # top_retrieved_text = [self.sequence[p[0]]['text'] for p in indexed_scores[:self.top_k]]

        # logging.info('[BM25Baseline] sparse check')
        # for t in top_retrieved_text:
        #     print(t)

        aggregated = defaultdict(list)
        for idx, score in indexed_scores:
            element = self.sequence[idx]
            item_id = element.get("item_id")
            review_id = element.get("review_id")
            text = element.get("text")
            aggregated[item_id].append(score)

        results = []
        for item_id, scores in aggregated.items():
            total_score = sum(scores)
            denom = self.item_length.get(item_id, 1)
            normalized_score = total_score / denom
            results.append((item_id, normalized_score))

        # Sort by normalized score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Top-k items only
        self.top_items = [t[0] for t in results[:self.top_k]]
        
    def _ensure_bm25_model(self, use_segment=False):
        div = "segments" if use_segment else "full reviews"
        logging.info(f"[BM25Baseline] Building BM25 model over {div}...")
        corpus_tokens = []
        for element in tqdm(self.sequence, ncols=88, desc='[sparse] collecting tokens for bm25...'):
            corpus_tokens.append(self._tokenize(element["text"]))
        return BM25Okapi(corpus_tokens)

    def _tokenize(self, text):
        cleaned = []
        for ch in text:
            cleaned.append(ch.lower() if ch.isalnum() else " ")
        return [tok for tok in "".join(cleaned).split() if tok]

