#dense.py
import numpy as np
from .base import BaseSystem
from collections import defaultdict

class DenseRetrieverBaseline(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        
        self.top_m = getattr(args, "dense_top_m", getattr(args, "bm25_top_m", 3))
        self.encode_batch_size = 64
        self.item_length = {k: len(v) for k, v in self.item_segments.items()}
        

    def recommend(self, request):
        query_vec = self._encode_query(request)
        scores = self.embedding @ query_vec

        # --- Rank by score ---
        order = np.argsort(scores)[::-1][:self.retrieve_k]

        # --- Aggregate by item_id ---
        aggregated = defaultdict(list)
        for idx in order:
            score = float(scores[idx])
            element = self.segments[idx] 
            item_id = element.get("item_id")
            aggregated[item_id].append(score)

        results = []
        for item_id, score_list in aggregated.items():
            total_score = sum(score_list)
            denom = self.item_length.get(item_id, 1)
            normalized_score = total_score / denom
            results.append((item_id, normalized_score))

        # --- Sort by normalized score descending ---
        results.sort(key=lambda x: x[1], reverse=True)

        self.top_items = [t[0] for t in results[:self.top_k]]