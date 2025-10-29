import re
import numpy as np
from rank_bm25 import BM25Okapi

class Searchable:
    def __init__(self, reviews):
        self.name = 'reviews'
        self.reviews = reviews 
        self.texts = [review['text'] for review in self.reviews]
        self._tokens = [self._tok(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

        self.length = len(self.texts)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.reviews[index]

    def __iter__(self):
        return iter(self.reviews)


    def _tok(self, t):
        return re.findall(r"\w+", (t or "").lower())

    def _make_snippet(self, text, q_tokens, window=100):
        low = text.lower()
        hit = next((t for t in q_tokens if t in low), None)
        if not hit:
            snippet = (text[:window*2] + "…") if len(text) > window*2 else text
            return snippet, None
        i = low.find(hit)
        start = max(0, i - window)
        end   = min(len(text), i + len(hit) + window)
        snippet = text[start:end]
        # bold the first hit
        snippet = snippet[:i-start] + "**" + text[i:i+len(hit)] + "**" + snippet[i-start+len(hit):]
        if start > 0: snippet = "…" + snippet
        if end < len(text): snippet = snippet + "…"
        return snippet, hit

    def search(self, query, topk=None, silent=False):
        if topk is None:
            topk = self.length

        q = self._tok(query)
        scores = self._bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:topk]
        if not silent: print(f"\n🔍 {self.name} — top {topk} for '{query}':\n")
        out = []
        for i in idx:
            review  = self.reviews[i]
            text = review['text']
            snippet, hit = self._make_snippet(text, q)
            if hit:
                rating_val = review.get("stars")
                rating_str = f"⭐ {rating_val}"
                score_str  = f"📈 {scores[i]:.2f}"

                prefix = f"{rating_str} | {score_str}" 
                if not silent: print(f"{prefix} | {snippet}")
                if not silent: print(f"→ keyword: {hit}\n")
                out.append({"review_id": review['review_id'], "item_id": review['item_id'], "review": review, "score": float(scores[i]), "hit": hit, "rating": rating_val, "snippet": snippet, "text": review['text']})
        return out