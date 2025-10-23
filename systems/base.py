import re
import logging
import numpy as np
from rank_bm25 import BM25Okapi

class Searchable:
    def __init__(self, docs, name="collection"):
        self.name = name
        self.docs = docs
        self.texts = [d["text"] for d in self.docs]
        self._tokens = [self._tok(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokens)

    def _tok(self, t):
        return re.findall(r"\w+", (t or "").lower())

    def _make_snippet(self, text, q_tokens, window=80):
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

    def search(self, query, topk=5):
        q = self._tok(query)
        scores = self._bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:topk]
        print(f"\n🔍 {self.name} — top {topk} for '{query}':\n")
        out = []
        for i in idx:
            doc  = self.docs[i]
            text = doc.get(self.text_key, "")
            snippet, hit = self._make_snippet(text, q)
            print(f"⭐ {scores[i]:.2f} | {snippet}")
            if hit: print(f"→ keyword: {hit}\n")
            out.append({"doc": doc, "score": float(scores[i]), "hit": hit})
        return out

class BaseSystem:
    '''
    provides
    self.reviews [list of reviews (dict)]
    self.segments [list of segments (dict)]
    self.embedding (segments)
    '''
    def __init__(self, args, data):
        self.args = args
        
        self.reviews = 

        
