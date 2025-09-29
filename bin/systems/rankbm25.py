from collections import defaultdict

from rank_bm25 import BM25Okapi

from .base import BaseSystem


class BM25Baseline(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cache = {}
        if self.default_city:
            self._ensure_city(self.default_city)

    def _ensure_city(self, city):
        key = self.normalize_city(city)
        if not key:
            return None
        if key in self.cache:
            return self.cache[key]
        payload = self.get_city_data(key)
        if not payload:
            self.cache[key] = None
            return None
        docs = self._build_documents(payload)
        item_ids = list(docs.keys())
        tokenized = [self._tokenize(docs[iid]) for iid in item_ids]
        pairs = [(iid, toks) for iid, toks in zip(item_ids, tokenized) if toks]
        if not pairs:
            self.cache[key] = None
            return None
        item_ids = [iid for iid, _ in pairs]
        tokenized = [toks for _, toks in pairs]
        bm25 = BM25Okapi(tokenized)
        filtered_docs = {iid: docs[iid] for iid in item_ids}
        model = {
            'bm25': bm25,
            'item_ids': item_ids,
            'documents': filtered_docs,
            'tokenized': tokenized,
        }
        self.cache[key] = model
        return model

    def _build_documents(self, payload):
        reviews = payload.get('REVIEWS')
        if not reviews:
            reviews = []
        info = payload.get('INFO')
        if not info:
            info = {}
        grouped = defaultdict(list)
        for entry in reviews:
            item_id = entry.get('item_id')
            if not item_id and entry.get('business_id'):
                item_id = entry.get('business_id')
            text = entry.get('text')
            if not item_id or not text:
                continue
            grouped[item_id].append(text.strip())
        docs = {}
        for item_id, texts in grouped.items():
            parts = []
            meta = info.get(item_id)
            if meta:
                name = meta.get('name')
                if name:
                    parts.append(str(name).strip())
                address = meta.get('address')
                if address:
                    parts.append(str(address).strip())
                categories = meta.get('categories')
                if isinstance(categories, list) and categories:
                    parts.append(' '.join(c.strip() for c in categories if c))
            parts.extend(texts)
            joined = '\n'.join(part for part in parts if part)
            docs[item_id] = joined
        return docs

    def _tokenize(self, text):
        if not text:
            return []
        cleaned = []
        for ch in text:
            if ch.isalnum():
                cleaned.append(ch.lower())
            else:
                cleaned.append(' ')
        tokens = [tok for tok in ''.join(cleaned).split() if tok]
        return tokens

    def rank_items(self, request, city=None, top_k=None):
        model = self._ensure_city(city or self.default_city)
        if not model or not request:
            return []
        tokens = self._tokenize(request)
        if not tokens:
            return []
        scores = model['bm25'].get_scores(tokens)
        pairs = sorted(zip(model['item_ids'], scores), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            if top_k <= 0:
                return []
            pairs = pairs[:top_k]
        return [item_id for item_id, _ in pairs]

    def recommend(self, request, city=None, top_k=10):
        return self.rank_items(request, city=city, top_k=top_k)
