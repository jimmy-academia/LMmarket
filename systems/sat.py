# https://github.com/segment-any-text/wtpsplit?tab=readme-ov-file

from .base import BaseSystem
from wtpsplit import SaT

class SATBaseline(BaseSystem):
    def __init__(self, args, reviews, tests):
        super().__init__(args, reviews, tests)

        self.cache_dir = Path(getattr(self.args, "cache_dir", "cache"))
        self.div_name = getattr(self.args, "div_name", "default")

        self.segment_batch_size = getattr(self.args, "segment_batch_size", 64)
        self.top_l = getattr(self.args, "top_l_segments", 3)
        self.doc_limit = getattr(self.args, "top_docs", 3)
        self.faiss_topk = getattr(self.args, "faiss_topk", 256)
        self.temperature = getattr(self.args, "segment_temperature", 0.1)

        self.segment_cache_path = self.cache_dir / f"segments_{self.div_name}.pkl"
        self.embedding_path = self.cache_dir / f"gte_segments_{self.div_name}.npy"
        self.sentiment_path = self.cache_dir / f"segment_sentiment_{self.div_name}.npy"
        self.index_path = self.cache_dir / f"gte_segments_{self.div_name}.faiss"

        self.embedder_name = getattr(self.args, "embedder_name", "Alibaba-NLP/gte-large-en-v1.5")
        use_gpu = getattr(self.args, "use_gpu", True)
        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.encoder = SentenceTransformer(self.embedder_name, device=device, trust_remote_code=True)

        self.positive_words = {
            "amazing",
            "awesome",
            "best",
            "clean",
            "delicious",
            "fantastic",
            "friendly",
            "great",
            "love",
            "perfect",
            "tasty",
            "wonderful",
        }
        self.negative_words = {
            "awful",
            "bad",
            "dirty",
            "disappointing",
            "horrible",
            "poor",
            "rude",
            "slow",
            "terrible",
            "unfriendly",
            "worst",
        }

        self.segments = load_or_build(self.segment_cache_path, dumpp, loadp, self._build_segments)
        self.segment_embeddings = load_or_build(self.embedding_path, _save_numpy, _load_numpy, self._build_embeddings)
        self.segment_sentiments = load_or_build(self.sentiment_path, _save_numpy, _load_numpy, self._build_sentiments)
        self.index = self._load_or_create_index(self.segment_embeddings)

        self.review_to_segments = defaultdict(list)
        self.item_to_segments = defaultdict(list)
        for idx, seg in enumerate(self.segments):
            rid = seg.get("review_id")
            iid = seg.get("item_id")
            self.review_to_segments[rid].append(idx)
            self.item_to_segments[iid].append(idx)

    def _build_segments(self):
        sat = SaT("sat-3l")
        rows = []
        total = len(self.reviews)
        step = max(1, self.segment_batch_size)
        for start in range(0, total, step):
            batch = self.reviews[start:start + step]
            texts = [r.get("text", "") for r in batch]
            segmented = list(sat.split(texts))
            for review, segments in zip(batch, segmented):
                rid = review.get("review_id")
                iid = review.get("item_id")
                uid = review.get("user_id")
                for pos, seg in enumerate(segments):
                    text = seg.strip()
                    if not text:
                        continue
                    rows.append({
                        "segment_id": f"{rid}::{pos}",
                        "review_id": rid,
                        "item_id": iid,
                        "user_id": uid,
                        "position": pos,
                        "text": text,
                    })
        return rows

    def _build_embeddings(self):
        texts = [s.get("text", "") for s in self.segments]
        embeddings = self.encoder.encode(texts, batch_size=64, normalize_embeddings=True)
        return embeddings.astype("float32")

    def _build_sentiments(self):
        scores = []
        for seg in self.segments:
            text = seg.get("text", "")
            tokens = [t for t in ''.join(ch.lower() if ch.isalpha() else ' ' for ch in text).split() if t]
            pos = sum(1 for t in tokens if t in self.positive_words)
            neg = sum(1 for t in tokens if t in self.negative_words)
            score = 0.0
            total = pos + neg
            if total:
                score = (pos - neg) / float(total)
            scores.append(score)
        return np.array(scores, dtype="float32")

    def _load_or_create_index(self, embeddings):
        if os.path.exists(self.index_path):
            return faiss.read_index(str(self.index_path))
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 80
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        return index

    def encode_query(self, text):
        vec = self.encoder.encode([text], batch_size=1, normalize_embeddings=True)
        return vec[0].astype("float32")

    def _softmax(self, scores):
        arr = np.array(scores, dtype=np.float32)
        arr = arr / max(self.temperature, 1e-6)
        arr = arr - arr.max()
        weights = np.exp(arr)
        denom = weights.sum()
        if denom <= 0:
            return np.zeros_like(weights)
        return weights / denom

    def _retrieve_segment_indices(self, query_vec, item_id):
        q = query_vec.astype("float32")[None, :]
        scores, indices = self.index.search(q, min(self.faiss_topk, len(self.segment_embeddings)))
        indices = indices[0]
        candidate = []
        seen = set()
        for idx in indices:
            if idx < 0:
                continue
            seg = self.segments[idx]
            if seg.get("item_id") == item_id:
                candidate.append(idx)
                seen.add(idx)
        if len(candidate) < self.top_l:
            extra = self.item_to_segments.get(item_id, [])
            if extra:
                sims = self.segment_embeddings[extra] @ query_vec
                order = np.argsort(sims)[::-1]
                for idx in order:
                    seg_idx = extra[idx]
                    if seg_idx not in seen:
                        candidate.append(seg_idx)
                        seen.add(seg_idx)
        return candidate

    def _gather_docs(self, query_vec, item_id):
        seg_indices = self._retrieve_segment_indices(query_vec, item_id)
        doc_map = defaultdict(list)
        for idx in seg_indices:
            seg = self.segments[idx]
            rid = seg.get("review_id")
            sim = float(np.dot(query_vec, self.segment_embeddings[idx]))
            sentiment = float(self.segment_sentiments[idx])
            doc_map[rid].append((sim, idx, sentiment))

        docs = []
        for rid, values in doc_map.items():
            sims = np.array([v[0] for v in values], dtype=np.float32)
            order = np.argsort(sims)[::-1][:self.top_l]
            top_indices = [values[i][1] for i in order]
            top_sims = sims[order]
            weights = self._softmax(top_sims)
            seg_embeds = self.segment_embeddings[top_indices]
            seg_sents = np.array([values[i][2] for i in order], dtype=np.float32)
            doc_vector = (weights[:, None] * seg_embeds).sum(axis=0)
            doc_strength = float((weights * top_sims).sum())
            doc_sentiment = float((weights * seg_sents).sum())
            docs.append({
                "review_id": rid,
                "vector": doc_vector.astype("float32"),
                "strength": doc_strength,
                "sentiment": doc_sentiment,
                "segments": top_indices,
                "segment_weights": weights,
            })
        docs.sort(key=lambda d: d["strength"], reverse=True)
        return docs

    def _aggregate_docs(self, docs):
        if not docs:
            return 0.0
        selected = docs[:self.doc_limit]
        weights = np.array([max(d.get("strength"), 0.0) for d in selected], dtype=np.float32)
        sentiments = np.array([d.get("sentiment") for d in selected], dtype=np.float32)
        denom = weights.sum()
        if denom <= 0:
            return 0.0
        return float(np.dot(weights, sentiments) / denom)

    def predict_given_aspects(self, user_id, item_id, aspects):
        outputs = []
        for aspect in aspects:
            q_vec = self.encode_query(aspect)
            docs = self._gather_docs(q_vec, item_id)
            score = self._aggregate_docs(docs)
            outputs.append(score)
        return outputs


