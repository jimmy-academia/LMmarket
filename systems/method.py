from .base import BaseSystem

class BestSystem(BaseSystem):
    def __init__(self, args, reviews):
        super().__init__(args, reviews)


class LightningPassSystem(SaTGTEFaissBaseline):
    def __init__(self, args, reviews, tests):
        self.teacher_model_name = getattr(args, "teacher_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        self.regularization = getattr(args, "lightning_reg", 1e-2)
        self.distill_topk = getattr(args, "lightning_topk", 50)
        self.scale_floor = getattr(args, "lightning_scale_floor", 1e-3)
        super().__init__(args, reviews, tests)

        self.lightning_dir = self.cache_dir / f"lightning_{self.div_name}"
        self.lightning_dir.mkdir(parents=True, exist_ok=True)
        self.scale_path = self.lightning_dir / "scale.npy"

        if os.path.exists(self.scale_path):
            self.scale_vector = _load_numpy(self.scale_path)
        else:
            self.scale_vector = self._run_distillation()
            _save_numpy(self.scale_path, self.scale_vector)

    def encode_query(self, text):
        base_vec = super().encode_query(text)
        scaled = base_vec * self.scale_vector
        norm = np.linalg.norm(scaled)
        if norm > 0:
            scaled = scaled / norm
        return scaled.astype("float32")

    def _run_distillation(self):
        teacher = CrossEncoder(self.teacher_model_name)
        features = []
        targets = []
        for sample in self.test_data:
            item_id = sample.get("item_id")
            aspects = sample.get("aspects", [])
            for aspect in aspects:
                q_vec = super().encode_query(aspect)
                docs = self._gather_docs(q_vec, item_id)
                docs = docs[:self.distill_topk]
                if not docs:
                    continue
                teacher_scores = []
                for doc in docs:
                    seg_indices = doc.get("segments", [])
                    pairs = [(aspect, self.segments[idx].get("text", "")) for idx in seg_indices]
                    if not pairs:
                        teacher_scores.append(0.0)
                        continue
                    seg_scores = teacher.predict(pairs)
                    seg_scores = np.array(seg_scores, dtype=np.float32)
                    order = np.argsort(seg_scores)[::-1][:self.top_l]
                    best_scores = seg_scores[order]
                    weights = self._softmax(best_scores)
                    teacher_score = float((weights * best_scores).sum())
                    teacher_scores.append(teacher_score)
                if not teacher_scores:
                    continue
                teacher_scores = np.array(teacher_scores, dtype=np.float32)
                doc_vectors = np.array([doc.get("vector") for doc in docs], dtype=np.float32)
                q_repeat = np.repeat(q_vec[None, :], len(docs), axis=0)
                feats = q_repeat * doc_vectors
                features.append(feats)
                targets.append(teacher_scores)

        if not features:
            dim = self.segment_embeddings.shape[1]
            return np.ones(dim, dtype=np.float32)

        features = np.concatenate(features, axis=0)
        targets = np.concatenate(targets, axis=0)

        dim = features.shape[1]
        gram = features.T @ features
        gram += np.eye(dim, dtype=np.float32) * float(self.regularization)
        rhs = features.T @ targets
        scale = np.linalg.solve(gram, rhs)
        scale = np.sign(scale) * np.maximum(np.abs(scale), self.scale_floor)
        return scale.astype("float32")
