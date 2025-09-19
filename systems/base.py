from collections import defaultdict

class BaseSystem:
    def __init__(self, args, reviews, tests):
        self.args = args
        self.tests = tests
        self.test_review_ids = [t["review_id"] for t in self.tests]

        self.reviews = [r for r in reviews if r["review_id"] not in self.test_review_ids]

        self.user_index = self._build_user_index(self.reviews)
        self.test_data, self.ground_truth = self._prep_tests(self.tests)

    def _build_user_index(self):
        index = defaultdict(list)
        for r in self.reviews:
            user_id = r.get("user_id")
            index[user_id].append(r)
        return index

    def _prep_tests(self, tests):
        prepared, ground_truth = [], []
        for t in tests:
            units   = t.get("opinion_units")
            aspects = [u.get("aspect") for u in units]
            scores  = [self._normalize_sentiment(u.get("sentiment")) for u in units]
            prepared.append({
                "user_id":     t.get("user_id"),
                "item_id":     t.get("item_id"),
                "review_text": t.get("review_text"),
                "aspects": aspects,
            })
            ground_truth.append(scores)
        return prepared, ground_truth

    # ---------- make predictions ----------

    def predict_with_aspect(self, user_id, item_id, aspects):
        """Return list of floats (same length as aspects)"""
        raise NotImplementedError("predict_with_aspect must be implemented by subclass")

    def predict_all(self):
        """Iterate over self.test_data and call predict_with_aspect for each aspect."""
        predictions = []
        for t in self.test_data:
            uid, iid, aspects = t["user_id"], t["item_id"], t["aspects"]
            sent_score = self.predict_with_aspect(uid, iid, aspects)
            predictions.append(sent_scores)
        return predictions

    def evaluate(self, predictions):
        sims, accs = [], []
        for y_pred, y_true in zip(predictions, self.ground_truth):
            sim = self._cosine(y_true, y_pred)
            acc = (sum(self._quantize(tp) == self._quantize(pp) for tp, pp in zip(y_true, y_pred)) / len(y_true))
            sims.append(sim); accs.append(acc)

        return {
            "similarity": (sum(sims) / len(sims)) if sims else 0.0,
            "accuracy":   (sum(accs) / len(accs)) if accs else 0.0,
        }

    # ---------- helper functions ----------

    @staticmethod
    def _quantize(v):
        v = float(v)
        if v <= -0.33: return -1
        if v <=  0.33: return  0
        return 1

    @staticmethod
    def _cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na  = sum(x*x for x in a) ** 0.5
        nb  = sum(y*y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0