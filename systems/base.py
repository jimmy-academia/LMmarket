from collections import defaultdict

from utils import load_or_build, dumpj, loadj

class BaseSystem:
    def __init__(self, args, DATA):
        self.args = args
        self.data = DATA

    def _build_user_index(self):
        index = defaultdict(list)
        for r in self.reviews:
            user_id = r.get("user_id")
            index[user_id].append(r)
        return index

    def serve_one_request(self, aspect_list):
        raise NotImplementedError("predict_with_aspect must be implemented by subclass")

    def serve(self, requests):
        for request, aspect_list in requests:
            self.serve_one_request(request, aspect_list)
    
    def evaluate(self):
        


    # def _prep_tests(self, tests):
    #     test_data, ground_truth = [], []
    #     for t in tests:
    #         units   = t.get("opinion_units")
    #         aspects = [u[0] for u in units]
    #         scores  = [u[2] for u in units]
    #         test_data.append({
    #             "user_id":     t.get("user_id"),
    #             "item_id":     t.get("item_id"),
    #             "aspects": aspects,
    #         })
    #         ground_truth.append(scores)
    #     return test_data, ground_truth

    # # ---------- make predictions ----------

    # def predict_given_aspects(self, user_id, item_id, aspects):
    #     """Return list of floats (same length as aspects)"""
    #     raise NotImplementedError("predict_with_aspect must be implemented by subclass")

    # def predict_all(self):
    #     """Iterate over self.test_data and call predict_with_aspect for each aspect."""
    #     predictions = []
    #     for t in self.test_data:
    #         uid, iid, aspects = t["user_id"], t["item_id"], t["aspects"]
    #         y_pred = self.predict_given_aspects(uid, iid, aspects)
    #         predictions.append(y_pred)
    #     return predictions

    def evaluate(self, predictions):
        sims, accs = [], []

        for t, y_pred, y_true in zip(self.test_data, predictions, self.ground_truth):
            sim = self._cosine(y_true, y_pred)
            acc = (sum(self._quantize(tp) == self._quantize(pp) for tp, pp in zip(y_true, y_pred)) / len(y_true))
            sims.append(sim); accs.append(acc)
            print(t['aspects'], y_pred, y_true, sim, acc)
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


class HeuristicBaseline(BaseSystem):
