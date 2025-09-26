from collections import defaultdict

from utils import load_or_build, dumpj, loadj

class BaseSystem:
    def __init__(self, args, reviews, tests):
        self.args = args
        self.tests = tests
        self.test_review_ids = [t["review_id"] for t in self.tests]

        self.reviews = [r for r in reviews if r["review_id"] not in self.test_review_ids]
        
        self.rich_reviews = load_or_build(args.rich_reviews_path, dumpj, loadj, lambda x:x, self.reviews)

        self.user_index = self._build_user_index()
        self.test_data, self.ground_truth = self._prep_tests(self.tests)

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
        
        


