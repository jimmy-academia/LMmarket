# data/hub.py
from collections import defaultdict
from pathlib import Path 
from tqdm import tqdm
import logging

class DataHub:
    def __init__(self, dset_root):
        self.dset_root = Path(dset_root)
        self.max_clean_passes = 3
        self.min_review_chars = 200
        self.min_reviews_per_user = 50
        self.min_reviews_per_item = 50

    def load_data(self):
        raise NotImplementedError("subclass must implement load_data()")

    def clean_data(self):

        self._drop_short_reviews()

        for __ in tqdm(range(self.max_clean_passes), ncols=88, desc='[clean_data]'):
            self._recompute_degrees()                # sets self._udeg / self._ideg from CURRENT reviews
            changed = False
            changed |= self._prune_users_by_degree() # uses self.min_reviews_per_user
            changed |= self._prune_items_by_degree() # uses self.min_reviews_per_item
            changed |= self._drop_dangling_reviews() # remove reviews referencing pruned/missing u/i
            logging.info("length >>> "+str([len(x) for x in [self.users, self.items, self.reviews]]))
            if not changed:
                break

    def _drop_short_reviews(self):
        self.reviews = {rid: r for rid, r in self.reviews.items() if len(r.get("text", "")) >= self.min_review_chars}

    def _recompute_degrees(self):
        udeg, ideg = defaultdict(int), defaultdict(int)    
        for r in self.reviews.values():
            uid, iid = r.get("user_id"), r.get("item_id")
            # Count only if both endpoints still exist
            if uid in self.users and iid in self.items:
                udeg[uid] += 1
                ideg[iid] += 1
        self._udeg, self._ideg = udeg, ideg

    def _prune_users_by_degree(self):
        prev_length = len(self.users)
        self.users = {u: rec for u, rec in self.users.items() if self._udeg.get(u, 0) >= self.min_reviews_per_user}
        return prev_length != len(self.users)

    def _prune_items_by_degree(self):
        prev_length = len(self.items)
        self.items = {i: rec for i, rec in self.items.items() if self._ideg.get(i, 0) >= self.min_reviews_per_item}
        return prev_length != len(self.items)

    def _drop_dangling_reviews(self):
        # only prune review without item
        prev_length = len(self.reviews)
        # self.reviews = {rid: r for rid, r in self.reviews.items() if r.get("item_id") in self.items}
        self.reviews = {rid: r for rid, r in self.reviews.items() if r.get("user_id") in self.users and r.get("item_id") in self.items}
        return prev_length != len(self.reviews)

    def postprocess(self):
        """Dataset-specific final step (override if needed)."""
        return

    def build(self):
        self.load_data()
        self.clean_data()
        self.postprocess()
        self.data = {"users": self.users, "items": self.items, "reviews": self.reviews}
        return self.data
