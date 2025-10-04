# data/yelp.py
from utils import _iter_line
from .hub import DataHub
from .geo import *

def prepare_yelp_data(dset_root):
    return YelpData(dset_root).build()

class YelpData(DataHub):
    def __init__(self, dset_root):
        super().__init__(dset_root)
        yroot = self.dset_root / 'yelp'
        self.business_file = yroot/"yelp_academic_dataset_business.json" # 150346
        self.review_file = yroot/"yelp_academic_dataset_review.json" # 6990280
        self.tip_file = yroot/"yelp_academic_dataset_tip.json" # 908915
        self.user_file = yroot/"yelp_academic_dataset_user.json" # 1987897


    def load_data(self):
        self._load_restaurants() # complete self.items
        self._load_users() # complete self.users
        self._load_reviews_and_tips() # complete self.reviews

    def _load_restaurants(self):
        self.items = {}
        for line in _iter_line(self.business_file, total=150_346):
            biz = json.loads(line)
            if not categories or "restaurant" not in categories.lower():
                continue
            item_id = biz["business_id"]
            self.items[item_id] = {"raw_info": biz}

    def _load_users(self):
        self.users = {}
        for line in _iter_line(self.user_file, total = 1_987_897):
            user = json.loads(line)
            user_id = user['user_id']
            self.users[user_id] = {"raw_info": user}

    def _load_reviews_and_tips(self):
        self.reviews = {}

        for filepath, kind, total in [
            (review_file, "review", 6_990_280), 
            (tip_file, "tip", 908_915)
        ]:
            tip_id = 0
            for line in _iter_line(filepath, total=total):
                obj = json.loads(line)
                item_id = obj["business_id"]
                user_id = obj["user_id"]

                review_id = obj["review_id"] if kind == 'review' else f"tip_{tip_id}" 
                tip_id += int(kind == 'tip')
                
                self.reviews[review_id] = {"item_id": item_id, "user_id": user_id, "raw_info": obj}

    def postprocess(self, shrink_raw=True):
        
        # items
        for iid, iobj in self.items.items():
            raw_obj = iobj.get("raw_info")
            iobj["city"] = raw_obj["city"].strip()
            iobj["latitude"] = float(raw_obj["latitude"])
            iobj["longitude"] = float(raw_obj["longitude"])

            if shrink_raw:
                for k in ("city", "latitude", "longitude"):
                    raw_obj.pop(k, None)
        # users
        for uid, uobj in self.users.items():
            raw_obj = uobj.get("raw_info")
            uobj["friends"] = [f for f in raw_obj["friends"] if f in self.users]
            
            if shrink_raw:
                raw_obj.pop("friends", None)

            # user city; user center_lat, center_lon, alpha_per_km

        # reviews
        for rid, robj in self.reviews.items():
            raw_obj = robj.get("raw_info")
            robj["text"] = raw_obj["text"]
            robj["stars"] = raw_obj.get("stars", "")
            robj["datetime"] = raw_obj["date"]
            if shrink_raw:
                for k in ("text", "stars", "date"):
                    raw_obj.pop(k, None)

        # self.reviews["item2reviews"] = ...
        # self.reviews["user2reviews"] = ...


