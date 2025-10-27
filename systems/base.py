import re
import logging
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi

from .searchable import Searchable
from networks.slm import SmallLM

class ItemSearchable:
    def __init__(self, items, review_searchable):
        self.name = 'items'
        self.item_star_name = {item['raw_info']['business_id']: [item['raw_info']['stars'], item['raw_info']['name']] for item in items}
        self.reviews = review_searchable

    def search(self, query, topk=5, topm=10, review_k=None, agg="sum", silent=False): 
        # agg = 'sum' | 'mean' | 'max' over topm review scores
        
        if review_k is None:
            review_k = topk*20

        hits = self.reviews.search(query, topk=review_k, silent=True)
        
        item_buckets = defaultdict(list)
        for h in hits:
            item_buckets[h['review']['item_id']].append(h)
        
        results = []
        for iid, hlist in item_buckets.items():
            hlist.sort(key=lambda z: z["score"], reverse=True)
            if agg == "max":
                item_score = hlist[0]["score"]
            elif agg == "mean":
                item_score = sum(x["score"] for x in hlist) / len(hlist)
            else:  # sum
                item_score = sum(x["score"] for x in hlist)

            best = hlist[0]
            snippet = best.get("snippet") 

            rating, item_name = self.item_star_name[iid]
            results.append({
                "item_id": iid,
                "score": float(item_score),
                "rating": rating,
                "snippet": snippet,
                "hits": hlist,     # all review hits for this item (desc by score)
                "item_name": item_name
            })

        # 4) Rank and print
        results.sort(key=lambda r: r["score"], reverse=True)
        if not silent: print(f"\n🔍 {self.name} — top {topk} for '{query}':\n")
        for r in results[:topk]:
            rating_str = f"⭐ {r['rating']}"
            score_str  = f"📈:{r['score']:.2f}"
            prefix = f"{rating_str} | {score_str}" 
            if not silent: print(f"{prefix} | name: {r["item_name"]} | {r['snippet']}")

        return results[:topk]

class BaseSystem:
    '''
    provides
    self.reviews [list of reviews (dict)]
    self.segments [list of segments (dict)]
    self.embedding (segments)
    '''
    def __init__(self, args, data):
        self.args = args
        
        self.reviews = Searchable(data['reviews'])
        self.items = ItemSearchable(data['items'], self.reviews)
        
        # self.model = SmallLM(args.model_name, args.device)

        

