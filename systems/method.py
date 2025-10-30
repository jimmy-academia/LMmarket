import json
import random
import logging
from pathlib import Path

from .base import BaseSystem
from .helper import _llm_judge_batch

class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend_a_query(self, query, aspect_infos):

        positive_sets = []
        for aspect_info in aspect_infos:
            positives = self.handle_one_aspect(query, aspect_info)
            logging.info(f"{aspect}, # positives={len(positives)}")
            positive_sets.append(positives)

        candidateset = set.intersection(*positives)
        
        return candidateset

    def handle_one_aspect(self, query, aspect_info):

        aspect = aspect_info['aspect']
        aspect_type = aspect_info['aspect_type']

        # --- phase 1 --- collect review to process ---
        ## todo: more sophisticated collection loops
        collected_reviews = self._collect_reviews(aspect)
        positives = self._process_reviews(aspect, aspect_type, query, collected_reviews)
        return positives
        
        # --- phase 2 todo --- brute force check remaining items ---
        

    def _collect_reviews(self, aspect)
        retrieved = self.reviews.search(aspect, silent=True)
        collected = [[obj['review_id'], obj['score'], obj['text'], obj['snippet']] for obj in retrieved]
        collected = sorted(collected, key=lambda x: x[0], reverse=True)
        return collected

    def _process_reviews(self, aspect, aspect_type, query, collected_reviews):
        '''
        LM operation on review persistent by self.review_cache
        item_set persistent by self.aspect_cache
        '''
        seen_items = self.aspect_cache.get(aspect, 'seen_items', set())
        positives = self.aspect_cache.get(aspect, 'positives', set())
        batch_size = 20
        while collected_reviews:
            batch_obj = []
            judgment_list = []
            for i, obj in enumerate(to_review):
                review_id, item_id, text, snippet = obj 
                if item_id not in seen_items:
                    seen_items.add(item_id)
                    judgement = self.review_cache.get(review_id, f'{aspect}_judgement', False)
                    if not judgement:
                        batch_obj.append(obj)
                    else:
                        judgment_list.append(judgement)

            judgment_list += _llm_judge_batch(aspect, aspect_type, query, batch_obj)
            
            for judgement in judgment_list:
                if judgment["is_conclusive"]:
                    seen_items.add(item_id)
                    if judgment["is_positive"]:
                        positives.add(item_id)

        self.aspect_cache.set(aspect, 'seen_items', seen_items)
        self.aspect_cache.set(aspect, 'positives', positives)
        return positives

