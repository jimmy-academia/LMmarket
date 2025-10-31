import json
import random
import logging
from pathlib import Path

from .base import BaseSystem
from .helper import _llm_judge_batch
from tqdm import tqdm

class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend_a_query(self, query, aspect_infos):

        positive_sets = []
        for aspect_info in aspect_infos:
            logging.info(f"[recommend_a_query]->{aspect_info['aspect']}")
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

    def _collect_reviews(self, aspect):
        retrieved = self.reviews.search(aspect, silent=True)
        collected = [[obj['review_id'], obj['score'], obj['text'], obj['snippet']] for obj in retrieved]
        collected = sorted(collected, key=lambda x: x[0], reverse=True)
        return collected

    def _process_reviews(self, aspect, aspect_type, query, collected_reviews, batch_size=20, verbose=True):
        '''
        LM operation on review persistent by self.review_cache
        item_set persistent by self.aspect_cache
        '''
        concluded = self.aspect_cache.get(aspect, 'concluded', set())
        positives = self.aspect_cache.get(aspect, 'positives', set())
        
        pbar = None
        if verbose: pbar = tqdm(total=len(collected_reviews), desc=f"{aspect}: judging", ncols=88)

        start_idx = 0
        while start_idx < len(collected_reviews):
            batch_obj = []

            for i, obj in enumerate(tqdm(collected_reviews[start_idx:], ncols=88, desc="collecting...", leave=False)):
                review_id, item_id, text, snippet = obj 

                if item_id in concluded or item_id in [x[1] for x in batch_obj]:
                    continue

                if self.review_cache.get(review_id, f'{aspect}_judgement', False):
                    continue                    
                
                batch_obj.append(obj)
                if len(batch_obj) >= batch_size:
                    break

            if batch_obj:
                judgment_list = _llm_judge_batch(aspect, aspect_type, query, batch_obj)
                for judgment, obj in zip(judgment_list, batch_obj):
                    review_id, item_id, text, snippet = obj 
                    self.review_cache.set(review_id, f'{aspect}_judgement', judgment)

                    if judgment["is_conclusive"]:
                        concluded.add(item_id)
                        if judgment["is_positive"]:
                            positives.add(item_id)

            start_idx += i + 1
            if pbar: pbar.update(i+1)

        self.aspect_cache.set(aspect, 'concluded', concluded)
        self.aspect_cache.set(aspect, 'positives', positives)
        
        if pbar: pbar.close()
        return positives

