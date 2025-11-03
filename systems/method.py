import json
import random
import logging
from pathlib import Path

from .base import BaseSystem
from networks.relevant_judge import _llm_judge_batch
from tqdm import tqdm

class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend_a_query(self, query, aspect_infos):
        
        positive_sets = []
        for aspect_info in aspect_infos:
            aspect = aspect_info['aspect']
            logging.info(f"[recommend_a_query]->{aspect}")
            positives = self.handle_one_aspect(query, aspect_info)
            logging.info(f"{aspect}, # positives={len(positives)}")
            positive_sets.append(positives)
        
        candidates = set.intersection(*positive_sets)
        logging.info(f"# final candidates={len(candidates)}")

        
        scoreset = self.score(query, aspect_infos, candidates)
        finallist = self.rank(scoreset)
        
        return finallist
            
    def handle_one_aspect(self, query, aspect_info):
        aspect = aspect_info['aspect']
        aspect_type = aspect_info['aspect_type']

        # --- phase 1 --- collect review to process ---
        ## todo: more sophisticated collection loops
        collected_reviews = self._collect_reviews(aspect)
        positives = self._identify_positives(aspect, aspect_type, query, collected_reviews)
        return positives
        
        # --- phase 2 todo --- brute force check remaining items ---

    def _collect_reviews(self, aspect):
        retrieved = self.reviews.search(aspect, silent=True)
        collected = [[obj['review_id'], obj['score'], obj['item_id'], obj['text'], obj['snippet']] for obj in retrieved]
        collected = sorted(collected, key=lambda x: x[1], reverse=True)
        return collected

    def _identify_positives(self, aspect, aspect_type, query, collected_reviews, batch_size=20, verbose=True):
        '''
        LM operation on review persistent by self.review_cache
        item_set persistent by self.aspect_cache
        '''
        concluded = self.aspect_cache.get(aspect, 'concluded', set())
        positives = self.aspect_cache.get(aspect, 'positives', set())
        concluded, positives = set(concluded), set(positives)
        
        def _apply(judgment, item_id):
            if judgment.get("is_conclusive"):
                concluded.add(item_id)
                if judgment.get("is_positive"):
                    positives.add(item_id)
        pbar = None
        if verbose: pbar = tqdm(total=len(collected_reviews), desc=f"{aspect}: judging", ncols=88)

        start_idx = 0
        while start_idx < len(collected_reviews):
            batch_obj = []

            for i, obj in enumerate(collected_reviews[start_idx:]):
                review_id, score, item_id, text, snippet = obj 

                if item_id in concluded or item_id in [x[1] for x in batch_obj]:
                    continue

                judgment = self.review_cache.get(review_id, f'{aspect}_judgment', False)
                if judgment:
                    _apply(judgment, item_id)
                    continue 

                batch_obj.append(obj)
                if len(batch_obj) >= batch_size:
                    break

            if batch_obj:
                judgment_list = _llm_judge_batch(aspect, aspect_type, query, batch_obj)
                for judgment, obj in zip(judgment_list, batch_obj):
                    review_id, score, item_id, text, snippet = obj 
                    self.review_cache.set(review_id, f'{aspect}_judgement', judgment)

                    _apply(judgment, item_id)
                    

            start_idx += i + 1
            if pbar: pbar.update(i+1)

        self.aspect_cache.set(aspect, 'concluded', list(concluded))
        self.aspect_cache.set(aspect, 'positives', list(positives))
        
        if pbar: pbar.close()
        return positives

