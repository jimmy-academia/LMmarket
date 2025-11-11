import json
import random
import logging
from pathlib import Path

from .base import BaseSystem
from networks.relevant_judge import _llm_judge_batch
from networks.tagger import extract_tags_batch
from tqdm import tqdm

class ASPECT_MATCH_Method(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend_a_query(self, query, aspect_infos):
        logging.info(f'ASPECT_MATCH recommending query: {query}')

        self.tag_reviews()

        candidates = set(self.result_cache.get_or_build("candidate:"+query, self._find_candidates, query, aspect_infos))
        
        scoreset = self.score(query, aspect_infos, candidates)
        finallist = self.rank(scoreset)
        
        return finallist

    def tag_reviews(self, batch_size=64, verbose=True):
        if verbose: pbar = tqdm(total=len(self.reviews), desc="tagging reviews", ncols=88)
        start_idx = 0
        while start_idx < len(self.reviews):
            batch_obj = []
            review_id_list = []
            for i, review in enumerate(self.reviews[start_idx:]):
                review_id = review['review_id']
                review_text = review["text"]
                tags = self.review_cache.get(review_id, "tags", False)
                if tags:
                    continue
                review_id_list.append(review_id)
                batch_obj.append(review_text)
                if len(batch_obj) >= batch_size:
                    break

            if batch_obj:
                tags_list = extract_tags_batch(batch_obj)
                for review_id, tags in zip(review_id_list, tags_list):
                    self.review_cache.set(review_id, "tags", tags)

            start_idx += i + 1
            if pbar: pbar.update(i+1)
            logging.info(review_text)
            logging.info(json.dumps(tags))


    def _find_candidates(self, query, aspect_infos):
        input('use aspect match')
        positive_sets = []
        for aspect_info in aspect_infos:
            aspect = aspect_info['aspect']
            logging.info(f"[recommend_a_query]->{aspect}")
            positives = self.handle_one_aspect(query, aspect_info)
            logging.info(f"{aspect}, # positives={len(positives)}")
            positive_sets.append(positives)
        
        candidates = set.intersection(*positive_sets)
        logging.info(f"# final candidates={len(candidates)}")

        return list(candidates)


    