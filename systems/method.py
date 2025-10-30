import json
import random
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from pathlib import Path
from networks.helper import _decompose_aspect, _generate_aspect_info, _llm_judge_batch

# self.item_aspect_status = {}
#         for item in self.items:
#             item_id = item['item_id']
#             self.item_aspect_status[item_id] = {aspect:{} for aspect in aspect_list}
#         self.review_aspect_labels = {}


class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend(self, query):

        self.query = query
        aspect_list = load_or_build('cache/tmp/aspect_list.json', dumpj, loadj, _decompose_aspect, query)
        aspect_list = [aspect_term.strip() for aspect_term in aspect_list.split(',')]
        aspect_info_list = load_or_build('cache/tmp/aspect_info_list.json', dumpj, loadj, _generate_aspect_info, aspect_list, query)

        itemsets_path = Path('cache/tmp/itemsets.json')
        if itemsets_path.exists():
            itemsets = loadj(itemsets_path)
        else:
            itemsets = {}
            for aspect_info in aspect_info_list:
                aspect = aspect_info["aspect"]
                positives = self.handle_one_aspect(aspect_info)
                itemsets[aspect] = positives
                logging.info(f"{aspect}, # positives={len(positives)}")
            dumpj(itemsets_path, itemsets)

        scoreset = {}
        candidateset = set.intersection(*itemsets.values())
        logging.info(f"{aspect}, # candidates={len(candidateset)}")
        from debug import check
        check()
        for candidate in candidateset:
            scoreset[candidate] = self.score_one_candidate(candidate, aspect_list)

        rankedset = self.ranking(scoreset)

    def handle_one_aspect(self, aspect_info):
        """
        LLM-guided retrieval for one aspect.
        Returns:
            set(item_id) that have POSITIVE performance for `aspect`.
        Side effects:
            - Updates self.aspect_states[aspect]
            - Updates self.review_aspect_labels[(review_id, aspect)]
            - Updates self.item_aspect_status[item_id][aspect]
        """
        self.aspect = aspect_info['aspect']
        self.aspect_type = aspect_info['aspect_type']
        logging.info(f'aspect: {self.aspect} type: {self.aspect_type}')

        # --- phase 1 --- obtain by search ---
        to_review = {}
        kw = self.aspect
        retrieved = self.reviews.search(kw, silent=True)
        to_review = [[obj['review_id'], obj['score'], obj['text'], obj['snippet']] for obj in retrieved]
        to_review = sorted(to_review, key=lambda x: x[0], reverse=True)

        positives = self.process_to_reviews(to_review)
        
        # --- phase 2 todo --- brute force check remaining items ---
        
    def process_to_reviews(self, to_review):
        seen_items = set()
        positive_items = set()
        batch_size = 20
        while to_review:
            logging.info('='*20)
            logging.info(f"{len(to_review)}, {len(self.reviews)}, {len(seen_items)}, {len(self.items)}")
            logging.info('='*20)

            item_ids = []
            batch_obj = []
            for i, obj in enumerate(to_review): 
                if obj[1] not in item_ids:
                    item_ids.append(obj[1])
                    batch_obj.append(to_review.pop(i))

            # review_id, item_id, text, snippet = obj
            # Step 1. LLM review and judgment
            judgment_list = _llm_judge_batch(self.aspect, self.aspect_type, self.query, batch_obj)
            
            for judgement in judgment_list:
                if judgment["is_conclusive"]:
                    seen_items.add(item_id)
                    to_review = [r for r in to_review if r[1] != item_id]
                    if judgment["is_positive"]:
                        positive_items.add(item_id)

        return positive_items

    def score_one_candidate(self, candidate, aspect_list):
        pass

    def ranking(self, scoreset):
        pass
