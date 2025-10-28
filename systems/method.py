import json
import random
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem

from network.helper import _decompose_aspect, _generate_aspect_info
from debug import check

class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend(self, query):

        aspect_list = _decompose_aspect(query)        
        aspect_list = aspect_list.split(',')

        self.item_aspect_status = {}
        for item in self.items:
            item_id = item['item_id']
            self.item_aspect_status[item_id] = {aspect:{} for aspect in aspect_list}
        self.review_aspect_labels = {}

        aspect_info_list = _generate_aspect_info(aspect_list, query)

        itemsets = {}
        for aspect_info in aspect_info_list:
            itemsets[aspect_info["aspect"]] = self.handle_one_aspect(aspect_info, query)

        scoreset = {}
        candidateset = set.intersection(*itemsets.values())
        for candidate in candidateset:
            scoreset[candidate] = self.score_one_candidate(candidate, aspect_list)

        rankedset = self.ranking(scoreset)

    def handle_one_aspect(self, aspect_info, query):
        """
        LLM-guided retrieval for one aspect.
        Returns:
            set(item_id) that have POSITIVE performance for `aspect`.
        Side effects:
            - Updates self.aspect_states[aspect]
            - Updates self.review_aspect_labels[(review_id, aspect)]
            - Updates self.item_aspect_status[item_id][aspect]
        """
        MAX_ROUNDS = 1
        GROWTH_PATIENCE = 2    # consecutive low-growth rounds allowed
        GROWTH_MIN = 0.02      # <2% new items => stagnation
        REMAINING_MIN = 0.1    # <10% items left => move to mode 2

        aspect = aspect_info['aspect']
        search_state = {
            "aspect": aspect, 
            "aspect_type": aspect_info['aspect_type'], # ontological, functional, teleological
            "mode": 1, 
            "round": 0,
            "keywords_que": aspect_info['starter_keywords']
            "keyword_stats": {}
            "items_total": len(self.items),
            "items_covered": 0,
        }

        # --- phase 1 --- obtain by search ---
        
        seen_items = set()
        positive_items = set()
        all_items = self.items.item_id_set
        patience = 0
        prev_items_covered = 0

        while (
            search_state["round"] < MAX_ROUNDS
            and patience < GROWTH_PATIENCE
            and (1 - len(seen_items) / search_state["items_total"]) > REMAINING_MIN
        ):
            search_state["round"] += 1
            last_count = len(seen_items)

            all_retrieved = {}
            for kw in search_state["keywords_que"]:
                retrieved = self.reviews.search(kw)
                retrieved = {obj for obj in retrieved if obj['item_id'] not in seen_items}
                all_retrieved[kw] = retrieved

            to_review = set.intersection(*all_retrieved.values())

            new_keywords = []
            while to_review:
                obj = to_review.pop()

                # Step 1. LLM review and judgment
                judgment = _llm_judge_item(aspect, query, obj['text'])

                # Step 2. Update state
                self.item_aspect_status[item_id][aspect] = judgment["aspect_status"]

                # Step 3. Remove redundant reviews if conclusive
                if judgment["is_conclusive"]:
                    item_id = obj['item_id']
                    to_review = {r for r in to_review if r["item_id"] != item_id}

                    if judgment["is_positive"]:
                        positive_items.add(item_id)
                    seen_items.add(item_id)

                # Step 4. collect keyword
                new_keywords.append(judgment['new_keywords'].split(','))

            search_state["keywords_que"] = new_keywords

            if (len(seen_items) - last_count)/last_count <= GROWTH_MIN:
                patience += 1

        # --- phase 2 --- brute force check remaining items ---
        remaining_items = all_items - seen_items
        BATCH_SIZE = 20   # don't overload the API
        for i in range(0, len(remaining_items), BATCH_SIZE):
            batch = remaining_items[i:i+BATCH_SIZE]
            item_texts = []
            for item in batch:
                reviews = [r for r in self.reviews if r['item_id'] == item['item_id']]
                # Optionally subsample or summarize reviews
                text = self._concat_reviews(reviews, limit=1200)  # helper to truncate long text
                item_texts.append({"item_id": item['item_id'], "text": text})

            judgments = _llm_batch_judge_items(aspect_info['aspect'], query, item_texts)

            for j in judgments:
                item_id = j['item_id']
                self.item_aspect_status[item_id][aspect_info['aspect']] = j['aspect_status']
                if j['is_conclusive'] and j['aspect_status'] == 'positive':
                    positive_items.add(item_id)

        return positive_items

    def score_one_candidate(self, candidate, aspect_list):
        pass

    def ranking(scoreset):
        pass
