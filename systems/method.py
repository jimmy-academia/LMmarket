import json
import random
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from pathlib import Path
from networks.helper import _decompose_aspect, _generate_aspect_info, _llm_judge_item

# self.item_aspect_status = {}
#         for item in self.items:
#             item_id = item['item_id']
#             self.item_aspect_status[item_id] = {aspect:{} for aspect in aspect_list}
#         self.review_aspect_labels = {}


class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

    def recommend(self, query):

        aspect_list = load_or_build('cache/tmp/aspect_list.json', dumpj, loadj, _decompose_aspect, query)
        aspect_list = [aspect_term.strip() for aspect_term in aspect_list.split(',')]
        aspect_info_list = load_or_build('cache/tmp/aspect_info_list.json', dumpj, loadj, _generate_aspect_info, aspect_list, query)

        itemsets_path = Path('cache/tmp/itemsets.json')
        if itemsets_path.exists():
            itemsets = loadj(itemsets_path)
        else:
            itemsets = {}
            for aspect_info in aspect_info_list:
                positive_set = self.handle_one_aspect(aspect_info, query)
                itemsets[aspect_info["aspect"]] = positive_set
                logging.info(f"{aspect}, num positive = {len(positive_set)}")
            dumpj(itemsets_path, itemsets)

        scoreset = {}
        candidateset = set.intersection(*itemsets.values())
        logging.info(f"{aspect}, num candidate = {len(candidateset)}")
        from debug import check
        check()
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
        aspect = aspect_info['aspect']
        aspect_type = aspect_info['aspect_type']
        MAX_ROUNDS = 1
        GROWTH_PATIENCE = 2    # consecutive low-growth rounds allowed
        GROWTH_MIN = 0.02      # <2% new items => stagnation
        REMAINING_MIN = 0.1    # <10% items left => move to mode 2

        # --- phase 1 --- obtain by search ---
        print('phase 1 start')

        seen_items = set()
        positive_items = set()
        all_items = self.items.item_id_set
        
        current_round = 0
        patience = 0
        prev_items_covered = 0
        keywords_que = aspect_info["starter_keywords"]

        while (
            current_round < MAX_ROUNDS
            and patience < GROWTH_PATIENCE
            and (1 - len(seen_items) / len(all_items)) > REMAINING_MIN
            and keywords_que
        ):
            current_round += 1
            last_count = len(seen_items) + 1e-5

            to_review = {}
            kw = keywords_que.pop(0)
            retrieved = self.reviews.search(kw, silent=True)
            to_review = [[obj['review_id'], obj['score'], obj['text'], obj['snippet']] for obj in retrieved]
            to_review = sorted(to_review, key=lambda x: x[0], reverse=True)

            while to_review:
                logging.info('='*20)
                logging.info(len(to_review), len(self.reviews), len(seen_items), len(all_items))
                logging.info('='*20)
                obj = to_review.pop()
                review_id, item_id, text, snippet = obj

                # Step 1. LLM review and judgment
                judgment = _llm_judge_item(aspect, aspect_type, query, text, snippet)
                logging.info(text)
                logging.info(snippet)
                logging.info(judgment)

                # Step 2. Remove redundant reviews if conclusive
                if judgment["is_conclusive"]:
                    seen_items.add(item_id)
                    to_review = [r for r in to_review if r[1] != item_id]

                    if judgment["is_positive"]:
                        positive_items.add(item_id)

            if (len(seen_items) - last_count)/last_count <= GROWTH_MIN:
                patience += 1

        # # --- phase 2 --- brute force check remaining items ---
        # remaining_items = all_items - seen_items
        # BATCH_SIZE = 20   # don't overload the API
        # for i in range(0, len(remaining_items), BATCH_SIZE):
        #     batch = remaining_items[i:i+BATCH_SIZE]
        #     item_texts = []
        #     for item in batch:
        #         reviews = [r for r in self.reviews if r['item_id'] == item['item_id']]
        #         # Optionally subsample or summarize reviews
        #         text = self._concat_reviews(reviews, limit=1200)  # helper to truncate long text
        #         item_texts.append({"item_id": item['item_id'], "text": text})

        #     judgments = _llm_batch_judge_items(aspect_info['aspect'], query, item_texts)

        #     for j in judgments:
        #         item_id = j['item_id']
        #         self.item_aspect_status[item_id][aspect_info['aspect']] = j['aspect_status']
        #         if j['is_conclusive'] and j['aspect_status'] == 'positive':
        #             positive_items.add(item_id)

        return positive_items

    def score_one_candidate(self, candidate, aspect_list):
        pass

    def ranking(self, scoreset):
        pass
