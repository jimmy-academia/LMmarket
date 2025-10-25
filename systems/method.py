import json
import random
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem

# from networks.revagent import ReviewAgent
from debug import check

class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.handle_one_aspect()
        # self.agent = ReviewAgent()

    def handle_one_aspect(self, aspect='cafe'):
        logging.info('we are doing cafe')
        search_result = self.reviews.search(aspect, topk=len(self.reviews), silent=True)

        N = len(search_result)
        K = 100
        k = 10
        seen_items = set()
        results_by_item = {}   # item_id -> examine result
        j = 0   # forward scan pointer

        print(N, len(self.reviews))
        for i in range(1, 10):
            print(N-i, search_result[-i])
            input()

        input('pause')
        # while True:
        #     batch = []
        #     while len(batch) < K and j < N:
        #         r = reviews[j]; j += 1
        #         if r['item_id'] in seen_items:
        #             continue
        #         batch.append(r)

        #     sample_batch = random.sample(batch, min(k, len(batch)))
        #     text_batch = [r['text'] for r in sample_batch]

        #     examine_result = self.agent.examine(text_batch)

        #     batch_items = set()
        #     for r, res in zip(sample_batch, examine_result):
        #         item_id = r['item_id']
        #         batch_items.add(item_id)
        #         results_by_item[item_id] = res

        #     seen_items.update(batch_items)
