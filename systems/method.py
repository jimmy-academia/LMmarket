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

    def handle_one_aspect(self, aspect='high chair'):
        logging.info(f'we are doing {aspect}')
        search_results = self.reviews.search(aspect, topk=len(self.reviews), silent=True)

        i = 0
        batch_size = 5
        print(f"number of reviews: {len(self.reviews)}")
        while True:
            batch = search_results[i:i+batch_size]
            print(f'=== {i} to {i+batch_size} of {len(search_results)} ===')
            for result in batch:
                print(result['score'], result['snippet'].replace('\n', ''))
            user_input = input('jump to num: ')
            if user_input == '':
                i+= 100
            else:
                i = int(user_input)
            print()

