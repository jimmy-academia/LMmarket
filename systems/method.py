import json
import random
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from api import run_llm_batch_api, query_llm, user_struct, system_struct, assistant_struct

# from networks.revagent import ReviewAgent
from debug import check

class MainMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)


    def recommend(self, query):
        aspect_list = self.decompose_query(query)
        check()
        aspect_list = aspect_list.split(',')

        for aspect in aspect_list:
            self.handle_one_aspect(aspect, query)
            input()
        # self.agent = ReviewAgent()

    def decompose_query(self, query):
        messages = [
            system_struct(
                "You are a text decomposition assistant. "
                "Your task is to extract the key aspects or attributes from a natural language query. "
                "Output them as a concise, comma-separated list of short phrases. "
                "Do not include filler words, conjunctions, or duplicates. "
                "Keep the phrasing natural but minimal."
                "If multiple ideas are combined split them into separate aspects. "
            ),
            user_struct("Find a romantic Italian restaurant with candlelight and outdoor seating, perfect for a date night."),
            assistant_struct("romantic, Italian restaurant, has candlelight, has outdoor seating, perfect for a date night"),
            user_struct("Looking for a family-friendly restaurant that serves vegetarian food and has a play area for kids."),
            assistant_struct("family-friendly, has vegetarian food, has play area for kids"),
            user_struct("Recommend a sushi spot with fast service and reasonable prices"),
            assistant_struct("sushi place, fast service, reasonable prices"),
            user_struct(query)
        ]
        output = query_llm(messages)
        return output

    def handle_one_aspect(self, aspect, query):
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

