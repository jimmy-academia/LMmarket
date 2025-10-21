import json
import random
import json
import logging
import numpy as np

from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from networks.aspect import infer_aspects_weights

from debug import check

class DevMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        logging.info("[dev] starting retrieval experiments")

        # self.examine_segments()
        # self.query_variants_overlap()
        # self.query_variants_overlap(False)

        print('todo: check segment cluster result!')


    def examine_segments(self, sample_reviews=5):
        for rid in random.sample(list(self.review_segments.keys()), sample_reviews):
            print()
            print()
            print(rid)
            print(self.id2reviews[rid]['text'])
            print('-----segments----')
            for segment in self.review_segments[rid]:
                print(segment['text'])

    def query_variants_overlap(self, check_type=True):

        if check_type:
            query = "Find a quiet, cozy cafe with comfortable seating and good natural light that's perfect for reading a book for a few hours."
            raw = load_or_build(self.args.cache_dir/'temp_aspect_dict.json', dumpj, loadj, infer_aspects_weights, query)
            aspect_dict = raw if isinstance(raw, dict) else json.loads(raw)

            aspects = aspect_dict.get('aspects')
            variants_output = {}
            aspect = aspects[0]

            name = aspect.get('name')
            sentence = aspect.get('sentence')
            positives = aspect.get('positives')
            queries = {
                'keyword': name.replace('_', ' ') if name else None,
                'description': sentence,
                'definition': f"{name} focuses on {sentence}" if name and sentence else None,
                'find_sentence': f"Find segments describing: {sentence}" if sentence else None,
                'synonyms': ' '.join(positives) if positives else None,
            }
        else:
            queries = {
                1: 'quiet', 2: 'silent', 3: 'not loud', 4: 'peaceful', 5: 'Tranquil'
            }
        print('============\n'*3)    

        aspect_runs = {}
        for __, text in queries.items():
            
            scores, idxs = self._collect_top_segments(text, 1000)
            results = set([self.segments[idx].get('text') for idx in idxs])

            aspect_runs[text] = results
        
        percentages = {'int': [], 'sep': []}

        for k in list(range(100, 1100, 100)):
            variant_sets = {
                v: set(list(results)[:k]) for v, results in aspect_runs.items()
            }

            union_set = set.union(*variant_sets.values())
            intersection_set = set.intersection(*variant_sets.values())
            length = len(union_set)
            percentages['int'].append(f'{len(intersection_set)/length:.2f}')
            percentages['sep'].append(f'{k/length:.2f}')
        
        for key, values in percentages.items():
            print(key)
            print(values)
        
        print('============\n'*3)    
        for key, values in variant_sets.items():
            print()
            print(">>>", key)
            for value in list(values)[:20]:
                print(value)

    def _collect_top_segments(self, text, topk):
        query_vec = self._encode_query(text)
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query_vec)
        if norm:
            query_vec /= norm
        scores, idxs = self._get_top_k(query_vec, min(topk, len(self.segments)))
        return scores, idxs

    def recommend(self, query):
        logging.info('[dev] recommend placeholder :: %s', query)

    def rr(self, *args, **kwargs):
        self.retrieve_similar_segments(*args, **kwargs)
