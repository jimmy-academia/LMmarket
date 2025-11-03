import logging
from statistics import mean

from utils import JSONCache, InternalCache
from .searchable import Searchable, ItemSearchable
from .helper import _decompose_aspect, _generate_aspect_info, _llm_score_batch

class BaseSystem:
    '''
    provides
    self.reviews [list of reviews (dict)]
    self.segments [list of segments (dict)]
    self.embedding (segments)
    '''
    def __init__(self, args, data):

        logging.info('[Base] initializing...')
        self.args = args
        self.reviews = Searchable(data['reviews'])
        self.items = ItemSearchable(data['items'], self.reviews)

        self.aspect_info_cache = JSONCache(args.output_dir/"query_aspect_infos.json")
        city_tag = "_"+ args.city.replace(" ", "_") if args.city else "full"
        self.result_cache = JSONCache(args.output_dir/f"{args.system}{city_tag}_query_result.json")

        self.review_cache = InternalCache(args.cache_dir/f"full_review")
        self.aspect_cache = InternalCache(args.cache_dir/f"{city_tag}_aspect")

    def _build_aspect_infos(self, query):
        aspect_list = _decompose_aspect(query)
        aspect_list = [t.strip() for t in aspect_list.split(',') if t.strip()]
        aspect_info = _generate_aspect_info(aspect_list, query)
        return aspect_info
    
    def recommend(self, query_list):
        for query in query_list:
            aspect_infos = self.aspect_info_cache.get_or_build(query, self._build_aspect_infos, query)

            finallist = self.result_cache.get_or_build(query, self.recommend_a_query, query, aspect_infos)

            logging.info(finallist)

    def recommend_a_query(self, query, aspect_infos):
        raise NotImplementedError

    def score(self, query, aspect_infos, candidates):
        scoreset = {}
        for candidate in candidates:
            all_scores = []
            for aspect_info in aspect_infos:
                dual_set_scores = self._score_a_case(query, aspect_info, candidate)
                all_scores.append(dual_set_scores)
            scoreset[candidate] = all_scores
        return scoreset

    def _score_a_case(self, query, aspect_info, candidate):
        scores = []
        aspect = aspect_info['aspect']
        retrieved = self.reviews.search(aspect, silent=True, item_id=candidate)

        review_id_list = []
        texts = []
        for obj in retrieved:
            review_id = obj['review_id']
            result = self.review_cache.get(review_id, f'{aspect}_score', False)
            if not result:
                review_id_list.append(review_id)
                texts.append(obj['text'])
            else:
                scores.append(result['score']) 
                
        results = _llm_score_batch(aspect, query, texts)
        for result, review_id in zip(results, review_id_list):
            self.review_cache.set(review_id, f'{aspect}_score', result)
            scores.append(result['score'])

        # reorg split positive, negative
        dual_set_scores = [[], []]
        for score in scores:
            if score > 0.5:
                dual_set_scores[0].append(score)
            elif score < 0.5:
                dual_set_scores[1].append(score)

        return dual_set_scores

    def rank(self, scoreset):
        logging.info('CAUSION: temporary ranking by average of positive set')
        full_list = []
        for candidate, all_scores in scoreset.items():
            avg_score = mean([mean(x[0]) if x[0] else 0 for x in all_scores]) 
            full_list.append((candidate, avg_score))
        rankedlist = [c for c, _ in sorted(full_list, key=lambda x: x[1], reverse=True)]

        return rankedlist



