from utils import JSONCache
from .helper import _decompose_aspect, _generate_aspect_info, 

class BaseSystem:
    '''
    provides
    self.reviews [list of reviews (dict)]
    self.segments [list of segments (dict)]
    self.embedding (segments)
    '''
    def __init__(self, args, data):
        self.args = args
        
        self.reviews = Searchable(data['reviews'])
        self.items = ItemSearchable(data['items'], self.reviews)

        self.aspect_info_cache = JSONCache(args.output_dir/"query_aspect_infos.json")
        self.candidate_cache = JSONCache(args.output_dir/f"{args.system}_query_candidates.json")
        self.score_cache = JSONCache(args.output_dir/f"query_item_scores.json")

    def _build_aspect_infos(self, query):
        aspect_list = _decompose_aspect(query)
        aspect_list = [t.strip() for t in aspect_list.split(',') if t.strip()]
        aspect_info = _generate_aspect_info(aspect_list, query)
        return aspect_info
    
    def recommend(self, query_list):
        for query in query_list:
            aspect_infos = self.aspect_info_cache.get_or_build(query, self._build_aspect_infos, query)

            candidates = self.candidate_cache.get_or_build(query, self.recommend_a_query, query, aspect_infos)

            scores = self.score_cache.get_or_build(query, self.examine)

    def recommend_a_query(self, query, aspect_infos)
        raise NotImplementedError

    def examine(self, candidate_list):
        for candidate in candidate_list:
            self._evaluate_one_candidate(candidate)

    def _evaluate_one_candidate(self, item_id):
        pass