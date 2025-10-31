from utils import JSONCache, InternalCache
from .searchable import Searchable, ItemSearchable
from .helper import _decompose_aspect, _generate_aspect_info

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
        city_tag = "_"+ args.city if args.city else ""
        self.candidate_cache = JSONCache(args.output_dir/f"{args.system}{city_tag}_query_candidates.json")

        self.review_cache = InternalCache(args.cache_dir/f"{args.system}{city_tag}_review")
        self.aspect_cache = InternalCache(args.cache_dir/f"aspect_{args.system}{city_tag}")

    def _build_aspect_infos(self, query):
        aspect_list = _decompose_aspect(query)
        aspect_list = [t.strip() for t in aspect_list.split(',') if t.strip()]
        aspect_info = _generate_aspect_info(aspect_list, query)
        return aspect_info
    
    def recommend(self, query_list):
        for query in query_list:
            aspect_infos = self.aspect_info_cache.get_or_build(query, self._build_aspect_infos, query)

            candidates = self.candidate_cache.get_or_build(query, self.recommend_a_query, query, aspect_infos)

            self.score(query, aspect_infos, candidates)

    def recommend_a_query(self, query, aspect_infos):
        raise NotImplementedError

    def score(self, query, aspect_infos, candidates):
        for candidate in candidate_list:
            for aspect_info in aspect_infos:
                self._score_a_case(query, aspect_info, candidate)

    def _score_a_case(self, query, aspect_info, candidate):
        pass






