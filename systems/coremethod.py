import json
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from networks.aspect import infer_aspects_weights

from network.sentiment import LLM_ASPECT_LABEL_PROMPT, LLM_ASPECT_LABEL_SCHEMA

class MainMethod(BaseSystem):
    def __init__(self, args, data):
            super().__init__(args, data)

    def recommend(self, query):

        gameplan = """
        [o] 1. aspect split
        [.] 2. retrieve by aspect
                - (over load => filter => score)
                - reasonable: no significant misses
                //////////// IDEAS ///////////////////
                /// 1a LLM produce, reuse aspect pool => definition, lexical hints //exclusion items?
                /// 1b aspect query vector
                /// 1c iterative threshold adjustment with LLM, save to aspect pool
                //////////////////////////////////////
        [ ] 3. score aspects
                //////////// IDEAS ///////////////////////////
                /// find the axis for positive to negative?///
                //////////////////////////////////////////////
                /// cluster and sample //////
                /// LM generate and match ///
                /////////////////////////////
        [ ] 4. combine into hesitant fuzzy set
                stable and reproduceable process?
        [ ] 5. rank by hasitant set operation
        """
        print(gameplan)
        
        raw = load_or_build(self.args.cache_dir/'temp_aspect_dict.json', dumpj, loadj, infer_aspects_weights, query)
        self.aspect_dict = raw if isinstance(raw, dict) else json.loads(raw)
        sum_weights = sum([float(a["weight"]) for a in self.aspect_dict["aspects"]])
        for a in self.aspect_dict["aspects"]:
            a["weight"] = float(a.get("weight", 0)) / sum_weights
        logging.info("[aspects] %s", json.dumps(self.aspect_dict, indent=2))
        
        # print(self.embedding.shape) (1492456, 384)

        for aspect in self.aspect_dict['aspects']:
            # self.work_one_aspect(aspect)        

