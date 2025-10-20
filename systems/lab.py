import json
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from networks.aspect import infer_aspects_weights

class DevMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        experiment plan = """EXPERIMENTS for RETRIEVAL
        1. check how it is segmented: determine whether a segment really only has 1 aspect. does it need nearby segments for context?
        2. how to retrieve? compare using as query: aspect (keyword); aspect description; aspect definition; "find"-sentences; synonym expansion sentences
        3. what is retrieved? Checked if using similar keywords retrieve same results (when N is large). Check if non-overlaping results are important
        4. Embedding vector: what happen when shift emerging vector? How close are similar queries?
        """
        print(experiment plan)
    
    def recommend(self, query):
        pass

    def rr(self, *args, **kwargs):
        self.retrieve_similar_segments(*args, **kwargs)
