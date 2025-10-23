import json
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from networks.aspect import infer_aspects_weights

from debug import check

class MainMethod(BaseSystem):
    def __init__(self, args, data):
            super().__init__(args, data)

    def recommend(self, query):

       
        logging.info('we are doing cafe')
        check()