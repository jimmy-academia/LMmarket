import random
from .base import BaseScheme

class Random(BaseScheme):
    def __init__(self, args):
        super().__init__(args)

    def run_method(self, uid, iids):
        retrieved_iids = iids + [str(i) for i in range(20)]
        ranked_iids = [iid for iid in random.sample(retrieved_iids, len(retrieved_iids))]
        return ranked_iids