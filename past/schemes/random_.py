import random
from .base import BaseScheme

class Random(BaseScheme):
    def __init__(self, args, task_loader):
        super().__init__(args, task_loader)

    def run_method(self, data):
        retrieved_iids = data['item_ids_rank'] + [str(i) for i in range(20)]
        ranked_iids = [iid for iid in random.sample(retrieved_iids, len(retrieved_iids))]
        return ranked_iids