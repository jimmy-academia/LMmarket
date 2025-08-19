from rank_bm25 import BM25Okapi
from .base import BaseScheme

class BM25(BaseScheme):
    def __init__(self, args, task_loader):
        super().__init__(args, task_loader)

    def run_method(self, data):
        iids = data['item_ids_rank']

        query = data['request']
        passages = [self.task_loader['iid2desc'][iid] for iid in iids]

        tokenized_docs = [psg.lower().split() for psg in passages]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query.lower().split())
        ranked_iid_score_pairs = sorted(zip(iids, scores), key=lambda x: x[1], reverse=True)

        ranked_iids = [iid for iid, _ in ranked_iid_score_pairs]
        return ranked_iids