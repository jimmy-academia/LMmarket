from sentence_transformers import SentenceTransformer
from .base import BaseScheme

class Embedding(BaseScheme):
    def __init__(self, args, task_loader):
        super().__init__(args, task_loader)

        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.instruction = "Represent this sentence for searching relevant passages:"

    def run_method(self, data):
        iids = data['item_ids_rank']

        query = self.instruction + data['request']
        passages = [self.task_loader['iid2desc'][iid] for iid in iids]

        q_embeddings = self.model.encode([query], normalize_embeddings=True)
        p_embeddings = self.model.encode(passages, normalize_embeddings=True)
        scores = (q_embeddings @ p_embeddings.T)[0]
        ranked_iid_score_pairs = sorted(zip(iids, scores), key=lambda x: x[1], reverse=True)

        ranked_iids = [iid for iid, _ in ranked_iid_score_pairs]
        return ranked_iids