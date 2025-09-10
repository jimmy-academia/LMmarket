import json
import numpy as np
from datasets import Dataset

class BaseScheme():
    def __init__(self, args):
        self.args = args
        self.task_loader = None

    def get_task_loader(self, args):
        if args.dset == "yelp":
            with open("cache/benchmark.json", "r") as fp:
                benchmark = json.load(fp)
            self.task_loader = benchmark
        else:
            raise NotImplementedError(f"prompt scheme {args.scheme} not implemented")
        
    def evaluate_ranking(self, labels, preds, k):
        """
        Evaluate ranking quality using multiple metrics at top-k.

        Args:
            labels (list): ground truth relevant item IDs
            preds (list): ranked predicted item IDs
            k (int): top-k threshold

        Returns:
            dict: {hit_rate, precision, recall, f1, ndcg}
        """
        labels_set = set(labels)
        if k <= 0:
            print(f"evaluate_ranking: k({k}) is invalid, return zero scores")
            return {'hit_rate': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'ndcg': 0}
        elif k > len(preds):
            print(f"evaluate_ranking: k({k}) is set to len(preds) = {len(preds)}")
        topk_preds = preds[:k]
        hits = [1 if p in labels_set else 0 for p in topk_preds]
        num_relevant = len(labels_set)
        num_hit = sum(hits)

        # Hit Rate@k
        hit_rate = 1 if num_hit > 0 else 0

        # Precision@k
        precision = num_hit / k

        # Recall@k
        recall = num_hit / num_relevant if num_relevant > 0 else 0

        # F1@k
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # nDCG@k
        dcg = sum([1 / np.log2(i + 2) for i, p in enumerate(topk_preds) if p in labels_set])
        ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(num_relevant, k))])
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        return {
            'hit_rate': hit_rate,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg
        }

    def operate(self, args, k):
        self.get_task_loader(args)
        res = []
        for data in self.task_loader:
            labels = data['item_ids_rank']
            preds = self.run_method(data['user_id'], data['item_ids_rank'])
            res.append(self.evaluate_ranking(labels, preds, k))
    
        res = Dataset.from_list(res)
        length = len(self.task_loader)
        res_str = f"""{self.__class__.__name__} Preformance (top-{k})
- hit_rate: {(sum(res['hit_rate']) / length):.4f}
- precision: {(sum(res['precision']) / length):.4f}
- recall: {(sum(res['recall']) / length):.4f}
- f1: {(sum(res['f1']) / length):.4f}
- ndcg: {(sum(res['ndcg']) / length):.4f}
"""
        print(res_str)
        