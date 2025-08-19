import numpy as np
from datasets import Dataset
from scipy.stats import spearmanr
from tqdm import tqdm

class BaseScheme():
    def __init__(self, args, task_loader):
        self.dset = args.dset
        self.top_m = args.top_m
        self.top_n = args.top_n
        self.task_loader = task_loader
        
    def evaluate_ranking(self, labels, preds):
        """
        Evaluate ranking quality using multiple metrics at top-k.

        Args:
            labels (list): ground truth relevant item IDs (ranked)
            preds (list): ranked predicted item IDs

        Returns:
            dict: {hit_rate, precision, recall, f1, ndcg, map, spearman}
        """
        top_labels = labels[:self.top_m]
        top_preds = preds[:self.top_n]

        labels_set = set(top_labels)
        hits = [1 if p in labels_set else 0 for p in top_preds]
        num_relevant = len(labels_set)
        num_hit = sum(hits)

        # Hit Rate@k
        hit_rate = 1 if num_hit > 0 else 0

        # Precision@k
        precision = num_hit / self.top_n

        # Recall@k
        recall = num_hit / num_relevant if num_relevant > 0 else 0

        # F1@k
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # nDCG@k
        dcg = sum([1 / np.log2(i + 2) for i, p in enumerate(top_preds) if p in labels_set])
        ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(num_relevant, self.top_n))])
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        # MAP@k
        avg_precision = 0
        hit_count = 0
        for i, p in enumerate(top_preds):
            if p in labels_set:
                hit_count += 1
                avg_precision += hit_count / (i + 1)
        map_score = avg_precision / num_relevant if num_relevant > 0 else 0

        # Spearman's rank correlation
        overlap_items = [top_label for top_label in top_labels if top_label in top_preds]
        if len(overlap_items) >= 2:
            # Get ranks in label and pred
            label_ranks = [top_labels.index(item) for item in overlap_items]
            pred_ranks = [top_preds.index(item) for item in overlap_items]
            spearman_corr, _ = spearmanr(label_ranks, pred_ranks)
        else:
            spearman_corr = 0.0  # undefined or low overlap

        return {
            'hit_rate': hit_rate,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg,
            'map': map_score,
            'spearman': spearman_corr
        }

    def operate(self):
        label_len = len(self.task_loader['data'][0]['item_ids_rank'])
        if self.top_m <= 0 or self.top_m > label_len:
            raise IndexError(f"Ground truth item count top_m({self.top_m}) is invalid, enter: [1, {label_len}]")
        if self.top_n <= 0 or self.top_n > label_len:
            raise IndexError(f"Retrieval item count top_n({self.top_n}) is invalid, enter: [1, {label_len}]")

        res = []
        for data in tqdm(self.task_loader['data'], desc="evalution progress"):
            labels = data['item_ids_rank']
            preds = self.run_method(data)
            res.append(self.evaluate_ranking(labels, preds))
    
        res = Dataset.from_list(res)
        length = len(self.task_loader['data'])
        res_str = f"""{self.__class__.__name__}
- hit_rate: {(sum(res['hit_rate']) / length):.4f}
- precision: {(sum(res['precision']) / length):.4f}
- recall: {(sum(res['recall']) / length):.4f}
- f1: {(sum(res['f1']) / length):.4f}
- ndcg: {(sum(res['ndcg']) / length):.4f}
- map: {(sum(res['map']) / length):.4f}
- spearman: {(sum(res['spearman']) / length):.4f}
"""
        print(res_str)
        