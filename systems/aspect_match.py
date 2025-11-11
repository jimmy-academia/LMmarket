import json
import random
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from api import embed_many
from utils import loadp, dumpp
from networks.relevant_judge import _llm_judge_batch
from networks.tagger import extract_tags_batch
from .base import BaseSystem


class ASPECT_MATCH_Method(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.persist_path = args.cache_dir/'tag_emb.pkl'
        self.tag_emb = loadp(self.persist_path) if self.persist_path.exists() else {}
        

    def recommend_a_query(self, query, aspect_infos):
        logging.info(f'ASPECT_MATCH recommending query: {query}')

        self._tag_reviews()
        self._embed_tags()

        candidates = set(self.result_cache.get_or_build("candidate:"+query, self._find_candidates, query, aspect_infos))
        
        scoreset = self.score(query, aspect_infos, candidates)
        finallist = self.rank(scoreset)
        
        return finallist

    def _tag_reviews(self, batch_size=64, verbose=True):
        if verbose: pbar = tqdm(total=len(self.reviews), desc="tagging reviews", ncols=88)
        start_idx = 0
        while start_idx < len(self.reviews):
            batch_obj = []
            review_id_list = []
            for i, review in enumerate(self.reviews[start_idx:]):
                review_id = review['review_id']
                review_text = review["text"]
                tags = self.review_cache.get(review_id, "tags", False)
                if tags:
                    continue
                review_id_list.append(review_id)
                batch_obj.append(review_text)
                if len(batch_obj) >= batch_size:
                    break

            if batch_obj:
                tags_list = extract_tags_batch(batch_obj)
                for review_id, tags in zip(review_id_list, tags_list):
                    self.review_cache.set(review_id, "tags", tags)

            start_idx += i + 1
            if pbar: pbar.update(i+1)
            logging.info(review_text)
            logging.info(json.dumps(tags))

    def _embed_tags(self):
        tag_set = set()
        for review in tqdm(self.reviews, ncols=88, desc='collect tags'):
            review_id = review['review_id']
            tags = set(self.review_cache.get(review_id, "tags", set()))
            tag_set |= tags

        to_embed = [tag for tag in tag_set if tag not in self.tag_emb]
        if not to_embed:
            return 

        CHUNK = 1024
        for i in range(0, len(to_embed), CHUNK):
            batch = to_embed[i:i+CHUNK]
            emb_mat = embed_many(batch)        # shape (B, d) np.ndarray
            for tag, emb in zip(batch, emb_mat): # emb is 1-D np.ndarray
                self.tag_emb[tag] = emb
            dumpp(self.persist_path, self.tag_emb) 

    def _find_candidates(self, query, aspect_infos):
        positive_sets = []
        for aspect_info in aspect_infos:
            aspect = aspect_info['aspect']
            logging.info(f"[recommend_a_query]->{aspect}")
            positives = self.handle_one_aspect(aspect)
            logging.info(f"{aspect}, # positives={len(positives)}")
            positive_sets.append(positives)
        
        candidates = set.intersection(*positive_sets)
        logging.info(f"# final candidates={len(candidates)}")

        return list(candidates)

    def handle_one_aspect(aspect):
        # use top aspect, or up to K = 20 reviews
        if aspect not in self.tag_emb:
            self.tag_emb[aspect] = emb
            dumpp(self.persist_path, self.tag_emb)
        else:
            emb = self.tag_emb[aspect]

        # rank by embedding similarity
        rid2item_id = {}
        tag2reviews = {}
        for review in self.reviews:
            rid = review["review_id"]
            rid2item_id[rid] = review['item_id']
            tags = self.review_cache.get(rid, "tags", [])
            if not tags:
                continue
            for t in set(tags):
                tag2reviews.setdefault(t, []).append(rid)

        # tags that actually appear in reviews AND have embeddings
        tags = [t for t in tag2reviews.keys() if t in self.tag_emb]
        if not tags:
            logging.info(f"[handle_one_aspect] no tags available for aspect='{aspect}'")
            return set()

        # stack tag embeddings -> (T, d)
        M = np.vstack([self.tag_emb[t].astype(np.float32, copy=False) for t in tags])
        # defensive normalize rows (cheap no-op if already unit)
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        a = a / max(np.linalg.norm(a), 1e-12)

        sims = M @ a  # (T,)

        K = 100
        min_sim = 0.0
        positives = []
        # best score per review = max over its tags’ similarity
        best_by_review = {}  # rid -> (score, best_tag)
        for tag, sim in zip(tags, sims):
            if sim < min_sim:
                continue
            for rid in tag2reviews[tag]:
                positives.append(rid2item_id[rid])
        return set(positives)

        