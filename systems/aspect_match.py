import json
import random
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from api import embed_one, embed_many
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
        self.tag_set = set()
        self.tag2reviews = defaultdict(list)
        self.rid2item_id = {}
        for review in tqdm(self.reviews, ncols=88, desc='collect tags'):
            review_id = review['review_id']
            item_id = review['item_id']
            self.rid2item_id[review_id] = item_id
            features = self.review_cache.get(review_id, "tags", {}).get("tags")
            if features:
                tags = set([feat['name'] for feat in features])
                for tag in tags:
                    self.tag2reviews[tag].append(review_id)
                self.tag_set |= tags

        to_embed = [tag for tag in self.tag_set if tag not in self.tag_emb]
        if not to_embed:
            return 

        CHUNK = 1024
        for i in tqdm(range(0, len(to_embed), CHUNK), ncols=88, desc='batch embed tags'):
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

        input()

        return list(candidates)

    def handle_one_aspect(self, aspect):
        # use top aspect, or up to K = 20 reviews
        if aspect not in self.tag_emb:
            self.tag_emb[aspect] = embed_one(aspect)
            dumpp(self.persist_path, self.tag_emb)

        a = self.tag_emb[aspect]
        # tags that actually appear in reviews AND have embeddings
        tags = [t for t in self.tag2reviews.keys() if t in self.tag_emb]
        if not tags:
            logging.info(f"[handle_one_aspect] no tags available for aspect='{aspect}'")
            return set()

        # stack tag embeddings -> (T, d)
        M = np.vstack([self.tag_emb[t].astype(np.float32, copy=False) for t in tags])
        # defensive normalize rows (cheap no-op if already unit)
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        a = a / max(np.linalg.norm(a), 1e-12)

        sims = M @ a  # (T,)
        ranked = sorted(zip(tags, sims), key=lambda x: x[1], reverse=True)

        top_n = max(1, int(0.1 * len(ranked)))
        ranked = ranked[:top_n]
        min_sim = 0.0
        positives = []

        for tag, sim in ranked:
            if sim < min_sim:
                continue
            for rid in self.tag2reviews[tag]:
                positives.append(self.rid2item_id[rid])

        logging.info(f"{aspect}, # tags={top_n} | {[x[0] for x in ranked[:10]]}")
        return set(positives)

        