import torch
import logging
import numpy as np

from utils import load_or_build, dumpp, loadp, dumpj, loadj
from networks.segmenter import segment_reviews
from networks.encoder import get_text_encoder

from networks.aspect import aspect_splitter
from tqdm import tqdm
class BaseSystem:
    '''
    provides
    self.reviews [list of reviews (dict)]
    self.segments [list of segments (dict)]
    self.embedding (segments)
    '''
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.id2reviews = data["reviews"]
        self.reviews = []
        for rev_id, review in self.id2reviews.items():
            review['review_id'] = rev_id
            self.reviews.append(review)
        
        self.segment_batch_size = 32
        self.flush_size = 10240
        self.top_k = args.top_k
        self.retrieve_k = 500
        
        # % --- segment any text segmentation ---
        segment_path = args.clean_dir / f"segments_{args.dset}.pkl"
        segment_payload = load_or_build(segment_path, dumpp, loadp, segment_reviews, self.reviews, self.segment_batch_size)
        self.segments = segment_payload["segments"]
        self.segment_lookup = segment_payload["segment_lookup"]
        self.review_segments = segment_payload["review_segments"] 
        self.item_segments = segment_payload["item_segments"] 
        self.item_reviews = segment_payload["item_reviews"] 

        # % --- embedding ---
        self.encoder = get_text_encoder(self.args.encoder_name, self.args.device)

        self.embedding_path = args.clean_dir / f"embeddings_{args.dset}_{args.enc}.pkl"
        self.batch_size = 32
        if self.args.enc == 'mini':
            self.batch_size = 1024
        self.embedding = load_or_build(self.embedding_path, dumpp, loadp, self.build_segment_embeddings)
        # % --- faiss ---
        # index_path = args.clean_dir / f"index_{args.dset}.pkl"
        # self.faiss_index = load_or_build(index_path, faiss_dump, faiss_load, build_faiss_ivfpq_ip, self.embedding)

        # self.normalize = args.normalize

    def _encode_query(self, text, show=False, is_query=None):
        with torch.no_grad():
            if type(text) != list:
                text = [text]
            kw = dict(
                normalize_embeddings=self.args.normalize,
                convert_to_numpy=True,
                show_progress_bar=show,
            )
            if is_query is not None:
                kw["is_query"] = False
            encoded = self.encoder.encode(text, **kw)
        return encoded

    def build_segment_embeddings(self, show_progress=True):
        
        start_i, embeddings, N = 0, [], len(self.segments)

        partial_path = self.embedding_path.with_name(self.embedding_path.name + ".partial")
        partial_save_frequency = max(N//self.batch_size//10, 10)
        if partial_path.exists():
            embeddings = loadp(partial_path)
            start_i = len(embeddings)

        it = range(start_i, N, self.batch_size)
        if show_progress: it = tqdm(it, desc=f"[encoder] from {start_i}", ncols=88)

        for i in it:
            batch = self.segments[i:i+self.batch_size]
            batch = [seg['text'] for seg in batch]
            embeddings.extend(self._encode_query(batch, is_query=False))

            if ((i-start_i)//self.batch_size+1) % partial_save_frequency == 0:
                dumpp(partial_path, embeddings)
                if show_progress:
                    it.set_postfix(note=f"saved@{len(embeddings)//self.batch_size}")

        matrix = np.asarray(embeddings, dtype="float32")
        matrix = np.ascontiguousarray(matrix)
        partial_path.unlink(missing_ok=True)
        return matrix

    def _get_top_k(self, query_vec, topk=3):
        scores = self.embedding @ query_vec
        order = np.argsort(scores)[::-1][:topk]
        return scores, order

    def rr(self, sentence):
        self.retrieve_similar_segments(sentence, 10)

    def retrieve_similar_segments(self, sentence, topk=None):
        query_vec = self._encode_query(sentence)
        query_vec = np.asarray(query_vec)[0].astype("float32", copy=False)
        limit = topk or self.top_k or 1
        if limit > len(self.segments):
            limit = len(self.segments)
        scores, order = self._get_top_k(query_vec, limit)
        results = []
        for idx in order:
            segment = self.segments[idx]
            results.append({
                "segment": segment,
                "score": float(scores[idx])
            })
            text = segment.get("text")
            if text:
                print(text)

    # ==== test =====

    def recommend(self, query):
        logging.info(query)
        test_aspect_path = 'data/test_aspect.json'
        aspect_list = load_or_build(test_aspect_path, dumpj, loadj, aspect_splitter, query)
        logging.info(aspect_list)

    def evaluate(self):
        logging.info(f'[Base] evaluating {self.args.system}')
        for item_id in self.top_items:
            logging.info(f'--- {item_id} --- >>>\n')
            for review_id in self.item_reviews[item_id]:
                print(self.id2reviews[review_id]['text']+'\n')
                input('>>> press any key for next review')
            input('>>> press any key for next item')  

