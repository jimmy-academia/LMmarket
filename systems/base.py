import torch
import logging
import numpy as np

from utils import load_or_build, dumpp, loadp, dumpj, loadj
from networks.segmenter import segment_reviews
from networks.encoder import build_segment_embeddings, build_faiss_ivfpq_ip, faiss_dump, faiss_load

from networks.aspect import aspect_splitter
from sentence_transformers import SentenceTransformer

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
        self.embedder_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_path = args.clean_dir / f"embeddings_{args.dset}.pkl"
        self.embedding = load_or_build(embedding_path, dumpp, loadp, build_segment_embeddings, self.segments, self.args, embedding_path)
        index_path = args.clean_dir / f"index_{args.dset}.pkl"
        self.faiss_index = load_or_build(index_path, faiss_dump, faiss_load, build_faiss_ivfpq_ip, self.embedding)

        self.normalize = args.normalize
        self.encoder = SentenceTransformer(self.embedder_name, device=self.args.device)

    def _encode_query(self, text):
        with torch.no_grad():
            encoded = self.encoder.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True,)
        
        if isinstance(encoded, np.ndarray):
            query = encoded[0]
        else:
            query = np.array(encoded)[0]
        query = query.astype("float32", copy=False)
        return query

    def _get_top_k(self, query_vec, topk=3):
        scores = self.embedding @ query_vec
        order = np.argsort(scores)[::-1][:topk]
        return scores, order


    def spellfix(self, text):
        return correct_spelling(self.symspell, text)

    def get_review_segments(self, review_id):
        return self.review_segments.get(review_id, [])

    def get_segment(self, segment_id):
        return self.segment_lookup.get(segment_id)

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

