import logging

from utils import load_or_build, dumpp, loadp, dumpj, loadj
from networks.symspell import build_symspell, correct_spelling, fix_review
from networks.segmenter import segment_reviews, apply_segment_data
from networks.encoder import build_segment_embeddings, build_faiss_ivfpq_ip, faiss_dump, faiss_load

from networks.aspect import aspect_splitter

class BaseSystem:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.reviews = data["reviews"]
        self.segment_batch_size = 32
        self.flush_size = 10240

        # % --- symspell fix typo ---
        symspell_path = args.clean_dir / f"symspell_{args.dset}.pkl"
        self.symspell = load_or_build(symspell_path, dumpp, loadp, build_symspell, self.reviews)
        fixed_path = args.clean_dir / f"reviews_spellfix_{args.dset}.pkl"
        self.reviews = load_or_build(fixed_path, dumpp, loadp, fix_review, self.symspell, self.reviews)
        
        # % --- segment any text segmentation ---
        segment_path = args.clean_dir / f"segments_{args.dset}.pkl"
        segment_payload = load_or_build(segment_path, dumpp, loadp, segment_reviews, self.reviews, self.segment_batch_size)
        segments, lookup, review_segments, item_segments = apply_segment_data(segment_payload)
        self.segments = segments
        self.segment_lookup = lookup
        self.review_segments = review_segments
        self.item_segments = item_segments
        # self.segment_ids = [s["segment_id"] for s in self.segments]

        # % --- embedding ---
        embedding_path = args.clean_dir / f"embeddings_{args.dset}.pkl"
        self.embedding = load_or_build(embedding_path, dumpp, loadp, build_segment_embeddings, self.segments, self.args.device)
        index_path = args.clean_dir / f"index_{args.dset}.pkl"
        self.emb_index = load_or_build(index_path, faiss_dump, faiss_load, build_faiss_ivfpq_ip, self.embedding)

    def spellfix(self, text):
        return correct_spelling(self.symspell, text)

    def get_review_segments(self, review_id):
        return self.review_segments.get(review_id, [])

    def get_segment(self, segment_id):
        return self.segment_lookup.get(segment_id)

    # ==== test =====

    def recommend(self, query):
        print(query)
        test_aspect_path = 'data/test_aspect.json'
        aspect_list = load_or_build(test_aspect_path, dumpj, loadj, aspect_splitter, query)
        print(aspect_list)

        