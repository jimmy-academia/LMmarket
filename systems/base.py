import torch
import numpy as np
import faiss
from tqdm import tqdm
import logging

from pathlib import Path

from utils import load_or_build, dumpp, loadp

from networks.encoder import Encoder
from networks.symspell import build_symspell, correct_spelling
from networks.segmenter import segment_reviews, apply_segment_data
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

class BaseSystem(Encoder):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.reviews = data['reviews']
        self._prepare_model()
        self.segment_batch_size = 32
        self.flush_size = 10240

        self.result = {}
        symspell_path = args.clean_dir / f"symspell_{args.dset}.pkl"
        self.symspell = load_or_build(symspell_path, dumpp, loadp, build_symspell, self.reviews)
        segment_path = args.clean_dir / f"segments_{args.dset}.pkl"
        segment_payload = load_or_build(segment_path, dumpp, loadp, segment_reviews, self.reviews, self.segment_batch_size)
        segments, segment_lookup, review_segments, item_segments = apply_segment_data(segment_payload)
        self.segments = segments
        self.segment_lookup = segment_lookup
        self.review_segments = review_segments
        self.item_segments = item_segments
        
        embedding_path = args.clean_dir / f"segment_embeddings_{args.dset}.pkl"
        self.embedding_path = embedding_path
        self.embedding_partial_path = Path(str(embedding_path) + ".partial")
        if embedding_path.exists():
            logging.info(f"[base] loading embeddings from {embedding_path}")
            embedding_payload = loadp(embedding_path)
        else:
            logging.info(f"[base] building embeddings → {embedding_path}")
            embedding_payload = self._build_segment_embeddings(self.segments)
            dumpp(embedding_path, embedding_payload)
        self._apply_segment_embeddings(embedding_payload)

        print(self.segment_embedding_matrix.shape)
        print(len(self.segments))
        print(len(self.reviews))
        print(self.data.keys())
        print([len(x) for x in self.data.values()])

    def spellfix(self, text):
        return correct_spelling(self.symspell, text)

    def get_review_segments(self, review_id):
        return self.review_segments.get(review_id, [])

    def get_segment(self, segment_id):
        return self.segment_lookup.get(segment_id)

    # % --- embedding ---

    def _build_segment_embeddings(self, segments):
        partial_path = self.embedding_partial_path
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        seen = set()
        if partial_path.exists():
            logging.info(f"[base] resuming embeddings from {partial_path}")
            cached = loadp(partial_path)
            if isinstance(cached, dict):
                stored = cached.get("entries")
                if isinstance(stored, list):
                    entries = stored
                    for entry in entries:
                        sid = entry.get("segment_id")
                        if sid:
                            seen.add(sid)

        def flush(records, texts):
            if not texts:
                return
            # logging.info(f"[base] encoding {len(texts)} segments")
            embeds = self._model_encode(texts)
            for idx, vector in enumerate(embeds):
                item = records[idx]
                item["embedding"] = vector.tolist()
                entries.append(item)
                sid = item.get("segment_id")
                if sid:
                    seen.add(sid)
            dumpp(partial_path, {"entries": entries})

        batch_records = []
        batch_texts = []
        for record in tqdm(segments, ncols=88, desc='[base] embed segments'):
            text = record.get("text")
            if not text:
                continue
            sid = record.get("segment_id")
            if sid and sid in seen:
                continue
            info = {
                "segment_id": sid,
                "review_id": record.get("review_id"),
                "item_id": record.get("item_id"),
                "text": text,
            }
            batch_records.append(info)
            batch_texts.append(text)
            if len(batch_texts) >= self.flush_size:
                flush(batch_records, batch_texts)
                batch_records = []
                batch_texts = []
        if batch_texts:
            flush(batch_records, batch_texts)

        partial_path.unlink(missing_ok=True)

        if not entries:
            return {"entries": [], "index": None, "matrix": None, "dimension": None}

        entry_map = {}
        for entry in entries:
            sid = entry.get("segment_id")
            if sid and sid not in entry_map:
                entry_map[sid] = entry
        ordered = []
        used = set()
        for record in segments:
            sid = record.get("segment_id")
            if not sid or sid in used:
                continue
            entry = entry_map.get(sid)
            if entry:
                ordered.append(entry)
                used.add(sid)
        entries = ordered
        if not entries:
            return {"entries": [], "index": None, "matrix": None, "dimension": None}

        matrix = np.asarray([entry.get("embedding") for entry in entries], dtype="float32")
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        serialized = faiss.serialize_index(index)
        return {
            "entries": entries,
            "index": serialized,
            "matrix": matrix,
            "dimension": dim,
        }

    def _apply_segment_embeddings(self, payload):
        data = payload or {}
        entries = data.get("entries")
        if entries is None:
            entries = []
        self.segment_embedding_entries = entries
        matrix = data.get("matrix")
        if matrix is None and entries:
            vectors = []
            for entry in entries:
                vector = entry.get("embedding")
                if vector is None:
                    continue
                vectors.append(vector)
            if vectors:
                matrix = np.array(vectors, dtype="float32")
        self.segment_embedding_matrix = matrix
        self.segment_embedding_dim = data.get("dimension")
        index_bytes = data.get("index")
        self.segment_faiss_index = faiss.deserialize_index( index_bytes)
