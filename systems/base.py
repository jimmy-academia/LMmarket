import torch
import numpy as np
import faiss
from tqdm import tqdm
import logging

from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


from symspellpy import SymSpell, Verbosity
from wtpsplit import SaT

from pathlib import Path

from utils import load_or_build, dumpp, loadp


from .encoder import Encoder
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from debug import check

class BaseSystem(Encoder):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.reviews = data['reviews']
        self._prepare_model()
        self.segment_batch_size = 32

        self.result = {}
        symspell_path = args.clean_dir / f"symspell_{args.dset}.pkl"
        self.symspell = load_or_build(symspell_path, dumpp, loadp, self._build_symspell, self.reviews)
        segment_path = args.clean_dir / f"segments_{args.dset}.pkl"
        segment_payload = load_or_build(segment_path, dumpp, loadp, self._segment_reviews, self.reviews)
        self._apply_segment_data(segment_payload)
        
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

    def _tokenize_for_spell(self, text):
        if not text:
            return []
        buffer = []
        for ch in str(text):
            if ch.isalpha():
                buffer.append(ch.lower())
            elif ch.isdigit():
                buffer.append(ch)
            else:
                buffer.append(" ")
        tokens = "".join(buffer).split()
        return [tok for tok in tokens if tok]

    def _build_symspell(self, reviews):
        if not reviews:
            return None
        counts = {}
        for review in tqdm(reviews.values(), ncols=88, desc="[base] _build_symspell"):
            try:
                text = review.get("text")
            except:
                check()

            for token in self._tokenize_for_spell(text):
                counts[token] = counts.get(token, 0) + 1
        if not counts:
            return None
        symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for token, freq in counts.items():
            symspell.create_dictionary_entry(token, freq)
        return symspell

    def _correct_spelling(self, text):
        if not text or not self.symspell:
            return text
        words = text.split()
        if not words:
            return text
        corrected = []
        for word in words:
            matches = self.symspell.lookup(word, Verbosity.CLOSEST)
            corrected.append(matches[0].term if matches else word)
        return " ".join(corrected)

    def spellfix(self, text):
        return self._correct_spelling(text)

    # % --- segment ---
    
    def _segment_reviews(self, reviews):
        segments = []
        segment_lookup = {}
        review_segments = {}
        item_segments = {}
        valid_reviews = [r for r in reviews.values() if isinstance(r, dict) and r.get("text")]
        if not valid_reviews:
            result = {
                "segments": segments,
                "segment_lookup": segment_lookup,
                "review_segments": review_segments,
                "item_segments": item_segments,
            }
            self._apply_segment_data(result)
            return result
        self.segment_model = SaT("sat-12l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        step = self.segment_batch_size 
        total = len(valid_reviews)
        for start in tqdm(range(0, total, step), ncols=88, desc="[base] _segment_reviews"):
            batch = valid_reviews[start:start + step]
            texts = [r.get("text") for r in batch]
            splits = list(self.segment_model.split(texts))
            for review, pieces in zip(batch, splits):
                rid = review.get("review_id")
                item_id = review.get("item_id")
                user_id = review.get("user_id")
                collected = []
                for pos, segment in enumerate(pieces):
                    content = segment.strip()
                    if not content:
                        continue
                    seg_id = f"{rid}::{pos}" if rid else f"seg::{len(segments)}"
                    record = {
                        "segment_id": seg_id,
                        "review_id": rid,
                        "item_id": item_id,
                        "user_id": user_id,
                        "position": pos,
                        "text": content,
                    }
                    segments.append(record)
                    segment_lookup[seg_id] = record
                    collected.append(record)
                if rid:
                    review_segments[rid] = list(collected)
                if item_id:
                    existing = item_segments.get(item_id)
                    if existing is None:
                        existing = []
                        item_segments[item_id] = existing
                    existing.extend(collected)
        result = {
            "segments": segments,
            "segment_lookup": segment_lookup,
            "review_segments": review_segments,
            "item_segments": item_segments,
        }
        return result

    def _apply_segment_data(self, payload):
        data = payload if isinstance(payload, dict) else {}
        segments_value = data.get("segments")
        segment_lookup_value = data.get("segment_lookup")
        review_segments_value = data.get("review_segments")
        item_segments_value = data.get("item_segments")
        segments = segments_value if isinstance(segments_value, list) else []
        segment_lookup = segment_lookup_value if isinstance(segment_lookup_value, dict) else {}
        review_segments = review_segments_value if isinstance(review_segments_value, dict) else {}
        item_segments = item_segments_value if isinstance(item_segments_value, dict) else {}
        self.segments = segments
        self.segment_lookup = segment_lookup
        self.review_segments = review_segments
        self.item_segments = item_segments

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
            logging.info(f"[base] encoding {len(texts)} segments")
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
            if len(batch_texts) >= 256:
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
        if index_bytes:
            self.segment_faiss_index = faiss.deserialize_index(index_bytes)
        else:
            self.segment_faiss_index = None
