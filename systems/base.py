import torch
import numpy as np
import faiss
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


from symspellpy import SymSpell, Verbosity
from wtpsplit import SaT

from pathlib import Path

from utils import load_or_build, dumpp, loadp


from .encoder import Encoder
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

SPECIAL_KEYS = {"test", "user_loc"}

class BaseSystem(Encoder):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.test = data.get("test")
        self.user_loc = data.get("user_loc")
        self._prepare_model()

        self.result = {}
        self.city_lookup = {}
        stats = []
        for key, payload in data.items():
            if key in SPECIAL_KEYS:
                continue
            if not isinstance(payload, dict):
                continue
            norm = key.strip().lower()
            if not norm or norm in self.city_lookup:
                continue
            self.city_lookup[norm] = key
            count = self._count_city_items(payload)
            stats.append((count, key))
        stats.sort(key=lambda pair: (pair[0], pair[1].strip().lower()))
        self.city_list = [name for _, name in stats]
        self.city_sizes = {name: count for count, name in stats}
        self.default_city = self.city_list[0] if self.city_list else None
        self.default_top_k = args.top_k
        self.segment_batch_size = 32
        self.all_reviews = self._collect_all_reviews()

        symspell_path = args.cache_dir / f"symspell_{args.dset}.pkl"
        self.symspell = load_or_build(symspell_path, dumpp, loadp, self._build_symspell, self.all_reviews)
        segment_path = args.cache_dir / f"segments_{args.dset}.pkl"
        segment_payload = load_or_build(segment_path, dumpp, loadp, self._segment_reviews, self.all_reviews)
        self._apply_segment_data(segment_payload)
        
        embedding_path = args.cache_dir / f"segment_embeddings_{args.dset}.pkl"
        embedding_payload = load_or_build(embedding_path, dumpp, loadp, self._build_segment_embeddings, self.segments)
        self._apply_segment_embeddings(embedding_payload)


    def dep_prepare_model(self):
        model_name = "nvidia/NV-Embed-v2"

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        word = models.Transformer(
            model_name,
            max_seq_length=256,
            config_args={"trust_remote_code": True, "return_dict": True},
            model_args={
                "quantization_config": bnb_config,
                "device_map": "auto",
                "attn_implementation": "eager",  # NV-Embed requires eager
                "trust_remote_code": True,
            },
            tokenizer_args={"trust_remote_code": True},
        )

        base = word.auto_model

        class DictToTupleWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.config = inner.config  # ST Pooling needs this

            def _norm_last_hidden(self, emb, seq_len: int):
                # Normalize to [B, L, H]
                if emb.dim() == 2:
                    # [B, H] -> [B, L, H]
                    return emb.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
                elif emb.dim() == 3:
                    # [B, L, H]
                    return emb
                elif emb.dim() == 4 and emb.size(1) == 1:
                    # [B, 1, L, H] -> [B, L, H]
                    return emb.squeeze(1).contiguous()
                else:
                    # Try to squeeze any singleton dims and re-check
                    squeezed = emb
                    for d in list(range(squeezed.dim()))[::-1]:
                        if squeezed.size(d) == 1:
                            squeezed = squeezed.squeeze(d)
                    if squeezed.dim() in (2, 3):
                        return self._norm_last_hidden(squeezed, seq_len)
                    raise RuntimeError(f"Unhandled embedding shape: {tuple(emb.shape)}")

            def forward(self, *args, **kwargs):
                # Infer sequence length from attention mask or input_ids
                attn = kwargs.get("attention_mask", None)
                if attn is not None:
                    seq_len = attn.shape[1]
                else:
                    input_ids = kwargs.get("input_ids", None)
                    seq_len = input_ids.shape[1] if input_ids is not None else 1

                out = self.inner(*args, **kwargs)
                if not isinstance(out, dict):
                    # Already a tuple/list; ST expects [0] to be [B, L, H]
                    return out

                # Prefer standard HF; otherwise handle NV-Embed outputs
                if "last_hidden_state" in out:
                    return (self._norm_last_hidden(out["last_hidden_state"], seq_len),)

                # NV-Embed variants
                for key in ("sentence_embeddings", "embeddings"):
                    if key in out:
                        return (self._norm_last_hidden(out[key], seq_len),)

                raise KeyError(f"Unexpected model outputs: {list(out.keys())}")

        word.auto_model = DictToTupleWrapper(base)

        pool = models.Pooling(
            word.get_word_embedding_dimension(),
            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
        )

        self.model = SentenceTransformer(modules=[word, pool])
        self.model.eval()



    def list_cities(self):
        return list(self.city_list)

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

    def get_city_key(self, city=None):
        if city:
            if city in self.data and city not in SPECIAL_KEYS:
                return city
            norm = city.strip().lower()
            if norm in self.city_lookup:
                return self.city_lookup[norm]
        return self.default_city

    def get_city_data(self, city=None):
        key = self.get_city_key(city)
        if not key:
            return None
        return self.data.get(key)

    def _count_city_items(self, payload):
        if not isinstance(payload, dict):
            return 0
        items = payload.get("ITEMS")
        if isinstance(items, dict):
            return len(items)
        return 0

    def _normalize_request(self, entry, index, group):
        base_id = index + 1
        prefix = group if group else "request"
        request_id = f"{prefix}_{base_id}"
        if isinstance(entry, str):
            query = entry.strip()
            if not query:
                return None
            query = self._correct_spelling(query)
            result = {
                "request_id": request_id,
                "query": query,
            }
            if group:
                result["group"] = group
            return result
        if isinstance(entry, dict):
            request = dict(entry)
            existing_id = request.get("request_id")
            if not existing_id:
                request["request_id"] = request_id
            if group and "group" not in request:
                request["group"] = group
            query = request.get("query")
            if isinstance(query, str):
                cleaned = query.strip()
                if cleaned:
                    request["query"] = self._correct_spelling(cleaned)
                else:
                    request.pop("query", None)
            query = request.get("query")
            if not query:
                return None
            topk = request.get("topk")
            if isinstance(topk, int) and topk > 0:
                request["topk"] = topk
            elif "topk" in request:
                request.pop("topk")
            return request
        return None

    def _prepare_requests(self):
        requests = []
        tests = self.test
        if isinstance(tests, list):
            for index, entry in enumerate(tests):
                normalized = self._normalize_request(entry, index, None)
                if normalized:
                    requests.append(normalized)
        elif isinstance(tests, dict):
            for group, entries in tests.items():
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    normalized = self._normalize_request(entry, index, group)
                    if normalized:
                        requests.append(normalized)
        return requests

    def evaluate(self, city=None, top_k=None):
        requests = self._prepare_requests()
        if not requests:
            print("[eval] no test requests available.")
            return
        if not hasattr(self, "recommend"):
            print("[eval] recommend() not implemented for this system.")
            return
        resolved_city = self.get_city_key(city)
        if not resolved_city:
            print("[eval] no city data available for evaluation.")
            return
        cutoff = top_k if isinstance(top_k, int) and top_k > 0 else self.default_top_k
        for request in requests:
            print('base.py', request)
            request_id = request.get("request_id")
            query = request.get("query")
            if not query:
                continue
            per_request_top_k = request.get("topk")
            if not isinstance(per_request_top_k, int) or per_request_top_k <= 0:
                per_request_top_k = cutoff
            results = self.recommend(query, city=resolved_city, top_k=per_request_top_k)
            if results is None:
                results = []
            packaged = []
            for entry in results:
                if isinstance(entry, dict):
                    item_id = entry.get("item_id")
                    if not item_id:
                        continue
                    score = entry.get("model_score")
                    evidence = entry.get("evidence")
                    if evidence is None:
                        evidence = []
                    short_excerpt = entry.get("short_excerpt")
                    full_explanation = entry.get("full_explanation")
                    packaged.append({
                        "item_id": item_id,
                        "model_score": score,
                        "evidence": evidence,
                        "short_excerpt": short_excerpt,
                        "full_explanation": full_explanation,
                    })
                else:
                    packaged.append({
                        "item_id": entry,
                        "model_score": None,
                        "evidence": [],
                        "short_excerpt": "",
                        "full_explanation": "",
                    })
            self.result[request_id] = {
                "request": {
                    "query": query,
                    "city": resolved_city,
                    "topk": per_request_top_k,
                },
                "candidates": packaged,
            }
            print(f"[eval] Request '{request_id}' in {resolved_city}")
            print(query)
            if not packaged:
                print("  (no candidates)")
                continue
            for rank, candidate in enumerate(packaged, 1):
                score = candidate.get("model_score")
                line = f"  {rank}. {candidate['item_id']}"
                if score is not None:
                    line += f" (score={float(score):.4f})"
                short_excerpt = candidate.get("short_excerpt")
                if short_excerpt:
                    line += f" — {short_excerpt}"
                print(line)

            print('pause after 1')
            break

    def _collect_all_reviews(self):
        rows = []
        for key, payload in self.data.items():
            if key in SPECIAL_KEYS:
                continue
            if not isinstance(payload, dict):
                continue
            reviews = payload.get("REVIEWS") if isinstance(payload, dict) else None
            if not isinstance(reviews, list):
                continue
            for review in reviews:
                if isinstance(review, dict):
                    rows.append(review)
        return rows

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

    def _normalize_text(self, text):
        if not text:
            return ""
        collapsed = " ".join(str(text).split())
        return collapsed.strip()

    def _compose_excerpt(self, summaries):
        if not summaries:
            return ""
        combined = " ".join(summaries)
        max_len = 160
        if len(combined) <= max_len:
            return combined
        trimmed = combined[:max_len].rstrip()
        return f"{trimmed}…"

    def _compose_explanation(self, summaries):
        if not summaries:
            return ""
        parts = []
        for idx, summary in enumerate(summaries, 1):
            parts.append(f"{idx}) {summary}")
        return " ".join(parts)

    def _build_symspell(self, reviews):
        if not reviews:
            return None
        counts = {}
        for review in tqdm(reviews, ncols=88, desc="[base] _build_symspell"):
            text = review.get("text")
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

    def _segment_reviews(self, reviews):
        segments = []
        segment_lookup = {}
        review_segments = {}
        item_segments = {}
        valid_reviews = [r for r in reviews if isinstance(r, dict) and r.get("text")]
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

    def get_review_segments(self, review_id):
        return self.review_segments.get(review_id, [])

    def get_segment(self, segment_id):
        return self.segment_lookup.get(segment_id)

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
        self.segment_faiss_index = faiss.deserialize_index(index_bytes)

    def _build_segment_embeddings(self, segments):
        usable = []
        texts = []
        for record in tqdm(segments, ncols=88, desc='[base] collect segments for embedding'):
            text = record.get("text")
            if not text:
                continue
            info = {
                "segment_id": record.get("segment_id"),
                "review_id": record.get("review_id"),
                "item_id": record.get("item_id"),
                "text": text,
            }
            usable.append(info)
            texts.append(text)
        
        # embeddings = self.model.encode(texts, batch_size=8,normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        embeddings = self._model_encode(texts)

        # batch_size = 4
        # all_embeddings = []

        # for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts", ncols=88):
            # Get the current batch of texts
            # batch_texts = texts[i:i + batch_size]
            
            # Encode the batch
            # batch_embeddings = self._model_encode(batch_texts)
            # batch_embeddings = self.model.encode(
            #     batch_texts,
            #     # batch_size is handled by our loop, so it's not needed here
            #     normalize_embeddings=True,
            #     convert_to_numpy=True
            # )
            
            # Store the results
            # all_embeddings.append(batch_embeddings)

        # embeddings = np.vstack(all_embeddings)

        matrix = np.asarray(embeddings, dtype="float32")
        for idx, vector in enumerate(matrix):
            usable[idx]["embedding"] = vector.tolist()
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        serialized = faiss.serialize_index(index)
        return {
            "entries": usable,
            "index": serialized,
            "matrix": matrix,
            "dimension": dim,
        }
