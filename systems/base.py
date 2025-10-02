from tqdm import tqdm

from symspellpy import SymSpell, Verbosity
from wtpsplit import SaT

from pathlib import Path

from utils import load_or_build, dumpp, loadp

SPECIAL_KEYS = {"test", "user_loc"}

class BaseSystem:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        test = data.get("test")
        self.test = test if test is not None else []
        user_loc = data.get("user_loc")
        self.user_loc = user_loc if user_loc is not None else {}
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
        top_k = getattr(args, "top_k", None)
        if top_k is None or top_k <= 0:
            top_k = 5
            setattr(args, "top_k", top_k)
        self.default_top_k = top_k
        batch_size = getattr(args, "segment_batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            batch_size = 32
        self.segment_batch_size = batch_size
        self.all_reviews = self._collect_all_reviews()

        symspell_path = args.cache_dir / f"symspell_{args.dset}.pkl"
        self.symspell = load_or_build(symspell_path, dumpp, loadp, self._build_symspell, self.all_reviews)
        self.segment_model = None
        segment_path = args.cache_dir / f"segments_{args.dset}.pkl"
        segment_payload = load_or_build(segment_path, dumpp, loadp, self._segment_reviews, self.all_reviews)
        self._apply_segment_data(segment_payload)

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
        items = payload.get("ITEMS") if isinstance(payload, dict) else None
        if isinstance(items, dict):
            return len(items)
        reviews = payload.get("REVIEWS") if isinstance(payload, dict) else None
        if isinstance(reviews, list):
            seen = set()
            for entry in reviews:
                if not isinstance(entry, dict):
                    continue
                item_id = entry.get("item_id")
                if not item_id:
                    item_id = entry.get("business_id")
                if item_id:
                    seen.add(item_id)
            return len(seen)
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
        if not self.segment_model:
            self.segment_model = SaT("sat-12l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        step = self.segment_batch_size if self.segment_batch_size > 0 else 32
        total = len(valid_reviews)
        for start in tqdm(range(0, total, step), ncols=88, desc="[base] _segment_reviews"):
            batch = valid_reviews[start:start + step]
            texts = [r.get("text") for r in batch]
            splits = list(self.segment_model.split(texts))
            for review, pieces in zip(batch, splits):
                rid = review.get("review_id")
                item_id = review.get("item_id")
                if not item_id:
                    item_id = review.get("business_id")
                user_id = review.get("user_id")
                collected = []
                for pos, segment in enumerate(pieces):
                    content = segment.strip()
                    if not content:
                        continue
                    seg_id = f"{rid}::{pos}" if rid else f"seg::{len(self.segments)}"
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
