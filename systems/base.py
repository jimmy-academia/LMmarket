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

    def list_cities(self):
        return list(self.city_list)

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
                request["query"] = query.strip()
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
